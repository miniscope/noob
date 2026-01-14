# Runners

No, not Usain Bolt. These are the kind of runners that run _other_ things.
In fact, these run the {class}`~noob.tube.Tube`.
To run tubes, a runner requires a plan of how it's going to run the tube.
It also has to awaken the tube and take inventory of what's inside the tube.
Then, while the tube is running, it needs to monitor how it's doing.
Finally, after the run is over, it needs to clean everything up.

You can grab your favorite runners from the top level of `noob`, like the following:

```python
from noob import Tube, SynchronousRunner

tube = Tube.from_specification("...")
runner = SynchronousRunner(tube=tube)
```

and start running your tube with it `n` times...

```python
results = runner.run(n=100)  # run tube 100 times
```

or use it like an _iterator_ of your tube:

```python
gen = runner.iter()

for result in gen():
    ...  # do things with result
```

or, use it like a _function_ and call it a single time:

```python
runner.init()
result = runner.process()
runner.deinit()
```

## States

A track and field runner can be in a few different states,
ranging from shooting up steroid pre-run,
to clearing the bandage post-run.

_Our_ runner, on the other hand, can also be in a few different states,
ranging from shooting up a {class}`~noob.tube.Tube` pre-run,
to clearing returns and garbage post-run.

Let's take a look at some of the more notable states that you will find a {class}`~noob.runner.base.TubeRunner` in.

1. **Pre-init**
    - The runner exists. It has accepted a `Tube`
2. **Inited**
    - The runner has run an `init` method on all nodes and assets,
      bringing them into a state that can be called if given inputs.
3. **Running**
    - The runner is picking ready nodes off the scheduler, gathering inputs for them from its EventStore,
      executing them, and storing their outputs in its EventStore for the next ready node.
4. **Deinited**
    - There may be no more ready nodes that runner can execute, or the user stopped the runner.
      Either way, all nodes and assets within the Tube has undergone a teardown process,
      reverting the runner state to the pre-init stage.

## Synchronous Runner

Unfortunately, there actually isn't much happening "in sync" in {class}`~noob.runner.sync.SynchronousRunner`.
In contrast, `SynchronousRunner` actually only ever does one thing at a time.
This is called a single-processor, single-threaded operation.
The word synchronous here would rather mean that every line of code "exists _in_ the same time frame."

Here, the order of operation is clearer. Let's take a look at a few examples:

```{mermaid}
flowchart LR
A --> B
A --> C
B --> D
C --> D
```

When a {class}`~noob.runner.sync.SynchronousRunner` first encounters a {class}`~noob.tube.Tube` like the above,
the first thing it does is performing a topological sort, using a {class}`~noob.scheduler.Scheduler`.

```{mermaid}
sequenceDiagram
participant runner
box Nodes
participant A@{ "type" : "entity" }
participant B@{ "type" : "entity" }
participant C@{ "type" : "entity" }
participant D@{ "type" : "entity" }
end

    runner ->> A: execute
    Activate A
    A --> runner: signal
    Deactivate A
    runner ->> B: execute
    Activate B
    B --> runner: signal
    Deactivate B
    runner ->> C: execute
    Activate C
    C --> runner: signal
    Deactivate C
    runner ->> D: execute
    Activate D
    D --> runner: signal
    Deactivate D
```

Based on this graph, all runners will start by executing node `A`.
As you can see, nodes `B` and `C` do not depend on each other.
`SynchronousRunner` will choose at random which one of the two will precede the other.
While one is processing, the other is simply idling.
If you'd like to multitask and process both at the same time, take a look at
[Asynchronous Runner](./runners.md#asynchronous-runner).
Once both `B` and `C` are fully processed, it will move onto node `D`.
After node `D` is finished, since the graph is complete,
we move onto the next epoch, generate another one of the graph, and repeat the process.

A strength of `SynchronousRunner` is a simpler architecture and thus having a more predictable control of the nodes.

It also is the only runner that currently supports dynamically enabling / disabling nodes:

```python
runner.enable_node(node_id="a")
runner.disable_node(node_id="a")
```

For more details on enabled / disabled nodes, refer to [Tube](./tubes.md#disabling-nodes).

Additionally, we strictly deal with only one epoch at a time.
Python debuggers will probably have an easier time debugging things in this runner,
so if asynchronous operation isn't part of the core logic of your pipeline,
it could prove helpful to try running it in a {class}`~noob.runner.sync.SynchronousRunner` first.

## Scheduler

When a runner accepts a {class}`~noob.tube.Tube`, and enters the running stage,
it needs to know which node to execute next, since some nodes depend on others.
{class}`~noob.scheduler.Scheduler` determines this order of execution.

Currently, the {class}`~noob.scheduler.Scheduler` class is a wrapper around {class}`~graphlib.TopologicalSorter`.
For each epoch, the `Scheduler` generate a new `TopologicalSorter` instance, and as runner executes the nodes,
the `Scheduler` return the next set of ready nodes to the runner. Once the ready nodes have been depleted,
the `Scheduler` marks the epoch complete.

### Epochs

The easiest, albeit oversimplified way to describe an `epoch` would be "a full cycle around a graph."
Consider the following graph:

```{mermaid}
flowchart LR
A --> B
A --> C
B --> D
C --> D
```

Here, the graph begins with `A`, and progresses through nodes `B` and `C`, and finally arrives in node `D`.
This would constitute an epoch.
However, once we introduce cardinality and multiple sources into the mix,
the definition of `epoch` starts becoming more nuanced. Take a look at the following graph:

```{mermaid}
flowchart LR
A1 --> Gather
A2 --> Gather
Gather --> C
```

Now, we have two sources, `A1` and `A2`.
Let's say `A1` fires 3 times during the span in which `A2` fires 5 times.
The `Gather` node collects 2 of `A1` and 3 of `A2`'s outputs and only then passes the group to C.
How shall we define an epoch, in this case?

In an effort to keep the definition from overextending, we define an epoch as
"when all source nodes has emitted _something_"
(without counting `NoEvent`. See [Optional Events](./events.md#optional-events).)

We recognize that this definition may end up with nonuniform epochs,
but the complexity of epoch definition can grow indefinitely
especially once asynchronous operations or undeterministic triggers join the tube.
Imagine a scenario where `A1` and `A2` are outputs from two sensors
whose sampling rates fluctuate a lot.
The output intervals between `A1` and `A2` will be also undeterministic,
making it impossible for us to define epoch around _how many_ signals the sources emitted.
The best we could come up with is _whether_ all sources have emitted anything at all,
and if all sources have, we consider those source events and all downstream events from those events an epoch.

## Asynchronous Runner

