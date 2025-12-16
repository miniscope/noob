# Runners

No, not Usain Bolt. These are the kind of runners that run _other_ things.
In fact, these run the {class}`~noob.tube.Tube`.
To run tubes, a runner requires a plan of how it's going to run the tube.
It also has to awaken the tube and take inventory of what's inside the tube.
Then, while the tube is running, it needs to monitor how it's doing.
Finally, after the run is over, it needs to clean everything up.

## States

A track and field runner can be in a few different states,
ranging from shooting up steroid pre-run,
to clearing the bandage after the race.

_Our_ runner, on the other hand, can also be in a few different states,
ranging from shooting up a {class}`~noob.tube.Tube`,
to clearing returns and garbage after the run.

Let's take a look at some of the more notable states that you will find a {class}`~noob.runner.base.TubeRunner` in.

1. Pre-init
2. Inited
3. Running
4. Deinited

## Synchronous Runner

Unfortunately, there actually isn't much happening "in sync" in {class}`~noob.runner.sync.SynchronousRunner`.
In contrast, `SynchronousRunner` actually does only one thing at a time.
This is called a single-threaded operation.
The word synchronous here means that every line of code "exists _in_ the same time frame."

Here, the order of operation is clearer in most cases. Let's take a look at a few examples:

```{mermaid}
flowchart LR
A --> B
A --> C
B --> D
C --> D
```

When a {class}`~noob.runner.sync.SynchronousRunner` first encounters a {class}`~noob.tube.Tube` like the above,
the first thing it does is performing a topological sort, using a {class}`~noob.scheduler.Scheduler`.

Based on this graph, all runners will start by executing node `A`.
As you can see, nodes `B` and `C` do not depend on each other.
`SynchronousRunner` will choose at random which one of the two will precede the other.
Once both `B` and `C` are fully processed, it will move onto node `D`.

This gives us the benefit of having more controllable nodes.
Additionally, we strictly deal with only one epoch at a time.
Python debuggers will probably have an easier time debugging things in this runner,
so if asynchronous operation isn't part of the core logic of your pipeline,
it could prove helpful to try running it in a {class}`~noob.runner.sync.SynchronousRunner` first.

## Asynchronous Runner