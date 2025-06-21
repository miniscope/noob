# Design

The basic graph structure of a pipeline consists of

- Nodes: The discrete processing stages of the pipeline.
- Slots: The events that a node can accept
- Signals: The events that a node emits
- Dependencies: The structured relationships between the signals of one or more nodes
    and a slot of a node.

```{mermaid}
flowchart TD
    NodeA
    NodeB
    slotA1 --> NodeA
    slotA2 --> NodeA
    NodeA --> signalA1
    NodeA --> signalA2
    slotB1 --> NodeB
    slotB2 --> NodeB
    NodeB --> signalB1
    NodeB --> signalB2
    signalA1 --> Dependency
    signalA2 --> Dependency
    Dependency --> slotB1
```

The graph is executed such that, in graph topological order
- Nodes that have their dependencies satisfied are executed
- The signals of those nodes update any dependencies
- once a dependency is completed, the node becomes available for execution
- if all nodes that are available have already been run and there are still
  unsatisfied dependencies that depend on them, 
  available nodes with outstanding dependencies are run again until the pipeline completes
- once the pipeline completes, the event state is cleared and run again.

`depends` declarations in the config get materialized as a bipartite dependency
graph from a `node` to a (potentially parameterized) `node.signal` pair.
Maybe intuitively, "to call a processing node, all of the things it depends on must have happened."
And since depended-on (upstream) nodes can be called without emitting an event (returning `None`),
and dependent (downstream) nodes can have dependencies that e.g. `gather` multiple events,
we can't model dependencies at the node-node level all that well.

So say we have a tube with a source node `A` and a few sink nodes with different dependency specifications:

```{mermaid}
flowchart TD
    NodeA("NodeA")
    NodeB("NodeB")
    NodeC("NodeC")
    NodeD("NodeD")
    NodeA.signal1
    NodeA.signal2

    style NodeA stroke:#00f;

    classDef signal2 stroke:#f00;

    class NodeA.signal2,NodeA.signal2a,NodeA.signal2b signal2

    NodeA --> NodeA.signal1
    NodeA --> NodeA.signal2

    NodeA.signal1 --> NodeB
    NodeA.signal2 --> NodeB
    NodeA.signal2 -- "gather:5" --> NodeC
    NodeA.signal2 -- "gather:3" --> NodeD

```

We might have a bipartite node - event graph like this:

*the arrows should be pointing `source node -> event` and `event -> sink node`
but mermaid has piss-poor control over hierarchy*

```{mermaid}
flowchart LR
    NodeA("NodeA")
    NodeB("NodeB")
    NodeC("NodeC")
    NodeD("NodeD")
    NodeA.signal1
    NodeA.signal2
    NodeA.signal2a["NodeA.signal2[type=gather,n=5]"]
    NodeA.signal2b["NodeA.signal2[type=gather,n=3]"]

    style NodeA stroke:#00f;

    classDef signal2 stroke:#f00;

    class NodeA.signal2,NodeA.signal2a,NodeA.signal2b signal2

    NodeA -. "emits" .-> NodeA.signal1
    NodeA -. "emits" .-> NodeA.signal2
    NodeA -. "emits" .-> NodeA.signal2a
    NodeA -. "emits" .-> NodeA.signal2b

    NodeB -- "depends" --> NodeA.signal1
    NodeB -- "depends" --> NodeA.signal2
    NodeC -- "depends" --> NodeA.signal2a
    NodeD -- "depends" --> NodeA.signal2b
```

so that we can call a node, process its events,
compute the status of dependencies,
which then gives us a fresh set of nodes that we can process next.
