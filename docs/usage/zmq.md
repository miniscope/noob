# ZMQ Runner

## Objects

Entities involved...

```{todo}
Jonny document this actually good
```

```{mermaid}
flowchart TD
    ZMQRunner
    CommandNode
    subgraph NodeRunners
      Node1
      Node2
      Node3
    end

    ZMQRunner -- "Spawns" --> NodeRunners
    ZMQRunner -- "Spawns" --> CommandNode

    CommandNode -- "Commands" --> NodeRunners
    Node1 -- "Event1" --> Node2
    Node2 -- "Event2" --> Node3
    Node1 --> CommandNode
    Node2 -- "Events" --> CommandNode
    Node3 --> CommandNode

```

### Routines

```{todo}
Just committing an in-progress diagram from call today,
this is not by any means final or anything. 
```

#### Instantiation

When given a tube, create a set of processes,
acquaint them with one another,
but don't do any work yet

```{mermaid}
sequenceDiagram
    participant User
    participant ZMQ as ZMQRunner
    participant Cmd as CommandRunner
    participant Node1 as NodeRunner1
    participant Node2 as NodeRunner2

    ZMQ --> Cmd: Initialize
    ZMQ --> Node1: Initialize
    ZMQ --> Node2: Initialize
    Node1 ->> Cmd: Identify
    Cmd ->> Node1: Announce
    Node2 ->> Cmd: Identify
    Cmd ->> Node1: Announce
    Cmd ->> Node2: Announce
    Node2 ->> Node1: Subscribe
```

#### Initialization

```{mermaid}
sequenceDiagram
    participant User
    participant ZMQ as ZMQRunner
    participant Cmd as CommandRunner
    participant Node1 as NodeRunner1
    participant Node2 as NodeRunner2

    ZMQ ->> Cmd: `Runner.init()`
    Cmd ->> Node1: `Node.init()`
    Cmd ->> Node2: `Node.init()`


```


#### Process

```{mermaid}
sequenceDiagram
    participant User
    participant ZMQ as ZMQRunner
    participant Cmd as CommandRunner
    participant Node1 as NodeRunner1
    participant Node2 as NodeRunner2
    
    User ->> ZMQ: `Runner.process(inputs)`
    ZMQ ->> Cmd: `Process[inputs]`
    Cmd ->> Node1: Process[inputs]
    activate Node1
    Node1 ->> Node2: Event1
    activate Node2
    Node1 ->> Cmd: Event1
    deactivate Node1
    Cmd ->> ZMQ: Callback[Event1]
    Node2 ->> Cmd: Event2
    deactivate Node2
    Cmd ->> ZMQ: Callback[Event2]
    ZMQ ->> User: Return
```

#### Free-Run

#### Deinit
