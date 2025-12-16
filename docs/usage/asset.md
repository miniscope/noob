# Assets

## What it is

Assets are elements within noob, who primarily identified as a Directed Acyclic Graph (DAG) processor, that gives
it an ability to handle cycles and persistence. It is, in a sense, a static node - does not process but instead holds
objects, connections, data, and whatever else you'd like to persist longer than a node processing event. You can
determine its lifespan with the scope setting.

## Why we made it

When we have an object that needs to span multiple epochs, nodes that emit massive arrays, or when we want to define a
connection that persists through some groups of nodes, we do not want to copy the object every single time it's
passed from one node to another, or disappear when we move onto the next epoch.

## How it works

A runner-scoped asset will be able to portal between two consecutive epochs, while traveling through the nodes
within each epoch that intakes / manipulates its values.

```{mermaid}
sequenceDiagram

    activate Assets
    Assets->>Epochs: Inject
    activate Epochs
    Epochs->>Assets: Update
    deactivate Epochs
    Assets->>Epochs: Inject
    activate Epochs
    Epochs->>Assets: Update
    deactivate Epochs
    Assets->>Epochs: Inject
    activate Epochs
    Epochs->>Assets: Update
    deactivate Epochs
    deactivate Assets
```

A process-scoped asset maintains its initial state every epoch remains stateful through the nodes within each epoch.

```{mermaid}
sequenceDiagram

    Assets->>Epochs: Inject
    activate Assets
    activate Epochs
    Epochs-->Assets: No Update
    deactivate Epochs
    deactivate Assets
    Assets->>Epochs: Inject
    activate Assets
    activate Epochs
    Epochs-->Assets: No Update
    deactivate Epochs
    deactivate Assets
    Assets->>Epochs: Inject
    activate Epochs
    activate Assets
    Epochs->>Assets: Update
    deactivate Epochs
    deactivate Assets
```

A node-scoped asset serves a similar purpose to an input, whose value gets initialized at every node process.

## Spec

An Asset is specified in the YAML format spec file under the `assets` like the following:

```yaml
assets:
  db: # unique asset id
    type: noob.testing.db_connection  # absolute Python path to initializer
    scope: runner
```