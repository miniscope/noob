# Assets

## What it is

Assets are elements within noob, which is primarily a Directed Acyclic Graph (DAG) processor, that give
it an ability to handle cycles and persistence. It is in a sense a static node that does not process but instead holds
objects, connections, data, and whatever else you'd like to persist longer than a node processing event. You can
determine its lifespan with the `scope` setting.

## Why we made it

When we have an object that needs to span multiple epochs, nodes that emit massive arrays, or when we want to define a
connection that persists through some groups of nodes, we do not want to copy the object every single time it's
passed from one node to another, or wipe it out when we move onto the next epoch.

## How it works

### Basics

All Python callables that outputs a stateful object can produce an asset. Usually this would take the form of a function
or a class. The way an asset persists depends on the `scope` of the asset.

### Spec

The yaml specification for an asset is almost identical to its {class}`~noob.node.base.Node` counterpart.

```yaml
asset_id:
  type: absolute.python.path
  params:
    param1: ...
  scope: runner  # or process or node
  depends: node.signal
```

`asset_id` must be unique. `type` is the absolute Python path to the callable. The others have additional nuances.

### Scopes

There are three different scopes for an asset: `runner`, `process`, and `node`.

#### Runner

A `runner`-scoped asset persists as long as the runner does. It will be able to portal between two consecutive epochs,
while remaining stateful throughout the entire run.

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

```yaml
assets:
  db: # unique asset id
    type: noob.testing.array  # absolute Python path to initializer
    scope: runner
    depends: z.result  # exits the process noob from the last node

nodes:
  a:
    type: noob.testing.row_sum
    params:
      row_index: 0
    depends:
      - right: assets.db  # enters the process loop via the first node

  # ...

  z:
    type: noob.testing.multiply
    params:
      multiplier: 2
    depends:
      array: y.output  # takes the asset directly from the previous node
```

```{mermaid}
flowchart LR
    asset_db -- "inject" --> node_a
    node_a --> ...
    ... --> node_z
    node_z -- "update" --> asset_db
    
```

##### Depends

The meaning of the `depends` entry of the asset spec is _different_ from its equivalent in {class}`~noob.node.base.Node`
and is _only_ used when `scope: runner`. Here, `depends` should point to the _last node that changes the value of the
asset_. The idea is that the `asset` enters the processing loop through the first {class}`~noob.node.base.Node` that
modifies its value, travels through the rest of the nodes, and the last node to modify it puts it back where it
came from. Therefore, an asset can only depend on a single node.

```{admonition} CAUTION
:class: caution

When running an asset in a `runner` scope, make sure the asset depends on the correct node (the last one to modify it), 
and be mindful of race-conditions, even in synchronous mode if your graph has branching / merging operations. For
example, if nodes `B`, `C` below both modify an asset in-place, it can cause a nondeterministic result by the time it 
reaches node `D`. 
```{mermaid}
flowchart LR    
  A --> B
  A --> C
  B --> D
  C --> D
```

#### Process

A `process`-scoped asset persists for the duration of a runner's {meth}`~noob.runner.Base.TubeRunner.process` method.
It is recreated on every epoch and remains stateful within that epoch.

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

Notice the missing `update` step here in contrast to the `runner` scoped asset.

```{mermaid}
flowchart LR
    asset_db -- "inject" --> node_a
    node_a --> ...
    ... --> node_z
    
```

#### Node

A node-scoped asset serves a similar purpose to an input, whose value gets initialized on every call to a node's
{meth}`~noob.node.base.Node.process` method.

```{mermaid}
flowchart LR
    asset_db -- "inject" --> node_a
    asset_db -- "inject" --> ...
    asset_db -- "inject" --> node_z
    node_a --> ...
    ... --> node_z
    
```

### Function Assets

Here's an example of an asset defined by a function:

```python
def asset_func(x, y) -> np.ndarray:
    return np.random.random((x, y))
```

The spec below declares the _output of the above function_ an asset:

```yaml
assets:
  array: # Unique asset ID
    type: my_package.assets.asset_func  # Absolute Python path
    scope: runner
    params:
      x: 2
      y: 5
    depends: # Roundtrip endpoint
      node.signal
```

The way it's scaffolded in spec is almost identical to the nodes. `array` is the unique ID (cannot duplicate with
another asset.) `type` is the absolute Python path to the function. The rest, however, diverges from the spec of
{class}`~noob.node.base.Node`. All function parameters should strictly be defined in `params` spec.

### Class Assets

Here's an example of an asset defined by a class:

```python
class AssetCls:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def some_method(self): ...
```

The spec below declares an _instance of the above class_ an asset:

```yaml
assets:
  array: # Unique asset ID
    type: my_package.assets.AssetCls  # Absolute Python path
    scope: runner
    params:
      x: 2
      y: 5
    depends: # Roundtrip endpoint
      node.signal
```

The format does not change much for a class-based Asset. Like its function-based twin, all `__init__` parameters must
be defined in the `params` section.

### How do you use an Asset?

The defined asset then can simply become a `depends` input for any node and used like any regular Python object,
like the following:

```python
def use_func_asset(
        param, event, asset: np.ndarray
) -> Annotated[np.ndarray, Name("modified_asset")]:
    asset[0][0] = asset.size + param * event
    return asset


def use_cls_asset(
        param, event, asset: AssetCls
) -> Annotated[AssetCls, Name("modified_asset")]:
    asset.x += param + event * asset.y
    return asset
```

```yaml
# just demonstrating use_cls_asset for brevity.

assets:
  array: # Unique asset ID
    type: my_package.assets.AssetCls  # Absolute Python path
    scope: runner
    params:
      x: 2
      y: 5
    depends: b.modified_asset  # Roundtrip endpoint 

nodes:
  a:
    type: my_package.nodes.constant  # a node that outputs a constant value
    params:
      value: 1
  b:
    type: my_package.nodes.use_cls_asset
    params:
      param: 2
    depends:
      - event: a.output
      - asset: assets.array
```

Here, node `b` will output an an instance of an `AssetCls` with an updated attribute `x` value `2 -> (2 + 1 * 5 = 9)`