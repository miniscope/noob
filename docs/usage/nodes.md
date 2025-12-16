# Nodes

```{index} node; structure
```

The basic elements of a noob tube are **nodes.**

Nodes are just normal python functions or classes - 
they do not require any special noob-specific syntax in order to be used in a tube.

Each node has a set of **{index}`slot`s** and **{index}`signal`s**[^qt]:

- **Slots** define the names of [events](./events.md) that a node accepts as inputs
- **Signals** define the names of [events](./events.md) that a node emits

```{mermaid}
flowchart LR
  signal_a@{ shape: sm-circ }
  signal_b@{ shape: sm-circ }
  slot_a@{ shape: sm-circ }
  slot_b@{ shape: sm-circ }
  
  
  node@{ shape: rounded }
  
  signal_a -- "signal_a" --> node 
  signal_b -- "signal_b" --> node
  node -- "slot_a" --> slot_a
  node -- "slot_b" --> slot_b
```

## Function Nodes

The simplest nodes are pure functions.

The function parameters and their types define the node's slots,
and the return value annotation defines its signals.

For example, this function:

```python
def concat(left: str, right: str) -> str:
    return left + right
```

has the node structure

```{mermaid}
flowchart LR
  left@{ shape: sm-circ }
  right@{ shape: sm-circ }
  value@{ shape: sm-circ }
  
  node@{ shape: rounded, label: "concat" }
  
  left -- "left" --> node 
  right -- "right" --> node
  node -- "value" --> value
```

The special signal name `value` is used whenever a node doesn't annotate its return type with a name.

### Named Signals

To give the signal a name, we can use the special annotation {class}`~noob.types.Name` :

```python
from typing import Annotated as A
from noob import Name

def concat(left: str, right: str) -> A[str, Name("catted")]:
    return left + right
```

```{mermaid}
flowchart LR
  left@{ shape: sm-circ }
  right@{ shape: sm-circ }
  value@{ shape: sm-circ }
  
  node@{ shape: rounded, label: "concat" }
  
  left -- "left" --> node 
  right -- "right" --> node
  node -- "catted" --> value
```

### Multiple Signals

Multiple signals can be emitted by returning a tuple:

```python
def concount(left: str, right: str) -> tuple[
        A[str, Name("catted")],
        A[int, Name("count")]
    ]:
    catted = left + right
    return catted, len(catted)
```

```{mermaid}
flowchart LR
  left@{ shape: sm-circ }
  right@{ shape: sm-circ }
  catted@{ shape: sm-circ }
  count@{ shape: sm-circ }
  
  node@{ shape: rounded, label: "concount" }
  
  left -- "left" --> node 
  right -- "right" --> node
  node -- "catted" --> catted
  node -- "count" --> count
```

See also [Optional Events](./events.md#optional-events) for how to emit events for only one (or zero) signals
and how `None`s are handled.

### Positional Slots

Positional-only arguments are named with integers that indicate their position,
for example a function using `args` could accept any (contiguous) set of integer-named events:

```python
def concat_all(*args) -> str:
    return ''.join(args)
```

```{mermaid}
flowchart LR
  zero@{ shape: sm-circ }
  one@{ shape: sm-circ }
  n@{ shape: sm-circ }
  value@{ shape: sm-circ }
  
  node@{ shape: rounded, label: "concat_all" }
  
  zero -- "0" --> node 
  one -- "1" --> node
  n -- "...{n}" --> node
  node -- "value" --> value
```

## Class Nodes

Some nodes are stateful! 
Some nodes are classes!

The simplest class node is a bare python class with a `process` method - 
no inheritance needed:

```python
class RollingSum:
    def __init__(self, x: int = 0):
        self.x = x
        
    def process(self, value: int) -> int:
        self.x += value
        return self.x
```

This node would be initialized with some `param` or `input` in the [tube specification](./tubes.md#tube-specifications)
when starting to run the tube,
and then would have its `process` method called in each iteration of the tube. 

```{admonition} Wrapper Nodes
:class: hint

When created in a tube, non-{class}`~noob.node.base.Node` nodes are wrapped
as two special classes:

- {class}`~noob.node.base.WrapClassNode` for classes
- {class}`~noob.node.base.WrapFuncNode` for functions
```


### Non-`process` processing methods

If the class's processing method doesnt happen to be named process, 
it can be decorated with {func}`~noob.node.base.process_method`:

```python
from noob import process_method

class RollingSum:
    def __init__(self, x: int = 0):
        self.x = x
        
    @process_method
    def add(self, value: int) -> int:
        self.x += value
        return self.x
```

### Class Lifespan Events

Subclassing the {class}`~noob.node.base.Node` class gives the most control over how a node is used within a tube.

Specifically, nodes have two lifespan methods that allow them to perform some actions or change their state
when [runners stop and start processing events](./runners.md#states).

So, say you had some tube that you only ran sporadically but didn't want to have to setup and teardown each time,
you might make some database accessing node open a connection only when the tube is processing:

```python
from noob import Node

class DBNode(Node):
    def __init__(self, db_address):
        self.db_address = db_address
        self._connection = None
        
    def init(self) -> None:
        self._connection = SomeDBClass(self.db_address).connect()
    
    def deinit(self) -> None:
        self._connection.close()
        
    def process(self, value: str) -> str:
        return self._connection.exec("SELECT * FROM table where something = ?", value)
```

Subclassed nodes can also manipulate how they are interpreted by tubes,
e.g. by overriding their {meth}`~noob.node.base.Node.signals` property.


## Generator Nodes

Generator functions can also be used as nodes. 
They can't have any slots (for now), but can be used as sources when e.g. reading data from disk.

Generator nodes are given any configured [params](tubes.md#params) and then called once per round of processing,
so e.g. one could cause output strings to become increasingly excited with a generator node like

```python
def hype(starting_n: int = 0) -> A[str, Name("exclamation")]:
    n = starting_n
    while True:
        n += 1
        yield '!'*n
```

Which is wrapped and called roughly like

```python
class HypeWrapper(Node):
    def __init__(self, params: dict):
        self._generator = hype(**params)

    def process(self) -> A[str, Name("exclamation")]:
        return next(self._generator)
```

## Async Nodes

```{todo}
Don't you worry, they are not supported yet, but we'll implement async nodes :)
```





[^qt]: Terminology borrowed from Qt
