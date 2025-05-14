# Notes

## Options for handling differing cardinality

- External signal

gather inputs from node 1 until we receive event from node 2,
and then call with all accumulated events.

```yaml
- node1:
- triggernode:
- node2:
    depends:
      - n2value:
          source: node1.n1value
          trigger: node1.n1trigger
          # or
          # trigger: triggernode.value
```

- Explicit number

gather n inputs from input node

In config, if it's variable within a node

```yaml
- node1:
- triggernode:
- node2:
    depends:
      - n2value:
          source: node1.n1value
          accumulate: 5         
```

In annotations if it's fixed for the node

```python
def processing_node(value: Annotated[list[int], Len(ge=5)]) -> Any: ...
```

- Callback function

Call the callback function with every event,
when the callback evaluates true, pass accumulated values

```yaml

- node1:
- node2:
    depends:
      - n2value:
          source: node1.n1value
          accumulate:
            callback: module.submodule.function
```

- Internal callback/accumulate

Always call function with input as it accumulates,
while it returns None, keep accumulating

```python
from typing import Annotated as A
def processing_node(value: list[int]) -> None | A[int, Name("hey")]:
    if len(value) > 5:
        # do something, returning the expected something
        return 10
    else:
        return None
```

which is then called by the runner like

```python
>>> node([1])
None
>>> node([1, 2])
None
>>> # ...
>>> node([1,2,3,4,5])
10
```

### Other odd cardinality cases

- Cycle input

E.g. if a node is a "config-like" node that emits some output once,
request that the same input is passed every call,
or "keep passing the last value that was emitted"

