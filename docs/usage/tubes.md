# Tubes

```{index} tube
```

Noob tubes are collections of nodes that their connections that form a *processing graph*.

Tubes have two representations:

- A {class}`~noob.tube.TubeSpecification` - a yaml[^orjson]-serializable form 
  that specifies the abstract structure of the tube
- A {class}`~noob.tube.Tube` - the in-memory, instantiated form of a tube that contains
  instantiates {class}`~noob.tube.node.Node` objects and the rest of the machinery used to run a tube.

We encourage people to write tubes as specifications and load them rather than defining tubes programmatically,
though it is possible to do so: 
one of the goals of noob is to support portable, publishable, lockable[^comingsoon] processing graphs
that don't require a python interpreter to be able to inspect.

```{noob-tube} docs-branch
```                   

## Tube Specifications

Tube specifications consist of a few sections:

- A [header](#header) with metadata that identifies the tube
- A [nodes](#nodes) dictionary that defines the nodes and their dependencies
- An optional [input](#input) dictionary that defines runtime input to the tube
- An optional [assets](#assets) dictionary that defines data objects that can persist across multiple runs of the same tube.

## Header

The header section contains information that allows a tube specification to be identified and instantiated.

It looks like this:

```yaml
noob_id: my-tube
noob_model: noob.tube.TubeSpecification
noob_version: 0.0.1
```

Where the

- `noob_id` gives a (locally) unique identifier to my tube so that it can be loaded and referred to by other tubes
  (See [Locating Tubes](./config.md#locating-tubes))
- `noob_model` gives an absolute identifier to which pydantic model this specification describes,
  in case downstream packages want to create their own extensions to the base tube specification
- `noob_version` indicates which version of noob the spec was created with in order to support
  evolution of the spec format over time.

`noob_id` *must* be provided. 
If `noob_model` or `noob_version` are absent the first time a tube is loaded,
they will be populated automatically (see {class}`~noob.yaml.ConfigYAMLMixin` ).

If a tube with a matching is found in the [config sources](./config.md#locating-tubes),
then we can instantiate the tube from its `noob_id` like this:

```python
from noob import Tube

tube = Tube.from_specification("my-tube")
```

## Nodes

The `nodes` dictionary is the heart of a tube.

Recall the [signals and slots](nodes.md) of a node. 
We can declare a set of nodes and the connectivity between the nodes like this:

```{literalinclude} ../assets/pipelines/linear.yaml
:language: yaml
```

That looks like this:

```{noob-tube} docs-linear
```

The basic requirement of a `node` entry is the `type` specification,
which is an absolute specifier of some node function or class within some python package in the environment.
Throughout these examples, we'll assume all code is written in the root of some python package named `pkg`

### Dependencies

Connections between nodes are specified as *dependencies* (with the `depends` key) -
a target node specifies the signals from a source node to plug into its slots.
This lets tubes grow!
A node knows what it needs to run, but it shouldn't know anything about what other nodes do with its outputs,
so new nodes that consume its outputs can be added freely.

Dependencies are typically specified as a list of dictionaries that map a signal from another node to one of its slots.

So 

```yaml
depends:
- slot_a: a.value 
```

maps the `value` output of node `a` to `slot_a`. 
Or, roughly,

```python
node_b(slot_a=node_a())
```

Dependencies for positional-only arguments can be specified with a list of string dependencies,
or list of dictionaries with integers for keys. 
The following are equivalent:

```yaml
depends:
- a.value
- b.value
- c.value
```

```yaml
depends:
- 0: a.value
- 1: b.value
- 2: c.value
```

### Params

Params are static parameters that don't change, and are part of the tube's definition.

- For **function** nodes, params effectively[^literally] work like a {func}`~functools.partial`:
  the parameters are always passed to the function when it is called.
- For **class** nodes, params are passed to the class's `__init__` method.
- For **generator** nodes, params are passed when the generator is created
  (generators cannot have dependencies)

So, for example, if one wanted to fix some behavior of a functional node,
like to make the following `multiply` node become a "multiply by 2" node,
one might specify one of its slots with a param and the other with a dependency:

```python
def multiply(left: float, right: float) -> float:
    return left * right
```

```yaml
nodes:
  int_source:
    type: itertools.count
  multiply_by_2:
    type: pkg.multiply
    depends:
      - left: int_source.value
    params:
      - right: 2
```

## Input

Inputs are *variable* parameters that do change, 
either every time the tube is created or every time the tube is run.

Inputs have a `scope` ({class}`~noob.input.InputScope`) that defines when they must be passed.

- `tube` scoped inputs have the same lifespan as a tube: they are passed when creating the tube.
- `process` scoped inputs have the same lifespan as a {meth}`~noob.runner.base.TubeRunner.process` call:
  they are passed every time `process` is called.

### Tube-scoped inputs

For example, a common need for pipelines is to run some operation over files within a directory.

Say we used some file iterator node like this:

```python
from pathlib import Path
from typing import Generator

def iter_directory(path: Path) -> Generator[Path, None, None]:
    for file in path.iterdir():
        if file.is_file():
            yield file
```

Then then we could specify that operation as a tube that has a `tube`-scoped `directory` input like this:

```yaml
noob_id: process-directory

input:
  directory:
    scope: tube
    type: pathlib.Path

nodes:
  file:
    type: pkg.iter_directory
    params:
      path: input.directory
  do_something:
    type: pkg.do_something
    depends:
      - a_file: file.value
  # ...other nodes...
```

that we then would pass when instantiating the tube from its specification like

```python
tube = Tube.from_specification(
    'process-directory', 
    input = {"directory": Path("my_data")}
)
runner = SyncRunner(tube)

for result in runner.iter():
    # do something with the result...
```

### Process-scoped inputs

If instead we wanted to vary some parameter every time we ran the tube,
we could specify the input with a `process` scope. 

So say we wanted to pass a new path manually, we could do

```yaml
noob_id: process-directory-manual

input:
  path:
    scope: process
    type: pathlib.Path
```

which we would provide like

```python
tube = Tube.from_specification('process-directory-manual')
runner = SyncRunner(tube)

for path in Path(my_data).iter_dir():
    result = runner.process(path=path)
```

### Mixing Scopes, Depends, Params

The scopes of inputs can be crossed with the implicit scopes of `depends` and `params`:

- `tube`-scoped inputs may be used for params, since `params` is a tube-level specification too: 
  
  ```yaml
  input:
    a:
     scope: tube
  
  nodes:
    b:
      params:
        c: input.a
  ```

- `process`-scoped inputs may *not* be used as `params`, 
  since they are not defined at the time `params` are used!
- `tube`-scoped inputs **and** `process`-scoped inputs may be used as dependencies.

Additionally, a `tube`-scoped input may be *overridden* by an input to a `process` call - 
the most local scope prevails.

```yaml
noob_id: mixed-scope
input:
  a:
    scope: tube

nodes:
  b:
    type: return
    depends: input.a
```

```python
tube = Tube.from_specification('mixed-scope', input={'a': 'x'})
runner = SyncRunner(tube)
result = runner.process(a="y")

print(result)
# y
```

## Assets

```{tip}
See the [assets](./assets.md) documentation
```

## Returning Values

Tubes have inputs, 
and for whatever reason people usually like to "do things" with the result of a pipeline,
so they also can return values.

The `return` node is a special node 
(see: {data}`~noob.node.SPECIAL_NODES`) 
that collects events from a tube to be returned from a `process` call.
Only *one* `return` node may be defined for a tube.

The `return` node uses the structure of its `depends` block to destructure the returned values:

For a **scalar** return value, specify `depends` as a single key/value pair

```yaml
nodes:
  # ...
  return:
    type: return
    depends: a.value
```

```python
runner.process()
# 0
```

For a **list** of returned values, specify `depends` as a list of signals

```yaml
nodes:
  # ...
  return:
    type: return
    depends:
      - a.value
      - b.value
      - c.value
```

```python
runner.process()
# [0, 1, 2]
```

For a **dictionary** of returned values, specify `depends` as a list of key/value pairs:

```yaml
nodes:
  # ...
  no_broke_boys:
    type: return
    depends:
      - no: a.value
      - new: b.value
      - friends: c.value
```

```python
runner.process()
# {"no": 0, "new": 1, "friends": 2}
```

## Patterns

noob supports the basic patterns one would expect of a DAG processing library,
including cardinality manipulation operations that are necessary for 
bridging arbitrary graph computation with practical programming conventions

- [**merge**](#merge) - Combine signals from multiple dependant nodes in a single node
- [**branch**](#branch) - Split a signal from a single node to the slots of several nodes
- [**gather**](#gather) - Reduce cardinality: collect events emitted 

### Merge

Multiple signals can be merged as inputs to a single node's slots:

```{literalinclude} ../assets/pipelines/merge.yaml
:language: yaml
```

```{noob-tube} docs-merge
```

### Branch

An event from a single signal can be branched an fed to multiple nodes:

```{literalinclude} ../assets/pipelines/branch.yaml
:language: yaml
```

```{noob-tube} docs-branch
```
 
### Gather

Events from multiple rounds of calling the `process` method (or, [epochs](runner.md#epochs)) can be gathered as input to another node in two ways. Both use the special `gather` node.

#### Gather `n`

With a fixed `n` value in the `gather` node's params, the gather node collects `n` events from the depended-on node and then emits them as a list `[e1, e2, ... e_n]` to the node that depends on it

```{literalinclude} ../assets/pipelines/gather_n.yaml
:language: yaml
```

```{noob-tube} docs-gather-n
```

Where here the `letter_source` emits individual letters, the `gather` node collects 5 of them at a time, and the `concat` node joins them back together

The output of each node, per epoch, looks like:

| epoch | a   | b   | c   |
| ---   | --- | --- | --- |
| 0     | "a" |     |     |
| 1     | "b" |     |     |
| 2     | "c" |     |     |
| 3     | "d" |     |     |
| 4     | "e" | `['a', 'b', 'c', 'd', 'e']` | `'abcde'` |

#### Gather `trigger`

The `gather` node can also depend on another node's output as a `trigger`,
emitting the events it has collected since the last trigger

```{literalinclude} ../assets/pipelines/gather_dependent.yaml
:language: yaml
```

```{noob-tube} docs-gather-dependent
```

Where the `gather` node collects numbers from the `a1` count source until the `a2` "sporadic_word" node returns a value.
The `dictify` node then converts the collected numbers into a dictionary using the value of the `a2` word as a key,

so a set of runs might look like:

| epoch | a1   | a2  | b   | c   |
| ---   | ---  | --- | --- | --- |
| 0     | 0    |     |     |     |
| 1     | 1    |     |     |     |
| 2     | 2    |     |     |     |
| 3     | 3    | "electricity"     | [0, 1, 2, 3] | {"electricity": [0, 1, 2, 3]} |


### Map

```{warning}
Map has not been implemented yet!

It is a slightly more complicated problem becaue of some ambiguities that map-like specifications have
that gather-like specifications don't. 

See: [`#61`](https://github.com/miniscope/noob/issues/61), [`#29`](https://github.com/miniscope/noob/issues/29)
```

Map spreads a single, iterable event out, passing it multiple times to a given node within an epoch.
This is useful for transforming data as well as for parallelization...


### Nesting

Tubes have inputs and return values, so what is a tube but a node?

A special `tube` node allows tubes to be nested within one another:

Say you have a "child" tube like this


```{literalinclude} ../assets/pipelines/recursive_child.yaml
:language: yaml
```

```{noob-tube} docs-recursive-child
```

You can include it in some "parent" tube like this:

```{literalinclude} ../assets/pipelines/recursive_parent.yaml
:language: yaml
```

```{noob-tube} docs-recursive-parent
```




[^orjson]: Or, JSON - we don't use any of yaml's fancier features and tube specifications.
[^comingsoon]: Coming soon... locking a tube specification with a [pylock.toml](https://packaging.python.org/en/latest/specifications/pylock-toml/)
    lockfile for reproducible tubes.
[^literally]: And, at least for now, literally