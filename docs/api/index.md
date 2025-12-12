# API

## Main modules

Core functionality, main public interface

- [**node**](./node/index.md) - Units of a processing graph
- [**tube**](./tube.md) - A whole processing graph!
- [**runner**](./runner/index.md) - The thing that executes the processing graph

## Secondary modules

Models, mixins, and helper classes that support the main modules and are also part of the public interface

- [**asset**](./asset.md) - Static objects that can persist through graph processing cycles
- [**config**](./config.md) - Control how noob works
- [**event**](./event.md) - Data models for events
- [**input**](./input.md) - Types and collections for handling tube inputs
- [**network**](./network/index.md) - Data models and support for networked runners
- [**scheduler**](./scheduler.md) - Keeps track of which nodes should run when
- [**state**](./state.md) - Manages [assets](./asset.md)
- [**store**](./store.md) - Manages [events](./event.md)
- [**types**](./types.md) - Annotated and validating types used throughout noob

## Utility modules

Internal or limited-use tools

- [**const**](./const.md) - constants!
- [**exceptions**](./exceptions.md) - Custom exceptions and warnings raised by noob
- [**introspection**](./introspection.md) - Helpers for working with python type annotations
- [**logging**](./logging.md) - what it says
- [**testing**](./testing/index.md) - Nodes and other code useful for downstream packages testing things built with noob
- [**utils**](./utils.md) - junk drawer
- [**yaml**](./yaml.md) - Mixin for locating and parsing tube config files


```{toctree}
:maxdepth: 2
:hidden:

asset
config
const
event
exceptions
input
introspection
logging
network/index
node/index
runner/index
scheduler
state
store
testing/index
tube
types
utils
yaml
```
