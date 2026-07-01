# Changelog

## Upcoming

**Added**

- [`#233`](https://github.com/miniscope/noob/pull/233) -
  Preliminary representation of assets as nodes in JS viewer
- [`#237`](https://github.com/miniscope/noob/pull/237) - 
  {class}`~noob.node.Node`s now have an {attr}`~noob.node.Node.logger` property 
  that lazily instantiates a logger, if accessed.

**Changed**

- [`#235`](https://github.com/miniscope/noob/pull/235/) -
  Assets (and other specifications) now include their `id` when being dumped to json

**Fixed**

- [`#233`](https://github.com/miniscope/noob/pull/233) -
  Correctly handle nexted prefixes in JS viewer
- [`#236`](https://github.com/miniscope/noob/pull/236) -
  `context: recursive` when loading a tube specification actually recurses more than two levels
- [`#236`](https://github.com/miniscope/noob/pull/236) -
  Validation errors for tube specifications are logged rather than suppressed so it's possible to know
  what is wrong with the spec while working on it
- [`238`](https://github.com/miniscope/noob/pull/238) - 
  No more validation errors on type annotations on node process functions from the
  {class}`~noob.edge.Signal` class.

**Perf**

- [`#224`](https://github.com/miniscope/noob/pull/224) - 
  Avoid calls to `Epoch.__eq__` when unnecessary.
  6-7% performance improvement on whole runner benchmark
- [`#226`](https://github.com/miniscope/noob/pull/226) -
  Cache `Node.edges`.
  2.5% performance improvement on whole runner benchmark.
- [`#230`](https://github.com/miniscope/noob/pull/230) ([@vaishnavidesai09](https://github.com/vaishnavidesai09)) -
  Use a sets for O(1) lookups in the epoch log rather than O(n) lookups in deque.
  ~6% scheduler performance improvement.
- [`#232`](https://github.com/miniscope/noob/pull/232) -
  Use a snowflake-like identifier for event IDs rather than UUIDs,
  add helpers for consistently making events.

## v1000.*

### v1000.1.0 - 26-05-18

**Added**

- [`#195`](https://github.com/miniscope/noob/pull/195),
  [`#196`](https://github.com/miniscope/noob/pull/196) -
  `extends` keyword - allow a tube to extend other tubes!
  reuse tubes, add on new nodes, override them, and so on.
  build bigger tubes out of smaller tube fragments.
- [`#196`](https://github.com/miniscope/noob/pull/196) -
  `NoEventable[]` convenience generic that indicates that a return type can also be NoEvent.
- [`#207`](https://github.com/miniscope/noob/pull/207),
  [`#209`](https://github.com/miniscope/noob/pull/209),
  [`#214`](https://github.com/miniscope/noob/pull/214),
  [`#222`](https://github.com/miniscope/noob/pull/222) - 
  Big improvements to the display of tubes. 
  A `noob view` cli command to show a live-updating display of a tube as it is edited on disk.
  Better representations of tube specs: nested tubes, correct signals and slots from inspecting nodes,
  better use of the ELK layout engine.

**Changed**

- [`#211`](https://github.com/miniscope/noob/pull/211) - 
  Allow accessing a node's signals and slots from a classmethod,
  avoids needing to instantiate a node in order to inspect its edge properties
- [`#212`](https://github.com/miniscope/noob/pull/212) - 
  Make the `signals` and `slots` accessors have the same type: dicts rather than dicts and lists
- [`#213`](https://github.com/miniscope/noob/pull/213) - 
  An additional `NodeInfo` dictionary contains metadata about signals and slots
  derived from the combination of the node specification and the node class.

### v1000.0.1 - 26-03-15

**Fix**
- [#192](https://github.com/miniscope/noob/issues/192), 
  [#201](https://github.com/miniscope/noob/issues/201),
  [#202](https://github.com/miniscope/noob/pull/202) - 
  Support event values that can't use the `==` operator by using `is` to check for NoEvents

### v1000.0.0 - 26-03-13

First "official" beta release with all target features working :).

ok NOW the changelog officials starts since we're now releasing versions regularly.

## v0.1.*

### v0.1.0 - 25-12-09

- Start actually publishing versions.
- Begin changelog

Recent changes

- [#54](https://github.com/miniscope/noob/pull/54) - ZMQ Runner
- [#72](https://github.com/miniscope/noob/pull/72) - Make `NoEvent` a `MetaSignal` enum
- [#51](https://github.com/miniscope/noob/pull/51) - Recursive Tubes

## v0.0.*

### v0.0.9999999

```{raw} html
:file: assets/important.html
```