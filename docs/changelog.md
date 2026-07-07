# Changelog

## Upcoming

**Added**

- [`#231`](https://github.com/miniscope/noob/pull/231) -
  `stateful` is propagated from {class}`~.Node` classes back to {class}`~.NodeSpecification` s.
  `stateful` can be specified explicitly in the spec, or computed automatically by the node class.
  So we need to make sure to propagate that with the spec for e.g. the `ZMQRunner`
- [`#231`](https://github.com/miniscope/noob/pull/231) - 
  {meth}`~.Scheduler.iter_epoch` and {meth}`~.Scheduler.iter_ready` methods
  to iterate over a single epoch's nodes, or any nodes, respectively, as they are ready.
  This will largely replace the previous pattern of repeatedly calling `get_ready` 
  within a `while scheduler.is_active()` loop.
- [`#231`](https://github.com/miniscope/noob/pull/231) - 
  A special `('meta', 'previous_epoch')` signal is added to {class}`.TopoSorter`s
  to model statefulness and inter-epoch dependencies within the graph directly.
  Nodes that are `stateful` will depend on this signal, 
  which is marked as done when the previous epoch completes.
  This ensures that nodes that must be run with ordered inputs are in all contexts,
  and dramatically simplifies scheduling while also making concurrent runners
  naturally to handle multiple epochs simultaneously with stateless tubes.
- [`#231`](https://github.com/miniscope/noob/pull/231) -
  A {attr}`.Scheduler.epoch` property was added to get the current epoch.
- [`#231`](https://github.com/miniscope/noob/pull/231) -
  {meth}`.TopoSorter.get_state` convenience method to get a summary of a topo sorter's state
  (for debugging purposes).
  This copies the internal sets to protect them from mutation,
  and so shouldn't be used in any perf-sensitive paths.
- [`#233`](https://github.com/miniscope/noob/pull/233) -
  Preliminary representation of assets as nodes in JS viewer
- [`#234`](https://github.com/miniscope/noob/pull/234) - 
  Inputs can be `optional` and have a `default`. 
  These defaults have to be basic yaml types at the moment,
  more elaborate types are still TODO.
- [`#234`](https://github.com/miniscope/noob/pull/234) -
  Nodes can use a tube-scoped `input` to control their enabledness
- [`#234`](https://github.com/miniscope/noob/pull/234) -
  The {class}`~noob.input.InputCollection` is now passed in the dependency-injected
  {class}`~noob.types.RunnerContext`
- [`#237`](https://github.com/miniscope/noob/pull/237) - 
  {class}`~noob.node.Node`s now have an {attr}`~noob.node.Node.logger` property 
  that lazily instantiates a logger, if accessed.
- [`#244`](https://github.com/miniscope/noob/issues/239) - 
  Tube nodes propagate the signals of the enclosed tube by reading the deps of its "return" node.

**Changed**

- [`#231`](https://github.com/miniscope/noob/pull/231) - 
  Moved scheduling logic out of runners and into the scheduler!
  Runners used to have to internally keep their own counters of the current/active epochs,
  which was a huge mess and a bad division of labor.
  Now the relationship between a runner and the scheduler is much less janky.
- [`#231`](https://github.com/miniscope/noob/pull/231) -
  The ZMQRunner was simplified and its terrible state management was streamlined.
  There are still some flaky test failures from threading deadlocks, but we are getting there.
- [`#231`](https://github.com/miniscope/noob/pull/231) - 
  the {class}`.TubeRunner`'s `_get_ready` method was made an iterator, yielding from the scheduler.
- [`#231`](https://github.com/miniscope/noob/pull/231) - 
  The {class}`.TopoSorter` no longer returns `NodeSignal`s as "ready".
  Instead it marks them as being "out" along with the node, returning only the node as ready.
  Previously, they could be returned with a warning in the event of a bug or an
  inconsistent scheduling state that marked a node as done without marking its signals as done.
  Signals can't be run, it makes no sense for them to be ready.
  They are always out at the same time that a node is,
  even if they can have different ending conditions
  (i.e. being marked "done" by emitting an event or "expired" by emitting `NoEvent`)
- [`#231`](https://github.com/miniscope/noob/pull/231) - 
  The {class}`.Scheduler`'s node completion methods all return a list of {class}`.MetaEvent`s
  rather than one or `None`, allowing multiple epochs to be marked as completed at once.
- [`#231`](https://github.com/miniscope/noob/pull/231) - 
  An epoch is considered completed by {meth}`.Scheduler.epoch_completed` 
  *only* when it is fully complete, rather than when no more progress can be made,
  but a completion event has not been emitted. 
  This clarifies the distinction and use between it and {meth}`.Scheduler.is_active`,
  which returns `False` when an epoch *should* be marked as completed 
  because no more progress can be made, 
  while `epoch_completed` checks if that completion event has been emitted. 
- [`#234`](https://github.com/miniscope/noob/pull/234) -
  The `type` of an input can be an arbitrary python type expression instead of an absolute identifier.
  These are still unused, but they will be used for static type checking in the future.
- [`#235`](https://github.com/miniscope/noob/pull/235/) -
  Assets (and other specifications) now include their `id` when being dumped to json

**Fixed**

- [`#231`](https://github.com/miniscope/noob/pull/231) - 
  The `ZMQRunner` now more correctly locks the state around its scheduler,
  avoiding race conditions between the command node thread and the main thread.
- [`#231`](https://github.com/miniscope/noob/pull/231) -
  Events in the parent epoch that are from a node that is *not* the one that induced subepochs
  can mark their counterparts in subepochs as expired.
  This is how it was supposed to work, and how `gather`ing works - 
  To decrease cardinality, a node emits an event in the parent.
- [`#231`](https://github.com/miniscope/noob/pull/231) -
  Empty sets are no longer created in the Scheduler's `_subepochs` dict for every epoch,
  even when there are no subepochs.
- [`#231`](https://github.com/miniscope/noob/pull/231) -
  Dynamically enabling/disabling a node with the scheduler now works in epochs that have already been scheduled,
  however this means that nodes should only be enabled/disabled in between running epochs,
  or else nodes may be re-ran in those epochs.
- [`#231`](https://github.com/miniscope/noob/pull/231) -
  Even stateless tubes run in a "best effort" epoch order -
  when a bunch of epochs are queued up out of order,
  the scheduler ensures that ready nodes are yielded in an epoch-sorted order.
- [`#231`](https://github.com/miniscope/noob/pull/231) -
  Fix!!!! a longstanding threadlocking problem in the ZMQRunner.
  Twofold: the socket iteration method never actually yielded the eventloop when it was flooeded
  with messages from an upstream node that is much faster than it.
  This meant noderunners would sometimes just never process any events because they were just receiving messages.
  Similarly, the command node would get flooded, and it would never actually call the coro used to quit it.
  So: force a context switch using sleep(0) in the socket receive iterator,
  and use a TaskGroup to more reliably force the command node to quit the thread,
  freeing the socket.
- [`#233`](https://github.com/miniscope/noob/pull/233) -
  Correctly handle nexted prefixes in JS viewer
- [`#234`](https://github.com/miniscope/noob/pull/234) -
  {class}`.node.Tube` nodes now correctly accept params in their specs
  and forward them and other inputs to the {class}`~noob.tube.Tube` they wrap.
- [`#234`](https://github.com/miniscope/noob/pull/234) -
  The {class}`.ZMQRunner` 's {class}`.zmq.node.NodeRunner` class now correctly injects
  a requested {class}`~noob.types.RunnerContext`
- [`#236`](https://github.com/miniscope/noob/pull/236) -
  `context: recursive` when loading a tube specification actually recurses more than two levels
- [`#236`](https://github.com/miniscope/noob/pull/236) -
  Validation errors for tube specifications are logged rather than suppressed so it's possible to know
  what is wrong with the spec while working on it
- [`238`](https://github.com/miniscope/noob/pull/238) - 
  No more validation errors on type annotations on node process functions from the
  {class}`~noob.edge.Signal` class.
- [`#200`](https://github.com/miniscope/noob/issues/200),
  [`#244`](https://github.com/miniscope/noob/pull/244) - 
  Allow multiple assets to be updated from the same node,
  either multiple signals to different assets or the same signal to multiple assets.
- [`#162`](https://github.com/miniscope/noob/issues/162)
  [`#239`](https://github.com/miniscope/noob/issues/239)
  [`#245`](https://github.com/miniscope/noob/issues/245) - 
  Correctly propagate NoEvents from nested tubes

**Removed**
- [`#231`](https://github.com/miniscope/noob/pull/231) - 
  {class}`.zmq.NodeRunner` 's `await_inputs` and `await_node` methods 
  were removed in favor of the scheduler iterators.

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
- [`#231`](https://github.com/miniscope/noob/pull/231) -
  Cached {attr}`.Node.signals` and {attr}`.Node.slots` as they are frequently accessed props
- [`#231`](https://github.com/miniscope/noob/pull/231) -
  More of the {class}`.TopoSorter`'s methods were adapted to be batched set operations
  rather than iterators.
- [`#231`](https://github.com/miniscope/noob/pull/231) -
  {class}`.Epoch` 's `__eq__` method was removed to avoid using it during lookups,
  using hash instead and normal tuple equality.
  Being able to `==` an integer wasn't worth the performance cost.
- [`#231`](https://github.com/miniscope/noob/pull/231) -
  {class}`.Epoch`'s creation was optimized to avoid remaking {class}`.EpochSegment`s.
- [`#231`](https://github.com/miniscope/noob/pull/231) -
  {class}`.Epoch` gained `__slots__`
- [`#232`](https://github.com/miniscope/noob/pull/232) -
  Use a snowflake-like identifier for event IDs rather than UUIDs,
  add helpers for consistently making events.

**CI**
- [`#231`](https://github.com/miniscope/noob/pull/231) -
  pytest hooks were added to set the debug logging level when re-running an action in debug mode.


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