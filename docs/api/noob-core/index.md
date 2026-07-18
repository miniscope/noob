(noob-core)=
# noob-core

`noob-core` is a rust crate exposed to python with [PyO3](https://pyo3.rs)
that implements the core graph modeling and scheduling routines.
These routines are (or, were) the largest source of performance overhead in noob,
and are exactly the kind of CPU-bound, instruction-intensive routines where Python becomes slow.

`noob-core`'s Scheduler and Topo Sorter are between 4x and 10x faster than their python equivalents,
and make the whole library roughly 2x as fast across the whole `process` cycle.


## Design Notes

While most of the code is a straightforward port of the prior python code,
there are some design decisions that are not obvious on their own
and are important for understanding how the package works:

### Python/Rust Border

The major cost with an extension module is the language barrier - 
if the cost of copying or converting data is greater than the time saved with the compiled code,
the extension isn't worth it.

An initial draft of `noob-core` that needed to convert `Epoch` objects in both directions
was *twice* as slow as the published version,
or, transiting between rust and python cost as much as the scheduler itself!

To keep costs low, 
we narrowed the language barrier to avoid as many string allocations on the heap as possible,
reduced all objects that need to be copied to ints,
and the only remaining allocations are the set/list expression allocations in the Python-side `update` method
(and even this can likely be removed).

The python/rust interface with the scheduler consists of three layers:

- The Rust {rust:struct}`noob_core::scheduler::Scheduler` struct,
  which is a pure-rust implementation of the scheduler - 
  it knows nothing about python or PyO3 and can be used independently without noob.
- The Rust/PyO3 {rust:struct}`noob_core::bridge::PyScheduler` is the rust-side adapter,
  that is importable in python from `noob_core.Scheduler`.
  It wraps the `Scheduler` struct.
  It is responsible for casting any data to the rust form,
  including [interning strings](#interning-strings),
  and converting rust exceptions to python exceptions.
- The Python {class}`noob.scheduler.Scheduler` object, 
  which is the python-side adapter.
  It is arguably the least necessary, and may be removed in future versions
  in favor of a purely PyO3-based class.
  It wraps the `PyScheduler` and exposes some python-specific methods,
  most notably handling rebuilding the rust scheduler on node mutations.

### Interning Strings

```{tip}
See the {rust:struct}`noob_core::item::Interner` struct
and generally the {rust:module}`noob_core::item` module
```

Strings are [relatively expensive heap allocations](https://nnethercote.github.io/perf-book/heap-allocations.html#string),
and since the rust scheduler needs to clone/copy objects when mutating the sorter,
we wouldn't be able to just borrow a reference from python.

The strategy `noob-core` takes is to keep a global [interning pool](https://en.wikipedia.org/wiki/Interning_(computer_science))
of strings used as node IDs,
mapping them to ints, and using the ints internally.
Strings from python only need to be allocated once[^item-allocation]
per python interpreter session,
and the rest are non-allocating borrows.

A global (rather than per-scheduler) interner is used for a few reasons:
- node ids are "write rarely, read often" data, where the interner entries are allocated once on tube creation,
  and they are relatively few in number, so the amount of memory held by the interner is negligible.
- Since writes are rare the risk of thread contention over the interner is low.
  Additionally, since the interner uses an [`IndexSet`](https://docs.rs/indexmap/latest/indexmap/set/struct.IndexSet.html)
  as an append-only log, readers don't need to hold the lock during writes.
- it greatly simplifies programming, especially error handling,
  where otherwise we would need to provide a mechanism everywhere for converting the ints back to strings when bubbling up to python.

Since both `node_id`s (strings) and and signals (`(string, string)` tuples) are used in the graph model,
both are interned separately, and the node_id for a signal is looked up via the interner again without allocation.

The interner is made thread safe using the combination of an [`RwLock`](https://doc.rust-lang.org/std/sync/struct.RwLock.html)
and an [`Arc`](https://doc.rust-lang.org/std/sync/struct.Arc.html).
The thing that is actually locked is the *reference* to the interner,
rather than the interner itself.

The {rust:fn}`~noob_core::item::interner` (read-only) method only acquires the lock within the function body,
returning a clone of the Arc. 
This blocks if there is an active writer holding the lock,
but doesn't block a writer acquiring it.

The {rust:fn}`~noob_core::item::interner_mut` (read-write) method acquires and holds the lock,
It makes a new mutable reference ([`Arc::make_mut`](https://doc.rust-lang.org/alloc/sync/struct.Arc.html#method.make_mut)) to the Arc,
so that if there are no reading references to the interner it can be mutated without cloning,
but does mean that the interner set will be cloned when interning a new object if a reading reference is held out.

This compromise means that under normal operation the interner has very little opportunity for deadlocking
at the expense of possibly expensive clone operations if nodes/signals are interned in non-standard ways,
e.g. mid-run of a tube.

### Epoch Design

Previously, the Epoch was a `tuple[tuple[NodeID, int], ...]` homogenous tuple of tuples.
Epochs support mapping operations, where one node emits multiple events that should all be run as if they were independent epochs in downstream nodes.
So, epochs are nested such that all epochs have some "root" epoch, 
and then potentially additional subepoch segments keyed by the node that induced them.

A dummy "tube" node id was reserved to indicate the root epoch, so a subepoch used to look like this:

```python
(("tube", 0), ("subepoch_node", 0), ("another_node", 1))
```

However, in most tubes, there are no subepochs, and the epoch was more expensive than it needed to be:

```python
(("tube", 0),)
```

rather than an integer.

Since allocating the tuples and interning the strings is costly and unnecessary for root epochs,
epochs were redesigned as either integers or inhomogenous tuples like this:

```python
root_epoch = 0
subepoch = (0, ("subepoch_node", 0))
```

and the implementation of the `Epoch` class was moved to noob core.
The new `Epoch` struct only stores interned ints,
but still has the same constructor behavior from python
(thanks to the global static interner, 
otherwise there would need to be some complex way of attaching an epoch to some scheduler's context).

Now only ints are passed between python and rust in most normal cases,
and otherwise there is only a single allocation when the epoch is created,
rather than at least 3 allocations when passing python->rust->python.


### Python->Rust->Python Roundtrip Example

So, for example, following a normal {class}`.Event` from python to rust and back again,
the data changes form like this:

The normal python event starts its life as a humble dictionary:

```python
event: Event = {
  "id": 0,
  "timestamp": datetime(...),
  "node_id": "a",
  "signal": "lyric",
  "epoch": Epoch(0),
  "value": "biggie smalls for mayor, the rap slayer"
}
```

Then within {meth}`.Scheduler.update`, it is reduced to a tuple of only the necessary items:

```python
(Epoch(0), "a", "lyric", event['value'] is MetaSignal.NoEvent)
# type: (Epoch, str, str, bool)
```

The {rust:fn}`noob_core::bridge::PyScheduler::update` method interns the `node_id` and the `(node_id, signal)`,
creating a {rust:struct}`~noob_core::bridge::UpdateEvent`

```rust
let event = UpdateEvent {
  epoch: Epoch(0),
  node: 123,
  signal: 456,
  no_event: false,
}
```

all the scheduler methods only emit {class}`.MetaEvent`s that declare when epochs end and nodes are ready to run,
so the rust scheduler returns Epochs (from, e.g. `done`, `update`, `expire`) or `Vec<(Epoch, ItemID: u32)>`
vecs (from, e.g. `get_ready`), rather than roundtripping the event.

```rust
let ended_epochs: Vec<Epoch> = scheduler.update(vec![event]);
```

which are propagated back up to the python Scheduler, rehydrated into meta events,
and combined with the passed events:

```python
def update(self, events: list[Event]) -> list[Event | MetaEvent]:
    # the previous event reduction stage
    reduced_events = ...
    
    ended_epochs = self._core.update(reduced_events)
    end_events = [self.event_maker.new_meta_event(EpochEnded, ep) for ep in ended_epochs]
    return [*events, *end_events]
```

```{toctree}
:maxdepth: 1

lib
python
bridge
epoch
exceptions
item
scheduler
sorter
```

[^item-allocation]: Well, they could be, but at the moment they are allocated when creating {rust:struct}`~noob_core::item::Item`s,
  but this is mostly an in implementation artifact from rust enums than a hard requirement.