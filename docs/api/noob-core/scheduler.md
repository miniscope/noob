# `mod scheduler`

:::::::{rust:module} noob_core::scheduler
:index: 0
:vis: pub

  :::
  :::
:::{rust:use} noob_core::scheduler
:used_name: self

:::
:::{rust:use} noob_core
:used_name: crate

:::
:::{rust:use} noob_core::bridge::UpdateEvent
:used_name: UpdateEvent

:::
:::{rust:use} noob_core::epoch::Epoch
:used_name: Epoch

:::
:::{rust:use} noob_core::exceptions::CoreError
:used_name: CoreError

:::
:::{rust:use} noob_core::exceptions::CoreResult
:used_name: CoreResult

:::
:::{rust:use} noob_core::item::Interner
:used_name: Interner

:::
:::{rust:use} noob_core::item::Item
:used_name: Item

:::
:::{rust:use} noob_core::item::ItemID
:used_name: ItemID

:::
:::{rust:use} noob_core::item::META_NODE
:used_name: META_NODE

:::
:::{rust:use} noob_core::item::PREVIOUS_EPOCH
:used_name: PREVIOUS_EPOCH

:::
:::{rust:use} noob_core::item::interner
:used_name: interner

:::
:::{rust:use} noob_core::item::interner_mut
:used_name: interner_mut

:::
:::{rust:use} noob_core::sorter::EdgeRec
:used_name: EdgeRec

:::
:::{rust:use} noob_core::sorter::NodeFlags
:used_name: NodeFlags

:::
:::{rust:use} noob_core::sorter::Sorter
:used_name: Sorter

:::
:::{rust:use} noob_core::sorter::SorterState
:used_name: SorterState

:::
:::{rust:use} noob_core::FxIndexMap
:used_name: FxIndexMap

:::
:::{rust:use} noob_core::FxIndexSet
:used_name: FxIndexSet

:::
:::{rust:use} rustc_hash::FxHashMap
:used_name: FxHashMap

:::
:::{rust:use} rustc_hash::FxHashSet
:used_name: FxHashSet

:::
:::{rust:use} std::cmp::Reverse
:used_name: Reverse

:::
:::{rust:use} std::collections::BTreeMap
:used_name: BTreeMap

:::
:::{rust:use} std::collections::BTreeSet
:used_name: BTreeSet

:::
:::{rust:use} std::iter::once
:used_name: once

:::
:::{rust:use} std::sync::Arc
:used_name: Arc

:::

:::{rubric} Functions
:::

::::::{rust:function} noob_core::scheduler::downstream_nodes
:index: 0
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"downstream_nodes"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'a"},{"type":"punctuation","value":">"},{"type":"punctuation","value":"("},{"type":"name","value":"edges"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"lifetime","value":"'a"},{"type":"space"},{"type":"punctuation","value":"["},{"type":"link","value":"EdgeRec","target":"EdgeRec"},{"type":"punctuation","value":"]"},{"type":"punctuation","value":", "},{"type":"name","value":"node"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"lifetime","value":"'a"},{"type":"space"},{"type":"link","value":"str","target":"str"},{"type":"punctuation","value":", "},{"type":"name","value":"exclude"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"FxIndexSet","target":"FxIndexSet"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"&"},{"type":"link","value":"str","target":"str"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"FxIndexSet","target":"FxIndexSet"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"&"},{"type":"lifetime","value":"'a"},{"type":"space"},{"type":"link","value":"str","target":"str"},{"type":"punctuation","value":">"}]

  :::
  Compute all the nodes downstream (that depend on) a given node.
  :::
::::::

:::{rubric} Structs and Unions
:::

::::::{rust:struct} noob_core::scheduler::EpochIter
:index: 1
:vis: pub
:toc: struct EpochIter
:layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"EpochIter"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'a"},{"type":"punctuation","value":">"}]

  :::
  Iterator for ready events within an epoch and its subepochs.
  Since the scheduler itself can't be mutably borrowed across iterations,
  creates a proxy iterator object with methods for updating the scheduler.
  
  Note that since iterating an epoch also iterates events from its subepochs,
  the epoch returned during iteration must be used when fetching the relevant inputs to run a node,
  and used when calling the `done` and `expire` methods -
  don't just recycle the epoch passed when starting iteration.
  
  # Examples
  
  ```rust
  # use noob_core::exceptions::CoreError;
  # use noob_core::scheduler::Scheduler;
  # use noob_core::FxIndexMap;
  # use noob_core::sorter::EdgeRec;
  let mut scheduler = Scheduler::from_graph(
      FxIndexMap::default(),
      vec![
          EdgeRec::from(("a", "a1", "b")),
          EdgeRec::from(("b", "b1", "c")),
      ]
  )?;
  
  let ep = scheduler.add_epoch();
  let mut batches = scheduler.iter_epoch_at(ep)?;
  while let Some(ready) = batches.next() {
      for (epoch, node) in ready {
          // in reality, actually do something
          // events = call_the_node()
          batches.done(&epoch, node, true)?;
      }
  }
  # Ok::<(), CoreError>(())
  :::

:::{rubric} Implementations
:::

:::::{rust:impl} noob_core::scheduler::EpochIter
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"EpochIter","target":"EpochIter"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":">"}]
:toc: impl EpochIter

  :::
  :::

:::{rubric} Functions
:::

::::{rust:function} noob_core::scheduler::EpochIter::done
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"done"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":", "},{"type":"name","value":"item"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":", "},{"type":"name","value":"with_signals"},{"type":"punctuation","value":": "},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"CoreResult","target":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

  :::
  Call [Scheduler.done]
  :::
::::
::::{rust:function} noob_core::scheduler::EpochIter::expire
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"expire"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":", "},{"type":"name","value":"item"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":", "},{"type":"name","value":"with_signals"},{"type":"punctuation","value":": "},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":", "},{"type":"name","value":"unlock_optionals"},{"type":"punctuation","value":": "},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"CoreResult","target":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

  :::
  Call [Scheduler.expire]
  :::
::::
:::::

:::{rubric} Traits implemented
:::

:::::{rust:impl} noob_core::scheduler::EpochIter::Iterator
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Iterator","target":"Iterator"},{"type":"space"},{"type":"keyword","value":"for"},{"type":"space"},{"type":"link","value":"EpochIter","target":"EpochIter"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":">"}]
:toc: impl Iterator for EpochIter

  :::
  :::
:::::
::::::
::::::{rust:struct} noob_core::scheduler::ReadyIter
:index: 1
:vis: pub
:toc: struct ReadyIter
:layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"ReadyIter"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'a"},{"type":"punctuation","value":">"}]

  :::
  Iterator over all nodes in any epoch that are ready.
  Ends iteration when no remaining nodes are available in any epoch,
  but since usually between calls to `next` the graph is advanced,
  callers have to implement breaking iteration themselves.
  :::

:::{rubric} Implementations
:::

:::::{rust:impl} noob_core::scheduler::ReadyIter
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"ReadyIter","target":"ReadyIter"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":">"}]
:toc: impl ReadyIter

  :::
  :::

:::{rubric} Functions
:::

::::{rust:function} noob_core::scheduler::ReadyIter::done
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"done"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":", "},{"type":"name","value":"item"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":", "},{"type":"name","value":"with_signals"},{"type":"punctuation","value":": "},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"CoreResult","target":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

  :::
  Call [Scheduler.done]
  :::
::::
::::{rust:function} noob_core::scheduler::ReadyIter::expire
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"expire"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":", "},{"type":"name","value":"item"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":", "},{"type":"name","value":"with_signals"},{"type":"punctuation","value":": "},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":", "},{"type":"name","value":"unlock_optionals"},{"type":"punctuation","value":": "},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"CoreResult","target":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

  :::
  Call [Scheduler.expire]
  :::
::::
:::::

:::{rubric} Traits implemented
:::

:::::{rust:impl} noob_core::scheduler::ReadyIter::Iterator
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Iterator","target":"Iterator"},{"type":"space"},{"type":"keyword","value":"for"},{"type":"space"},{"type":"link","value":"ReadyIter","target":"ReadyIter"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":">"}]
:toc: impl Iterator for ReadyIter

  :::
  :::
:::::
::::::
::::::{rust:struct} noob_core::scheduler::Scheduler
:index: 1
:vis: pub
:toc: struct Scheduler
:layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"Scheduler"}]

  :::
  The thing that says what nodes run, when.
  Coordinates a set of [Sorter]s, keyed by [Epoch]s.
  
  Typical use is to receive one from an instantiated (python) `Tube`,
  and then to use one of its two iteration modes in a loop with the `update` method.
  
  In general, for actions that can either take place in a specific epoch or any epoch,
  the naming convention is something like
  
  - {do_something} - in any epoch
  - {do_something}_at - in a specific, given epoch.
  
  the `_at` form is usually fallible since it's possible that the epoch has already been completed
  or is otherwise invalid for the operation,
  and the "any epoch" form is infallible and adds new epochs if necessary.
  
  # Examples
  
  ```rust
  # use noob_core::exceptions::CoreError;
  # use noob_core::scheduler::Scheduler;
  # use noob_core::FxIndexMap;
  # use noob_core::sorter::EdgeRec;
  let mut scheduler = Scheduler::from_graph(
      FxIndexMap::default(),
      vec![
          EdgeRec::from(("a", "a1", "b")),
          EdgeRec::from(("b", "b1", "c")),
      ]
  )?;
  
  let ep = scheduler.add_epoch();
  let mut batches = scheduler.iter_epoch_at(ep)?;
  while let Some(ready) = batches.next() {
      for (epoch, node) in ready {
          // in reality, call the node somehow and have events
          // events = node_calling_thing()
          // scheduler.update(events)
          // for docs, just trivially mark done
          batches.done(&epoch, node, true)?;
      }
  }
  # Ok::<(), CoreError>(())
  ```
  :::
:::::{rust:variable} noob_core::scheduler::Scheduler::graph_items
:index: 2
:vis: crate
:toc: graph_items
:layout: [{"type":"name","value":"graph_items"},{"type":"punctuation","value":": "},{"type":"link","value":"FxHashSet","target":"FxHashSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  All the nodes and signals that we care about for updates -
  i.e. all those that are present in our template sorter
  :::
:::::

:::{rubric} Implementations
:::

:::::{rust:impl} noob_core::scheduler::Scheduler
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Scheduler","target":"Scheduler"}]
:toc: impl Scheduler

  :::
  :::

:::{rubric} Functions
:::

::::{rust:function} noob_core::scheduler::Scheduler::add_epoch
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"add_epoch"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Epoch","target":"Epoch"}]

  :::
  Add the next epoch, after the last highest created root epoch
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::add_epoch_at
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"add_epoch_at"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Into","target":"Into"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"CoreResult","target":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"}]

  :::
  Add a specific epoch, can fail if the epoch has already been created or completed
  If `epoch` is not a root epoch, dispatches to [Scheduler.add_subepoch]
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::add_subepoch
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"add_subepoch"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Into","target":"Into"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"CoreResult","target":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"}]

  :::
  Add a subepoch consisting of a subgraph of the nodes that are
  downstream from the node that induced the epoch
  (the node id in the [Epoch.leaf]).
  
  The state of the graph items in the parent epoch is transferred
  to any nodes that also exist in the subepoch,
  and the node that induced the subepoch is expired in the parent.
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::done
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"done"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":", "},{"type":"name","value":"item"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":", "},{"type":"name","value":"with_signals"},{"type":"punctuation","value":": "},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"CoreResult","target":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

  :::
  Mark a node or signal as having been run to completion in a given epoch.
  If `with_signals` is passed and `item` is a node,
  and signals belonging to the node will also be marked done.
  
  Special care has to be taken when handling subepochs -
  creating subepochs by marking a node as done in a subep
  does not necessarily handle expiring the signals of the node in the parent epoch.
  Handle the parent epoch state explicitly.
  (The preferred interface for downstream use is to update the scheduler from events,
  rather than directly marking nodes done)
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::end_epoch
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"end_epoch"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Into","target":"Into"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"CoreResult","target":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

  :::
  Declare that an epoch has been completed,
  performing bookkeeping, cleanup, and emitting the epoch completion meta event in python.
  Only root epochs cause epoch sorters to be dropped -
  a parent epoch is not completed until all of its subepochs are,
  and the completion of a parent epoch means that all its subepochs are completed.
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::epoch_completed
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"epoch_completed"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"bool","target":"bool"}]

  :::
  Whether the given epoch and all of its subepochs have been completed.
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::epoch_log
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"epoch_log"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"BTreeSet","target":"BTreeSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"u32","target":"u32"},{"type":"punctuation","value":">"}]

  :::
  Clone of the internal log of epochs used to track completed epochs,
  since epochs can be completed out of order.
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::expire
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"expire"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":", "},{"type":"name","value":"item"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":", "},{"type":"name","value":"with_signals"},{"type":"punctuation","value":": "},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":", "},{"type":"name","value":"unlock_optionals"},{"type":"punctuation","value":": "},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"CoreResult","target":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

  :::
  Mark a node as having been completed without any result,
  without making its successors ready,
  effectively canceling the tree of obligate nodes beneath it.
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::first_active_epoch
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"first_active_epoch"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Option","target":"Option"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"}]

  :::
  The lowest root epoch that is active, directly or via subepochs -
  python's `iter_epoch` no-arg resolution (scheduler.py:111-120)
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::from_graph
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"from_graph"},{"type":"punctuation","value":"("},{"type":"name","value":"nodes"},{"type":"punctuation","value":": "},{"type":"link","value":"FxIndexMap","target":"FxIndexMap"},{"type":"punctuation","value":"<"},{"type":"link","value":"String","target":"String"},{"type":"punctuation","value":", "},{"type":"link","value":"NodeFlags","target":"NodeFlags"},{"type":"punctuation","value":">"},{"type":"punctuation","value":", "},{"type":"name","value":"edges"},{"type":"punctuation","value":": "},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"EdgeRec","target":"EdgeRec"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"CoreResult","target":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Scheduler","target":"Scheduler"},{"type":"punctuation","value":">"}]

  :::
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::generations
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"generations"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

  :::
  Topological generations of the sorter:
  sets of graph items that would be yielded if all of the items in the preceding generation were marked done.
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::get_epoch_state
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"get_epoch_state"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"CoreResult","target":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"SorterState","target":"SorterState"},{"type":"punctuation","value":">"}]

  :::
  Get the state of a sorter for a given epoch
  Extremely expensive, clones everything
  only to be used when debugging
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::get_ready
:index: -1
:vis: crate
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"get_ready"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"("},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":", "},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":")"},{"type":"punctuation","value":">"}]

  :::
  Get nodes that are ready in any epoch
  This and get_ready_at are not preferred calling patterns,
  as they require a continual check of epoch completion
  rather than just iterating, which also naturally closes.
  They are public only within the crate to expose them to python,
  but prefer the iterators.
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::get_ready_at
:index: -1
:vis: crate
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"get_ready_at"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"("},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":", "},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":")"},{"type":"punctuation","value":">"}]

  :::
  Get nodes that are ready in a specific epoch.
  See the note in [Scheduler.get_ready]
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::has_cycle
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"has_cycle"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"bool","target":"bool"}]

  :::
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::is_active
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"is_active"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"bool","target":"bool"}]

  :::
  Is the scheduler active in any epoch?
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::is_active_at
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"is_active_at"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"bool","target":"bool"}]

  :::
  Is the scheduler active in a specific epoch?
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::iter_epoch
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"iter_epoch"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"EpochIter","target":"EpochIter"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":">"}]

  :::
  Create an iterator for ready events in the active epoch and its subepochs
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::iter_epoch_at
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"iter_epoch_at"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Into","target":"Into"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"CoreResult","target":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"EpochIter","target":"EpochIter"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

  :::
  Create an iterator for ready events in the given epoch and its subepochs,
  creating it if it doesn't exist.
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::iter_ready
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"iter_ready"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"ReadyIter","target":"ReadyIter"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":">"}]

  :::
  Create and iterator that iterates over all ready events until there are no more left,
  in any epoch.
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::node_is_done
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"node_is_done"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"node"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"bool","target":"bool"}]

  :::
  Whether the node is done (expired or ran) in an epoch and all of its subepochs
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::node_is_ready
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"node_is_ready"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"node"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":", "},{"type":"name","value":"subepochs"},{"type":"punctuation","value":": "},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"bool","target":"bool"}]

  :::
  Check if a node is ready without marking it as out
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::source_nodes
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"source_nodes"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"FxIndexSet","target":"FxIndexSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  Any nodes that have no dependencies within the graph.
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::sources_finished
:index: -1
:vis: crate
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"sources_finished"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"bool","target":"bool"}]

  :::
  Whether all nodes that have no dependencies, or "source" nodes,
  have been finished (either expired or done) in a given epoch.
  When all the sources are completed in a (root) epoch,
  then the successor epoch is created, as that would be the earliest
  that any nodes within it could run, if stateless.
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::subepochs
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"subepochs"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"FxHashMap","target":"FxHashMap"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":", "},{"type":"link","value":"FxIndexSet","target":"FxIndexSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

  :::
  Clone of the internal mapping between an epoch and any subepochs it has (at any depth)
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::update
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"update"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"name","value":"events"},{"type":"punctuation","value":": "},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"UpdateEvent","target":"UpdateEvent"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"CoreResult","target":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

  :::
  After a node(s) emit a set of events, update the state of the scheduler,
  returning any epochs that were completed by the updates.
  
  This is the primary way that the scheduler should consume the results of processing:
  in a loop:
  - get available nodes
  - get their inputs from previously stored events
  - run them
  - store their values -> creating events
  - update the scheduler with events.
  :::
::::
::::{rust:function} noob_core::scheduler::Scheduler::upstream_nodes
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"upstream_nodes"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"node"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"CoreResult","target":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"FxHashSet","target":"FxHashSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

  :::
  The collection of nodes and signals that influence a node's readiness:
  both direct predecessors,
  and any predecessors that might reach us through a chain of optional dependencies.
  :::
::::
:::::
::::::
:::::::
