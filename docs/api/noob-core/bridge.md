# `mod bridge`

:::::::{rust:module} noob_core::bridge
:index: 0
:vis: pub

  :::
  :::
:::{rust:use} noob_core::bridge
:used_name: self

:::
:::{rust:use} noob_core
:used_name: crate

:::
:::{rust:use} noob_core::FxIndexMap
:used_name: FxIndexMap

:::
:::{rust:use} noob_core::epoch::Epoch
:used_name: Epoch

:::
:::{rust:use} noob_core::exceptions::CoreError
:used_name: CoreError

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
:::{rust:use} noob_core::item::interner
:used_name: interner

:::
:::{rust:use} noob_core::scheduler::Scheduler
:used_name: Scheduler

:::
:::{rust:use} noob_core::sorter::EdgeRec
:used_name: EdgeRec

:::
:::{rust:use} noob_core::sorter::NodeFlags
:used_name: NodeFlags

:::
:::{rust:use} pyo3::exceptions::PyValueError
:used_name: PyValueError

:::
:::{rust:use} pyo3::import_exception
:used_name: import_exception

:::
:::{rust:use} rustc_hash::FxHashMap
:used_name: FxHashMap

:::
:::{rust:use} rustc_hash::FxHashSet
:used_name: FxHashSet

:::
:::{rust:use} std::collections::BTreeSet
:used_name: BTreeSet

:::
:::{rust:use} std::collections::HashSet
:used_name: HashSet

:::

:::{rubric} Enums
:::

::::::{rust:enum} noob_core::bridge::EpochArg
:index: 1
:vis: crate
:layout: [{"type":"keyword","value":"enum"},{"type":"space"},{"type":"name","value":"EpochArg"}]

  :::
  :::
:::::{rust:struct} noob_core::bridge::EpochArg::Handle
:index: 2
:vis: crate
:toc: Handle
:layout: [{"type":"name","value":"Handle"},{"type":"punctuation","value":"("},{"type":"link","value":"Py","target":"Py"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"}]

  :::
  :::
:::::
:::::{rust:struct} noob_core::bridge::EpochArg::Root
:index: 2
:vis: crate
:toc: Root
:layout: [{"type":"name","value":"Root"},{"type":"punctuation","value":"("},{"type":"link","value":"u32","target":"u32"},{"type":"punctuation","value":")"}]

  :::
  :::
:::::
::::::

:::{rubric} Structs and Unions
:::

::::::{rust:struct} noob_core::bridge::PyScheduler
:index: 1
:vis: pub
:toc: struct PyScheduler
:layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"PyScheduler"},{"type":"punctuation","value":"("},{"type":"link","value":"Scheduler","target":"Scheduler"},{"type":"punctuation","value":")"}]

  :::
  PyO3 class that bridges between the pure-rust Scheduler and its python counterpart
  Only those methods that do something beyond forwarding calls with string interning
  have public docstrings.
  For the rest, see either the Python or Rust scheduler.
  (Methods being marked public is purely a documentation detail because the PyO3 macro makes them so anyway)
  :::

:::{rubric} Implementations
:::

:::::{rust:impl} noob_core::bridge::PyScheduler
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"PyScheduler","target":"PyScheduler"}]
:toc: impl PyScheduler

  :::
  :::

:::{rubric} Functions
:::

::::{rust:function} noob_core::bridge::PyScheduler::node_is_done
:index: -1
:vis: crate
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"node_is_done"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"node"},{"type":"punctuation","value":": "},{"type":"link","value":"String","target":"String"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":">"}]

  :::
  Check if a node has been either run or expired in the given epoch
  
  Similarly to node_is_ready, raises NotAddedError if node has not been previously added.
  :::
::::
::::{rust:function} noob_core::bridge::PyScheduler::node_is_ready
:index: -1
:vis: crate
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"node_is_ready"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"node"},{"type":"punctuation","value":": "},{"type":"link","value":"String","target":"String"},{"type":"punctuation","value":", "},{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":", "},{"type":"name","value":"subepochs"},{"type":"punctuation","value":": "},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":">"}]

  :::
  Check if a node is ready in a given epoch without marking it as `out`
  
  Raises a NotAdded error if the node has not been previously added to the graph,
  rather than automatically interning:
  differentiates simply not being ready from not existing at all
  :::
::::
::::{rust:function} noob_core::bridge::PyScheduler::update
:index: -1
:vis: crate
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"update"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"events"},{"type":"punctuation","value":": "},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"("},{"type":"link","value":"EpochArg","target":"EpochArg"},{"type":"punctuation","value":", "},{"type":"link","value":"String","target":"String"},{"type":"punctuation","value":", "},{"type":"link","value":"String","target":"String"},{"type":"punctuation","value":", "},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":")"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

  :::
  Accept recast events from the python scheduler,
  intern the strings to ints,
  filter the events to only those whose signals are in the graph,
  (e.g., not disabled, etc.)
  and pass to the internal update method.
  :::
::::
:::::
::::::
::::::{rust:struct} noob_core::bridge::UpdateEvent
:index: 1
:vis: pub
:toc: struct UpdateEvent
:layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"UpdateEvent"}]

  :::
  A single event to update the scheduler state from emitted by a node.
  After interning python strings to ints and extracting other string values.
  :::
:::::{rust:variable} noob_core::bridge::UpdateEvent::epoch
:index: 2
:vis: pub
:toc: epoch
:layout: [{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"link","value":"Epoch","target":"Epoch"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::bridge::UpdateEvent::node
:index: 2
:vis: pub
:toc: node
:layout: [{"type":"name","value":"node"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::bridge::UpdateEvent::signal
:index: 2
:vis: pub
:toc: signal
:layout: [{"type":"name","value":"signal"},{"type":"punctuation","value":": "},{"type":"link","value":"Option","target":"Option"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::bridge::UpdateEvent::no_event
:index: 2
:vis: pub
:toc: no_event
:layout: [{"type":"name","value":"no_event"},{"type":"punctuation","value":": "},{"type":"link","value":"bool","target":"bool"}]

  :::
  :::
:::::
::::::
:::::::
