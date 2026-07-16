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

:::{rubric} Structs and Unions
:::

::::::{rust:struct} noob_core::bridge::PyScheduler
:index: 1
:vis: pub
:toc: struct PyScheduler
:layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"PyScheduler"},{"type":"punctuation","value":"("},{"type":"link","value":"Scheduler","target":"Scheduler"},{"type":"punctuation","value":")"}]

  :::
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
:::::
::::::
::::::{rust:struct} noob_core::bridge::PySorterState
:index: 1
:vis: pub
:toc: struct PySorterState
:layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"PySorterState"}]

  :::
  :::
:::::{rust:variable} noob_core::bridge::PySorterState::ready
:index: 2
:vis: pub
:toc: ready
:layout: [{"type":"name","value":"ready"},{"type":"punctuation","value":": "},{"type":"link","value":"FxHashSet","target":"FxHashSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"Item","target":"Item"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::bridge::PySorterState::out
:index: 2
:vis: pub
:toc: out
:layout: [{"type":"name","value":"out"},{"type":"punctuation","value":": "},{"type":"link","value":"FxHashSet","target":"FxHashSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"Item","target":"Item"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::bridge::PySorterState::done
:index: 2
:vis: pub
:toc: done
:layout: [{"type":"name","value":"done"},{"type":"punctuation","value":": "},{"type":"link","value":"FxHashSet","target":"FxHashSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"Item","target":"Item"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::bridge::PySorterState::disabled
:index: 2
:vis: pub
:toc: disabled
:layout: [{"type":"name","value":"disabled"},{"type":"punctuation","value":": "},{"type":"link","value":"FxHashSet","target":"FxHashSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"Item","target":"Item"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::bridge::PySorterState::ran
:index: 2
:vis: pub
:toc: ran
:layout: [{"type":"name","value":"ran"},{"type":"punctuation","value":": "},{"type":"link","value":"FxHashSet","target":"FxHashSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"Item","target":"Item"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::bridge::PySorterState::pending
:index: 2
:vis: pub
:toc: pending
:layout: [{"type":"name","value":"pending"},{"type":"punctuation","value":": "},{"type":"link","value":"FxHashSet","target":"FxHashSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"Item","target":"Item"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::bridge::PySorterState::npassedout
:index: 2
:vis: pub
:toc: npassedout
:layout: [{"type":"name","value":"npassedout"},{"type":"punctuation","value":": "},{"type":"link","value":"i64","target":"i64"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::bridge::PySorterState::nfinished
:index: 2
:vis: pub
:toc: nfinished
:layout: [{"type":"name","value":"nfinished"},{"type":"punctuation","value":": "},{"type":"link","value":"i64","target":"i64"}]

  :::
  :::
:::::
::::::
::::::{rust:struct} noob_core::bridge::UpdateEvent
:index: 1
:vis: pub
:toc: struct UpdateEvent
:layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"UpdateEvent"}]

  :::
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
