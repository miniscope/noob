# `mod sorter`

:::::::{rust:module} noob_core::sorter
:index: 0
:vis: pub

  :::
  :::
:::{rust:use} noob_core::sorter
:used_name: self

:::
:::{rust:use} noob_core
:used_name: crate

:::
:::{rust:use} rustc_hash::FxHashMap
:used_name: FxHashMap

:::
:::{rust:use} rustc_hash::FxHashSet
:used_name: FxHashSet

:::
:::{rust:use} noob_core::exceptions::CoreError
:used_name: CoreError

:::
:::{rust:use} noob_core::exceptions::CoreResult
:used_name: CoreResult

:::
:::{rust:use} noob_core::item::ASSETS_NODE
:used_name: ASSETS_NODE

:::
:::{rust:use} noob_core::item::INPUT_NODE
:used_name: INPUT_NODE

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
:::{rust:use} noob_core::item::PREVIOUS_EPOCH
:used_name: PREVIOUS_EPOCH

:::
:::{rust:use} noob_core::FxIndexMap
:used_name: FxIndexMap

:::
:::{rust:use} noob_core::FxIndexSet
:used_name: FxIndexSet

:::

:::{rubric} Structs and Unions
:::

::::::{rust:struct} noob_core::sorter::EdgeRec
:index: 1
:vis: pub
:toc: struct EdgeRec
:layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"EdgeRec"}]

  :::
  The fields of `noob.edge.Edge` the sorter cares about.
  The boundary layer is responsible for extracting these from python.
  :::
:::::{rust:variable} noob_core::sorter::EdgeRec::source_node
:index: 2
:vis: pub
:toc: source_node
:layout: [{"type":"name","value":"source_node"},{"type":"punctuation","value":": "},{"type":"link","value":"String","target":"String"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::EdgeRec::source_signal
:index: 2
:vis: pub
:toc: source_signal
:layout: [{"type":"name","value":"source_signal"},{"type":"punctuation","value":": "},{"type":"link","value":"String","target":"String"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::EdgeRec::target_node
:index: 2
:vis: pub
:toc: target_node
:layout: [{"type":"name","value":"target_node"},{"type":"punctuation","value":": "},{"type":"link","value":"String","target":"String"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::EdgeRec::required
:index: 2
:vis: pub
:toc: required
:layout: [{"type":"name","value":"required"},{"type":"punctuation","value":": "},{"type":"link","value":"bool","target":"bool"}]

  :::
  :::
:::::

:::{rubric} Traits implemented
:::

:::::{rust:impl} noob_core::sorter::EdgeRec::From
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"From","target":"From"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"link","value":"str","target":"str"},{"type":"punctuation","value":", "},{"type":"punctuation","value":"&"},{"type":"link","value":"str","target":"str"},{"type":"punctuation","value":", "},{"type":"punctuation","value":"&"},{"type":"link","value":"str","target":"str"},{"type":"punctuation","value":", "},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":")"},{"type":"punctuation","value":">"},{"type":"space"},{"type":"keyword","value":"for"},{"type":"space"},{"type":"link","value":"EdgeRec","target":"EdgeRec"}]
:toc: impl From for EdgeRec

  :::
  :::
:::::
:::::{rust:impl} noob_core::sorter::EdgeRec::From
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"From","target":"From"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"link","value":"str","target":"str"},{"type":"punctuation","value":", "},{"type":"punctuation","value":"&"},{"type":"link","value":"str","target":"str"},{"type":"punctuation","value":", "},{"type":"punctuation","value":"&"},{"type":"link","value":"str","target":"str"},{"type":"punctuation","value":")"},{"type":"punctuation","value":">"},{"type":"space"},{"type":"keyword","value":"for"},{"type":"space"},{"type":"link","value":"EdgeRec","target":"EdgeRec"}]
:toc: impl From for EdgeRec

  :::
  Default edge to required
  :::
:::::
::::::
::::::{rust:struct} noob_core::sorter::NodeFlags
:index: 1
:vis: pub
:toc: struct NodeFlags
:layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"NodeFlags"}]

  :::
  The fields of `noob.node.NodeSpecification` the sorter cares about.
  `stateful` is `bool | None` in python - `None` (unresolved) is treated
  as stateless, see [`NodeFlags::is_stateful`].
  :::
:::::{rust:variable} noob_core::sorter::NodeFlags::enabled
:index: 2
:vis: pub
:toc: enabled
:layout: [{"type":"name","value":"enabled"},{"type":"punctuation","value":": "},{"type":"link","value":"bool","target":"bool"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::NodeFlags::stateful
:index: 2
:vis: pub
:toc: stateful
:layout: [{"type":"name","value":"stateful"},{"type":"punctuation","value":": "},{"type":"link","value":"Option","target":"Option"},{"type":"punctuation","value":"<"},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::

:::{rubric} Implementations
:::

:::::{rust:impl} noob_core::sorter::NodeFlags
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"NodeFlags","target":"NodeFlags"}]
:toc: impl NodeFlags

  :::
  :::

:::{rubric} Functions
:::

::::{rust:function} noob_core::sorter::NodeFlags::is_stateful
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"is_stateful"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"bool","target":"bool"}]

  :::
  Statefulness with the `None` (unresolved) default flattened to stateless
  :::
::::
:::::
::::::
::::::{rust:struct} noob_core::sorter::NodeRec
:index: 1
:vis: pub
:toc: struct NodeRec
:layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"NodeRec"}]

  :::
  :::
:::::{rust:variable} noob_core::sorter::NodeRec::nqueue
:index: 2
:vis: pub
:toc: nqueue
:layout: [{"type":"name","value":"nqueue"},{"type":"punctuation","value":": "},{"type":"link","value":"i64","target":"i64"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::NodeRec::successors
:index: 2
:vis: pub
:toc: successors
:layout: [{"type":"name","value":"successors"},{"type":"punctuation","value":": "},{"type":"link","value":"FxIndexSet","target":"FxIndexSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::NodeRec::predecessors
:index: 2
:vis: pub
:toc: predecessors
:layout: [{"type":"name","value":"predecessors"},{"type":"punctuation","value":": "},{"type":"link","value":"FxIndexSet","target":"FxIndexSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::NodeRec::optional_predecessors
:index: 2
:vis: pub
:toc: optional_predecessors
:layout: [{"type":"name","value":"optional_predecessors"},{"type":"punctuation","value":": "},{"type":"link","value":"FxIndexSet","target":"FxIndexSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::NodeRec::optional_successors
:index: 2
:vis: pub
:toc: optional_successors
:layout: [{"type":"name","value":"optional_successors"},{"type":"punctuation","value":": "},{"type":"link","value":"FxIndexSet","target":"FxIndexSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
::::::
::::::{rust:struct} noob_core::sorter::Sorter
:index: 1
:vis: pub
:toc: struct Sorter
:layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"Sorter"}]

  :::
  Port of `noob.toposort.TopoSorter`, operating on interned item ids.
  
  Uses `IndexSet`/`IndexMap` rather than the std hash containers
  because their iteration is deterministic (reproducible scheduling and
  directly comparable test output, where std's per-instance random hash
  seeds are not) and iterates a dense vec rather than sparse hash buckets.
  These sets are iterated constantly.
  :::
:::::{rust:variable} noob_core::sorter::Sorter::signals
:index: 2
:vis: pub
:toc: signals
:layout: [{"type":"name","value":"signals"},{"type":"punctuation","value":": "},{"type":"link","value":"FxHashMap","target":"FxHashMap"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":", "},{"type":"link","value":"FxIndexSet","target":"FxIndexSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

  :::
  node item id -> signal items emitted by that node that the graph depends on
  :::
:::::
:::::{rust:variable} noob_core::sorter::Sorter::info
:index: 2
:vis: pub
:toc: info
:layout: [{"type":"name","value":"info"},{"type":"punctuation","value":": "},{"type":"link","value":"FxIndexMap","target":"FxIndexMap"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":", "},{"type":"link","value":"NodeRec","target":"NodeRec"},{"type":"punctuation","value":">"}]

  :::
  mirrors `TopoSorter._node2info`
  :::
:::::
:::::{rust:variable} noob_core::sorter::Sorter::ready
:index: 2
:vis: pub
:toc: ready
:layout: [{"type":"name","value":"ready"},{"type":"punctuation","value":": "},{"type":"link","value":"FxIndexSet","target":"FxIndexSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::Sorter::out
:index: 2
:vis: pub
:toc: out
:layout: [{"type":"name","value":"out"},{"type":"punctuation","value":": "},{"type":"link","value":"FxIndexSet","target":"FxIndexSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::Sorter::done
:index: 2
:vis: pub
:toc: done
:layout: [{"type":"name","value":"done"},{"type":"punctuation","value":": "},{"type":"link","value":"FxIndexSet","target":"FxIndexSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::Sorter::disabled
:index: 2
:vis: pub
:toc: disabled
:layout: [{"type":"name","value":"disabled"},{"type":"punctuation","value":": "},{"type":"link","value":"FxIndexSet","target":"FxIndexSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::Sorter::ran
:index: 2
:vis: pub
:toc: ran
:layout: [{"type":"name","value":"ran"},{"type":"punctuation","value":": "},{"type":"link","value":"FxIndexSet","target":"FxIndexSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::Sorter::npassedout
:index: 2
:vis: pub
:toc: npassedout
:layout: [{"type":"name","value":"npassedout"},{"type":"punctuation","value":": "},{"type":"link","value":"i64","target":"i64"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::Sorter::nfinished
:index: 2
:vis: pub
:toc: nfinished
:layout: [{"type":"name","value":"nfinished"},{"type":"punctuation","value":": "},{"type":"link","value":"i64","target":"i64"}]

  :::
  :::
:::::

:::{rubric} Implementations
:::

:::::{rust:impl} noob_core::sorter::Sorter
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Sorter","target":"Sorter"}]
:toc: impl Sorter

  :::
  :::

:::{rubric} Functions
:::

::::{rust:function} noob_core::sorter::Sorter::add
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"add"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"interner"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"link","value":"Interner","target":"Interner"},{"type":"punctuation","value":", "},{"type":"name","value":"node"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":", "},{"type":"name","value":"predecessors"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"punctuation","value":"["},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":"]"},{"type":"punctuation","value":", "},{"type":"name","value":"required"},{"type":"punctuation","value":": "},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"CoreResult","target":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"("},{"type":"punctuation","value":")"},{"type":"punctuation","value":">"}]

  :::
  :::
::::
::::{rust:function} noob_core::sorter::Sorter::clone_state
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"clone_state"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"SorterState","target":"SorterState"}]

  :::
  Get a cloned version of the internal sorter state
  This is an expensive method, since it clones... everything
  To be used when debugging
  :::
::::
::::{rust:function} noob_core::sorter::Sorter::done
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"done"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"interner"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Interner","target":"Interner"},{"type":"punctuation","value":", "},{"type":"name","value":"nodes"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"punctuation","value":"["},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":"]"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"CoreResult","target":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"("},{"type":"punctuation","value":")"},{"type":"punctuation","value":">"}]

  :::
  :::
::::
::::{rust:function} noob_core::sorter::Sorter::find_cycle
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"find_cycle"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Option","target":"Option"},{"type":"punctuation","value":"<"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

  :::
  :::
::::
::::{rust:function} noob_core::sorter::Sorter::from_graph
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"from_graph"},{"type":"punctuation","value":"("},{"type":"name","value":"interner"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"link","value":"Interner","target":"Interner"},{"type":"punctuation","value":", "},{"type":"name","value":"nodes"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"FxIndexMap","target":"FxIndexMap"},{"type":"punctuation","value":"<"},{"type":"link","value":"String","target":"String"},{"type":"punctuation","value":", "},{"type":"link","value":"NodeFlags","target":"NodeFlags"},{"type":"punctuation","value":">"},{"type":"punctuation","value":", "},{"type":"name","value":"edges"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"punctuation","value":"["},{"type":"link","value":"EdgeRec","target":"EdgeRec"},{"type":"punctuation","value":"]"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"CoreResult","target":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Sorter","target":"Sorter"},{"type":"punctuation","value":">"}]

  :::
  Port of `TopoSorter.__init__` from a node map and edge list
  :::
::::
::::{rust:function} noob_core::sorter::Sorter::get_nodeinfo
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"get_nodeinfo"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"id"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"link","value":"NodeRec","target":"NodeRec"}]

  :::
  :::
::::
::::{rust:function} noob_core::sorter::Sorter::get_ready
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"get_ready"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"interner"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Interner","target":"Interner"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  :::
::::
::::{rust:function} noob_core::sorter::Sorter::is_active
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"is_active"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"bool","target":"bool"}]

  :::
  :::
::::
::::{rust:function} noob_core::sorter::Sorter::mark_expired
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"mark_expired"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"nodes"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"punctuation","value":"["},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":"]"},{"type":"punctuation","value":", "},{"type":"name","value":"unlock_optionals"},{"type":"punctuation","value":": "},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":")"}]

  :::
  :::
::::
::::{rust:function} noob_core::sorter::Sorter::mark_out
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"mark_out"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"nodes"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"FxIndexSet","target":"FxIndexSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"}]

  :::
  :::
::::
::::{rust:function} noob_core::sorter::Sorter::mark_ready
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"mark_ready"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"nodes"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"punctuation","value":"["},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":"]"},{"type":"punctuation","value":")"}]

  :::
  :::
::::
::::{rust:function} noob_core::sorter::Sorter::resurrect
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"resurrect"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"interner"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Interner","target":"Interner"},{"type":"punctuation","value":", "},{"type":"name","value":"nodes"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"punctuation","value":"["},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":"]"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"CoreResult","target":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"("},{"type":"punctuation","value":")"},{"type":"punctuation","value":">"}]

  :::
  :::
::::
::::{rust:function} noob_core::sorter::Sorter::source_nodes
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"source_nodes"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"FxIndexSet","target":"FxIndexSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  Nodes within the graph that have no dependencies (except PREVIOUS_EPOCH)
  :::
::::
:::::
::::::
::::::{rust:struct} noob_core::sorter::SorterState
:index: 1
:vis: pub
:toc: struct SorterState
:layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"SorterState"}]

  :::
  Independent, cloned state of the sorter to be used when debugging
  :::
:::::{rust:variable} noob_core::sorter::SorterState::ready
:index: 2
:vis: pub
:toc: ready
:layout: [{"type":"name","value":"ready"},{"type":"punctuation","value":": "},{"type":"link","value":"FxHashSet","target":"FxHashSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::SorterState::out
:index: 2
:vis: pub
:toc: out
:layout: [{"type":"name","value":"out"},{"type":"punctuation","value":": "},{"type":"link","value":"FxHashSet","target":"FxHashSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::SorterState::done
:index: 2
:vis: pub
:toc: done
:layout: [{"type":"name","value":"done"},{"type":"punctuation","value":": "},{"type":"link","value":"FxHashSet","target":"FxHashSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::SorterState::disabled
:index: 2
:vis: pub
:toc: disabled
:layout: [{"type":"name","value":"disabled"},{"type":"punctuation","value":": "},{"type":"link","value":"FxHashSet","target":"FxHashSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::SorterState::ran
:index: 2
:vis: pub
:toc: ran
:layout: [{"type":"name","value":"ran"},{"type":"punctuation","value":": "},{"type":"link","value":"FxHashSet","target":"FxHashSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::SorterState::pending
:index: 2
:vis: pub
:toc: pending
:layout: [{"type":"name","value":"pending"},{"type":"punctuation","value":": "},{"type":"link","value":"FxHashSet","target":"FxHashSet"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::SorterState::npassedout
:index: 2
:vis: pub
:toc: npassedout
:layout: [{"type":"name","value":"npassedout"},{"type":"punctuation","value":": "},{"type":"link","value":"i64","target":"i64"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::sorter::SorterState::nfinished
:index: 2
:vis: pub
:toc: nfinished
:layout: [{"type":"name","value":"nfinished"},{"type":"punctuation","value":": "},{"type":"link","value":"i64","target":"i64"}]

  :::
  :::
:::::
::::::
:::::::
