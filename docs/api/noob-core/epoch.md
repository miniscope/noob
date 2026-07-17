# `mod epoch`

:::::::{rust:module} noob_core::epoch
:index: 0
:vis: pub

  :::
  :::
:::{rust:use} noob_core::epoch
:used_name: self

:::
:::{rust:use} noob_core
:used_name: crate

:::
:::{rust:use} noob_core::item::ItemID
:used_name: ItemID

:::
:::{rust:use} noob_core::item::TUBE_NODE
:used_name: TUBE_NODE

:::
:::{rust:use} noob_core::item::interner
:used_name: interner

:::
:::{rust:use} noob_core::item::resolve_or_intern_node
:used_name: resolve_or_intern_node

:::
:::{rust:use} pyo3::exceptions::PyIndexError
:used_name: PyIndexError

:::
:::{rust:use} pyo3::exceptions::PyValueError
:used_name: PyValueError

:::
:::{rust:use} pyo3::types::IntoPyDict
:used_name: IntoPyDict

:::
:::{rust:use} pyo3::types::PyType
:used_name: PyType

:::
:::{rust:use} std::fmt
:used_name: fmt

:::
:::{rust:use} std::iter::once
:used_name: once

:::
:::{rust:use} std::ops::Add
:used_name: Add

:::
:::{rust:use} std::ops::Div
:used_name: Div

:::
:::{rust:use} std::ops::Sub
:used_name: Sub

:::

:::{rubric} Structs and Unions
:::

::::::{rust:struct} noob_core::epoch::Epoch
:index: 1
:vis: pub
:toc: struct Epoch
:layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"Epoch"}]

  :::
  The basic unit of event alignment in Noob:
  Events emitted within the same epoch are passed to a node's slots together.
  
  Epochs are hierarchical: by default, all events exist in an integer-valued root epoch.
  However if a node like `Map` expands cardinality by emitting multiple events per event taken in,
  Epochs grow "layers" of *subepochs* labeled with the ID of the node that emitted them.
  :::
:::::{rust:variable} noob_core::epoch::Epoch::root
:index: 2
:vis: pub
:toc: root
:layout: [{"type":"name","value":"root"},{"type":"punctuation","value":": "},{"type":"link","value":"u32","target":"u32"}]

  :::
  The common, tube-level epoch
  :::
:::::
:::::{rust:variable} noob_core::epoch::Epoch::path
:index: 2
:vis: pub
:toc: path
:layout: [{"type":"name","value":"path"},{"type":"punctuation","value":": "},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"EpochSegment","target":"EpochSegment"},{"type":"punctuation","value":">"}]

  :::
  Subepoch segments induced by cardinality expanding Maplike events.
  :::
:::::

:::{rubric} Implementations
:::

:::::{rust:impl} noob_core::epoch::Epoch
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Epoch","target":"Epoch"}]
:toc: impl Epoch

  :::
  :::

:::{rubric} Functions
:::

::::{rust:function} noob_core::epoch::Epoch::child
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"child"},{"type":"punctuation","value":"("},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"subep"},{"type":"punctuation","value":": "},{"type":"link","value":"EpochSegment","target":"EpochSegment"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Epoch","target":"Epoch"}]

  :::
  :::
::::
::::{rust:function} noob_core::epoch::Epoch::is_root
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"is_root"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"bool","target":"bool"}]

  :::
  :::
::::
::::{rust:function} noob_core::epoch::Epoch::leaf
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"leaf"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Option","target":"Option"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"&"},{"type":"link","value":"EpochSegment","target":"EpochSegment"},{"type":"punctuation","value":">"}]

  :::
  The last available subepoch segment, if any.
  `None` for root epochs.
  :::
::::
::::{rust:function} noob_core::epoch::Epoch::make_subepochs
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"make_subepochs"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"node"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":", "},{"type":"name","value":"n"},{"type":"punctuation","value":": "},{"type":"link","value":"u32","target":"u32"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"}]

  :::
  Create a collection of `n` sequential subepochs induced by events from the given node
  
  # Example
  
  ```
  Epoch::from(0).make_subepochs(6, 2)
  // vec![Epoch(0) / (6, 0), Epoch(0) / (6, 1)]
  ```
  :::
::::
::::{rust:function} noob_core::epoch::Epoch::n_segments
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"n_segments"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"usize","target":"usize"}]

  :::
  1 for root epochs, 1 + subepoch segments otherwise.
  :::
::::
::::{rust:function} noob_core::epoch::Epoch::new
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"new"},{"type":"punctuation","value":"("},{"type":"name","value":"root"},{"type":"punctuation","value":": "},{"type":"link","value":"u32","target":"u32"},{"type":"punctuation","value":", "},{"type":"name","value":"path"},{"type":"punctuation","value":": "},{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"IntoIterator","target":"IntoIterator"},{"type":"punctuation","value":"<"},{"type":"name","value":"Item"},{"type":"punctuation","value":" = "},{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Into","target":"Into"},{"type":"punctuation","value":"<"},{"type":"link","value":"EpochSegment","target":"EpochSegment"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Epoch","target":"Epoch"}]

  :::
  :::
::::
::::{rust:function} noob_core::epoch::Epoch::parent
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"parent"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Option","target":"Option"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"}]

  :::
  :::
::::
::::{rust:function} noob_core::epoch::Epoch::parents
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"parents"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Iterator","target":"Iterator"},{"type":"punctuation","value":"<"},{"type":"name","value":"Item"},{"type":"punctuation","value":" = "},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"},{"type":"punctuation","value":" + "},{"type":"lifetime","value":"'_"}]

  :::
  :::
::::
::::{rust:function} noob_core::epoch::Epoch::root
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"root"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"u32","target":"u32"}]

  :::
  :::
::::
::::{rust:function} noob_core::epoch::Epoch::segments
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"segments"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Iterator","target":"Iterator"},{"type":"punctuation","value":"<"},{"type":"name","value":"Item"},{"type":"punctuation","value":" = "},{"type":"link","value":"EpochSegment","target":"EpochSegment"},{"type":"punctuation","value":">"},{"type":"punctuation","value":" + "},{"type":"lifetime","value":"'_"}]

  :::
  :::
::::
:::::
:::::{rust:impl} noob_core::epoch::Epoch
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Epoch","target":"Epoch"}]
:toc: impl Epoch

  :::
  :::
:::::

:::{rubric} Traits implemented
:::

:::::{rust:impl} noob_core::epoch::Epoch::From
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"From","target":"From"},{"type":"punctuation","value":"<"},{"type":"link","value":"u32","target":"u32"},{"type":"punctuation","value":">"},{"type":"space"},{"type":"keyword","value":"for"},{"type":"space"},{"type":"link","value":"Epoch","target":"Epoch"}]
:toc: impl From for Epoch

  :::
  ```
  Epoch::from(0)
  ```
  :::
:::::
:::::{rust:impl} noob_core::epoch::Epoch::Div
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"punctuation","value":"<"},{"type":"name","value":"T"},{"type":"punctuation","value":": "},{"type":"link","value":"Into","target":"Into"},{"type":"punctuation","value":"<"},{"type":"link","value":"EpochSegment","target":"EpochSegment"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"},{"type":"space"},{"type":"link","value":"Div","target":"Div"},{"type":"punctuation","value":"<"},{"type":"link","value":"T","target":"T"},{"type":"punctuation","value":">"},{"type":"space"},{"type":"keyword","value":"for"},{"type":"space"},{"type":"link","value":"Epoch","target":"Epoch"}]
:toc: impl Div for Epoch

  :::
  ```
  Epoch::from(0) / (1, 2)
  Epoch::from(0) / EpochSegment{ node: 1, epoch: 2 }
  ```
  :::
:::::
:::::{rust:impl} noob_core::epoch::Epoch::Add
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Add","target":"Add"},{"type":"punctuation","value":"<"},{"type":"link","value":"u32","target":"u32"},{"type":"punctuation","value":">"},{"type":"space"},{"type":"keyword","value":"for"},{"type":"space"},{"type":"link","value":"Epoch","target":"Epoch"}]
:toc: impl Add for Epoch

  :::
  Increment the lowest layer of the epoch.
  
  # Examples:
  ```
  Epoch::from(0) + 1
  // Epoch(1)
  (Epoch::from(0) / (2, 3)) + 1
  // Epoch(1, (2, 4))
  ```
  :::
:::::
:::::{rust:impl} noob_core::epoch::Epoch::Sub
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Sub","target":"Sub"},{"type":"punctuation","value":"<"},{"type":"link","value":"u32","target":"u32"},{"type":"punctuation","value":">"},{"type":"space"},{"type":"keyword","value":"for"},{"type":"space"},{"type":"link","value":"Epoch","target":"Epoch"}]
:toc: impl Sub for Epoch

  :::
  Decrement the lowest layer of the epoch.
  
  # Examples:
  ```
  Epoch::from(1) - 1
  // Epoch(0)
  (Epoch::from(0) / (2, 3)) - 1
  // Epoch(1, (2, 2))
  ```
  
  :::
:::::
:::::{rust:impl} noob_core::epoch::Epoch::Display
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"fmt","target":"fmt"},{"type":"punctuation","value":"::"},{"type":"name","value":"Display"},{"type":"space"},{"type":"keyword","value":"for"},{"type":"space"},{"type":"link","value":"Epoch","target":"Epoch"}]
:toc: impl Display for Epoch

  :::
  :::
:::::
::::::
::::::{rust:struct} noob_core::epoch::EpochSegment
:index: 1
:vis: pub
:toc: struct EpochSegment
:layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"EpochSegment"}]

  :::
  A single layer of a subepoch
  :::
:::::{rust:variable} noob_core::epoch::EpochSegment::node
:index: 2
:vis: pub
:toc: node
:layout: [{"type":"name","value":"node"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"}]

  :::
  The node that created this layer of the subepoch
  :::
:::::
:::::{rust:variable} noob_core::epoch::EpochSegment::epoch
:index: 2
:vis: pub
:toc: epoch
:layout: [{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"link","value":"u32","target":"u32"}]

  :::
  The index of this subepoch within its layer
  :::
:::::

:::{rubric} Traits implemented
:::

:::::{rust:impl} noob_core::epoch::EpochSegment::From
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"From","target":"From"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"("},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":", "},{"type":"link","value":"u32","target":"u32"},{"type":"punctuation","value":")"},{"type":"punctuation","value":">"},{"type":"space"},{"type":"keyword","value":"for"},{"type":"space"},{"type":"link","value":"EpochSegment","target":"EpochSegment"}]
:toc: impl From for EpochSegment

  :::
  :::
:::::
:::::{rust:impl} noob_core::epoch::EpochSegment::Display
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"fmt","target":"fmt"},{"type":"punctuation","value":"::"},{"type":"name","value":"Display"},{"type":"space"},{"type":"keyword","value":"for"},{"type":"space"},{"type":"link","value":"EpochSegment","target":"EpochSegment"}]
:toc: impl Display for EpochSegment

  :::
  Takes the interner arc lock - don't be string formatting while mutating interns
  :::
:::::
::::::
:::::::
