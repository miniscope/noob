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
  :::

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

::::{rust:function} noob_core::epoch::Epoch::checked_sub
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"checked_sub"},{"type":"punctuation","value":"("},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"rhs"},{"type":"punctuation","value":": "},{"type":"link","value":"u32","target":"u32"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Option","target":"Option"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"}]

  :::
  :::
::::
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
  :::
::::
::::{rust:function} noob_core::epoch::Epoch::make_subepochs
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"make_subepochs"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"node"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":", "},{"type":"name","value":"n"},{"type":"punctuation","value":": "},{"type":"link","value":"u32","target":"u32"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":">"}]

  :::
  :::
::::
::::{rust:function} noob_core::epoch::Epoch::n_segments
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"n_segments"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"usize","target":"usize"}]

  :::
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
  :::
:::::
:::::{rust:impl} noob_core::epoch::Epoch::Div
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"punctuation","value":"<"},{"type":"name","value":"T"},{"type":"punctuation","value":": "},{"type":"link","value":"Into","target":"Into"},{"type":"punctuation","value":"<"},{"type":"link","value":"EpochSegment","target":"EpochSegment"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"},{"type":"space"},{"type":"link","value":"Div","target":"Div"},{"type":"punctuation","value":"<"},{"type":"link","value":"T","target":"T"},{"type":"punctuation","value":">"},{"type":"space"},{"type":"keyword","value":"for"},{"type":"space"},{"type":"link","value":"Epoch","target":"Epoch"}]
:toc: impl Div for Epoch

  :::
  :::
:::::
:::::{rust:impl} noob_core::epoch::Epoch::Add
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Add","target":"Add"},{"type":"punctuation","value":"<"},{"type":"link","value":"u32","target":"u32"},{"type":"punctuation","value":">"},{"type":"space"},{"type":"keyword","value":"for"},{"type":"space"},{"type":"link","value":"Epoch","target":"Epoch"}]
:toc: impl Add for Epoch

  :::
  :::
:::::
:::::{rust:impl} noob_core::epoch::Epoch::Sub
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Sub","target":"Sub"},{"type":"punctuation","value":"<"},{"type":"link","value":"u32","target":"u32"},{"type":"punctuation","value":">"},{"type":"space"},{"type":"keyword","value":"for"},{"type":"space"},{"type":"link","value":"Epoch","target":"Epoch"}]
:toc: impl Sub for Epoch

  :::
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
  :::
:::::{rust:variable} noob_core::epoch::EpochSegment::node
:index: 2
:vis: pub
:toc: node
:layout: [{"type":"name","value":"node"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"}]

  :::
  :::
:::::
:::::{rust:variable} noob_core::epoch::EpochSegment::epoch
:index: 2
:vis: pub
:toc: epoch
:layout: [{"type":"name","value":"epoch"},{"type":"punctuation","value":": "},{"type":"link","value":"u32","target":"u32"}]

  :::
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
