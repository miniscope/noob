# `mod exceptions`

:::::::{rust:module} noob_core::exceptions
:index: 0
:vis: pub

  :::
  :::
:::{rust:use} noob_core::exceptions
:used_name: self

:::
:::{rust:use} noob_core
:used_name: crate

:::
:::{rust:use} noob_core::epoch::Epoch
:used_name: Epoch

:::
:::{rust:use} std::error::Error
:used_name: Error

:::
:::{rust:use} std::fmt
:used_name: fmt

:::

:::{rubric} Types
:::

::::::{rust:type} noob_core::exceptions::CoreResult
:index: 0
:vis: pub
:layout: [{"type":"keyword","value":"type"},{"type":"space"},{"type":"name","value":"CoreResult"},{"type":"punctuation","value":"<"},{"type":"name","value":"T"},{"type":"punctuation","value":">"}]

  :::
  Shorthand for fallible core operations, like `PyResult<T>` is for pyo3.
  :::
::::::

:::{rubric} Enums
:::

::::::{rust:enum} noob_core::exceptions::CoreError
:index: 1
:vis: pub
:layout: [{"type":"keyword","value":"enum"},{"type":"space"},{"type":"name","value":"CoreError"}]

  :::
  Errors raised by the pure-rust core.
  
  These mirror `noob.exceptions`: the scheduler's python boundary layer is
  responsible for translating them into the corresponding python exception
  types. Keeping this enum free of pyo3 lets the sorter and its tests stay
  pure rust.
  :::
:::::{rust:struct} noob_core::exceptions::CoreError::AlreadyDone
:index: 2
:vis: pub
:toc: AlreadyDone
:layout: [{"type":"name","value":"AlreadyDone"},{"type":"punctuation","value":"("},{"type":"link","value":"String","target":"String"},{"type":"punctuation","value":")"}]

  :::
  `noob.exceptions.AlreadyDoneError`
  :::
:::::
:::::{rust:struct} noob_core::exceptions::CoreError::NotAdded
:index: 2
:vis: pub
:toc: NotAdded
:layout: [{"type":"name","value":"NotAdded"},{"type":"punctuation","value":"("},{"type":"link","value":"String","target":"String"},{"type":"punctuation","value":")"}]

  :::
  `noob.exceptions.NotAddedError`
  :::
:::::
:::::{rust:struct} noob_core::exceptions::CoreError::EpochExists
:index: 2
:vis: pub
:toc: EpochExists
:layout: [{"type":"name","value":"EpochExists"},{"type":"punctuation","value":"("},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":")"}]

  :::
  :::
:::::
:::::{rust:struct} noob_core::exceptions::CoreError::EpochCompleted
:index: 2
:vis: pub
:toc: EpochCompleted
:layout: [{"type":"name","value":"EpochCompleted"},{"type":"punctuation","value":"("},{"type":"link","value":"Epoch","target":"Epoch"},{"type":"punctuation","value":")"}]

  :::
  :::
:::::
:::::{rust:struct} noob_core::exceptions::CoreError::Value
:index: 2
:vis: pub
:toc: Value
:layout: [{"type":"name","value":"Value"},{"type":"punctuation","value":"("},{"type":"link","value":"String","target":"String"},{"type":"punctuation","value":")"}]

  :::
  `ValueError`
  :::
:::::

:::{rubric} Traits implemented
:::

:::::{rust:impl} noob_core::exceptions::CoreError::Display
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"fmt","target":"fmt"},{"type":"punctuation","value":"::"},{"type":"name","value":"Display"},{"type":"space"},{"type":"keyword","value":"for"},{"type":"space"},{"type":"link","value":"CoreError","target":"CoreError"}]
:toc: impl Display for CoreError

  :::
  :::
:::::
:::::{rust:impl} noob_core::exceptions::CoreError::Error
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Error","target":"Error"},{"type":"space"},{"type":"keyword","value":"for"},{"type":"space"},{"type":"link","value":"CoreError","target":"CoreError"}]
:toc: impl Error for CoreError

  :::
  :::
:::::
::::::
:::::::
