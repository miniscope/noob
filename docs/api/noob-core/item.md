# `mod item`

:::::::{rust:module} noob_core::item
:index: 0
:vis: pub

  :::
  :::
:::{rust:use} noob_core::item
:used_name: self

:::
:::{rust:use} noob_core
:used_name: crate

:::
:::{rust:use} noob_core::FxIndexSet
:used_name: FxIndexSet

:::
:::{rust:use} pyo3::IntoPyObject
:used_name: IntoPyObject

:::
:::{rust:use} std::fmt
:used_name: fmt

:::
:::{rust:use} std::sync::Arc
:used_name: Arc

:::
:::{rust:use} std::sync::LazyLock
:used_name: LazyLock

:::
:::{rust:use} std::sync::RwLock
:used_name: RwLock

:::
:::{rust:use} std::sync::RwLockWriteGuard
:used_name: RwLockWriteGuard

:::

:::{rubric} Types
:::

::::::{rust:type} noob_core::item::ItemID
:index: 0
:vis: pub
:layout: [{"type":"keyword","value":"type"},{"type":"space"},{"type":"name","value":"ItemID"}]

  :::
  :::
::::::

:::{rubric} Variables
:::

::::::{rust:variable} noob_core::item::ASSETS_NODE
:index: 0
:vis: pub
:toc: const ASSETS_NODE
:layout: [{"type":"keyword","value":"const"},{"type":"space"},{"type":"name","value":"ASSETS_NODE"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"}]

  :::
  "assets"
  :::
::::::
::::::{rust:variable} noob_core::item::INPUT_NODE
:index: 0
:vis: pub
:toc: const INPUT_NODE
:layout: [{"type":"keyword","value":"const"},{"type":"space"},{"type":"name","value":"INPUT_NODE"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"}]

  :::
  "input"
  :::
::::::
::::::{rust:variable} noob_core::item::META_NODE
:index: 0
:vis: pub
:toc: const META_NODE
:layout: [{"type":"keyword","value":"const"},{"type":"space"},{"type":"name","value":"META_NODE"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"}]

  :::
  :::
::::::
::::::{rust:variable} noob_core::item::PREVIOUS_EPOCH
:index: 0
:vis: pub
:toc: const PREVIOUS_EPOCH
:layout: [{"type":"keyword","value":"const"},{"type":"space"},{"type":"name","value":"PREVIOUS_EPOCH"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"}]

  :::
  The interned id of the `("meta", "previous_epoch")` signal.
  
  Stateful nodes depend on it so they can't run before their previous
  epoch completes; the scheduler controls when it is marked done.
  Every [`Interner`] interns it at construction, so it is always id 0.
  :::
::::::
::::::{rust:variable} noob_core::item::TUBE_NODE
:index: 0
:vis: pub
:toc: const TUBE_NODE
:layout: [{"type":"keyword","value":"const"},{"type":"space"},{"type":"name","value":"TUBE_NODE"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"}]

  :::
  "tube" - The marker that indicates the root of an epoch, Epoch(("tube", 0))
  :::
::::::

:::{rubric} Functions
:::

::::::{rust:function} noob_core::item::interner
:index: 0
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"interner"},{"type":"punctuation","value":"("},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Arc","target":"Arc"},{"type":"punctuation","value":"<"},{"type":"link","value":"Interner","target":"Interner"},{"type":"punctuation","value":">"}]

  :::
  Get a read-only reference to the global python<->rust interner
  Acquired the RwLock only within the function call, releasing it immediately.
  See the top-level design docs for more details.
  :::
::::::
::::::{rust:function} noob_core::item::interner_mut
:index: 0
:vis: crate
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"interner_mut"},{"type":"punctuation","value":"("},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"RwLockWriteGuard","target":"RwLockWriteGuard"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'static"},{"type":"punctuation","value":", "},{"type":"link","value":"Arc","target":"Arc"},{"type":"punctuation","value":"<"},{"type":"link","value":"Interner","target":"Interner"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

  :::
  Get a read/write reference to the global python<->rust interner,
  holding the lock.
  
  Update the interner by getting a new mutable Arc from it:
  
  ```ignore
  let mut interner_slot = interner_mut();
  let interner = Arc::make_mut(&mut interner_slot);
  ```
  :::
::::::
::::::{rust:function} noob_core::item::resolve_or_intern_node
:index: 0
:vis: crate
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"resolve_or_intern_node"},{"type":"punctuation","value":"("},{"type":"name","value":"node_id"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"str","target":"str"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"ItemID","target":"ItemID"}]

  :::
  For use with Epoch - fast way to handle interned node_id -> int mappings
  We assume that most of the time we are constructing Epochs in the context of a scheduler,
  where all the node_ids will already be interned,
  so all we need to do is get a read-only view into the interner.
  To support random, python-like creation of Epochs, though,
  we automatically intern node ids that haven't been:
  if there are no other references to the interner,
  as cheap as mutating the IndexMap. Otherwise, requires a copy.
  :::
::::::

:::{rubric} Enums
:::

::::::{rust:enum} noob_core::item::Item
:index: 1
:vis: pub
:layout: [{"type":"keyword","value":"enum"},{"type":"space"},{"type":"name","value":"Item"}]

  :::
  A graph item: either a node id, or a (node id, signal name) pair.
  
  The native representation of `noob.types.NodeID | noob.types.NodeSignal`:
  a `str` or a 2-tuple of `str` on the python side.
  :::
:::::{rust:struct} noob_core::item::Item::Node
:index: 2
:vis: pub
:toc: Node
:layout: [{"type":"name","value":"Node"},{"type":"punctuation","value":"("},{"type":"link","value":"String","target":"String"},{"type":"punctuation","value":")"}]

  :::
  :::
:::::
:::::{rust:struct} noob_core::item::Item::Signal
:index: 2
:vis: pub
:toc: Signal
:layout: [{"type":"name","value":"Signal"},{"type":"punctuation","value":"("},{"type":"link","value":"String","target":"String"},{"type":"punctuation","value":", "},{"type":"link","value":"String","target":"String"},{"type":"punctuation","value":")"}]

  :::
  :::
:::::

:::{rubric} Implementations
:::

:::::{rust:impl} noob_core::item::Item
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Item","target":"Item"}]
:toc: impl Item

  :::
  :::

:::{rubric} Functions
:::

::::{rust:function} noob_core::item::Item::is_signal
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"is_signal"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"bool","target":"bool"}]

  :::
  :::
::::
::::{rust:function} noob_core::item::Item::node_id
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"node_id"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"punctuation","value":"&"},{"type":"link","value":"str","target":"str"}]

  :::
  The node id: itself for nodes, the node part for signals
  :::
::::
:::::

:::{rubric} Traits implemented
:::

:::::{rust:impl} noob_core::item::Item::Display
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"fmt","target":"fmt"},{"type":"punctuation","value":"::"},{"type":"name","value":"Display"},{"type":"space"},{"type":"keyword","value":"for"},{"type":"space"},{"type":"link","value":"Item","target":"Item"}]
:toc: impl Display for Item

  :::
  :::
:::::
::::::

:::{rubric} Structs and Unions
:::

::::::{rust:struct} noob_core::item::Interner
:index: 1
:vis: pub
:toc: struct Interner
:layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"Interner"}]

  :::
  Interns [`Item`]s to dense `ItemID` ids shared by all sorters in a scheduler,
  so that all graph algorithms operate on integers rather than strings.
  :::

:::{rubric} Implementations
:::

:::::{rust:impl} noob_core::item::Interner
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Interner","target":"Interner"}]
:toc: impl Interner

  :::
  :::

:::{rubric} Functions
:::

::::{rust:function} noob_core::item::Interner::get
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"get"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"item"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Item","target":"Item"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Option","target":"Option"},{"type":"punctuation","value":"<"},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":">"}]

  :::
  :::
::::
::::{rust:function} noob_core::item::Interner::intern
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"intern"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"item"},{"type":"punctuation","value":": "},{"type":"link","value":"Item","target":"Item"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"ItemID","target":"ItemID"}]

  :::
  :::
::::
::::{rust:function} noob_core::item::Interner::intern_node
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"intern_node"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"id"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"str","target":"str"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"ItemID","target":"ItemID"}]

  :::
  :::
::::
::::{rust:function} noob_core::item::Interner::intern_signal
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"intern_signal"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"node"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"str","target":"str"},{"type":"punctuation","value":", "},{"type":"name","value":"signal"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"str","target":"str"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"ItemID","target":"ItemID"}]

  :::
  Intern both the (node, signal) tuple and the node within it.
  :::
::::
::::{rust:function} noob_core::item::Interner::is_signal
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"is_signal"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"id"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"bool","target":"bool"}]

  :::
  :::
::::
::::{rust:function} noob_core::item::Interner::node_part
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"node_part"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"id"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"ItemID","target":"ItemID"}]

  :::
  For a signal item, the interned id of its node part.
  For a node item, its own id.
  :::
::::
::::{rust:function} noob_core::item::Interner::resolve
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"resolve"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"id"},{"type":"punctuation","value":": "},{"type":"link","value":"ItemID","target":"ItemID"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"punctuation","value":"&"},{"type":"link","value":"Item","target":"Item"}]

  :::
  Return an integer ItemId to the python string form
  Panics if no item is present in the interner
  :::
::::
:::::

:::{rubric} Traits implemented
:::

:::::{rust:impl} noob_core::item::Interner::Default
:index: -1
:vis: pub
:layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Default","target":"Default"},{"type":"space"},{"type":"keyword","value":"for"},{"type":"space"},{"type":"link","value":"Interner","target":"Interner"}]
:toc: impl Default for Interner

  :::
  :::
:::::
::::::
:::::::
