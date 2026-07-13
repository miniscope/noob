use std::fmt;
use std::sync::{Arc, LazyLock, RwLock, RwLockWriteGuard};

use crate::FxIndexSet;

pub type ItemID = u32;

static INTERNER: LazyLock<RwLock<Arc<Interner>>> =
    LazyLock::new(|| RwLock::new(Arc::new(Interner::default())));

pub fn interner() -> Arc<Interner> {
    INTERNER
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}
pub(crate) fn interner_mut() -> RwLockWriteGuard<'static, Arc<Interner>> {
    INTERNER
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// For use with Epoch - fast way to handle interned node_id -> int mappings
/// We assume that most of the time we are constructing Epochs in the context of a scheduler,
/// where all the node_ids will already be interned,
/// so all we need to do is get a read-only view into the interner.
/// To support random, python-like creation of Epochs, though,
/// we automatically intern node ids that haven't been:
/// if there are no other references to the interner,
/// as cheap as mutating the IndexMap. Otherwise, requires a copy.
pub(crate) fn resolve_or_intern_node(node_id: &str) -> ItemID {
    let item = Item::Node(node_id.to_string());
    if let Some(id) = interner().get(&item) {
        return id;
    }
    // take the lock and try to get again in case interned in the tiny race window
    let mut interner_slot = interner_mut();
    if let Some(id) = interner_slot.get(&item) {
        return id;
    }
    Arc::make_mut(&mut interner_slot).intern(item)
}

/// A graph item: either a node id, or a (node id, signal name) pair.
///
/// The native representation of `noob.types.NodeID | noob.types.NodeSignal`:
/// a `str` or a 2-tuple of `str` on the python side.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Item {
    Node(String),
    Signal(String, String),
}

impl Item {
    pub fn is_signal(&self) -> bool {
        matches!(self, Item::Signal(..))
    }

    /// The node id: itself for nodes, the node part for signals
    pub fn node_id(&self) -> &str {
        match self {
            Item::Node(n) => n,
            Item::Signal(n, _) => n,
        }
    }
}

impl fmt::Display for Item {
    /// Match the python repr: `'node'` for node ids (str repr),
    /// `('node', 'signal')` for signals (NodeSignal repr)
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Item::Node(n) => write!(f, "'{n}'"),
            Item::Signal(n, s) => write!(f, "('{n}', '{s}')"),
        }
    }
}

/// The interned id of the `("meta", "previous_epoch")` signal.
///
/// Stateful nodes depend on it so they can't run before their previous
/// epoch completes; the scheduler controls when it is marked done.
/// Every [`Interner`] interns it at construction, so it is always id 0.
pub const PREVIOUS_EPOCH: ItemID = 0;
pub const META_NODE: ItemID = 1;
/// The marker that indicates the root of an epoch, Epoch(("tube", 0))
pub const TUBE_NODE: ItemID = 2;
pub const INPUT_NODE: ItemID = 3;
pub const ASSETS_NODE: ItemID = 4;

/// Interns [`Item`]s to dense `ItemID` ids shared by all sorters in a scheduler,
/// so that all graph algorithms operate on integers rather than strings.
#[derive(Clone, Debug)]
pub struct Interner {
    items: FxIndexSet<Item>,
}

impl Default for Interner {
    /// Start with the [`PREVIOUS_EPOCH`] signal interned, guaranteeing its id
    fn default() -> Self {
        let mut interner = Interner {
            items: FxIndexSet::default(),
        };
        interner.intern_signal("meta", "previous_epoch");
        interner.intern_node("tube");
        interner.intern_node("input");
        interner.intern_node("assets");
        interner
    }
}

impl Interner {
    pub fn intern(&mut self, item: Item) -> ItemID {
        self.items.insert_full(item).0 as ItemID
    }

    pub fn intern_node(&mut self, id: &str) -> ItemID {
        self.intern(Item::Node(id.to_owned()))
    }

    pub fn intern_signal(&mut self, node: &str, signal: &str) -> ItemID {
        let id = self.intern(Item::Signal(node.to_owned(), signal.to_owned()));
        self.intern_node(node);
        id
    }

    pub fn get(&self, item: &Item) -> Option<ItemID> {
        self.items.get_index_of(item).map(|i| i as ItemID)
    }

    pub fn resolve(&self, id: ItemID) -> &Item {
        self.items
            .get_index(id as usize)
            .expect("interner ids are never removed, and always added before they are resolved")
    }

    pub fn is_signal(&self, id: ItemID) -> bool {
        self.resolve(id).is_signal()
    }

    /// For a signal item, the interned id of its node part.
    /// For a node item, its own id.
    pub fn node_part(&self, id: ItemID) -> ItemID {
        let node = self.resolve(id).node_id().to_owned();
        self.get(&Item::Node(node)).unwrap()
    }
}

#[cfg(test)]
#[path = "tests/item.rs"]
mod tests;
