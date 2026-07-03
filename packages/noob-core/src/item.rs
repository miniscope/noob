use std::fmt;

use indexmap::IndexSet;

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
pub const PREVIOUS_EPOCH: u16 = 0;

/// Interns [`Item`]s to dense `u16` ids shared by all sorters in a scheduler,
/// so that all graph algorithms operate on integers rather than strings.
#[derive(Clone, Debug)]
pub struct Interner {
    items: IndexSet<Item>,
}

impl Default for Interner {
    /// Start with the [`PREVIOUS_EPOCH`] signal interned, guaranteeing its id
    fn default() -> Self {
        let mut interner = Interner {
            items: IndexSet::new(),
        };
        interner.intern_signal("meta", "previous_epoch");
        interner
    }
}

impl Interner {
    pub fn intern(&mut self, item: Item) -> u16 {
        self.items.insert_full(item).0 as u16
    }

    pub fn intern_node(&mut self, id: &str) -> u16 {
        self.intern(Item::Node(id.to_owned()))
    }

    pub fn intern_signal(&mut self, node: &str, signal: &str) -> u16 {
        self.intern(Item::Signal(node.to_owned(), signal.to_owned()))
    }

    pub fn get(&self, item: &Item) -> Option<u16> {
        self.items.get_index_of(item).map(|i| i as u16)
    }

    pub fn resolve(&self, id: u16) -> &Item {
        self.items
            .get_index(id as usize)
            .expect("interner ids are never removed")
    }

    pub fn is_signal(&self, id: u16) -> bool {
        self.resolve(id).is_signal()
    }

    /// For a signal item, the interned id of its node part.
    /// For a node item, its own id.
    pub fn node_part(&mut self, id: u16) -> u16 {
        let node = self.resolve(id).node_id().to_owned();
        self.intern_node(&node)
    }
}
