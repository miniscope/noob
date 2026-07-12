use crate::epoch::Epoch;
use crate::item::ItemID;

pub struct UpdateEvent {
    pub epoch: Epoch,
    pub node: ItemID,
    pub signal: Option<ItemID>,
    pub no_event: bool,
}
