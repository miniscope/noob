use crate::epoch::Epoch;

pub struct UpdateEvent {
    pub epoch: Epoch,
    pub node: u16,
    pub signal: Option<u16>,
    pub no_event: bool,
}
