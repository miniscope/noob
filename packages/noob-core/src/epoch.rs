use std::ops::Div;

use crate::item::TUBE_NODE;

// TODO: Make some frozen "SmallEpoch" with a burned in tube ID?
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct EpochSegment {
    node: u16,
    epoch: u32,
}

impl From<(u16, u32)> for EpochSegment {
    fn from(segment: (u16, u32)) -> EpochSegment {
        EpochSegment {
            node: segment.0,
            epoch: segment.1,
        }
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct Epoch(Vec<EpochSegment>);

impl Epoch {
    pub fn segments(&self) -> &[EpochSegment] {
        &self.0
    }

    pub fn parent(&self) -> Option<Epoch> {
        if self.0.len() > 1 {
            Some(Epoch(self.0[..self.0.len() - 1].to_vec()))
        } else {
            None
        }
    }

    pub fn child(mut self, subep: EpochSegment) -> Epoch {
        self.0.push(subep);
        self
    }

    pub fn root(&self) -> u32 {
        self.0[0].epoch
    }
}

impl From<u32> for Epoch {
    fn from(number: u32) -> Self {
        Epoch(vec![EpochSegment {
            node: TUBE_NODE,
            epoch: number,
        }])
    }
}

impl<T: Into<EpochSegment>> Div<T> for Epoch {
    type Output = Epoch;
    fn div(self, rhs: T) -> Epoch {
        self.child(rhs.into())
    }
}

impl<T: Into<EpochSegment>> Div<T> for &Epoch {
    type Output = Epoch;
    fn div(self, rhs: T) -> Epoch {
        let epoch = self.clone();
        epoch.child(rhs.into())
    }
}

#[cfg(test)]
#[path = "tests/epoch.rs"]
mod tests;
