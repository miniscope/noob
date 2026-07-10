use std::fmt;
use std::ops::{Add, Div};

use crate::item::TUBE_NODE;

// TODO: Make some frozen "SmallEpoch" with a burned in tube ID?
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct EpochSegment {
    pub node: u16,
    pub epoch: u32,
}

impl From<(u16, u32)> for EpochSegment {
    fn from(segment: (u16, u32)) -> EpochSegment {
        EpochSegment {
            node: segment.0,
            epoch: segment.1,
        }
    }
}

impl fmt::Display for EpochSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.node, self.epoch)
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

    pub fn parents(&self) -> impl Iterator<Item = Epoch> + '_ {
        (1..self.0.len()).rev().map(|i| Epoch(self.0[..i].to_vec()))
    }

    pub fn child(mut self, subep: EpochSegment) -> Epoch {
        self.0.push(subep);
        self
    }

    pub fn make_subepochs(&self, node: u16, n: u32) -> Vec<Epoch> {
        (0..n)
            .map(|i| self.clone().child(EpochSegment { epoch: i, node }))
            .collect()
    }

    pub fn root(&self) -> u32 {
        self.0[0].epoch
    }

    pub fn leaf(&self) -> &EpochSegment {
        self.0.last().expect("Epoch can't be empty")
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

impl Add<u32> for Epoch {
    type Output = Epoch;
    fn add(mut self, rhs: u32) -> Epoch {
        self.0.last_mut().expect("Epoch can't be empty").epoch += rhs;
        self
    }
}

impl Add<u32> for &Epoch {
    type Output = Epoch;
    fn add(self, rhs: u32) -> Epoch {
        self.clone() + rhs
    }
}

impl fmt::Display for Epoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.len() == 1 {
            write!(f, "{}", self.root())
        } else {
            write!(f, "(")?;
            for (i, seg) in self.0.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{seg}")?;
            }
            write!(f, ")")
        }
    }
}

#[cfg(test)]
#[path = "tests/epoch.rs"]
mod tests;
