use crate::item::{ItemID, TUBE_NODE};
use std::fmt;
use std::iter::once;
use std::ops::{Add, Div};

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct EpochSegment {
    pub node: ItemID,
    pub epoch: u32,
}

impl From<(ItemID, u32)> for EpochSegment {
    fn from(segment: (ItemID, u32)) -> EpochSegment {
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
pub struct Epoch {
    root: u32,
    path: Vec<EpochSegment>,
}

impl Epoch {
    pub fn new(root: u32, path: impl IntoIterator<Item = impl Into<EpochSegment>>) -> Epoch {
        Epoch {
            root,
            path: path.into_iter().map(Into::into).collect(),
        }
    }

    pub fn is_root(&self) -> bool {
        self.path.is_empty()
    }

    pub fn n_segments(&self) -> usize {
        self.path.len() + 1
    }

    pub fn segments(&self) -> impl Iterator<Item = EpochSegment> + '_ {
        once(EpochSegment {
            node: TUBE_NODE,
            epoch: self.root,
        })
        .chain(self.path.iter().copied())
    }

    pub fn parent(&self) -> Option<Epoch> {
        self.path.split_last().map(|(_leaf, rest)| Epoch {
            root: self.root,
            path: rest.to_vec(),
        })
    }

    pub fn parents(&self) -> impl Iterator<Item = Epoch> + '_ {
        (0..self.path.len()).rev().map(|i| Epoch {
            root: self.root,
            path: self.path[..i].to_vec(),
        })
    }

    pub fn child(mut self, subep: EpochSegment) -> Epoch {
        self.path.push(subep);
        self
    }

    pub fn make_subepochs(&self, node: ItemID, n: u32) -> Vec<Epoch> {
        (0..n)
            .map(|i| self.clone().child(EpochSegment { epoch: i, node }))
            .collect()
    }

    pub fn root(&self) -> u32 {
        self.root
    }

    pub fn leaf(&self) -> Option<&EpochSegment> {
        self.path.last()
    }
}

impl From<u32> for Epoch {
    fn from(number: u32) -> Self {
        Epoch {
            root: number,
            path: Vec::new(),
        }
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
        match self.path.last_mut() {
            Some(seg) => seg.epoch += rhs,
            None => self.root += rhs,
        }
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
        if self.is_root() {
            write!(f, "{}", self.root())
        } else {
            write!(f, "(")?;
            write!(f, "{}", self.root)?;
            for (i, seg) in self.path.iter().enumerate() {
                write!(f, ", {seg}")?;
            }
            write!(f, ")")
        }
    }
}

#[cfg(test)]
#[path = "tests/epoch.rs"]
mod tests;
