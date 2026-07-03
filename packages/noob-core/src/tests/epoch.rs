use super::*;

use crate::item::TUBE_NODE;

/// Epochs can be created by an int
#[test]
fn test_from_int() {
    assert_eq!(
        Epoch::from(10),
        Epoch(vec![EpochSegment {
            node: TUBE_NODE,
            epoch: 10
        }])
    )
}

#[test]
fn test_root_has_no_parent() {
    assert_eq!(Epoch::from(0).parent(), None);
}

#[test]
fn test_subepochs_have_parent() {
    let subepoch = Epoch(vec![
        EpochSegment {
            node: TUBE_NODE,
            epoch: 10,
        },
        EpochSegment {
            node: 100,
            epoch: 0,
        },
    ]);
    assert_eq!(subepoch.parent(), Some(Epoch::from(10)));
}

#[test]
fn test_child() {
    let subepoch = Epoch(vec![
        EpochSegment {
            node: TUBE_NODE,
            epoch: 10,
        },
        EpochSegment {
            node: 100,
            epoch: 0,
        },
    ]);
    let parent = Epoch::from(10);
    assert_eq!(
        parent.child(EpochSegment {
            node: 100,
            epoch: 0
        }),
        subepoch
    );
}

#[test]
fn test_child_from_div() {
    let subepoch = Epoch(vec![
        EpochSegment {
            node: TUBE_NODE,
            epoch: 10,
        },
        EpochSegment {
            node: 100,
            epoch: 0,
        },
    ]);
    let parent = Epoch::from(10);
    assert_eq!(
        &parent
            / EpochSegment {
                node: 100,
                epoch: 0
            },
        subepoch
    );
    assert_eq!(&parent / (100, 0), subepoch);
}

#[test]
fn test_epoch_order() {
    assert!(Epoch::from(2) < Epoch::from(3));
    assert!(Epoch::from(4) > Epoch::from(3));
}

#[test]
fn test_subepoch_greater_than_parent() {
    let subepoch = Epoch::from(10) / (100, 0);
    assert!(Epoch::from(10) < subepoch);
}

#[test]
fn test_subepoch_ordered_by_nodeid() {
    let parent = Epoch::from(10);
    let subep_a = &parent / (100, 0);
    let subep_b = &parent / (101, 0);
    assert!(subep_a < subep_b);
}
