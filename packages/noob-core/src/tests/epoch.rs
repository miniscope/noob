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

#[test]
fn test_parents() {
    let root = Epoch::from(10);
    let parents: Vec<Epoch> = root.parents().collect();
    assert_eq!(parents, vec![]);
}

#[test]
fn test_parents_subepoch() {
    let subep = Epoch(vec![
        EpochSegment { node: 0, epoch: 0 },
        EpochSegment { node: 1, epoch: 0 },
        EpochSegment { node: 2, epoch: 0 },
    ]);
    let parent1 = Epoch(vec![
        EpochSegment { node: 0, epoch: 0 },
        EpochSegment { node: 1, epoch: 0 },
    ]);
    let parent2 = Epoch(vec![EpochSegment { node: 0, epoch: 0 }]);

    let parents: Vec<Epoch> = subep.parents().collect();
    assert_eq!(parents, vec![parent1, parent2]);
}

#[test]
fn test_add() {
    let root = Epoch::from(10);
    assert_eq!(root + 1, Epoch::from(11));

    let root = Epoch::from(10);
    let borrowed = &root + 1;
    assert_eq!(root, Epoch::from(10));
    assert_eq!(borrowed, Epoch::from(11));
}

#[test]
fn test_add_subepoch() {
    let subep = Epoch(vec![
        EpochSegment { node: 0, epoch: 0 },
        EpochSegment { node: 1, epoch: 0 },
        EpochSegment { node: 2, epoch: 0 },
    ]);

    let expected = Epoch(vec![
        EpochSegment { node: 0, epoch: 0 },
        EpochSegment { node: 1, epoch: 0 },
        EpochSegment { node: 2, epoch: 1 },
    ]);

    assert_eq!(subep + 1, expected);
}

#[test]
fn test_make_subepochs() {
    let parent = Epoch::from(0);
    let expected = vec![
        parent.clone().child(EpochSegment { node: 1, epoch: 0 }),
        parent.clone().child(EpochSegment { node: 1, epoch: 1 }),
        parent.clone().child(EpochSegment { node: 1, epoch: 2 }),
    ];
    assert_eq!(parent.make_subepochs(1, 3), expected);
}
