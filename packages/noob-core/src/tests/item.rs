use super::*;

#[test]
fn test_const_ids() {
    let interner = Interner::default();
    assert_eq!(
        interner.resolve(PREVIOUS_EPOCH),
        &Item::Signal("meta".into(), "previous_epoch".into())
    );
    assert_eq!(interner.resolve(TUBE_NODE), &Item::Node("tube".into()));
}
