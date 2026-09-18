use super::*;

#[test]
fn test_const_ids() {
    let interner = Interner::default();
    assert_eq!(
        interner.resolve(PREVIOUS_EPOCH),
        &Item::Signal("meta".into(), "previous_epoch".into())
    );
    assert_eq!(interner.resolve(META_NODE), &Item::Node("meta".into()));
    assert_eq!(interner.resolve(TUBE_NODE), &Item::Node("tube".into()));
    assert_eq!(interner.resolve(INPUT_NODE), &Item::Node("input".into()));
    assert_eq!(interner.resolve(ASSETS_NODE), &Item::Node("assets".into()));
}

/// Slots can be disambiguated from signals
#[test]
fn test_signals_vs_slots() {
    let mut interner = Interner::default();
    let sig = interner.intern_signal("a", "value");
    let slot = interner.intern_slot("a", "value");
    assert_ne!(sig, slot);

    assert!(interner.is_signal(sig));
    assert!(!interner.is_slot(sig));
    assert!(interner.is_slot(slot));
    assert!(!interner.is_signal(slot));
}
