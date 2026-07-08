use super::*;

// TODO: consolidate test helpers
fn edge(source: &str, signal: &str, target: &str, required: bool) -> EdgeRec {
    EdgeRec {
        source_node: source.into(),
        source_signal: signal.into(),
        target_node: target.into(),
        required,
    }
}

fn diamond() -> Vec<EdgeRec> {
    vec![
        edge("a", "a1", "b", true),
        edge("a", "a2", "c", true),
        edge("b", "b1", "d", true),
        edge("c", "c1", "d", true),
    ]
}

#[test]
fn test_from_graph() {
    let edges = diamond();
    let scheduler = Scheduler::from_graph(IndexMap::new(), edges.clone())
        .expect("couldnt even create the most basic graph");
    assert_eq!(scheduler.edges, edges);
    assert!(scheduler.epochs.is_empty())
}

#[test]
fn test_add_epoch() {
    let mut scheduler = Scheduler::from_graph(IndexMap::new(), Vec::new()).unwrap();
    let epoch = scheduler.add_epoch();
    assert_eq!(epoch.root(), 0);
    assert!(scheduler.epochs.contains_key(&epoch));
    let epoch2 = scheduler.add_epoch();
    assert_eq!(epoch2.root(), 1);
    assert!(scheduler.epochs.contains_key(&epoch2));
}

#[test]
fn test_add_epoch_at() {
    let mut scheduler = Scheduler::from_graph(IndexMap::new(), Vec::new()).unwrap();
    let expected = Epoch::from(10);
    let epoch = scheduler.add_epoch_at(expected.clone()).unwrap();
    assert_eq!(epoch, expected);

    // Can't add existing epochs
    let err = scheduler.add_epoch_at(expected.clone()).unwrap_err();
    assert_eq!(err, CoreError::EpochExists(expected.clone()));

    // Or completed epochs
    scheduler.epoch_log.insert(6);
    let err = scheduler.add_epoch_at(6).unwrap_err();
    assert_eq!(err, CoreError::EpochCompleted(Epoch::from(6)));

    // Next epoch increments correctly
    let next_epoch = scheduler.add_epoch();
    assert_eq!(next_epoch, Epoch::from(expected.root() + 1));
}

#[test]
fn test_add_epoch_at_out_of_order() {
    let mut scheduler = Scheduler::from_graph(IndexMap::new(), Vec::new()).unwrap();
    scheduler.add_epoch_at(Epoch::from(10)).unwrap();
    scheduler.add_epoch_at(Epoch::from(7)).unwrap();
    let epoch = scheduler.add_epoch();
    assert_eq!(epoch.root(), 11);
}

#[test]
fn test_previous_epoch_completed() {
    let mut nodes = IndexMap::new();
    nodes.insert(
        "a".to_string(),
        NodeFlags {
            enabled: true,
            stateful: Some(true),
        },
    );
    let mut scheduler = Scheduler::from_graph(nodes, Vec::new()).unwrap();
    scheduler.add_epoch();

    // Previous epoch marked done on zeroth epoch
    assert!(scheduler.epochs[&Epoch::from(0)]
        .done
        .contains(&PREVIOUS_EPOCH));

    // But not on a successive epoch whose previous epoch hasn't been completed
    scheduler.add_epoch();
    assert!(!scheduler.epochs[&Epoch::from(1)]
        .done
        .contains(&PREVIOUS_EPOCH));

    // And is marked done when a later epoch is marked done
    scheduler.epoch_log.insert(1);
    scheduler.add_epoch();
    assert!(scheduler.epochs[&Epoch::from(2)]
        .done
        .contains(&PREVIOUS_EPOCH));
}

#[test]
fn test_is_active() {
    let mut nodes = IndexMap::new();
    nodes.insert(
        "a".to_string(),
        NodeFlags {
            enabled: true,
            stateful: Some(false),
        },
    );
    let mut scheduler = Scheduler::from_graph(nodes, Vec::new()).unwrap();
    assert!(!scheduler.is_active());
    scheduler.add_epoch();
    assert!(scheduler.is_active());
    scheduler.add_epoch();

    // still active if one of the sorters is no longer active
    let node_int = scheduler.interner.intern_node("a");
    let sorter = scheduler.epochs.get_mut(&Epoch::from(0)).unwrap();
    sorter.done(&scheduler.interner, &[node_int]).unwrap();
    assert!(scheduler.is_active());

    // no longer active when both are inactive
    let sorter = scheduler.epochs.get_mut(&Epoch::from(1)).unwrap();
    sorter.done(&scheduler.interner, &[node_int]).unwrap();
    assert!(!scheduler.is_active());
}

#[test]
fn test_is_active_at() {
    let mut nodes = IndexMap::new();
    nodes.insert(
        "a".to_string(),
        NodeFlags {
            enabled: true,
            stateful: Some(true),
        },
    );
    let mut scheduler = Scheduler::from_graph(nodes, Vec::new()).unwrap();

    scheduler.add_epoch();
    scheduler.add_epoch();

    assert!(scheduler.is_active_at(&Epoch::from(0)));
    let node_int = scheduler.interner.intern_node("a");
    let sorter = scheduler.epochs.get_mut(&Epoch::from(0)).unwrap();
    sorter.done(&scheduler.interner, &[node_int]).unwrap();
    assert!(!scheduler.is_active_at(&Epoch::from(0)));

    // Not active for an epoch that doesn't exist
    assert!(!scheduler.is_active_at(&Epoch::from(99)));
}

#[test]
fn test_get_ready() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(IndexMap::new(), edges).unwrap();
    let a = scheduler.interner.intern_node("a");
    let a1 = scheduler.interner.intern_signal("a", "a1");
    let a2 = scheduler.interner.intern_signal("a", "a2");
    let b = scheduler.interner.intern_node("b");
    let c = scheduler.interner.intern_node("c");

    scheduler.add_epoch();
    scheduler.add_epoch();

    let sorter = scheduler.epochs.get_mut(&Epoch::from(0)).unwrap();
    sorter.done(&scheduler.interner, &[a, a1, a2]).unwrap();
    let ready = scheduler.get_ready();

    assert_eq!(
        ready,
        [
            (Epoch::from(0), b),
            (Epoch::from(0), c),
            (Epoch::from(1), a)
        ]
    );
}

#[test]
fn test_get_ready_at() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(IndexMap::new(), edges).unwrap();
    let a = scheduler.interner.intern_node("a");
    let a1 = scheduler.interner.intern_signal("a", "a1");
    let a2 = scheduler.interner.intern_signal("a", "a2");

    scheduler.add_epoch();
    scheduler.add_epoch();

    let sorter = scheduler.epochs.get_mut(&Epoch::from(0)).unwrap();
    sorter.done(&scheduler.interner, &[a, a1, a2]).unwrap();
    let ready = scheduler.get_ready_at(&Epoch::from(1));

    assert_eq!(ready, [(Epoch::from(1), a)]);

    // epoch that doesn't exist is just empty
    let ready = scheduler.get_ready_at(&Epoch::from(2));
    assert_eq!(ready, []);
}

#[test]
fn test_done_without_signals() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(IndexMap::new(), edges).unwrap();
    let a = scheduler.interner.intern_node("a");
    let a1 = scheduler.interner.intern_signal("a", "a1");
    let a2 = scheduler.interner.intern_signal("a", "a2");
    let ep = scheduler.add_epoch();
    scheduler.done(&ep, a, false).unwrap();

    let sorter = scheduler.epochs.get(&ep).unwrap();

    assert!(sorter.done.contains(&a));

    assert!(sorter.ready.contains(&a1));
    assert!(sorter.ready.contains(&a2));
    assert!(!sorter.done.contains(&a1));
    assert!(!sorter.done.contains(&a2));
}

#[test]
fn test_done_with_signals() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(IndexMap::new(), edges).unwrap();
    let a = scheduler.interner.intern_node("a");
    let a1 = scheduler.interner.intern_signal("a", "a1");
    let a2 = scheduler.interner.intern_signal("a", "a2");
    let ep = scheduler.add_epoch();
    scheduler.done(&ep, a, true).unwrap();

    let sorter = scheduler.epochs.get(&ep).unwrap();

    assert!(sorter.done.contains(&a));

    assert!(!sorter.ready.contains(&a1));
    assert!(!sorter.ready.contains(&a2));
    assert!(sorter.done.contains(&a1));
    assert!(sorter.done.contains(&a2));
}

/// Done creates a missing epoch and increments next_epoch
#[test]
fn test_done_on_missing_epoch() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(IndexMap::new(), edges).unwrap();
    let a = scheduler.interner.intern_node("a");
    let ep = Epoch::from(10);

    scheduler.done(&ep, a, true).unwrap();

    assert!(scheduler.epochs.contains_key(&ep));
    assert_eq!(scheduler.next_epoch, 11);
}

#[test]
fn test_done_on_completed_epoch() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(IndexMap::new(), edges).unwrap();
    let a = scheduler.interner.intern_node("a");

    let ep = scheduler.add_epoch();
    scheduler.end_epoch(ep.clone()).unwrap();

    let done = scheduler.done(&ep, a, false).unwrap();
    assert_eq!(done, vec![]);
    assert!(!scheduler.epochs.contains_key(&ep));
}

#[test]
fn test_done_ends_epoch() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(IndexMap::new(), edges).unwrap();
    let a = scheduler.interner.intern_node("a");
    let b = scheduler.interner.intern_node("b");
    let c = scheduler.interner.intern_node("c");
    let d = scheduler.interner.intern_node("d");
    let ep = scheduler.add_epoch();
    scheduler.done(&ep, a, true).unwrap();
    scheduler.done(&ep, b, true).unwrap();
    scheduler.done(&ep, c, true).unwrap();

    let done_ep = scheduler.done(&ep, d, true).unwrap();
    assert_eq!(done_ep, vec![ep])
}

/// Basic behavior: end epoch...
/// - garbage collects the epoch
/// - adds it to epoch_log
/// - returns it in a vec
#[test]
fn test_end_epoch() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(IndexMap::new(), edges).unwrap();

    let ep = scheduler.add_epoch();
    let done = scheduler.end_epoch(ep.clone()).unwrap();
    assert!(!scheduler.epochs.contains_key(&ep));
    assert!(scheduler.epoch_log.contains(&ep.root()));
    assert_eq!(done, vec![ep]);
}

/// when stateful nodes, previous_epoch marked done
#[test]
fn test_end_epoch_stateful() {
    let mut nodes = IndexMap::new();
    nodes.insert(
        "a".to_string(),
        NodeFlags {
            enabled: true,
            stateful: Some(true),
        },
    );
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(nodes, edges).unwrap();

    let ep = scheduler.add_epoch();
    assert_eq!(ep.root(), 0);

    scheduler.end_epoch(ep.clone()).unwrap();
    let next = Epoch::from(1);
    assert!(scheduler
        .epochs
        .get(&next)
        .unwrap()
        .done
        .contains(&PREVIOUS_EPOCH));
}
