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
    let scheduler = Scheduler::from_graph(FxIndexMap::default(), edges.clone())
        .expect("couldnt even create the most basic graph");
    assert_eq!(scheduler.edges, edges);
    assert!(scheduler.epochs.is_empty());
}

#[test]
fn test_add_epoch() {
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), Vec::new()).unwrap();
    let epoch = scheduler.add_epoch();
    assert_eq!(epoch.root(), 0);
    assert!(scheduler.epochs.contains_key(&epoch));
    let epoch2 = scheduler.add_epoch();
    assert_eq!(epoch2.root(), 1);
    assert!(scheduler.epochs.contains_key(&epoch2));
}

#[test]
fn test_add_epoch_at() {
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), Vec::new()).unwrap();
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

/// an epoch can't be created when it would fall below the epoch log, even if it's never actually been run
#[test]
fn test_add_epoch_at_below_log() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    scheduler.epoch_log_len = 5;

    // epoch below log
    for _ in 1..10 {
        let ep = scheduler.add_epoch();
        scheduler.end_epoch(ep.clone()).unwrap();
    }

    let err = scheduler.add_epoch_at(Epoch::from(0)).unwrap_err();
    assert_eq!(err, CoreError::EpochCompleted(Epoch::from(0)));
}

#[test]
fn test_add_epoch_at_out_of_order() {
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), Vec::new()).unwrap();
    scheduler.add_epoch_at(Epoch::from(10)).unwrap();
    scheduler.add_epoch_at(Epoch::from(7)).unwrap();
    let epoch = scheduler.add_epoch();
    assert_eq!(epoch.root(), 11);
}

#[test]
fn test_previous_epoch_completed() {
    let mut nodes = FxIndexMap::default();
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
    assert!(
        scheduler.epochs[&Epoch::from(0)]
            .done
            .contains(&PREVIOUS_EPOCH)
    );

    // But not on a successive epoch whose previous epoch hasn't been completed
    let ep = scheduler.add_epoch();
    assert!(!scheduler.epochs[&ep].done.contains(&PREVIOUS_EPOCH));

    // Yes marked done in the normal case
    scheduler.end_epoch(ep.clone()).unwrap();
    assert!(scheduler.epochs[&(ep + 1)].done.contains(&PREVIOUS_EPOCH));

    // And is also marked done if out of order
    let ep = scheduler.add_epoch_at(Epoch::from(10)).unwrap();
    assert!(!scheduler.epochs[&ep].done.contains(&PREVIOUS_EPOCH));
    scheduler.end_epoch(Epoch::from(9)).unwrap();
    assert!(scheduler.epochs[&ep].done.contains(&PREVIOUS_EPOCH));

    // for subepochs, sibling subepochs are marked as done
    let root = Epoch::from(20);
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    scheduler.add_epoch_at(root.clone()).unwrap();
    scheduler.add_epoch_at(&root / (a, 0)).unwrap();
    scheduler.add_epoch_at(&root / (a, 1)).unwrap();
    assert!(
        !scheduler.epochs[&(&root / (a, 1))]
            .done
            .contains(&PREVIOUS_EPOCH)
    );
    scheduler.end_epoch(&root / (a, 0)).unwrap();
    assert!(
        scheduler.epochs[&(&root / (a, 1))]
            .done
            .contains(&PREVIOUS_EPOCH)
    );
}

#[test]
fn test_is_active() {
    let mut nodes = FxIndexMap::default();
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
    let node_int = interner().get(&Item::Node("a".to_string())).unwrap();
    let sorter = scheduler.epochs.get_mut(&Epoch::from(0)).unwrap();
    sorter.done(&interner(), &[node_int]).unwrap();
    assert!(scheduler.is_active());

    // no longer active when both are inactive
    let sorter = scheduler.epochs.get_mut(&Epoch::from(1)).unwrap();
    sorter.done(&interner(), &[node_int]).unwrap();
    assert!(!scheduler.is_active());
}

#[test]
fn test_is_active_at() {
    let mut nodes = FxIndexMap::default();
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
    let node_int = interner().get(&Item::Node("a".to_string())).unwrap();
    let sorter = scheduler.epochs.get_mut(&Epoch::from(0)).unwrap();
    sorter.done(&interner(), &[node_int]).unwrap();
    assert!(!scheduler.is_active_at(&Epoch::from(0)));

    // Not active for an epoch that doesn't exist
    assert!(!scheduler.is_active_at(&Epoch::from(99)));
}

#[test]
fn test_get_ready() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let a1 = interner()
        .get(&Item::Signal("a".to_string(), "a1".to_string()))
        .unwrap();
    let a2 = interner()
        .get(&Item::Signal("a".to_string(), "a2".to_string()))
        .unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();
    let c = interner().get(&Item::Node("c".to_string())).unwrap();

    scheduler.add_epoch();
    scheduler.add_epoch();

    let sorter = scheduler.epochs.get_mut(&Epoch::from(0)).unwrap();
    sorter.done(&interner(), &[a, a1, a2]).unwrap();
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
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let a1 = interner()
        .get(&Item::Signal("a".to_string(), "a1".to_string()))
        .unwrap();
    let a2 = interner()
        .get(&Item::Signal("a".to_string(), "a2".to_string()))
        .unwrap();

    scheduler.add_epoch();
    scheduler.add_epoch();

    let sorter = scheduler.epochs.get_mut(&Epoch::from(0)).unwrap();
    sorter.done(&interner(), &[a, a1, a2]).unwrap();
    let ready = scheduler.get_ready_at(&Epoch::from(1));

    assert_eq!(ready, [(Epoch::from(1), a)]);

    // epoch that doesn't exist is just empty
    let ready = scheduler.get_ready_at(&Epoch::from(2));
    assert_eq!(ready, []);
}

#[test]
fn test_done_without_signals() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let a1 = interner()
        .get(&Item::Signal("a".to_string(), "a1".to_string()))
        .unwrap();
    let a2 = interner()
        .get(&Item::Signal("a".to_string(), "a2".to_string()))
        .unwrap();
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
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let a1 = interner()
        .get(&Item::Signal("a".to_string(), "a1".to_string()))
        .unwrap();
    let a2 = interner()
        .get(&Item::Signal("a".to_string(), "a2".to_string()))
        .unwrap();
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
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let ep = Epoch::from(10);

    scheduler.done(&ep, a, true).unwrap();

    assert!(scheduler.epochs.contains_key(&ep));
    // since 'a' is a source node, epoch 11 should have been added
    assert!(scheduler.epochs.contains_key(&Epoch::from(11)));
    assert_eq!(scheduler.next_epoch, 12);
}

#[test]
fn test_done_on_completed_epoch() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();

    let ep = scheduler.add_epoch();
    scheduler.end_epoch(ep.clone()).unwrap();

    let done = scheduler.done(&ep, a, false).unwrap();
    assert_eq!(done, vec![]);
    assert!(!scheduler.epochs.contains_key(&ep));
}

#[test]
fn test_done_ends_epoch() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();
    let c = interner().get(&Item::Node("c".to_string())).unwrap();
    let d = interner().get(&Item::Node("d".to_string())).unwrap();
    let ep = scheduler.add_epoch();
    scheduler.done(&ep, a, true).unwrap();
    scheduler.done(&ep, b, true).unwrap();
    scheduler.done(&ep, c, true).unwrap();

    let done_ep = scheduler.done(&ep, d, true).unwrap();
    assert_eq!(done_ep, vec![ep]);
}

/// Basic behavior: end epoch...
/// - garbage collects the epoch
/// - adds it to epoch_log
/// - returns it in a vec
#[test]
fn test_end_epoch() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();

    let ep = scheduler.add_epoch();
    let done = scheduler.end_epoch(ep.clone()).unwrap();
    assert!(!scheduler.epochs.contains_key(&ep));
    assert!(scheduler.epoch_log.contains(&ep.root()));
    assert_eq!(done, vec![ep]);
}

/// when stateful nodes, previous_epoch marked done
#[test]
fn test_end_epoch_stateful() {
    let mut nodes = FxIndexMap::default();
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
    assert!(
        scheduler
            .epochs
            .get(&next)
            .unwrap()
            .done
            .contains(&PREVIOUS_EPOCH)
    );
}

#[test]
fn test_sources_finished() {
    let edges = diamond();
    let mut nodes = FxIndexMap::default();
    nodes.insert(
        "a".to_string(),
        NodeFlags {
            enabled: true,
            stateful: Some(true),
        },
    );
    let mut scheduler = Scheduler::from_graph(nodes, edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    assert_eq!(scheduler.source_nodes, FxIndexSet::from_iter(vec![a]));

    let ep = scheduler.add_epoch();
    assert!(!scheduler.sources_finished(&ep));
    scheduler.done(&ep, a, true).unwrap();
    assert!(scheduler.sources_finished(&ep));
    assert!(scheduler.epochs.contains_key(&Epoch::from(ep.root() + 1)));

    // false for epochs that haven't been added yet
    assert!(!scheduler.sources_finished(&Epoch::from(99)));
}

#[test]
fn test_eager_epoch_creation_when_sources_done() {
    let edges = diamond();
    let mut nodes = FxIndexMap::default();
    nodes.insert(
        "a".to_string(),
        NodeFlags {
            enabled: true,
            stateful: Some(true),
        },
    );
    let mut scheduler = Scheduler::from_graph(nodes, edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    assert_eq!(scheduler.source_nodes, FxIndexSet::from_iter(vec![a]));

    // does not fire when next epoch already exists
    let ep = scheduler.add_epoch();
    let next = scheduler.add_epoch();
    scheduler.done(&ep, a, true).unwrap();
    assert!(!scheduler.epochs.contains_key(&Epoch::from(next.root() + 1)));

    // does not fire when next epoch already done
    let ep = scheduler.add_epoch();
    let next = scheduler.add_epoch();
    scheduler.end_epoch(next.clone()).unwrap();
    scheduler.done(&ep, a, true).unwrap();
    assert!(!scheduler.epochs.contains_key(&next));

    // does not fire when not a source node
    let b = interner().get(&Item::Node("b".to_string())).unwrap();
    let ep = scheduler.add_epoch();
    let next = Epoch::from(ep.root() + 1);
    scheduler.done(&ep, b, true).unwrap();
    assert!(!scheduler.epochs.contains_key(&next));
}

#[test]
fn test_epoch_log_trim() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    scheduler.epoch_log_len = 5;
    for i in 0..20 {
        scheduler.add_epoch_at(Epoch::from(i)).unwrap();
    }

    // normal behavior, in-order epoch completions keep the latest
    for i in 0..10 {
        scheduler.end_epoch(Epoch::from(i)).unwrap();
    }
    assert_eq!(BTreeSet::from_iter(5..10), scheduler.epoch_log);

    // out of order - we still keep the highest vals
    for i in [18, 17, 10, 19, 12, 11, 15, 16, 13, 14] {
        scheduler.end_epoch(Epoch::from(i)).unwrap();
    }
    assert_eq!(BTreeSet::from_iter(15..20), scheduler.epoch_log);
}

#[test]
fn test_epoch_completed() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    scheduler.epoch_log_len = 5;

    // empty log, nothing completed
    assert!(!scheduler.epoch_completed(&Epoch::from(0)));

    // active epoch
    let ep = scheduler.add_epoch();
    assert!(!scheduler.epoch_completed(&ep));

    // completed epoch
    scheduler.end_epoch(ep.clone()).unwrap();
    assert!(scheduler.epoch_completed(&ep));

    // epoch below log
    for _ in 0..10 {
        let ep = scheduler.add_epoch();
        scheduler.end_epoch(ep.clone()).unwrap();
    }
    assert!(scheduler.epoch_completed(&Epoch::from(0)));
}

/// one signal can be expired while the other lives
#[test]
fn test_expire_signal() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let a1 = interner()
        .get(&Item::Signal("a".to_string(), "a1".to_string()))
        .unwrap();
    let a2 = interner()
        .get(&Item::Signal("a".to_string(), "a2".to_string()))
        .unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();
    let ep = scheduler.add_epoch();
    scheduler.get_ready_at(&ep);
    scheduler.done(&ep, a, false).unwrap();
    scheduler.done(&ep, a1, false).unwrap();
    scheduler.expire(&ep, a2, false, false).unwrap();

    let ready = scheduler.get_ready_at(&ep);
    assert_eq!(vec![(ep, b)], ready);
}

/// expiring a node with its signals handles ending the epoch only once
#[test]
fn test_expire_node() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let ep = scheduler.add_epoch();
    scheduler.get_ready_at(&ep);
    let ended = scheduler.expire(&ep, a, true, true).unwrap();
    assert_eq!(ended, vec![ep]);
}

/// cleanup from python - expiring a node in a completed epoch is suppressed
#[test]
fn test_expire_completed_epoch() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let ep = scheduler.add_epoch();
    scheduler.end_epoch(ep.clone()).unwrap();
    scheduler.expire(&ep, a, true, true).unwrap();
}

#[test]
fn test_iter_epoch_to_completion() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();
    let c = interner().get(&Item::Node("c".to_string())).unwrap();
    let d = interner().get(&Item::Node("d".to_string())).unwrap();

    let mut batches: Vec<Vec<ItemID>> = Vec::new();
    let mut it = scheduler.iter_epoch();
    while let Some(batch) = it.next() {
        let mut nodes = Vec::new();
        for (epoch, node) in batch {
            it.done(&epoch, node, true).unwrap();
            nodes.push(node);
        }
        batches.push(nodes);
    }

    assert_eq!(batches, vec![vec![a], vec![b, c], vec![d]]);
    assert!(scheduler.epoch_completed(&Epoch::from(0)));
}

#[test]
fn test_iter_epoch_at_with_subepochs() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();
    let c = interner().get(&Item::Node("c".to_string())).unwrap();
    let d = interner().get(&Item::Node("d".to_string())).unwrap();

    let root = Epoch::from(0);
    let mut it = scheduler.iter_epoch_at(root.clone()).unwrap();

    // first batch is a, ez
    let batch = it.next().unwrap();
    assert!(batch.len() == 1);
    it.done(&batch[0].0, batch[0].1, true).unwrap();

    // then it's b and c
    let batch = it.next().unwrap();
    assert_eq!(vec![(root.clone(), b), (root.clone(), c)], batch);
    // make B induce some subepochs
    let eps = batch[0].0.make_subepochs(batch[0].1, 3);
    it.done(&batch[1].0, batch[1].1, true).unwrap();
    for ep in eps.clone() {
        it.done(&ep, batch[0].1, true).unwrap();
    }
    // Have to expire b manually here, since `done` calls the sorter `done`,
    // not recursively on the scheduler
    it.expire(&batch[0].0, b, true, false).unwrap();

    // now we should get subepochs in d
    let batch = it.next().unwrap();
    let interner = interner();
    assert_eq!(
        vec![
            (eps[0].clone(), d),
            (eps[1].clone(), d),
            (eps[2].clone(), d)
        ],
        batch.clone(),
        "{:?} - b: {}, c: {}, d: {}, interner: {:?}",
        scheduler.epochs.get(&eps[0]).unwrap().clone_state(),
        b,
        c,
        d,
        interner.resolve(12)
    );
    assert_eq!(
        eps,
        batch
            .iter()
            .map(|(epoch, _)| epoch)
            .cloned()
            .collect::<Vec<Epoch>>()
    );
    assert_eq!(
        vec![d, d, d],
        batch
            .iter()
            .map(|(_, node)| node)
            .cloned()
            .collect::<Vec<ItemID>>()
    );

    for (ep, node) in batch {
        it.done(&ep, node, true).unwrap();
    }

    assert!(
        scheduler.epoch_completed(&root),
        "root: {:?}, subeps: {:?}, 11: {}",
        scheduler.epochs.get(&root).unwrap().clone_state(),
        scheduler.epochs.get(&eps[0]).unwrap().clone_state(),
        interner.resolve(11)
    );
}

#[test]
fn test_iter_epoch_at_completed_epoch() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let ep = scheduler.add_epoch();
    scheduler.end_epoch(ep.clone()).unwrap();

    let Err(err) = scheduler.iter_epoch_at(ep.clone()) else {
        panic!("iterating a completed epoch should fail");
    };
    assert_eq!(err, CoreError::EpochCompleted(ep));
}

/// iter_epoch_at on an epoch that doesn't exist yet creates it, like done()
#[test]
fn test_iter_epoch_at_missing_epoch() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();

    let mut it = scheduler.iter_epoch_at(5).unwrap();
    assert_eq!(it.next().unwrap(), vec![(Epoch::from(5), a)]);

    assert!(scheduler.epochs.contains_key(&Epoch::from(5)));
    assert_eq!(scheduler.next_epoch, 6);
}

/// iter_epoch trusts the caller: an epoch stalled by an unreported batch
/// keeps yielding empty batches rather than terminating
#[test]
fn test_iter_epoch_stalls_with_empty_batches() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();

    let mut it = scheduler.iter_epoch();
    assert_eq!(it.next().unwrap(), vec![(Epoch::from(0), a)]);
    assert_eq!(it.next().unwrap(), vec![]);
    assert_eq!(it.next().unwrap(), vec![]);
}

#[test]
fn test_iter_ready_stops_on_stall() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();

    let mut it = scheduler.iter_ready();
    assert_eq!(it.next().unwrap(), vec![(Epoch::from(0), a)]);
    assert_eq!(it.next(), None);
}

/// iter_ready keeps yielding across an epoch boundary: ending epoch 0
/// unlocks the stateful source in the eagerly-created epoch 1
#[test]
fn test_iter_ready_crosses_epochs() {
    let edges = diamond();
    let mut nodes = FxIndexMap::default();
    nodes.insert(
        "a".to_string(),
        NodeFlags {
            enabled: true,
            stateful: Some(true),
        },
    );
    let mut scheduler = Scheduler::from_graph(nodes, edges).unwrap();

    let mut epochs_seen: Vec<u32> = Vec::new();
    let mut it = scheduler.iter_ready();
    for _ in 0..4 {
        let batch = it.next().unwrap();
        epochs_seen.push(batch[0].0.root());
        for (ep, node) in batch {
            it.done(&ep, node, true).unwrap();
        }
    }

    assert_eq!(epochs_seen, vec![0, 0, 0, 1]);
}

/// a root epoch has no parent to hang a subepoch on
#[test]
fn test_add_subepoch_root() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();

    let err = scheduler.add_subepoch(Epoch::from(5)).unwrap_err();
    assert!(matches!(err, CoreError::Value(_)));
}

#[test]
fn test_add_subepoch() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let a1 = interner()
        .get(&Item::Signal("a".to_string(), "a1".to_string()))
        .unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();
    let c = interner().get(&Item::Node("c".to_string())).unwrap();

    let ep = scheduler.add_epoch();
    scheduler.get_ready_at(&ep);
    scheduler.done(&ep, a, true).unwrap();
    // pull [b, c] - both pass out, as they would be mid-update
    scheduler.get_ready_at(&ep);

    let subep = scheduler.add_subepoch(&ep / (b, 0)).unwrap();

    // registered under the parent
    assert!(scheduler.subepochs[&ep].contains(&subep));

    // nothing is ready to issue in the fresh subepoch
    let subgraph = scheduler.epochs.get(&subep).unwrap();
    assert!(subgraph.done.contains(&a1));
    assert!(subgraph.out.contains(&b));
    assert!(subgraph.out.contains(&c));
    assert_eq!(scheduler.get_ready_at(&subep), vec![]);

    // inducing node expired in the parent: done there, but never ran
    let parent = scheduler.epochs.get(&ep).unwrap();
    assert!(parent.done.contains(&b));
    assert!(!parent.ran.contains(&b));
}

/// nodes that are expired in the parent carry over
/// so the subepoch doesn't wait on signals that will never fire.
/// c stays out in the parent so the parent survives the inducing-node expiry
#[test]
fn test_add_subepoch_expired_state() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();
    let c1 = interner()
        .get(&Item::Signal("c".to_string(), "c1".to_string()))
        .unwrap();
    let d = interner().get(&Item::Node("d".to_string())).unwrap();

    let ep = scheduler.add_epoch();
    scheduler.get_ready_at(&ep);
    scheduler.done(&ep, a, true).unwrap();
    scheduler.get_ready_at(&ep);
    // c ran but emitted NoEvent on c1
    scheduler.expire(&ep, c1, false, false).unwrap();

    let subep = scheduler.add_subepoch(&ep / (b, 0)).unwrap();

    // c1 expires in the parent carries into the subepoch as expired
    let subgraph = scheduler.epochs.get(&subep).unwrap();
    assert!(subgraph.done.contains(&c1));
    assert!(!subgraph.ran.contains(&c1));

    // b's events complete cleanly, but d stays blocked: an expired required
    // predecessor never unblocks its successors - the subepoch mirrors the
    // parent's fate for d
    scheduler.done(&subep, b, true).unwrap();
    assert!(!scheduler.epochs[&subep].ready.contains(&d));
    assert_eq!(scheduler.get_ready_at(&subep), vec![]);
}

/// subgraph templates are cached per inducing node: sibling subepochs
/// share one template
#[test]
fn test_subgraph_template_cached() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();

    let ep = scheduler.add_epoch();
    scheduler.get_ready_at(&ep);
    scheduler.done(&ep, a, true).unwrap();

    scheduler.add_subepoch(&ep / (b, 0)).unwrap();
    scheduler.add_subepoch(&ep / (b, 1)).unwrap();

    assert_eq!(scheduler.subgraph_templates.len(), 1);
    assert!(scheduler.epochs.contains_key(&(&ep / (b, 0))));
    assert!(scheduler.epochs.contains_key(&(&ep / (b, 1))));
}

/// out-of-order subepoch creation materializes the missing parent chain
#[test]
fn test_add_subepoch_missing_parent() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();

    let subep = scheduler.add_subepoch(Epoch::from(0) / (b, 0)).unwrap();

    assert!(scheduler.epochs.contains_key(&Epoch::from(0)));
    assert!(scheduler.epochs.contains_key(&subep));
    assert!(scheduler.subepochs[&Epoch::from(0)].contains(&subep));
}

#[test]
fn test_is_active_at_parent_survives_via_subepoch() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();
    let c = interner().get(&Item::Node("c".to_string())).unwrap();

    let ep = scheduler.add_epoch();
    scheduler.get_ready_at(&ep);
    scheduler.done(&ep, a, true).unwrap();
    scheduler.expire(&ep, c, true, true).unwrap();

    // expiring b (the inducing node) drains the parent's own work entirely
    let subep = scheduler.add_subepoch(&ep / (b, 0)).unwrap();

    assert!(!scheduler.epoch_log.contains(&ep.root()));
    assert!(scheduler.epochs.contains_key(&ep));
    // active purely through the subepoch
    assert!(scheduler.is_active_at(&ep));

    // fully draining the subepoch cascades:
    // subepoch inactive -> parent inactive -> root ends.
    // Parent's end events precede the subepoch's own
    let ended = scheduler.done(&subep, b, true).unwrap();
    assert_eq!(ended, vec![ep.clone(), subep.clone()]);

    assert!(scheduler.epoch_log.contains(&ep.root()));
    assert!(!scheduler.epochs.contains_key(&ep));
    // root-end GCs the subepoch SORTERS too
    assert!(!scheduler.epochs.contains_key(&subep));
    // and prunes the registry
    assert!(!scheduler.subepochs.contains_key(&ep));
}

/// a stale registry entry (subepoch GC'd, registration lingering) is
/// tolerated - python KeyErrors here (scheduler.py:236)
#[test]
fn test_is_active_at_stale_registry() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();

    scheduler
        .subepochs
        .entry(Epoch::from(7))
        .or_default()
        .insert(Epoch::from(7) / (b, 0));

    assert!(!scheduler.is_active_at(&Epoch::from(7)));
}

/// get_ready over an epoch includes its subepochs' ready nodes,
/// parent first (Epoch's Ord: a prefix sorts before its extensions).
#[test]
fn test_get_ready_at_includes_subepochs() {
    let mut edges = diamond();
    edges.push(edge("x", "x1", "y", true));
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();
    let c = interner().get(&Item::Node("c".to_string())).unwrap();
    let d = interner().get(&Item::Node("d".to_string())).unwrap();
    let x = interner().get(&Item::Node("x".to_string())).unwrap();
    let y = interner().get(&Item::Node("y".to_string())).unwrap();

    let ep = scheduler.add_epoch();
    scheduler.get_ready_at(&ep); // [a, x] out
    scheduler.done(&ep, a, true).unwrap();
    scheduler.get_ready_at(&ep); // [b, c] out

    // c completes in the parent BEFORE the subepoch exists, so the state
    // sync carries c1 over as genuinely done (with decrement) - the only
    // way d can ever become ready inside the subepoch
    scheduler.done(&ep, c, true).unwrap();

    let subep = scheduler.add_subepoch(&ep / (b, 0)).unwrap();
    scheduler.done(&subep, b, true).unwrap(); // unblocks d in the subepoch
    scheduler.done(&ep, x, true).unwrap(); // unblocks y in the parent

    let ready = scheduler.get_ready_at(&ep);
    assert_eq!(ready, vec![(ep.clone(), y), (subep.clone(), d)]);
}

/// done() on a missing subepoch key materializes a SUBGRAPH via the
/// add_epoch_at fork, not a root-template clone: upstream nodes are absent
#[test]
fn test_done_creates_subgraph_for_subepoch_key() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();
    let a2 = interner()
        .get(&Item::Signal("a".to_string(), "a2".to_string()))
        .unwrap();

    let ep = scheduler.add_epoch();
    scheduler.get_ready_at(&ep);
    scheduler.done(&ep, a, true).unwrap();

    let subep = &ep / (b, 0);
    scheduler.done(&subep, b, false).unwrap();

    // a2's edge targets c, which is outside downstream(b) - its absence is
    // what distinguishes a subgraph from a root-template clone (a itself IS
    // present, as a1's signal-linkage parent)

    let subgraph = scheduler.epochs.get(&subep).unwrap();
    assert!(!subgraph.info.contains_key(&a2));
    assert!(subgraph.done.contains(&b));
    // the get-or-create ran the full add_subepoch semantics:
    // registration + inducing-node expiry in the parent
    assert!(scheduler.subepochs[&ep].contains(&subep));
    assert!(scheduler.epochs[&ep].done.contains(&b));
    assert!(!scheduler.epochs[&ep].ran.contains(&b));
}

/// ending a subepoch under a still-active parent: no root logging, no GC of
/// anything (only root-end removes sorters, scheduler.py:547-562), and no
/// eager sibling creation - we only make subepochs when told they exist,
/// unlike root epochs, which always "theoretically exist" i guess,
/// and we only gc when the whole epoch is done, including subepochs.
#[test]
fn test_end_subepoch_live_parent() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();

    let ep = scheduler.add_epoch();
    scheduler.get_ready_at(&ep);
    scheduler.done(&ep, a, true).unwrap();
    scheduler.get_ready_at(&ep); // [b, c] out - parent stays active via c
    let subep = scheduler.add_subepoch(&ep / (b, 0)).unwrap();

    let ended = scheduler.end_epoch(subep.clone()).unwrap();
    assert_eq!(ended, vec![subep.clone()]);

    assert!(!scheduler.epoch_log.contains(&ep.root()));
    assert!(scheduler.epochs.contains_key(&ep));
    // the subepoch's sorter survives until the ROOT ends
    assert!(scheduler.epochs.contains_key(&subep));
    // and no sibling was eagerly created
    assert!(!scheduler.epochs.contains_key(&(&ep / (b, 1))));
}

/// AlreadyDone propagates on a plain epoch but is suppressed when the epoch
/// has subepochs (scheduler.py:415-417) - subepoch bookkeeping may
/// legitimately have marked the node first
#[test]
fn test_already_done_suppressed_with_subepochs() {
    // plain epoch: second done errors
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let ep = scheduler.add_epoch();
    scheduler.done(&ep, a, false).unwrap();
    let err = scheduler.done(&ep, a, false).unwrap_err();
    assert!(matches!(err, CoreError::AlreadyDone(_)));

    // epoch with subepochs: second done is swallowed
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();
    let ep = scheduler.add_epoch();
    scheduler.get_ready_at(&ep);
    scheduler.done(&ep, a, true).unwrap();
    scheduler.get_ready_at(&ep);
    scheduler.add_subepoch(&ep / (b, 0)).unwrap();

    assert_eq!(scheduler.done(&ep, a, false).unwrap(), vec![]);
}

/// expiring a non-inducing node in the parent recurses into subepochs
#[test]
fn test_expire_recurses_subepochs() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();
    let c = interner().get(&Item::Node("c".to_string())).unwrap();

    let ep = scheduler.add_epoch();
    scheduler.get_ready_at(&ep);
    scheduler.done(&ep, a, true).unwrap();
    scheduler.get_ready_at(&ep); // [b, c] out
    let subep = scheduler.add_subepoch(&ep / (b, 0)).unwrap();

    // c did not induce the subepoch: its expiry propagates there
    scheduler.expire(&ep, c, false, false).unwrap();
    let subgraph = scheduler.epochs.get(&subep).unwrap();
    assert!(subgraph.done.contains(&c));
    assert!(!subgraph.ran.contains(&c));

    // b induced it: expiring b in the parent leaves the subepoch's b alone
    scheduler.expire(&ep, b, false, false).unwrap();
    assert!(!scheduler.epochs[&subep].done.contains(&b));
}

/// the inducing-node skip goes by the item's NODE: expiring a *signal* of
/// the inducing node in the parent must also skip that node's subepoch
/// (python compares node_id, scheduler.py:477)
#[test]
fn test_expire_signal_skips_inducing_subepoch() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();
    let b1 = interner()
        .get(&Item::Signal("b".to_string(), "b1".to_string()))
        .unwrap();

    let ep = scheduler.add_epoch();
    scheduler.get_ready_at(&ep);
    scheduler.done(&ep, a, true).unwrap();
    scheduler.get_ready_at(&ep);
    let subep = scheduler.add_subepoch(&ep / (b, 0)).unwrap();

    // b1 belongs to b, the inducing node - the subepoch handles its own b1
    scheduler.expire(&ep, b1, false, false).unwrap();
    assert!(!scheduler.epochs[&subep].done.contains(&b1));
}

/// a node expired in the parent carries into the
/// subepoch as expired; when the node later completes for real, the parent
/// hits AlreadyDone (suppressed - subepochs exist), and done_subepochs
/// resurrects the subepoch's expired entry into properly-done.
#[test]
fn test_done_subepochs_resurrects_expired() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(FxIndexMap::default(), edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();
    let c = interner().get(&Item::Node("c".to_string())).unwrap();

    let ep = scheduler.add_epoch();
    scheduler.get_ready_at(&ep);
    scheduler.done(&ep, a, true).unwrap();
    scheduler.get_ready_at(&ep); // [b, c] out
    scheduler.expire(&ep, c, false, false).unwrap();

    let subep = scheduler.add_subepoch(&ep / (b, 0)).unwrap();
    // carried over as expired
    assert!(scheduler.epochs[&subep].done.contains(&c));
    assert!(!scheduler.epochs[&subep].ran.contains(&c));

    // the late "c actually completed" event
    scheduler.done(&ep, c, false).unwrap();
    assert!(scheduler.epochs[&subep].ran.contains(&c));
}

/// Exclusive-downstream expiry (the gather case):
/// with a subepoch induced by b, completing a marks c - downstream of a but
/// unreachable from b - as expired in the subepoch: c only runs in the
/// parent. Deliberate divergence: python filters our_subgraph through the
/// spec'd-nodes map purely for convention reasons, noob tubes always have node specs;
/// rust moves nodes by topology alone, so edge-only nodes are treated the same as spec'd ones.
#[test]
fn test_done_subepochs_exclusive_expiry() {
    let edges = diamond();
    let nodes: FxIndexMap<String, NodeFlags> = ["a", "b", "c", "d"]
        .into_iter()
        .map(|n| {
            (
                n.to_string(),
                NodeFlags {
                    enabled: true,
                    stateful: None,
                },
            )
        })
        .collect();
    let mut scheduler = Scheduler::from_graph(nodes, edges).unwrap();
    let a = interner().get(&Item::Node("a".to_string())).unwrap();
    let b = interner().get(&Item::Node("b".to_string())).unwrap();
    let c = interner().get(&Item::Node("c".to_string())).unwrap();

    let ep = scheduler.add_epoch();
    // subepoch exists before a completes
    let subep = scheduler.add_subepoch(&ep / (b, 0)).unwrap();

    scheduler.done(&ep, a, false).unwrap();

    let subgraph = scheduler.epochs.get(&subep).unwrap();
    // a propagated as properly done
    assert!(subgraph.ran.contains(&a));
    // c is exclusively downstream of a (unreachable from b): expired
    assert!(subgraph.done.contains(&c));
    assert!(!subgraph.ran.contains(&c));

    // skip-if-ran: a second completion event is inert, not an error
    scheduler.done(&ep, a, false).unwrap();
}

#[test]
fn test_downstream_nodes() {
    let edges = diamond();

    let downstream = downstream_nodes(&edges, "a", &FxIndexSet::default());
    assert_eq!(downstream, FxIndexSet::from_iter(["a", "b", "c", "d"]));

    let downstream = downstream_nodes(&edges, "b", &FxIndexSet::default());
    assert_eq!(downstream, FxIndexSet::from_iter(["b", "d"]));

    // a leaf is its own downstream set
    let downstream = downstream_nodes(&edges, "d", &FxIndexSet::default());
    assert_eq!(downstream, FxIndexSet::from_iter(["d"]));
}

/// exclude prunes paths *through* the excluded node, not everything below it:
/// d is still reachable via c
#[test]
fn test_downstream_nodes_exclude() {
    let edges = diamond();
    let exclude = FxIndexSet::from_iter(["b"]);

    let downstream = downstream_nodes(&edges, "a", &exclude);
    assert_eq!(downstream, FxIndexSet::from_iter(["a", "c", "d"]));
}
