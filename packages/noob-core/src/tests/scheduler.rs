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

/// an epoch can't be created when it would fall below the epoch log, even if it's never actually been run
#[test]
fn test_add_epoch_at_below_log() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(IndexMap::default(), edges).unwrap();
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
    // since 'a' is a source node, epoch 11 should have been added
    assert!(scheduler.epochs.contains_key(&Epoch::from(11)));
    assert_eq!(scheduler.next_epoch, 12);
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

#[test]
fn test_sources_finished() {
    let edges = diamond();
    let mut nodes = IndexMap::new();
    nodes.insert(
        "a".to_string(),
        NodeFlags {
            enabled: true,
            stateful: Some(true),
        },
    );
    let mut scheduler = Scheduler::from_graph(nodes, edges).unwrap();
    let a = scheduler.interner.intern_node("a");
    assert_eq!(scheduler.source_nodes, FxIndexSet::from_iter(vec![a]));

    let ep = scheduler.add_epoch();
    assert!(!scheduler.sources_finished(&ep));
    scheduler.done(&ep, a, true).unwrap();
    assert!(scheduler.sources_finished(&ep));
    assert!(scheduler.epochs.contains_key(&Epoch::from(ep.root() + 1)));

    // false for epochs that haven't been added yet
    assert!(!scheduler.sources_finished(&Epoch::from(99)))
}

#[test]
fn test_eager_epoch_creation_when_sources_done() {
    let edges = diamond();
    let mut nodes = IndexMap::new();
    nodes.insert(
        "a".to_string(),
        NodeFlags {
            enabled: true,
            stateful: Some(true),
        },
    );
    let mut scheduler = Scheduler::from_graph(nodes, edges).unwrap();
    let a = scheduler.interner.intern_node("a");
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
    let b = scheduler.interner.intern_node("b");
    let ep = scheduler.add_epoch();
    let next = Epoch::from(ep.root() + 1);
    scheduler.done(&ep, b, true).unwrap();
    assert!(!scheduler.epochs.contains_key(&next));
}

#[test]
fn test_epoch_log_trim() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(IndexMap::default(), edges).unwrap();
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
    let mut scheduler = Scheduler::from_graph(IndexMap::default(), edges).unwrap();
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
    let mut scheduler = Scheduler::from_graph(IndexMap::default(), edges).unwrap();
    let a = scheduler.interner.intern_node("a");
    let a1 = scheduler.interner.intern_signal("a", "a1");
    let a2 = scheduler.interner.intern_signal("a", "a2");
    let b = scheduler.interner.intern_node("b");
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
    let mut scheduler = Scheduler::from_graph(IndexMap::default(), edges).unwrap();
    let a = scheduler.interner.intern_node("a");
    let ep = scheduler.add_epoch();
    scheduler.get_ready_at(&ep);
    let ended = scheduler.expire(&ep, a, true, true).unwrap();
    assert_eq!(ended, vec![ep]);
}

/// cleanup from python - expiring a node in a completed epoch is suppressed
#[test]
fn test_expire_completed_epoch() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(IndexMap::default(), edges).unwrap();
    let a = scheduler.interner.intern_node("a");
    let ep = scheduler.add_epoch();
    scheduler.end_epoch(ep.clone()).unwrap();
    scheduler.expire(&ep, a, true, true).unwrap();
}

#[test]
fn test_iter_epoch_to_completion() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(IndexMap::default(), edges).unwrap();
    let a = scheduler.interner.intern_node("a");
    let b = scheduler.interner.intern_node("b");
    let c = scheduler.interner.intern_node("c");
    let d = scheduler.interner.intern_node("d");

    let mut batches: Vec<Vec<u16>> = Vec::new();
    let mut it = scheduler.iter_epoch();
    while let Some(batch) = it.next() {
        let nodes: Vec<u16> = batch.iter().map(|(_, node)| *node).collect();
        for &node in &nodes {
            it.done(node, true).unwrap();
        }
        batches.push(nodes);
    }

    assert_eq!(batches, vec![vec![a], vec![b, c], vec![d]]);
    assert!(scheduler.epoch_completed(&Epoch::from(0)));
}

#[test]
fn test_iter_epoch_at_completed_epoch() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(IndexMap::default(), edges).unwrap();
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
    let mut scheduler = Scheduler::from_graph(IndexMap::default(), edges).unwrap();
    let a = scheduler.interner.intern_node("a");

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
    let mut scheduler = Scheduler::from_graph(IndexMap::default(), edges).unwrap();
    let a = scheduler.interner.intern_node("a");

    let mut it = scheduler.iter_epoch();
    assert_eq!(it.next().unwrap(), vec![(Epoch::from(0), a)]);
    assert_eq!(it.next().unwrap(), vec![]);
    assert_eq!(it.next().unwrap(), vec![]);
}

#[test]
fn test_iter_ready_stops_on_stall() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(IndexMap::default(), edges).unwrap();
    let a = scheduler.interner.intern_node("a");

    let mut it = scheduler.iter_ready();
    assert_eq!(it.next().unwrap(), vec![(Epoch::from(0), a)]);
    assert_eq!(it.next(), None);
}

/// iter_ready keeps yielding across an epoch boundary: ending epoch 0
/// unlocks the stateful source in the eagerly-created epoch 1
#[test]
fn test_iter_ready_crosses_epochs() {
    let edges = diamond();
    let mut nodes = IndexMap::new();
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
    let mut scheduler = Scheduler::from_graph(IndexMap::default(), edges).unwrap();

    let err = scheduler.add_subepoch(Epoch::from(5)).unwrap_err();
    assert!(matches!(err, CoreError::Value(_)));
}


#[test]
fn test_add_subepoch() {
    let edges = diamond();
    let mut scheduler = Scheduler::from_graph(IndexMap::default(), edges).unwrap();
    let a = scheduler.interner.intern_node("a");
    let a1 = scheduler.interner.intern_signal("a", "a1");
    let b = scheduler.interner.intern_node("b");
    let c = scheduler.interner.intern_node("c");

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
    let mut scheduler = Scheduler::from_graph(IndexMap::default(), edges).unwrap();
    let a = scheduler.interner.intern_node("a");
    let b = scheduler.interner.intern_node("b");
    let c1 = scheduler.interner.intern_signal("c", "c1");
    let d = scheduler.interner.intern_node("d");

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
    let mut scheduler = Scheduler::from_graph(IndexMap::default(), edges).unwrap();
    let a = scheduler.interner.intern_node("a");
    let b = scheduler.interner.intern_node("b");

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
    let mut scheduler = Scheduler::from_graph(IndexMap::default(), edges).unwrap();
    let b = scheduler.interner.intern_node("b");

    let subep = scheduler.add_subepoch(Epoch::from(0) / (b, 0)).unwrap();

    assert!(scheduler.epochs.contains_key(&Epoch::from(0)));
    assert!(scheduler.epochs.contains_key(&subep));
    assert!(scheduler.subepochs[&Epoch::from(0)].contains(&subep));
}
