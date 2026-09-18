use indexmap::IndexSet;

use super::*;

fn chain_graph(stateful: Option<bool>) -> (FxIndexMap<String, NodeFlags>, Vec<EdgeRec>) {
    let nodes: FxIndexMap<String, NodeFlags> = ["a", "b", "c"]
        .into_iter()
        .map(|n| {
            (
                n.to_string(),
                NodeFlags {
                    enabled: true,
                    stateful,
                },
            )
        })
        .collect();
    let edges = vec![
        EdgeRec {
            source_node: "a".into(),
            source_signal: "sig".into(),
            target_node: "b".into(),
            target_slot: "slot".into(),
            required: true,
        },
        EdgeRec {
            source_node: "b".into(),
            source_signal: "sig".into(),
            target_node: "c".into(),
            target_slot: "slot".into(),
            required: true,
        },
    ];
    (nodes, edges)
}

fn chain(stateful: Option<bool>) -> (Interner, Sorter) {
    let (nodes, edges) = chain_graph(stateful);
    let mut interner = Interner::default();
    let sorter = Sorter::from_graph(&mut interner, &nodes, &edges).unwrap();
    (interner, sorter)
}

/// Port of `_static_order_with_groups`: drive the sorter to completion,
/// collecting each `get_ready` generation as a sorted group of node names.
fn static_order_with_groups(interner: &mut Interner, sorter: &mut Sorter) -> Vec<Vec<String>> {
    let mut groups = Vec::new();
    while sorter.is_active() {
        let ready = sorter.get_ready(interner);
        let out: Vec<ItemID> = sorter.out.iter().copied().collect();
        if ready.is_empty() && out.is_empty() {
            // python's generator would loop forever here; fail loudly instead
            panic!("sorter is_active() but nothing is ready or out");
        }
        sorter.done(interner, &out).unwrap();
        let mut group: Vec<String> = ready
            .iter()
            .map(|id| interner.resolve(*id).node_id().to_string())
            .collect();
        group.sort();
        groups.push(group);
    }
    groups
}

/// Port of `_test_graph`: build a sorter from `(node, dependencies)` pairs
/// (bare node -> node dependencies, no signals, like the graphlib tests),
/// run it, and compare the generations against `expected`.
fn test_graph(graph: &[(&str, &[&str])], expected: &[&[&str]]) {
    let mut interner = Interner::default();
    let mut sorter = Sorter::default();
    for (node, deps) in graph {
        let node_id = interner.intern_node(node);
        let dep_ids: Vec<ItemID> = deps.iter().map(|d| interner.intern_node(d)).collect();
        sorter.add(&mut interner, node_id, &dep_ids, true).unwrap();
    }

    let actual = static_order_with_groups(&mut interner, &mut sorter);
    let expected: Vec<Vec<String>> = expected
        .iter()
        .map(|group| {
            let mut group: Vec<String> = group.iter().map(|s| s.to_string()).collect();
            group.sort();
            group
        })
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_simple_cases() {
    test_graph(
        &[
            ("2", &["11"]),
            ("9", &["11", "8"]),
            ("10", &["11", "3"]),
            ("11", &["7", "5"]),
            ("8", &["7", "3"]),
        ],
        &[&["3", "5", "7"], &["8", "11"], &["2", "9", "10"]],
    );

    test_graph(&[("1", &[])], &[&["1"]]);

    // python builds this one with a comprehension; a literal chain
    // 0 <- 1 <- ... <- 10 avoids an ownership dance over dynamic strings
    test_graph(
        &[
            ("0", &["1"]),
            ("1", &["2"]),
            ("2", &["3"]),
            ("3", &["4"]),
            ("4", &["5"]),
            ("5", &["6"]),
            ("6", &["7"]),
            ("7", &["8"]),
            ("8", &["9"]),
            ("9", &["10"]),
        ],
        &[
            &["10"],
            &["9"],
            &["8"],
            &["7"],
            &["6"],
            &["5"],
            &["4"],
            &["3"],
            &["2"],
            &["1"],
            &["0"],
        ],
    );

    test_graph(
        &[
            ("2", &["3"]),
            ("3", &["4"]),
            ("4", &["5"]),
            ("5", &["1"]),
            ("11", &["12"]),
            ("12", &["13"]),
            ("13", &["14"]),
            ("14", &["15"]),
        ],
        &[
            &["1", "15"],
            &["5", "14"],
            &["4", "13"],
            &["3", "12"],
            &["2", "11"],
        ],
    );

    test_graph(
        &[
            ("0", &["1", "2"]),
            ("1", &["3"]),
            ("2", &["5", "6"]),
            ("3", &["4"]),
            ("4", &["9"]),
            ("5", &["3"]),
            ("6", &["7"]),
            ("7", &["8"]),
            ("8", &["4"]),
            ("9", &[]),
        ],
        &[
            &["9"],
            &["4"],
            &["3", "8"],
            &["1", "5", "7"],
            &["6"],
            &["2"],
            &["0"],
        ],
    );

    test_graph(
        &[("0", &["1", "2"]), ("1", &[]), ("2", &["3"]), ("3", &[])],
        &[&["1", "3"], &["2"], &["0"]],
    );

    test_graph(
        &[
            ("0", &["1", "2"]),
            ("1", &[]),
            ("2", &["3"]),
            ("3", &[]),
            ("4", &["5"]),
            ("5", &["6"]),
            ("6", &[]),
        ],
        &[&["1", "3", "6"], &["2", "5"], &["0", "4"]],
    );
}

/// Port of `_assert_cycle`: build a sorter from `(node, dependencies)`
/// pairs and assert `find_cycle` reports exactly `cycle`.
fn assert_cycle(graph: &[(&str, &[&str])], cycle: &[&str]) {
    let mut interner = Interner::default();
    let mut sorter = Sorter::default();
    for (node, deps) in graph {
        let node_id = interner.intern_node(node);
        let dep_ids: Vec<ItemID> = deps.iter().map(|d| interner.intern_node(d)).collect();
        sorter.add(&mut interner, node_id, &dep_ids, true).unwrap();
    }
    let found: Option<Vec<String>> = sorter.find_cycle().map(|c| {
        c.iter()
            .map(|id| interner.resolve(*id).node_id().to_string())
            .collect()
    });
    let expected: Vec<String> = cycle.iter().map(|s| s.to_string()).collect();
    assert_eq!(found, Some(expected));
}

#[test]
fn test_cycle() {
    // self cycle
    assert_cycle(&[("1", &["1"])], &["1", "1"]);
    // simple cycle
    assert_cycle(&[("1", &["2"]), ("2", &["1"])], &["2", "1", "2"]);
    // indirect cycle
    assert_cycle(
        &[("1", &["2"]), ("2", &["3"]), ("3", &["1"])],
        &["2", "1", "3", "2"],
    );
    // not all elements involved in a cycle
    assert_cycle(
        &[
            ("1", &["2"]),
            ("2", &["3"]),
            ("3", &["1"]),
            ("5", &["4"]),
            ("4", &["6"]),
        ],
        &["2", "1", "3", "2"],
    );
    // multiple cycles
    assert_cycle(
        &[
            ("1", &["2"]),
            ("2", &["1"]),
            ("3", &["4"]),
            ("4", &["5"]),
            ("6", &["7"]),
            ("7", &["6"]),
        ],
        &["2", "1", "2"],
    );
    // cycle in the middle of the graph
    assert_cycle(
        &[
            ("1", &["2"]),
            ("2", &["3"]),
            ("3", &["2", "4"]),
            ("4", &["5"]),
        ],
        &["2", "3", "2"],
    );
}

#[test]
fn test_no_cycle() {
    // a DAG has no cycle; also exercises the `chain` fixture
    let (_interner, sorter) = chain(None);
    assert_eq!(sorter.find_cycle(), None);
}

#[test]
fn test_no_dependencies() {
    test_graph(
        &[("1", &["2"]), ("3", &["4"]), ("5", &["6"])],
        &[&["2", "4", "6"], &["1", "3", "5"]],
    );

    test_graph(&[("1", &[]), ("3", &[]), ("5", &[])], &[&["1", "3", "5"]]);
}

fn edge(source: &str, signal: &str, target: &str, slot: &str, required: bool) -> EdgeRec {
    EdgeRec {
        source_node: source.into(),
        source_signal: signal.into(),
        target_node: target.into(),
        target_slot: slot.into(),
        required,
    }
}

/// Port of the `optional_graph` fixture (tests/fixtures/sorter.py)
fn optional_graph() -> (Interner, Sorter) {
    let mut interner = Interner::default();
    let edges = vec![
        edge("a", "a1", "only_optional", "x", false),
        edge("a", "a1", "mixed", "x", false),
        edge("a", "a2", "mixed", "y", true),
        edge("mixed", "value", "two_hop", "x", false),
        edge("a", "a2", "two_hop", "y", true),
        // linear chain to test upstream node/long range dependencies
        edge("a", "a1", "b", "x", false),
        edge("b", "b1", "c", "x", true),
        edge("c", "c1", "d", "x", false),
    ];
    let sorter = Sorter::from_graph(&mut interner, &FxIndexMap::default(), &edges).unwrap();
    (interner, sorter)
}

/// Port of `test_derive_optional_adjacency`: optional predecessors and
/// successors are derived correctly - successors found up to the nearest
/// optional and no further, and only signals get optional successors.
#[test]
fn test_derive_optional_adjacency() {
    let (mut interner, sorter) = optional_graph();
    let a = interner.intern_node("a");
    let a_a1 = interner.intern_signal("a", "a1");
    let mixed = interner.intern_node("mixed");
    let mixed_value = interner.intern_signal("mixed", "value");
    let mixed_slot = interner.intern_slot("mixed", "x");
    let only_optional = interner.intern_node("only_optional");
    let opt_slot = interner.intern_slot("only_optional", "x");
    let two_hop = interner.intern_node("two_hop");
    let two_hop_slot = interner.intern_slot("two_hop", "x");
    let b_slot = interner.intern_slot("b", "x");

    assert_eq!(
        sorter.info[&only_optional].optional_predecessors,
        FxHashMap::from_iter([(a_a1, opt_slot)])
    );
    assert_eq!(
        sorter.info[&mixed].optional_predecessors,
        FxHashMap::from_iter([(a_a1, mixed_slot)])
    );
    assert_eq!(
        sorter.info[&two_hop].optional_predecessors,
        FxHashMap::from_iter([(mixed_value, two_hop_slot)])
    );

    assert_eq!(
        sorter.info[&a_a1].optional_successors,
        IndexSet::from([opt_slot, mixed_slot, b_slot]),
        "{:?}",
        sorter.info[&a_a1]
            .optional_successors
            .iter()
            .cloned()
            .map(|id| interner.resolve(id).to_string())
            .fold(String::new(), |a, b| a + &*b + ",")
    );
    assert_eq!(
        sorter.info[&mixed_value].optional_successors,
        IndexSet::from([two_hop_slot])
    );
    // nodes do not get optional successors,
    // signals are the things that are NoEvent or not
    assert!(sorter.info[&a].optional_successors.is_empty());
    assert!(sorter.info[&mixed].optional_successors.is_empty());
}

/// Port of `test_optional_dependencies`: nodes with optional dependencies
/// run when those upstream nodes are expired.
#[test]
fn test_optional_dependencies() {
    let (mut interner, mut sorter) = optional_graph();
    let a = interner.intern_node("a");
    let a_a1 = interner.intern_signal("a", "a1");
    let a_a2 = interner.intern_signal("a", "a2");
    let mixed = interner.intern_node("mixed");
    let mixed_value = interner.intern_signal("mixed", "value");
    let only_optional = interner.intern_node("only_optional");
    let two_hop = interner.intern_node("two_hop");
    let b = interner.intern_node("b");
    let b_b1 = interner.intern_signal("b", "b1");

    // nodes with only optional dependencies still wait for those to be decided
    let ready = sorter.get_ready(&interner);
    assert_eq!(ready, vec![a]);
    sorter.done(&interner, &ready).unwrap();
    assert_eq!(sorter.out, IndexSet::from([a_a1, a_a2]));

    sorter.mark_expired(&interner, &[a_a1], true);
    sorter.done(&interner, &[a_a2]).unwrap();
    let ready: IndexSet<ItemID> = sorter.get_ready(&interner).into_iter().collect();
    assert_eq!(ready, IndexSet::from([only_optional, mixed, b]));

    sorter.done(&interner, &[only_optional, mixed]).unwrap();
    assert_eq!(sorter.out, IndexSet::from([b, b_b1, mixed_value]));

    sorter.mark_expired(&interner, &[mixed_value], true);
    let ready = sorter.get_ready(&interner);
    assert_eq!(ready, vec![two_hop]);
}

/// Port of `test_unlock_optionals`: the `unlock_optionals` arg controls
/// whether expiring nodes makes their downstream optional dependents ready.
/// (pytest parametrizes over the bool; here it's a helper called twice.)
fn unlock_optionals_case(unlock_optionals: bool) {
    let (mut interner, mut sorter) = optional_graph();
    let ready = sorter.get_ready(&interner);
    sorter.done(&interner, &ready).unwrap();

    let out: Vec<ItemID> = sorter.out.iter().copied().collect();
    sorter.mark_expired(&interner, &out, unlock_optionals);
    let ready: IndexSet<ItemID> = sorter.get_ready(&interner).into_iter().collect();
    if unlock_optionals {
        let expected = IndexSet::from([
            interner.intern_node("only_optional"),
            interner.intern_node("b"),
        ]);
        assert_eq!(ready, expected);
    } else {
        assert!(ready.is_empty());
    }
}

#[test]
fn test_unlock_optionals_true() {
    unlock_optionals_case(true);
}

#[test]
fn test_unlock_optionals_false() {
    unlock_optionals_case(false);
}

/// When a node has mixed required and optional deps, unlocking the optional doesn't unlock the node
#[test]
fn test_unlock_optionals_mixed() {
    let (mut interner, mut sorter) = optional_graph();
    let a = interner.get(&Item::Node("a".into())).unwrap();
    let a1 = interner
        .get(&Item::Signal("a".into(), "a1".into()))
        .unwrap();
    let a2 = interner
        .get(&Item::Signal("a".into(), "a2".into()))
        .unwrap();
    sorter.done(&interner, &[a]).unwrap();

    let expected = IndexSet::from([a1, a2]);
    assert_eq!(sorter.ready, expected);

    sorter.mark_expired(&interner, &[a1], true);
    let expected = IndexSet::from([
        a2,
        interner.intern_node("only_optional"),
        interner.intern_node("b"),
    ]);
    assert_eq!(sorter.ready, expected);

    sorter.mark_expired(&interner, &[a2], true);
    let expected = IndexSet::from([
        interner.intern_node("only_optional"),
        interner.intern_node("b"),
    ]);
    assert_eq!(sorter.ready, expected);
}

/// Regression - optional chains do not double-decrement for a single optional slot
/// Using concrete node and slot names just to keep them straight in my head
/// https://github.com/miniscope/noob/issues/259
#[test]
fn test_unlock_optionals_bookkeeping() {
    let graph = vec![
        edge("trim", "header", "target", "required", true),
        edge("combine", "frame_idx", "target", "optional", false),
        edge("trim", "buffer", "combine", "buffer", true),
        edge("parse_header", "header", "combine", "header", true),
        edge("parse_header", "buffer", "trim", "buffer", true),
        edge("parse_header", "header", "trim", "header", true),
    ];

    let mut interner = Interner::default();
    let mut sorter = Sorter::from_graph(&mut interner, &FxIndexMap::default(), &graph).unwrap();

    let parse_header = interner.get(&("parse_header".into())).unwrap();
    let parse_header_header = interner.get(&("parse_header", "header").into()).unwrap();
    let parse_header_buffer = interner.get(&("parse_header", "buffer").into()).unwrap();
    let trim_header = interner.intern_signal("trim", "header");
    let target = interner.intern_node("target");
    let target_optional = interner.intern_slot("target", "optional");
    let combine_frame_idx = interner.intern_signal("combine", "frame_idx");

    assert_eq!(sorter.get_nodeinfo(target).nqueue, 2);
    assert_eq!(
        sorter.get_nodeinfo(combine_frame_idx).optional_successors,
        IndexSet::from([target_optional])
    );
    assert_eq!(
        sorter.get_nodeinfo(parse_header_header).optional_successors,
        IndexSet::from([target_optional])
    );
    assert_eq!(
        sorter.get_nodeinfo(parse_header_buffer).optional_successors,
        IndexSet::from([target_optional])
    );

    // when parse_header.header and .buffer are both NoEvent, we should *not* incorrectly make the target ready
    let ready = sorter.get_ready(&interner);
    assert_eq!(vec![parse_header], ready);
    sorter.done(&interner, &ready).unwrap();
    sorter.mark_expired(&interner, &[parse_header_header, parse_header_buffer], true);
    assert_eq!(sorter.get_nodeinfo(target).nqueue, 1);

    // "target" should not be ready here
    let ready = sorter.get_ready(&interner);
    assert_eq!(vec![] as Vec<ItemID>, ready);

    // if we force the required dep to be done, though, now it should be
    sorter.done(&interner, &[trim_header]).unwrap();
    let ready = sorter.get_ready(&interner);
    assert_eq!(vec![target] as Vec<ItemID>, ready);
}

/// Optional dependencies should only propagate to signal paths that are directly upstream,
/// rather than leaking to other signals when propagated through a branch and merge
#[test]
fn test_update_optionals_is_constrained_to_direct_paths() {
    let graph = vec![
        edge("switch", "a", "branch_a", "value", true),
        edge("switch", "b", "branch_b", "value", true),
        edge("switch", "c", "branch_c", "value", true),
        edge("branch_a", "value", "merge", "slot_a", false),
        edge("branch_b", "value", "merge", "slot_b", false),
        edge("branch_c", "value", "merge", "slot_c", false),
    ];

    let mut interner = Interner::default();
    let sorter = Sorter::from_graph(&mut interner, &FxIndexMap::default(), &graph).unwrap();

    let a = interner.intern_signal("switch", "a");
    let b = interner.intern_signal("switch", "b");
    let c = interner.intern_signal("switch", "c");
    let slot_a = interner.intern_slot("merge", "slot_a");
    let slot_b = interner.intern_slot("merge", "slot_b");
    let slot_c = interner.intern_slot("merge", "slot_c");

    assert_eq!(
        sorter.info.get(&a).unwrap().optional_successors,
        IndexSet::from([slot_a])
    );
    assert_eq!(
        sorter.info.get(&b).unwrap().optional_successors,
        IndexSet::from([slot_b])
    );
    assert_eq!(
        sorter.info.get(&c).unwrap().optional_successors,
        IndexSet::from([slot_c])
    );
}

/// Regression - ensure that nodes that are disabled are not added to the graph even when stateful
#[test]
fn test_disabled_stateful_not_added() {
    let mut nodes: FxIndexMap<String, NodeFlags> = FxIndexMap::default();
    nodes.insert(
        "a".to_string(),
        NodeFlags {
            enabled: false,
            stateful: Some(true),
        },
    );
    let mut interner = Interner::default();
    let sorter = Sorter::from_graph(&mut interner, &nodes, &Vec::new()).unwrap();
    assert!(!sorter.info.contains_key(&interner.intern_node("a")));
}

#[test]
fn test_source_nodes() {
    let (mut interner, sorter) = chain(Some(true));
    let a = interner.intern_node("a");
    assert_eq!(FxIndexSet::from_iter(vec![a]), sorter.source_nodes());
}

/// A disabled node is not considered a source node
/// e.g. in the graph
/// a -> b
/// c -> d
/// where `a` is disabled, `b` will never run,
/// but a new graph should be created when `c` is done.
#[test]
fn test_source_nodes_disabled() {
    let (mut nodes, edges) = chain_graph(Some(true));
    nodes["a"].enabled = false;
    let mut interner = Interner::default();
    let sorter = Sorter::from_graph(&mut interner, &nodes, &edges).unwrap();
    assert!(sorter.source_nodes().is_empty());
}

/// Sorters are exhausted when they have no nodes that can be reached,
/// or only meta nodes are reachable.
#[test]
fn test_exhausted_disabled() {
    let (mut nodes, edges) = chain_graph(Some(true));
    nodes["a"].enabled = false;

    let mut interner = Interner::default();
    let sorter = Sorter::from_graph(&mut interner, &nodes, &edges).unwrap();
    assert!(sorter.exhausted);
}

#[test]
fn test_exhausted_meta_nodes() {
    let (mut nodes, mut edges) = chain_graph(Some(true));
    edges.push(EdgeRec {
        source_node: "input".to_string(),
        source_signal: "a".to_string(),
        target_node: "a".to_string(),
        target_slot: "x".to_string(),
        required: true,
    });
    // connected to 'b', which remains active, and thus in the graph when a is disabled
    edges.push(EdgeRec {
        source_node: "assets".to_string(),
        source_signal: "a".to_string(),
        target_node: "b".to_string(),
        target_slot: "x".to_string(),
        required: true,
    });

    let mut interner = Interner::default();
    let sorter = Sorter::from_graph(&mut interner, &nodes, &edges).unwrap();
    let input = interner.get(&Item::Node("input".into())).unwrap();
    let assets = interner.get(&Item::Node("assets".into())).unwrap();

    // not exhausted just because the meta nodes are first
    assert_eq!(
        sorter.ready,
        FxIndexSet::from_iter(vec![input, assets, PREVIOUS_EPOCH])
    );
    assert!(!sorter.exhausted);

    // now when they're the only reachable nodes, should be exhausted
    nodes["a"].enabled = false;
    let sorter = Sorter::from_graph(&mut interner, &nodes, &edges).unwrap();
    // input is not present, because it is only added by "a", which is not added because it's disabled.
    assert_eq!(
        sorter.ready,
        FxIndexSet::from_iter(vec![assets, PREVIOUS_EPOCH]),
        "generations: {:?}",
        generations(sorter.clone(), &interner)
    );
    assert!(
        sorter.exhausted,
        "generations: {:?}",
        generations(sorter.clone(), &interner)
    );
}
