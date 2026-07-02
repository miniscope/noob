use super::*;

fn chain() -> (Interner, Sorter) {
    let mut interner = Interner::default();
    let nodes: IndexMap<String, NodeFlags> = ["a", "b", "c"]
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
    let edges = vec![
        EdgeRec {
            source_node: "a".into(),
            source_signal: "sig".into(),
            target_node: "b".into(),
            required: true,
        },
        EdgeRec {
            source_node: "b".into(),
            source_signal: "sig".into(),
            target_node: "c".into(),
            required: true,
        },
    ];
    let sorter = Sorter::from_graph(&mut interner, &nodes, &edges).unwrap();
    (interner, sorter)
}

/// Port of `_static_order_with_groups`: drive the sorter to completion,
/// collecting each `get_ready` generation as a sorted group of node names.
fn static_order_with_groups(interner: &mut Interner, sorter: &mut Sorter) -> Vec<Vec<String>> {
    let mut groups = Vec::new();
    while sorter.is_active() {
        let ready = sorter.get_ready(interner, None);
        let out: Vec<u32> = sorter.out.iter().copied().collect();
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
        let dep_ids: Vec<u32> = deps.iter().map(|d| interner.intern_node(d)).collect();
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
        let dep_ids: Vec<u32> = deps.iter().map(|d| interner.intern_node(d)).collect();
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
    let (_interner, sorter) = chain();
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

fn edge(source: &str, signal: &str, target: &str, required: bool) -> EdgeRec {
    EdgeRec {
        source_node: source.into(),
        source_signal: signal.into(),
        target_node: target.into(),
        required,
    }
}

/// Port of the `optional_graph` fixture (tests/fixtures/sorter.py)
fn optional_graph() -> (Interner, Sorter) {
    let mut interner = Interner::default();
    let edges = vec![
        edge("a", "a1", "only_optional", false),
        edge("a", "a1", "mixed", false),
        edge("a", "a2", "mixed", true),
        edge("mixed", "value", "two_hop", false),
        edge("a", "a2", "two_hop", true),
        // linear chain to test upstream node/long range dependencies
        edge("a", "a1", "b", false),
        edge("b", "b1", "c", true),
        edge("c", "c1", "d", false),
    ];
    let sorter = Sorter::from_graph(&mut interner, &IndexMap::new(), &edges).unwrap();
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
    let only_optional = interner.intern_node("only_optional");
    let two_hop = interner.intern_node("two_hop");
    let b = interner.intern_node("b");

    assert_eq!(
        sorter.info[&only_optional].optional_predecessors,
        IndexSet::from([a_a1])
    );
    assert_eq!(
        sorter.info[&mixed].optional_predecessors,
        IndexSet::from([a_a1])
    );
    assert_eq!(
        sorter.info[&two_hop].optional_predecessors,
        IndexSet::from([mixed_value])
    );

    assert_eq!(
        sorter.info[&a_a1].optional_successors,
        IndexSet::from([only_optional, mixed, b])
    );
    assert_eq!(
        sorter.info[&mixed_value].optional_successors,
        IndexSet::from([two_hop])
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
    let ready = sorter.get_ready(&interner, None);
    assert_eq!(ready, vec![a]);
    sorter.done(&interner, &ready).unwrap();
    assert_eq!(sorter.out, IndexSet::from([a_a1, a_a2]));

    sorter.mark_expired(&[a_a1], true);
    sorter.done(&interner, &[a_a2]).unwrap();
    let ready: IndexSet<u32> = sorter.get_ready(&interner, None).into_iter().collect();
    assert_eq!(ready, IndexSet::from([only_optional, mixed, b]));

    sorter.done(&interner, &[only_optional, mixed]).unwrap();
    assert_eq!(sorter.out, IndexSet::from([b, b_b1, mixed_value]));

    sorter.mark_expired(&[mixed_value], true);
    let ready = sorter.get_ready(&interner, None);
    assert_eq!(ready, vec![two_hop]);
}

/// Port of `test_unlock_optionals`: the `unlock_optionals` arg controls
/// whether expiring nodes makes their downstream optional dependents ready.
/// (pytest parametrizes over the bool; here it's a helper called twice.)
fn unlock_optionals_case(unlock_optionals: bool) {
    let (mut interner, mut sorter) = optional_graph();
    let ready = sorter.get_ready(&interner, None);
    sorter.done(&interner, &ready).unwrap();

    let out: Vec<u32> = sorter.out.iter().copied().collect();
    sorter.mark_expired(&out, unlock_optionals);
    let ready: IndexSet<u32> = sorter.get_ready(&interner, None).into_iter().collect();
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
