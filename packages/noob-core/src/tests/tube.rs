use super::*;

// TODO: consolidate test helpers
fn edge(source: &str, signal: &str, target: &str) -> EdgeRec {
    EdgeRec {
        source_node: source.into(),
        source_signal: signal.into(),
        target_node: target.into(),
        required: true,
    }
}

fn diamond() -> Vec<EdgeRec> {
    vec![
        edge("a", "a1", "b"),
        edge("a", "a2", "c"),
        edge("b", "b1", "d"),
        edge("c", "c1", "d"),
    ]
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
