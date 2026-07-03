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

#[test]
fn test_from_graph() {
    let edges = vec![
        edge("a", "a1", "b", true),
        edge("a", "a2", "c", true),
        edge("b", "b1", "d", true),
        edge("c", "c1", "d", true),
    ];
    let scheduler = Scheduler::from_graph(IndexMap::new(), edges.clone())
        .expect("couldnt even create the most basic graph");
    assert_eq!(scheduler.edges, edges);
    assert!(scheduler.epochs.is_empty())
}
