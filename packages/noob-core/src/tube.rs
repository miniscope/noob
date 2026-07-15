//! Tube helpers from rust :)
//! Just mirroring the python structure for now for ported code

use crate::FxIndexMap;
use crate::FxIndexSet;
use crate::toposort::EdgeRec;

/// Compute all the nodes downsream (that depend on) a given node.
pub fn downstream_nodes<'a>(
    edges: &'a [EdgeRec],
    node: &'a str,
    exclude: &FxIndexSet<&str>,
) -> FxIndexSet<&'a str> {
    let adjacency: FxIndexMap<&str, Vec<&str>> =
        edges.iter().fold(FxIndexMap::default(), |mut adj, edge| {
            adj.entry(edge.source_node.as_str())
                .or_default()
                .push(edge.target_node.as_str());
            adj
        });

    let mut downstream: FxIndexSet<&'a str> = FxIndexSet::from_iter(vec![node]);
    let mut queue = vec![node];
    while let Some(current) = queue.pop() {
        if let Some(successors) = adjacency.get(current) {
            for successor in successors {
                if !downstream.contains(successor) && !exclude.contains(*successor) {
                    downstream.insert(successor);
                    queue.push(successor);
                }
            }
        }
    }
    downstream
}

#[cfg(test)]
#[path = "tests/tube.rs"]
mod tests;
