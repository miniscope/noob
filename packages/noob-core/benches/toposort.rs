//! Sorter benchmarks, mirroring the TopoSorter benchmarks in
//! packages/noob/tests/bench.py - the PRNG and graph generation must stay
//! exactly in sync with `_random_graph_edges` there.
use criterion::{Criterion, criterion_group, criterion_main};
use noob_core::FxIndexMap;
use noob_core::item::{Interner, ItemID};
use noob_core::sorter::{EdgeRec, Sorter};

/// Tiny deterministic PRNG, implemented identically in python
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) % n
    }
}

/// Deterministic pseudo-random layered DAG using all the sorter features:
/// 1-3 deps per node spanning up to 4 layers back, 3 signals per source
/// node, ~25% optional edges, and a full-depth optional chain in column 0.
fn random_graph_edges(layers: u64, width: u64, seed: u64) -> Vec<EdgeRec> {
    let mut rng = Lcg(seed);
    let mut edges = Vec::new();
    for layer in 1..layers {
        for col in 0..width {
            let ndeps = 1 + rng.below(3);
            for _ in 0..ndeps {
                let span = 1 + rng.below(layer.min(4));
                let src_col = rng.below(width);
                let sig = rng.below(3);
                let required = rng.below(4) != 0;
                edges.push(EdgeRec {
                    source_node: format!("n{}_{}", layer - span, src_col),
                    source_signal: format!("s{sig}"),
                    target_node: format!("n{layer}_{col}"),
                    required,
                });
            }
            if col == 0 {
                // optional chain many links deep down column 0
                edges.push(EdgeRec {
                    source_node: format!("n{}_0", layer - 1),
                    source_signal: "s0".to_string(),
                    target_node: format!("n{layer}_0"),
                    required: false,
                });
            }
        }
    }
    edges
}

fn drive(interner: &Interner, sorter: &mut Sorter) {
    while sorter.is_active() {
        sorter.get_ready(interner);
        let out: Vec<ItemID> = sorter.out.iter().copied().collect();
        sorter.done(interner, &out).unwrap();
    }
}

fn bench_sorter(c: &mut Criterion) {
    for (name, layers, width) in [("tiny", 5, 2), ("small", 10, 5), ("large", 50, 20)] {
        let edges = random_graph_edges(layers, width, 42);

        c.bench_function(&format!("random_graph_creation/{name}"), |b| {
            b.iter(|| {
                let mut interner = Interner::default();
                Sorter::from_graph(&mut interner, &FxIndexMap::default(), &edges).unwrap()
            });
        });

        let mut interner = Interner::default();
        let template = Sorter::from_graph(&mut interner, &FxIndexMap::default(), &edges).unwrap();

        // counterpart of python's deepcopy: per-epoch frozen-template copy
        c.bench_function(&format!("random_graph_clone/{name}"), |b| {
            b.iter(|| template.clone());
        });

        c.bench_function(&format!("random_graph_iteration/{name}"), |b| {
            b.iter(|| {
                let mut sorter = template.clone();
                drive(&interner, &mut sorter);
                sorter
            });
        });
    }
}

criterion_group!(benches, bench_sorter);
criterion_main!(benches);
