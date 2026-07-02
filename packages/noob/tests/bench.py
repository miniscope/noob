import contextlib
from copy import deepcopy

import pytest
from pytest_codspeed.plugin import BenchmarkFixture

from noob import Tube
from noob.edge import Edge
from noob.exceptions import AlreadyDoneError
from noob.runner.base import TubeRunner
from noob.toposort import PREVIOUS_EPOCH, TopoSorter


def test_load_tube(benchmark: BenchmarkFixture) -> None:
    benchmark(lambda: Tube.from_specification("testing-kitchen-sink"))


@pytest.mark.parametrize("loaded_tube", ["testing-kitchen-sink"], indirect=True)
def test_kitchen_sink_process(benchmark: BenchmarkFixture, runner: TubeRunner) -> None:
    benchmark(lambda: runner.process())


@pytest.mark.parametrize("loaded_tube", ["testing-kitchen-sink"], indirect=True)
def test_kitchen_sink_run(benchmark: BenchmarkFixture, runner: TubeRunner) -> None:
    benchmark(lambda: runner.run(n=10))


@pytest.mark.parametrize("loaded_tube", ["testing-long-add"], indirect=True)
def test_long_add(benchmark: BenchmarkFixture, runner: TubeRunner) -> None:
    """
    ZMQ runner should be faster for tubes where nodes take a long time
    and there's lots of concurrency possibilities
    """
    benchmark(lambda: runner.process())


@pytest.mark.parametrize("loaded_tube", ["testing-kitchen-sink"], indirect=True)
def test_topo_sorter(benchmark: BenchmarkFixture, loaded_tube: Tube) -> None:
    """
    Our TopoSorter should not get uh slower
    """
    benchmark(lambda: _run_sorter(loaded_tube))


def _run_sorter(tube: Tube) -> None:
    epoch = tube.scheduler.add_epoch()
    sorter = tube.scheduler._epochs[epoch]
    with contextlib.suppress(AlreadyDoneError):
        sorter.done(PREVIOUS_EPOCH)
    while sorter.is_active():
        sorter.get_ready()
        sorter.done(*sorter.out_nodes)


# --------------------------------------------------
# TopoSorter vs noob-core Sorter comparison benchmarks.
# The PRNG and graph generation must stay exactly in sync with
# packages/noob-core/benches/toposort.rs
# --------------------------------------------------

_MASK64 = 0xFFFFFFFFFFFFFFFF


class _Lcg:
    """Tiny deterministic PRNG, implemented identically in rust"""

    def __init__(self, seed: int) -> None:
        self.state = seed

    def below(self, n: int) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & _MASK64
        return (self.state >> 33) % n


def _random_graph_edges(layers: int, width: int, seed: int = 42) -> list[Edge]:
    """
    Deterministic pseudo-random layered DAG using all the sorter features:
    1-3 deps per node spanning up to 4 layers back, 3 signals per source
    node, ~25% optional edges, and a full-depth optional chain in column 0.
    """
    rng = _Lcg(seed)
    edges = []
    for layer in range(1, layers):
        for col in range(width):
            ndeps = 1 + rng.below(3)
            for _ in range(ndeps):
                span = 1 + rng.below(min(layer, 4))
                src_col = rng.below(width)
                sig = rng.below(3)
                required = rng.below(4) != 0
                edges.append(
                    Edge(
                        source_node=f"n{layer - span}_{src_col}",
                        source_signal=f"s{sig}",
                        target_node=f"n{layer}_{col}",
                        target_slot="value",
                        required=required,
                    )
                )
            if col == 0:
                # optional chain many links deep down column 0
                edges.append(
                    Edge(
                        source_node=f"n{layer - 1}_0",
                        source_signal="s0",
                        target_node=f"n{layer}_0",
                        target_slot="value",
                        required=False,
                    )
                )
    return edges


_SIZES = {"small": (10, 5), "large": (50, 20)}


def _drive(sorter: TopoSorter) -> None:
    while sorter.is_active():
        sorter.get_ready()
        sorter.done(*sorter.out_nodes)


@pytest.mark.parametrize("size", _SIZES)
def test_random_graph_creation(benchmark: BenchmarkFixture, size: str) -> None:
    edges = _random_graph_edges(*_SIZES[size])
    benchmark(lambda: TopoSorter(edges=edges))


@pytest.mark.parametrize("size", _SIZES)
def test_random_graph_deepcopy(benchmark: BenchmarkFixture, size: str) -> None:
    template = TopoSorter(edges=_random_graph_edges(*_SIZES[size]))
    benchmark(lambda: deepcopy(template))


@pytest.mark.parametrize("size", _SIZES)
def test_random_graph_iteration(benchmark: BenchmarkFixture, size: str) -> None:
    """deepcopy the template (mirrors per-epoch use) and drive to completion"""
    template = TopoSorter(edges=_random_graph_edges(*_SIZES[size]))
    benchmark(lambda: _drive(deepcopy(template)))
