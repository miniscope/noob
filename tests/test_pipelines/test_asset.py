import pytest

from noob import SynchronousRunner, Tube


def test_runner_scoped():
    """
    'runner' scoped assets are persistent across process calls
    """
    tube = Tube.from_specification("testing-runner-asset")
    runner = SynchronousRunner(tube=tube)

    n = 5
    output = runner.run(n=n)
    counters = []
    nexts = []
    for d in output:
        counters.append(d["counter"])
        nexts.append(d["next"])
    assert nexts == list(range(1, n + 1))
    assert all(c is counters[0] for c in counters)


def test_process_scoped():
    """
    'process' scoped assets are recreated every process call.
    However, a given asset is shared by nodes within a single process call.
    """
    tube = Tube.from_specification("testing-process-asset")
    runner = SynchronousRunner(tube=tube)

    runner.init()
    for _ in range(5):
        output = runner.process()
        assert output["next"] == 1  # renewed each process call
        assert next(output["counter"]) == 2  # but shared among nodes


@pytest.mark.xfail(raises=NotImplementedError)
def test_node_scoped():
    """
    new object created each time, ensure object ids are all different
    """
    raise NotImplementedError()


def test_asset_depends():
    """
    a tube-scoped asset can have a depends key,
    which functions to support inter-epoch dependency
    i.e. asset should not be used in an epoch
    if the last asset-modifying node of the previous epoch has not finished processing.
    """
    tube = Tube.from_specification("testing-depends-asset")
    runner = SynchronousRunner(tube=tube)

    n = 5
    runner.init()

    prev_asset = None

    for epoch in range(n):
        out = runner.process()
        assert out["from_asset"] == out["from_node"]
        if prev_asset is not None:
            assert prev_asset is runner.store.events[epoch]["jump"]["skirttt"][0]["value"]
        prev_asset = out["from_asset"]
