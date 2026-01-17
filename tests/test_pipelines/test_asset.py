import pytest

from noob import SynchronousRunner, Tube

pytestmark = pytest.mark.assets


def test_runner_scoped():
    """
    'runner' scoped assets are persistent across process calls
    """
    tube = Tube.from_specification("testing-runner-asset")
    runner = SynchronousRunner(tube=tube)

    runner.init()
    for i in range(5):
        runner.process()
        store = runner.store.events[i]
        assert store["increment"]["next"][0]["value"] == 2 ** (i + 1) - 1
        assert store["more_increment"]["next"][0]["value"] == 2 * (2 ** (i + 1) - 1)


def test_process_scoped():
    """
    'process' scoped assets are recreated every process call.
    However, a given asset is shared by nodes within a single process call.
    """
    tube = Tube.from_specification("testing-process-asset")
    runner = SynchronousRunner(tube=tube)

    runner.init()
    for i in range(5):
        runner.process()
        store = runner.store.events[i]
        assert store["increment"]["next"][0]["value"] == 3  # renewed each process call
        assert store["more_increment"]["next"][0]["value"] == 6  # but shared among nodes


def test_node_scoped():
    """
    new object created each time asset is called by a node.
    """
    tube = Tube.from_specification("testing-node-asset")
    runner = SynchronousRunner(tube=tube)

    runner.init()
    for i in range(5):
        runner.process()
        store = runner.store.events[i]
        assert store["increment"]["next"][0]["value"] == 3
        assert store["more_increment"]["next"][0]["value"] == 3


@pytest.mark.xfail(raises=NotImplementedError)
def test_asset_depends_ooo():
    """
    when a wrong order epoch begins,
    the asset should not be injected to the node,
    hanging the process call.
    """
    raise NotImplementedError()


def test_asset_copy_post_depends():
    """
    make sure the asset is copied when injected to nodes
    that are of later generations than the asset's dependency
    within the same epoch.

    The spec is identical to :func:`.test_runner_scoped`,
    except an additional node after the `go_to_asset` node,
    which takes the asset output and modifies it.
    The asset mutation within this last node
    should have no effect on the pre-asset-depends nodes
    (`increment` and `more_increment`) of the following epoch.
    """
    tube = Tube.from_specification("testing-depends-asset")
    runner = SynchronousRunner(tube=tube)

    runner.init()

    assert set(runner.tube.state.need_copy.keys()) == {"go_to_asset"}
    for i in range(5):
        runner.process()
        store = runner.store.events[i]
        # identical asset as `depends` returned from the previous epoch
        assert store["increment"]["next"][0]["value"] == 2 ** (i + 1) - 1
        # same asset within process
        assert store["more_increment"]["next"][0]["value"] == 2 * (2 ** (i + 1) - 1)
