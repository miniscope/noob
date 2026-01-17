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

    assert set(runner.tube.state.dependencies.keys()) == {"b"}
    last_asset_id = id(runner.tube.state.assets["counter"].obj)
    for i in range(5):
        result = runner.process()

        # we should have deepcopied the asset and made a new one
        this_asset_id = id(runner.tube.state.assets["counter"].obj)
        assert this_asset_id != last_asset_id
        last_asset_id = this_asset_id

        # within the epoch, all values are computed normally
        # between epochs, we should only advance by **2** rather than **3**
        # because the asset it deepcopied after `b`
        start = 1 + (i * 2)
        assert result["a_value"] == start
        assert result["b_value"] == start + 1
        assert result["post_value"] == start + 2

        # within the epoch, the generator should be unchanged and passed between nodes
        assert id(result["a_iterator"]) == id(result["b_iterator"]) == id(result["post_iterator"])


def test_asset_nocopy_when_unused():
    """
    Don't copy assets when there is no chance for them to be mutated after they are stored
    """
    tube = Tube.from_specification("testing-depends-asset-nocopy")
    runner = SynchronousRunner(tube=tube)

    runner.init()

    assert runner.tube.state.dependencies == {}
    first_asset_id = id(runner.tube.state.assets["counter"].obj)
    for i in range(5):
        result = runner.process()

        # we should **not** have deepcopied the asset
        assert id(runner.tube.state.assets["counter"].obj) == first_asset_id

        # within the epoch, all values are computed normally
        # between epochs, we should advance by 2 because the asset is unmutated
        # nodes downstream from the dependency still work as normal
        start = 1 + (i * 2)
        assert result["a_value"] == start
        assert result["b_value"] == start + 1
        assert result["post_value"] == (start + 1) * 2
