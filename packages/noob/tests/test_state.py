import pytest

from noob.asset import AssetScope, AssetSpecification
from noob.state import State

pytestmark = pytest.mark.assets


def test_non_equivalent_event(non_equivalent_event):
    """
    Regression - we can handle events that have a value that can't use ==

    References:
        https://github.com/miniscope/noob/issues/192
        https://github.com/miniscope/noob/issues/201
    """
    state = State.from_specification(
        specs={
            "any": AssetSpecification(
                id="any",
                type="noob.testing.AnyAsset",
                scope=AssetScope.runner,
                depends="default.value",
            )
        }
    )
    # no error is thrown
    state.update([non_equivalent_event])


def test_edgeless_nodes_dont_init():
    """
    Regression - ensure that nodes with falsy edge collections
    don't cause all node-scoped assets to init
    """
    spec = AssetSpecification(
        id="counter", type="noob.testing.counter_cm", scope="node", params={"start": 0}
    )

    state = State.from_specification(specs={"counter": spec})

    state.init(AssetScope.node, edges=None)
    assert state.assets["counter"].obj is None

    state.init(AssetScope.node, edges=[])
    assert state.assets["counter"].obj is None
