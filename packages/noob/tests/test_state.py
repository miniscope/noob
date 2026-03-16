from noob.asset import AssetScope, AssetSpecification
from noob.state import State


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
