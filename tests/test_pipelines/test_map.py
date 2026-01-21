import pytest
from pydantic import ValidationError

from noob import SynchronousRunner, Tube

pytestmark = pytest.mark.map


@pytest.mark.xfail(reason="map not implemented")
def test_map_basic():
    """
    Map splits up an iterator and processes its elements individually
    """
    tube = Tube.from_specification("testing-map-basic")
    runner = SynchronousRunner(tube)
    with runner:
        for _ in range(5):
            val = runner.process()
            assert isinstance(val["letter"], list)
            assert val["letter"] == [letter + "!" for letter in val["word"]]


@pytest.mark.xfail(reason="map not implemented")
def test_map_depends():
    """
    A node that depends on a normal event and a mapped one has the normal event repeated
    """
    tube = Tube.from_specification("testing-map-depends")
    runner = SynchronousRunner(tube)
    with runner:
        for _ in range(5):
            val = runner.process()
            assert isinstance(val["letter"], list)
            assert val["letter"] == [letter + ("!" * val["count"]) for letter in val["word"]]


@pytest.mark.xfail(reason="map not implemented")
def test_map_double_depends():
    """
    A node cannot be downstream of multiple, unrelated maps
    """
    with pytest.raises(ValidationError):
        Tube.from_specification("testing-map-depends-double")


@pytest.mark.xfail(reason="map not implemented")
def test_map_gather():
    """
    Gathering after a map collapses the sub-epoch
    """
    tube = Tube.from_specification("testing-map-gather")
    runner = SynchronousRunner(tube)
    with runner:
        for _ in range(5):
            val = runner.process()
            assert val["letter"] == [letter + ("!" * val["count"]) for letter in val["word"]]
            assert val["reconstructed"] == "".join(val["letter"])
            raise NotImplementedError(
                "When epochs are refactored, assert that after the gather node, "
                "the epoch number returns to normal"
            )


@pytest.mark.xfail(reason="map not implemented")
def test_map_nested():
    """
    Maps can map mappings
    """
    tube = Tube.from_specification("testing-map-nested")
    runner = SynchronousRunner(tube)
    with runner:
        for _ in range(5):
            val = runner.process()
            # mapping a list and then returning it just... reconstructs the list
            assert val["word"] == val["words"]
            # letters is a double-nested list, [['c', 'a', 't'], ['d', 'o', 'g'], ...]
            for i, letters in enumerate(val["letters"]):
                assert letters == [letter for letter in val["words"][i]]
            raise NotImplementedError("When epochs are refactored, check the epochs are correct")
