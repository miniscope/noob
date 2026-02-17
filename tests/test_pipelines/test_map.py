import pytest
from pydantic import ValidationError

from noob import SynchronousRunner, Tube
from noob.runner import TubeRunner
from noob.runner.zmq import ZMQRunner
from noob.types import Epoch
from noob.utils import iscoroutinefunction_partial

pytestmark = pytest.mark.map


@pytest.mark.asyncio
@pytest.mark.parametrize("loaded_tube", ["testing-map-basic"], indirect=True)
async def test_map_basic(loaded_tube: Tube, all_runners: TubeRunner):
    """
    Map splits up an iterator and processes its elements individually
    """
    runner = all_runners
    for _ in range(5):
        if iscoroutinefunction_partial(runner.process):
            val = await runner.process()
        else:
            val = runner.process()
        assert isinstance(val["letter"], list)
        assert val["letter"] == [letter + "!" for letter in val["word"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("loaded_tube", ["testing-map-depends"], indirect=True)
async def test_map_depends(loaded_tube: Tube, all_runners: TubeRunner):
    """
    A node that depends on a normal event and a mapped one has the normal event repeated
    """
    runner = all_runners
    for i in range(5):
        if iscoroutinefunction_partial(runner.process):
            val = await runner.process()
        else:
            val = runner.process()
        assert isinstance(val["letters"], list)
        assert val["count"] == i
        assert val["letters"] == [letter + ("!" * val["count"]) for letter in val["word"]]


def test_map_double_depends():
    """
    A node cannot be downstream of multiple, unrelated maps
    """
    with pytest.raises(ValidationError):
        Tube.from_specification("testing-map-depends-double")


@pytest.mark.asyncio
@pytest.mark.parametrize("loaded_tube", ["testing-map-gather"], indirect=True)
async def test_map_gather(loaded_tube: Tube, all_runners: TubeRunner):
    """
    Gathering after a map collapses the sub-epoch
    """
    runner = all_runners
    for i in range(5):
        if iscoroutinefunction_partial(runner.process):
            val = await runner.process()
        else:
            val = runner.process()
        assert val["letter"] == [letter + "!" for letter in val["word"]]
        assert val["reconstructed"] == "".join(val["letter"])
        assert "c" not in runner.store.events[Epoch(i)]
        assert runner.store.events[Epoch(i)]["d"]["value"][0]["value"] == val["letter"]
        assert runner.store.events[Epoch(i)]["e"]["value"][0]["value"] == val["reconstructed"]
        assert not runner.store.events[Epoch(i) / ("b", 0)]["d"]["value"]
        assert not runner.store.events[Epoch(i) / ("b", 0)]["e"]["value"]


@pytest.mark.asyncio
@pytest.mark.parametrize("loaded_tube", ["testing-map-nested"], indirect=True)
async def test_map_nested(loaded_tube: Tube, all_runners: TubeRunner):
    """
    Maps can map mappings
    """
    runner = all_runners
    for a_i in range(5):
        if iscoroutinefunction_partial(runner.process):
            val = await runner.process()
        else:
            val = runner.process()
        # mapping a list and then returning it just... reconstructs the list
        # map is stateless, so order doesn't matter
        assert val["word"] == val["words"]
        # letters just the flattened list of all letters
        assert val["letter"] == [letter for letter in "".join(val["words"])]
        # emit a list of words at the top level
        assert runner.store.events[Epoch(a_i)]["a"]["multi_words"][0]["value"] == val["words"]
        # first map emits individual words
        for b_i in range(len(val["words"])):
            assert (
                runner.store.events[Epoch(a_i) / ("b", b_i)]["b"]["value"][0]["value"]
                == val["words"][b_i]
            )
            # second map emits individual letters
            for c_i in range(len(val["words"][b_i])):
                ep_events = runner.store.events[Epoch(a_i) / ("b", b_i) / ("c", c_i)]
                assert ep_events["c"]["value"][0]["value"] == val["words"][b_i][c_i]
