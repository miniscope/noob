import pytest
from noob_core import Epoch, Scheduler


@pytest.fixture()
def diamond() -> Scheduler:
    """
    this is sort of pointless because this isn't actually what noob will pass,
     but hey i guess i'm getting scolded into doing it
    """
    nodes = [(node, True, True) for node in ("a", "b", "c", "d")]
    edges = [
        ("a", "a1", "b", True),
        ("a", "a2", "c", True),
        ("b", "b1", "d", True),
        ("c", "c1", "d", True),
    ]
    return Scheduler(nodes, edges)


def test_add_epoch(diamond):
    ep = diamond.add_epoch()
    assert ep == Epoch(0)
