from datetime import UTC, datetime

import pytest

from noob.event import Event
from noob.store import EventStore
from noob.types import Epoch


def test_store_get_by_epoch():
    """
    When a store has events from multiple epochs, we can get events from only one specific epoch
    """
    store = EventStore()
    for i in range(5):
        store.add(
            Event(
                id=i,
                timestamp=datetime.now(UTC),
                node_id="a",
                signal="b",
                epoch=Epoch(i),
                value=None,
            )
        )
    # get a specific epoch
    event = store.get(node_id="a", signal="b", epoch=Epoch(2))
    assert event
    assert event["epoch"] == 2

    # get the latest epoch
    event = store.get(node_id="a", signal="b", epoch=-1)
    assert event
    assert event["epoch"] == 4


@pytest.mark.xfail()
def test_store_iter():
    """
    Store can iterate over all events in its nested dictionary
    """
    raise NotImplementedError()


def test_store_get_collapses_subepochs():
    """
    When getting events for a subepoch, get events from parent epoch when none exist in subepoch
    """
    store = EventStore()
    parent = Epoch(0)
    store.add(
        Event(id=0, timestamp=datetime.now(UTC), node_id="a", signal="b", epoch=parent, value=None)
    )
    evt = store.get(node_id="a", signal="b", epoch=parent / ("something", 0))
    assert evt
    assert evt["epoch"] == parent
