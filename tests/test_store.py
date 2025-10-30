from datetime import UTC, datetime

from noob.event import Event
from noob.store import EventStore


def test_store_get_by_epoch():
    """
    When a store has events from multiple epochs, we can get events from only one specific epoch
    """
    store = EventStore()
    for i in range(5):
        store.add(
            Event(id=i, timestamp=datetime.now(UTC), node_id="a", signal="b", epoch=i, value=None)
        )
    # get a specific epoch
    event = store.get(node_id="a", signal="b", epoch=2)
    assert event
    assert event["epoch"] == 2

    # get the latest epoch
    event = store.get(node_id="a", signal="b", epoch=-1)
    assert event
    assert event["epoch"] == 4
