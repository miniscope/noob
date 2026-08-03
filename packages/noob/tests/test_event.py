import random

from noob.event import _ID_PREFIX_BITS, _ID_SEQUENCE_BITS, EventMaker, event_id, event_id_parts
from noob.types import Epoch


def test_eventmaker_sequence():
    """
    An eventmaker's events can have the prefix and event sequence decoded,
    and events are monotonically increasing
    """
    prefix = 12345
    maker = EventMaker(prefix=12345, node_id="a")
    events = [maker.new_event(value=0, epoch=Epoch(0))]
    assert all(event_id_parts(e["id"])[1] == i for e, i in zip(events, range(len(events))))
    assert all(event_id_parts(e["id"])[0] == prefix for e in events)


def test_eventid_parts_roundtrip():
    """
    Making an event ID from prefix and sequence can be reversed
    """
    n_reps = 50
    for _ in range(n_reps):
        prefix = random.getrandbits(_ID_PREFIX_BITS)
        sequence = random.getrandbits(_ID_SEQUENCE_BITS)
        eid = event_id(prefix, sequence)
        recovered_prefix, recovered_sequence = event_id_parts(eid)
        assert recovered_prefix == prefix
        assert recovered_sequence == sequence
