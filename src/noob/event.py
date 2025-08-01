import sys
from datetime import datetime
from typing import Any

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class Event(TypedDict):
    """
    Container for a single value returned from a single :meth:`.Node.process` call
    """

    id: int
    """Unique ID for each event"""
    timestamp: datetime
    """Timestamp of when the event was received by the :class:`.TubeRunner`"""
    node_id: str
    """ID of node that emitted the event"""
    slot: str
    """name of the slot that emitted the event"""
    value: Any
    """Value emitted by the processing node"""
