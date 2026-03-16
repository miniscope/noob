from datetime import UTC, datetime
from typing import Any

import pytest

from noob.event import Event
from noob.types import Epoch


class NonEquivalent:
    def __eq__(self, other: Any) -> bool:
        raise ValueError("Cant equals me!")


@pytest.fixture(scope="session")
def non_equivalent_event() -> Event:
    return Event(
        id=0,
        timestamp=datetime.now(UTC),
        node_id="default",
        signal="value",
        epoch=Epoch(0),
        value=NonEquivalent(),
    )
