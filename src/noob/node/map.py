import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated as A
from typing import TypeVar

from noob.event import Event
from noob.node.base import Node
from noob.types import Epoch, Name

_TInput = TypeVar("_TInput")


class Map(Node):
    """
    Cardinality expansion

    Given a node that emits 1 (iterable) event, split it into separate events.

    ```{admonition} Implementation Note
    Map is a *special node* -
    it is the only node where returning a list of values is interpreted as multiple events.
    For all other nodes,
    returning a list of values is interpreted as a single event with a list value.
    To return multiple events from a single node, chain a `map` after it.
    ```
    """

    def process(
        self, value: Sequence[_TInput], epoch: Epoch
    ) -> tuple[A[list[Event[_TInput]], Name("value")], A[int, Name("n")]]:
        subepochs = epoch.make_subepochs(self.id, len(value))
        now = datetime.now(UTC)
        ret = []

        for item, subepoch in zip(value, subepochs):
            ret.append(
                Event(
                    id=uuid.uuid4().int,
                    timestamp=now,
                    node_id=self.id,
                    signal="value",
                    epoch=subepoch,
                    value=item,
                )
            )
        return ret, len(ret)
