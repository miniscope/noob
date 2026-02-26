from typing import Any, TypeVar

from pydantic import PrivateAttr

from noob.node import Node

_TStatic = TypeVar("_TStatic", bound=Any)


class static(Node):
    """
    Emits some stored value.
    Optionally receives some value in its ``value`` slot.
    If it receives a new value, updates the stored value.
    Otherwise just keep emitting the same value.

    E.g. if some input source only emits events sporadically,
    but we want to continuously control another node.

    To set an initial value before any events are received,
    set `params.stored` in the node specification
    """

    stored: _TStatic = PrivateAttr(default=None)

    def process(self, value: _TStatic | None = None) -> _TStatic:
        if value is not None:
            self.stored = value
        return self.stored
