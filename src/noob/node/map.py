from collections.abc import Sequence
from typing import TypeVar

from noob.node.base import Node

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

    def process(self, value: Sequence[_TInput]) -> list[_TInput]:
        return [item for item in value]
