from multiprocessing import Lock
from typing import Any, TypeVar

from pydantic import PrivateAttr

from noob.node.base import Node

_TInput = TypeVar("_TInput")


class Gather(Node):
    """
    Cardinality reduction.

    Given a node that emits >1 events, gather them into a single iterable.

    Two (mutually exclusive) modes:

    - gather a fixed number of events

    ```yaml
    nodename:
      type: gather
      params:
        n: 5
      depends:
        - value: othernode.signal
    ```

    - gather events until a trigger is received

    ```yaml
    nodename:
      type: gather
      depends:
        - value: othernode.signal_1
        - trigger: thirdnode.signal_2
    ```
    """

    n: int | None = None
    _items: list[_TInput] = PrivateAttr(default_factory=list)
    _lock: Lock = PrivateAttr(default_factory=Lock)

    def process(self, value: _TInput, trigger: Any | None = None) -> list[_TInput] | None:
        """Collect value in a list, emit if `n` is met or `trigger` is present"""
        if trigger is not None and self.n is not None:
            raise ValueError("Cannot use trigger mode while `n` is set")
        with self._lock:
            self._items.append(value)
            if self._should_return(trigger):
                try:
                    return self._items
                finally:
                    # clear list after returning
                    self._items = []

    def _should_return(self, trigger: Any | None) -> bool:

        return (self.n is not None and len(self._items) >= self.n) or (
            self.n is None and trigger is not None
        )
