from collections.abc import Generator
from typing import Any


class counter(Generator):
    """
    custom generator because itertools is removing deepcopy in 3.14
    https://docs.python.org/3/deprecations/pending-removal-in-3.14.html
    """

    def __init__(self, start: int = 0) -> None:
        self.start = start
        self._current = start

    def send(self, _: Any) -> int:
        current = self._current
        self._current += 1
        return current

    def throw(self, type: Any = None, value: Any = None, traceback: Any = None) -> None:
        raise StopIteration()

    @property
    def current(self) -> int:
        return self._current
