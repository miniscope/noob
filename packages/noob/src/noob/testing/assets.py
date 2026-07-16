from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from noob.asset import Asset


class AnyAsset(Asset):
    obj: Any = None


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


class LifecycleCounter(counter):
    """Counter that records when its context manager has been entered and exited"""

    def __init__(self, start: int = 0) -> None:
        super().__init__(start)
        self.entered = 0
        self.exited = 0


@contextmanager
def counter_cm(start: int = 0) -> Generator[LifecycleCounter, None, None]:
    c = LifecycleCounter(start)
    c.entered += 1
    _ = next(c)
    try:
        yield c
    finally:
        _ = next(c)
        c.exited += 1


class Initializer(Asset):
    """True when initialized, false when deinitialized"""

    obj: bool = False

    def init(self) -> None:
        self.obj = True

    def deinit(self) -> None:
        self.obj = False
