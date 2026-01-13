from itertools import count


def counter(start: int) -> int:
    return count(start)


class Counter:
    def __init__(self, start: int = 0) -> None:
        self.start = start
        self._current = start
        self.counter = count(start)

    def __next__(self) -> int:
        self._current = next(self.counter)
        return self._current

    @property
    def current(self) -> int:
        return self._current
