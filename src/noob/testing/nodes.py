import random
import sqlite3
import string
from collections.abc import Generator
from datetime import datetime
from itertools import count, cycle
from typing import Annotated as A
from typing import Any

import xarray as xr
from faker import Faker

from noob import Name, process_method
from noob.node import Node


def count_source(limit: int = 1000, start: int = 0) -> Generator[A[int, Name("index")], None, None]:
    counter = count(start=start)
    while (val := next(counter)) < limit:
        yield val


def letter_source() -> Generator[A[str, Name("letter")]]:
    yield from cycle(string.ascii_lowercase)


def word_source() -> Generator[A[str, Name("word")]]:
    fake = Faker()
    while True:
        yield fake.unique.word()


def multi_words_source(n: int) -> Generator[A[list[str], Name("multi_words")]]:
    fake = Faker()
    while True:
        yield [fake.unique.word() for _ in range(n)]


def sporadic_word(every: int = 3) -> Generator[A[str, Name("word")] | None, None, None]:
    fake = Faker()
    i = 0
    while True:
        i += 1
        if i % every == 0:
            yield fake.unique.word()
        else:
            yield None


def word_counts() -> Generator[tuple[A[str, Name("word")], A[list[int], Name("counts")]]]:
    fake = Faker()
    while True:
        n_counts = random.randint(2, 5)
        yield fake.unique.word(), [random.randint(1, 100) for _ in range(n_counts)]


def multiply(left: int, right: int = 2) -> int:
    """
    Return value purposely unnamed,
    to be used as `{nodename}.value`
    """
    return left * right


def divide(numerator: int, denominator: int = 5) -> A[float, Name("ratio")]:
    return numerator / denominator


def concat(strings: list[str]) -> str:
    return "".join(strings)


def exclaim(string: str) -> str:
    return string + "!"


def repeat(string: str, times: int) -> str:
    return string * times


def dictify(key: str, items: list[Any]) -> dict[str, Any]:
    return {key: items}


class CountSource(Node):
    limit: int = 1000
    start: int = 0

    def process(self) -> Generator[A[int, Name("index")], None, None]:
        counter = count(start=self.start)
        while (val := next(counter)) < self.limit:
            yield val


class Multiply(Node):
    def process(self, left: int, right: int = 2) -> A[int, Name("product")]:
        return multiply(left=left, right=right)


class VolumeProcess:
    def __init__(self, height: int = 2):
        self.height = height

    def process(self, width: int, depth: int) -> A[int, Name("volume")]:
        return self.height * multiply(left=width, right=depth)


class Volume:
    def __init__(self, height: int = 2):
        self.height = height

    @process_method
    def volume(self, width: int, depth: int) -> A[int, Name("volume")]:
        return self.height * multiply(left=width, right=depth)


class Now:
    def __init__(self):
        self.now = datetime.now()

    @process_method
    def print(self, prefix: str = "Now: ") -> A[str, Name("timestamp")]:
        return f"{prefix}{self.now.isoformat()}"


def array_add_to_left(left: xr.DataArray, right: xr.DataArray) -> xr.DataArray:
    left += right
    return left


class CountSourceDecor:
    def __init__(self, start: int = 0) -> None:
        self.gen = count(start=start)

    @process_method
    def process(self) -> Generator[A[int, Name("count")], None, None]:
        yield from self.gen


def input_party(
    one: int, two: float, three: str, four: bool, five: list, six: dict, seven: set
) -> A[bool, Name("works")]:
    return True


def read_db(conn: sqlite3.Connection) -> A[tuple[int, str], Name("payload")]:
    return conn.cursor().execute("SELECT * FROM users").fetchone()
