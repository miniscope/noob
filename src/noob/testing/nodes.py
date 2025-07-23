import random
import string
from collections.abc import Generator
from itertools import count, cycle
from typing import Annotated as A
from typing import Any

from faker import Faker

from noob import Name
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
        return count_source(limit=self.limit, start=self.start)


class Multiply(Node):

    def process(self, left: int, right: int = 2) -> A[int, Name("product")]:
        return multiply(left=left, right=right)
