import asyncio
import random
import string
from collections.abc import Generator, Iterator
from datetime import datetime
from itertools import count, cycle
from time import sleep
from typing import Annotated as A
from typing import Any

from faker import Faker

from noob import Name, process_method
from noob.node import Node


def count_source(limit: int = 1000, start: int = 0) -> Generator[A[int, Name("index")], None, None]:
    counter = count(start=start)
    if limit == 0:
        while True:
            yield next(counter)
    else:
        while (val := next(counter)) < limit:
            yield val


def letter_source() -> Generator[A[str, Name("letter")]]:
    yield from cycle(string.ascii_lowercase)


def word_source() -> Generator[A[str, Name("word")]]:
    fake = Faker()
    while True:
        yield fake.word()


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
            yield fake.word()
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


def multi_concat(**kwargs: list[str]) -> str:
    return "".join("".join(letter for letter in word) for word in kwargs.values())


def exclaim(string: str, hype: int = 1) -> str:
    return string + ("!" * hype)


def repeat(string: str, times: int) -> str:
    return string * times


def dictify(key: str, items: list[Any]) -> dict[str, Any]:
    return {key: items}


def error(value: Any) -> None:
    raise ValueError("This node just emits errors")


class CountSource(Node):
    limit: int = 1000
    start: int = 0

    def process(self) -> Generator[A[int, Name("index")], None, None]:
        counter = count(start=self.start)
        while (val := next(counter)) < self.limit:
            yield val


class UnannotatedGenerator(Node):
    limit: int = 1000
    start: int = 0

    def process(self):  # noqa: ANN201
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


def long_add(value: float) -> float:
    sleep(0.5)
    return value + 1


async def number_to_letter(number: int, offset: int = 0) -> str:
    sleep_for = random.random() / 10
    await asyncio.sleep(sleep_for)
    return string.ascii_lowercase[(number + offset) % len(string.ascii_lowercase)]


class NumberToLetterCls:
    def __init__(self, offset: int = 0):
        self.offset = offset

    async def process(self, number: int) -> str:
        sleep_for = random.random() / 10
        await asyncio.sleep(sleep_for)
        return string.ascii_lowercase[(number + self.offset) % len(string.ascii_lowercase)]


async def async_error(value: Any) -> None:
    """Just raise an error!"""
    raise ValueError("This is the error that should be raised")


class StatefulMultiply:
    def __init__(self, start: int = 0) -> None:
        self.start = start
        self.current = self.start

    def process(self, left: float, right: float = 1) -> float:
        value = left * right * self.current
        self.current += 1
        return value


def fast_forward(generator: count, n: int = 1) -> tuple[A[int, Name("next")]]:
    for _ in range(n):
        val = next(generator)
    return val


def jump(generator: count, n: int = 1) -> tuple[A[count, Name("skirttt")], A[int, Name("next")]]:
    value = 0
    for _ in range(n):
        value = next(generator)
    return generator, value


def rewind(generator: count, n: int = 1) -> A[count, Name("skrittt")]:
    """Purposely designed to mutate the input and return a new object"""
    return count(next(generator) - n)


def zip_iter(*args: Iterator) -> tuple[Any, ...]:
    return tuple(next(a) for a in args)


def increment(
    iterator: Iterator[int],
) -> tuple[A[Iterator[int], Name("iterator")], A[int, Name("value")]]:
    value = next(iterator)
    return iterator, value


def passthrough(value: Any) -> Any:
    return value
