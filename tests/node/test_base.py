import inspect
from collections.abc import Generator
from itertools import count
from typing import Annotated as A

import pytest
from faker import Faker

from noob import Name
from noob.node.base import WrapNode


class TestWrapNode:
    @staticmethod
    def one_return(name: str) -> A[str, Name("hey")]:
        return name

    @staticmethod
    def two_returns(name1: str, name2: str) -> tuple[A[str, Name("hey")], A[str, Name("you")]]:
        return name1, name2

    @staticmethod
    def single_generator() -> Generator[A[str, Name("you")], None, None]:
        fake = Faker()
        while True:
            yield fake.unique.word()

    @staticmethod
    def double_generator(
        start: int, limit: int = 10
    ) -> Generator[
        tuple[A[str, Name("i can be your girlfriend")], A[int, Name("you can be my boyfriend")]]
    ]:
        fake = Faker()
        counter = count(start=start)
        while val := next(counter) < limit:
            yield fake.unique.word(), val

    @staticmethod
    def unnamed_return(name: str) -> str:
        return name

    @staticmethod
    def unnamed_tuple(num1: int, name2: str) -> tuple[int, str]:
        return num1, name2

    @staticmethod
    def unnamed_gen() -> Generator[str]:
        fake = Faker()
        while True:
            yield fake.unique.word()

    @staticmethod
    def return_none() -> None:
        return None

    @staticmethod
    def return_empty():  # noqa: ANN205
        return False

    @pytest.mark.parametrize(
        "func, target",
        [
            (one_return, ["hey"]),
            (two_returns, ["hey", "you"]),
            (single_generator, ["you"]),
            (double_generator, ["i can be your girlfriend", "you can be my boyfriend"]),
            (unnamed_return, ["value"]),
            (unnamed_tuple, ["value"]),
            (unnamed_gen, ["value"]),
            (return_none, []),
            (return_empty, []),
        ],
    )
    def test_collect_slot_names(self, func, target) -> None:
        return_annot = inspect.signature(func).return_annotation
        names = WrapNode._collect_slot_names(return_annot)
        assert names == target
