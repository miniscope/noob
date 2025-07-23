import inspect
from collections.abc import Generator
from itertools import count
from typing import Annotated as A

import pytest
from faker import Faker

from noob import Name
from noob.node import NodeSpecification
from noob.node.base import Node, Signal


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
    def union_return(num1: int, name2: str) -> A[str | int, Name("sk8r-boi")]:
        if num1 > 0:
            return num1
        else:
            return name2

    @staticmethod
    def unnamed_return(name: str) -> str:
        return name

    @staticmethod
    def unnamed_tuple(num1: int, name2: str) -> tuple[int, str]:
        return num1, name2

    @staticmethod
    def unnamed_union(num1: int, name2: str) -> int | str:
        if num1 > 0:
            return num1
        else:
            return name2

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
            (one_return, [("hey", str)]),
            (two_returns, [("hey", str), ("you", str)]),
            (single_generator, [("you", str)]),
            (
                double_generator,
                [("i can be your girlfriend", str), ("you can be my boyfriend", int)],
            ),
            (union_return, [("sk8r-boi", str | int)]),
            (unnamed_return, [("value", str)]),
            (unnamed_tuple, [("value", tuple[int, str])]),
            (unnamed_union, [("value", int | str)]),
            (unnamed_gen, [("value", str)]),
            (return_none, []),
            (return_empty, []),
        ],
    )
    def test_collect_signal_names(self, func, target) -> None:
        return_annot = inspect.signature(func).return_annotation
        names = Signal._collect_signal_names(return_annot)
        assert names == target


class TestNode:
    @pytest.mark.parametrize(
        "type, params, expected",
        [
            ("noob.testing.CountSource", {"limit": 10, "start": 5}, ["index"]),
            ("noob.testing.Multiply", {}, ["product"]),
        ],
    )
    def test_node_subclass_signal(self, type, params, expected):
        node = Node.from_specification(
            spec=NodeSpecification(
                id="test_node_subclass_signal",
                type=type,
                params=params,
                depends=None,
            )
        )

        assert node.signals == expected
