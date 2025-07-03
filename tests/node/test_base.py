import inspect
from itertools import count
from typing import Generator, Annotated as A

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
    def return_empty():
        return False

    def test_one_return(self):
        return_annot = inspect.signature(self.one_return).return_annotation
        names = WrapNode._collect_slot_names(return_annot)
        assert names == ["hey"]

    def test_two_returns(self):
        return_annot = inspect.signature(self.two_returns).return_annotation
        names = WrapNode._collect_slot_names(return_annot)
        assert names == ["hey", "you"]

    def test_single_generator(self):
        return_annot = inspect.signature(self.single_generator).return_annotation
        names = WrapNode._collect_slot_names(return_annot)
        assert names == ["you"]

    def test_double_generator(self):
        return_annot = inspect.signature(self.double_generator).return_annotation
        names = WrapNode._collect_slot_names(return_annot)
        assert names == ["i can be your girlfriend", "you can be my boyfriend"]

    def test_unnamed_return(self):
        return_annot = inspect.signature(self.unnamed_return).return_annotation
        names = WrapNode._collect_slot_names(return_annot)
        assert names == ["value"]

    def test_unnamed_tuple(self):
        return_annot = inspect.signature(self.unnamed_tuple).return_annotation
        names = WrapNode._collect_slot_names(return_annot)
        assert names == ["value"]

    def test_unnamed_gen(self):
        return_annot = inspect.signature(self.unnamed_gen).return_annotation
        names = WrapNode._collect_slot_names(return_annot)
        assert names == ["value"]

    def test_return_none(self):
        return_annot = inspect.signature(self.return_none).return_annotation
        names = WrapNode._collect_slot_names(return_annot)
        assert names == []

    def test_return_empty(self):
        return_annot = inspect.signature(self.return_empty).return_annotation
        names = WrapNode._collect_slot_names(return_annot)
        assert names == []
