from typing import Unpack, Any, TypeVar

from noob.node.base import Transform, TInput, TOutput

TListItems = TypeVar("TListItems")

class Concat(Transform):
    def process(self, *args: TListItems) -> list[TListItems]:
        return list(args)
