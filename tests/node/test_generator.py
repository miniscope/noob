from collections.abc import Generator

from noob.node import Node, NodeSpecification, process_method
from noob.node.base import WrapClassNode, WrapFuncNode

_annoying_kwargs = dict(id="gen-node", spec=NodeSpecification(type="a.b", id="zzz"))


def test_subclass_generator():
    """
    Subclasses of node that have a generator function for `process` should
    have it wrapped at instantiation time so `process` can be called like a normal function
    """

    class GenNode(Node):
        def process(self) -> Generator[int, None, None]:
            i = 9
            while i < 15:
                i += 1
                yield i

    node = GenNode(**_annoying_kwargs)
    items = []
    for _ in range(5):
        items.append(node.process())
    assert items == [10, 11, 12, 13, 14]


def test_wrapped_fn_generator():
    """
    A wrapped fn that is a generator is ... as above
    """

    def fn() -> Generator[int, None, None]:
        i = 9
        while i < 15:
            i += 1
            yield i

    node = WrapFuncNode(fn=fn, **_annoying_kwargs)
    items = []
    for _ in range(5):
        items.append(node.process())
    assert items == [10, 11, 12, 13, 14]


def test_wrapped_cls_generator():
    """
    A wrapped class that is a generator is ... as above
    """

    class NonSubclassNode:
        @process_method
        def not_process(self) -> Generator[int, None, None]:
            i = 9
            while i < 15:
                i += 1
                yield i

    node = WrapClassNode(cls=NonSubclassNode, **_annoying_kwargs)
    items = []
    for _ in range(5):
        items.append(node.process())
    assert items == [10, 11, 12, 13, 14]
