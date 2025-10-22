from noob.node import NodeSpecification
from noob.node.base import WrapClassNode, WrapFuncNode
from noob.testing import CountSource, CountSourceDecor, count_source

_annoying_kwargs = dict(id="gen-node", spec=NodeSpecification(type="a.b", id="zzz", enabled=True))


def test_subclass_generator():
    """
    Subclasses of node that have a generator function for `process` should
    have it wrapped at instantiation time so `process` can be called like a normal function
    """

    node = CountSource(**_annoying_kwargs)
    items = []
    for _ in range(5):
        items.append(node.process())
    assert items == [0, 1, 2, 3, 4]


def test_wrapped_fn_generator():
    """
    A wrapped fn that is a generator is ... as above
    """
    node = WrapFuncNode(fn=count_source, **_annoying_kwargs)
    items = []
    for _ in range(5):
        items.append(node.process())
    assert items == [0, 1, 2, 3, 4]


def test_wrapped_cls_generator():
    """
    A wrapped class that is a generator is ... as above
    """

    node = WrapClassNode(cls=CountSourceDecor, **_annoying_kwargs)
    items = []
    for _ in range(5):
        items.append(node.process())
    assert items == [0, 1, 2, 3, 4]
