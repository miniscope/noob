import pytest

from noob import Tube
from noob.exceptions import SchedulerExhaustedError
from noob.node import NodeSpecification
from noob.node.base import WrapClassNode, WrapFuncNode
from noob.runner import SynchronousRunner
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


def test_generator_exhaustion():
    """
    Generator nodes should properly emit Exhaustion events rather than throwing
    """
    tube = Tube.from_specification("testing-exhaustion")
    runner = SynchronousRunner(tube)

    # three epochs run fine
    with runner:
        for i in range(3):
            assert runner.process() == i

        # then we get nothing, we learn that the generator is exhausted here.
        assert runner.process() is None

        # then we throw since we know the scheduler is exhausted
        with pytest.raises(SchedulerExhaustedError):
            runner.process()
