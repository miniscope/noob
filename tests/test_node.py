from collections.abc import Generator

import pytest

from noob.node.base import WrapClassNode, WrapFuncNode


class SomeNode:
    def process(self) -> int:
        return 1


def func_node() -> int:
    return 1


def generator_node() -> Generator[int, None, None]:
    yield 1


@pytest.mark.xfail
def test_node_missing_param_input():
    """
    When a node specifies an input in its params, but is not provided an input collection,
    it raises an error.
    """
    raise NotImplementedError()


@pytest.mark.parametrize(
    "node,key,wrapper,default",
    (
        (SomeNode, "cls", WrapClassNode, True),
        (func_node, "fn", WrapFuncNode, False),
        (generator_node, "fn", WrapFuncNode, True),
    ),
)
def test_node_default_statefulness(node, key, wrapper, default):
    """
    Class nodes and generators default stateful, function nodes default stateless,
    and all can be overridden if passed explicitly
    """
    kwargs = {key: node, "id": "tmp"}
    wrapped = wrapper(**kwargs)
    assert wrapped.stateful == default

    assert wrapper(stateful=True, **kwargs).stateful
    assert not wrapper(stateful=False, **kwargs).stateful
