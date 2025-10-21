import pytest


@pytest.mark.xfail(raises=NotImplementedError)
def test_runner_scoped():
    """
    say we have an asset that just provides an integer, and the node just increments it.
    we should be able to call Runner.process multiple times, and the asset should
    increment without resetting.
    """
    raise NotImplementedError


@pytest.mark.xfail(raises=NotImplementedError)
def test_process_scoped():
    """
    same setup as above, except we should get back the same integer each time
    (it is reset between process calls, but also test that it is mutable within a
    process call - e.g. if we had two nodes that increment the value, it should return 2)
    """
    raise NotImplementedError


@pytest.mark.xfail(raises=NotImplementedError)
def test_node_scoped():
    """
    new object created each time, ensure object ids are all different
    """
    raise NotImplementedError
