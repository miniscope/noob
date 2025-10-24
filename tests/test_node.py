import pytest


@pytest.mark.xfail
def test_node_missing_param_input():
    """
    When a node specifies an input in its params, but is not provided an input collection,
    it raises an error.
    """
    raise NotImplementedError()
