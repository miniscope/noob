import pytest


@pytest.mark.xfail
def test_tube_input_params():
    raise NotImplementedError()


@pytest.mark.xfail
def test_tube_input_depends():
    """tube-scoped input can be used as depends"""
    raise NotImplementedError()


@pytest.mark.xfail
def test_process_input_params():
    """process-scoped input can NOT be used as params"""
    raise NotImplementedError()


@pytest.mark.xfail
def test_process_input_depends():
    """process-scoped input can be used as params"""
    raise NotImplementedError()


@pytest.mark.xfail
def test_tube_input_missing():
    """Input scoped as tube input raises an error when missing"""
    raise NotImplementedError()


@pytest.mark.xfail
def test_process_input_missing():
    """Input scoped as process input raises an error when missing"""
    raise NotImplementedError()


@pytest.mark.xfail
def test_scope_chaining():
    """Inputs from an outer scope should be accessible from an inner scope"""
    raise NotImplementedError()


@pytest.mark.xfail
def test_scope_overrides():
    """
    When a tube-scoped input is provided in both the process and tube input, the process overrides.
    """
    raise NotImplementedError()
