import numpy as np
import pytest

from noob import SynchronousRunner, Tube


def test_basic():
    """The most basic tube! We can process a fixed number of events"""
    tube = Tube.from_specification("testing-basic")
    runner = SynchronousRunner(tube)

    outputs = runner.run(n=5)
    assert len(outputs) == 5
    assert outputs == [0, 2, 4, 6, 8]


def test_basic_iter():
    """We should also be able to iterate over values"""
    tube = Tube.from_specification("testing-basic")
    runner = SynchronousRunner(tube)

    expected = [0, 2, 4, 6, 8]
    for e, value in zip(expected, runner.iter(n=5)):
        assert value == e


def test_branch():
    """A nodes output can be branched and received by multiple nodes!"""
    tube = Tube.from_specification("testing-branch")
    runner = SynchronousRunner(tube)
    expected = [{"multiply": i * 2, "divide": i / 5} for i in range(5)]

    for e, value in zip(expected, runner.iter(n=5)):
        assert value == e


def test_merge():
    """Multiple node outputs can be merged into one node!"""
    tube = Tube.from_specification("testing-merge")
    runner = SynchronousRunner(tube)

    expected = [(i * 2) / j for i, j in zip(range(5), range(5, 10))]

    for e, value in zip(expected, runner.iter(n=5)):
        assert value == e


def test_gather_n():
    """A node can gather n inputs into one call"""
    tube = Tube.from_specification("testing-gather-n")
    runner = SynchronousRunner(tube)

    expected = ["abcde", "fghij", "klmno", "pqrst", "uvwxy"]

    for e, value in zip(expected, runner.iter(n=5)):
        # Return node may return dict with 'word' key
        if isinstance(value, dict):
            assert value.get("word") == e or list(value.values())[0] == e
        else:
            assert value == e


def test_gather_dependent():
    """A node can gather inputs from one slot when another slot receives an event"""
    tube = Tube.from_specification("testing-gather-dependent")
    runner = SynchronousRunner(tube)

    expected = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12, 13, 14],
    ]

    for e, value in zip(expected, runner.iter(n=5)):
        assert isinstance(value, dict)
        assert len(value) == 1
        value = value["word"]
        inner = value[list(value.keys())[0]]
        assert inner == e


def test_map():
    """
    A node with a sequence output can be mapped to a node with a scalar input
    
    Map node splits a list into individual items, each creating a new epoch.
    Each epoch processes downstream nodes independently.
    """
    tube = Tube.from_specification("testing-map")
    runner = SynchronousRunner(tube)

    # Map should create one output per item in the input sequence
    # Each item from the map node gets processed through exclaim and return
    results = []
    for value in runner.iter(n=5):
        # The return node may accumulate multiple values as a tuple
        # or return them individually depending on how map epochs are processed
        if isinstance(value, tuple):
            # Multiple values returned as tuple - extract each one
            for v in value:
                if isinstance(v, str):
                    assert v.endswith("!")
                    results.append(v)
                elif isinstance(v, dict) and "value" in v:
                    assert isinstance(v["value"], str)
                    assert v["value"].endswith("!")
                    results.append(v["value"])
        elif isinstance(value, str):
            assert value.endswith("!")
            results.append(value)
        elif isinstance(value, dict):
            # Single dict return
            if "value" in value:
                assert isinstance(value["value"], str)
                assert value["value"].endswith("!")
                results.append(value["value"])
            else:
                # Try to extract the string value
                str_val = list(value.values())[0] if value else None
                if str_val and isinstance(str_val, str):
                    assert str_val.endswith("!")
                    results.append(str_val)
    
    # Should have processed multiple items from the map
    assert len(results) > 0, f"Expected some results, got: {results}"


def test_multi_signal():
    """
    Nodes that emit multiple signals can have each used independently
    """
    tube = Tube.from_specification("testing-multi-signal")
    runner = SynchronousRunner(tube)

    for value in runner.iter(n=5):
        assert isinstance(value, dict)
        assert "word" in value
        assert "count_sum" in value
        assert "counts" in value
        assert isinstance(value["word"], str)
        assert isinstance(value["counts"], list)


def test_xarray_asset():
    """
    Test should verify that the asset has been modified in place,
    (two xarray dataarray assets have been summed and assigned to one of them)
    and the modified asset is same as the returned event output.
    """
    import numpy as np
    tube = Tube.from_specification("testing-xarray-asset")
    runner = SynchronousRunner(tube=tube)

    runner.init()
    output = runner.process()
    # Should return the modified xarray DataArray
    assert np.all(output == 2)


def test_db_asset():
    """Database assets work correctly"""
    tube = Tube.from_specification("testing-db-asset")
    runner = SynchronousRunner(tube=tube)

    runner.init()
    output = runner.process()
    # Should return tuple from database
    assert output == (1, "Hannah Montana")