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
        assert value == {"word": e}


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


@pytest.mark.xfail(reason="map has not been implemented.")
def test_map():
    """
    A node with a sequence output can be mapped to a node with a scalar input

    In this case, "process" should know to iterate over the mapped values
    and return them one by one, so we should get n of the mapped values,
    not n calls of the source node -> n sets of mapped values.
    """
    tube = Tube.from_specification("testing-map")
    runner = SynchronousRunner(tube)

    for value in runner.iter(n=5):
        assert len(value) == 2
        assert isinstance(value, str)
        assert value[1] == "!"


def test_multi_signal():
    """
    Nodes that emit multiple signals can have each used independently
    """
    tube = Tube.from_specification("testing-multi-signal")
    runner = SynchronousRunner(tube)

    for value in runner.iter(n=5):
        assert isinstance(value, dict)
        assert isinstance(value["word"], str)
        assert value["count_sum"] == sum(value["counts"])


def test_xarray_asset():
    tube = Tube.from_specification("testing-xarray-asset")
    runner = SynchronousRunner(tube)

    outputs = runner.process()


def test_server_asset():
    tube = Tube.from_specification("testing-server-asset")
    runner = SynchronousRunner(tube)

    outputs = runner.process()
