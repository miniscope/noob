def test_mark_out():
    """
    manually marking something as out should behave exactly the same
    as if we called `get_ready()`
    """
    raise NotImplementedError()


def test_cancel_or_whatever_we_name_it():
    """
    Marks a node as having been completed without making nodes that depend on it ready.
    Should also cause the graph to be considered inactive/completed if no more nodes remain
    """
    raise NotImplementedError()


def test_dynamic_add():
    """
    Adding nodes while a sorter is halfway-through correct sorts it:
    if the dependencies are already completed, place in ready_nodes...
    """
    raise NotImplementedError()


def test_adding_merges():
    """
    Adding more predecessors (by calling add again) to a node that's already in the graph
    merges the predecessors with the existing ones and updates the predecessor count
    and takes it out of `ready_nodes` if applicable
    """
    raise NotImplementedError()


def test_invalid_dynamic_add():
    """
    If the node is out or completed, disallow adding additional predecessors to it
    """
    raise NotImplementedError()
