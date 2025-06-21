from noob.tube import Tube


def test_dep_graph():
    """
    Dependency graphs are correctly resolved
    """
    tube = Tube.from_config("test-depends")
    graph = tube.graph()
    graph.prepare()
    ready = graph.get_ready()

    breakpoint()

