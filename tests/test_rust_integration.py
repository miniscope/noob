"""
Test Rust noob_core integration

Verifies that the Rust extension builds, imports, and works correctly.
"""

import pytest


def test_import_noob_core():
    """Test that noob_core can be imported"""
    import noob_core
    assert noob_core.__version__ == "0.3.0"  # Updated with P2P features
    assert hasattr(noob_core, 'FastEventStore')
    assert hasattr(noob_core, 'FastScheduler')
    assert hasattr(noob_core, 'FastEvent')
    # P2P components
    assert hasattr(noob_core, 'ContentId')
    assert hasattr(noob_core, 'ContentStore')
    assert hasattr(noob_core, 'TaskMetadata')
    assert hasattr(noob_core, 'TaskRegistry')
    assert hasattr(noob_core, 'P2PNode')


def test_fast_event_store():
    """Test FastEventStore basic operations"""
    import noob_core

    store = noob_core.FastEventStore()

    # Add events
    id1 = store.add_event("node1", "signal1", 0, 1.0)
    id2 = store.add_event("node1", "signal1", 0, 2.0)
    id3 = store.add_event("node2", "signal2", 1, 3.0)

    assert id1 == 0
    assert id2 == 1
    assert id3 == 2

    # Get events
    events = store.get_events("node1", "signal1", 0)
    assert len(events) == 2
    assert events[0].node_id == "node1"
    assert events[0].signal == "signal1"
    assert events[0].epoch == 0

    # Event count
    assert store.event_count() == 3

    # Clear (counter stays monotonic, doesn't reset)
    store.clear()
    events_after_clear = store.get_events("node1", "signal1", 0)
    assert len(events_after_clear) == 0  # Events cleared
    assert store.event_count() == 3  # Counter remains (monotonic IDs)


def test_fast_scheduler():
    """Test FastScheduler topological sorting"""
    import noob_core

    # Create a simple dependency graph:
    # a -> b -> d
    # a -> c -> d
    dependencies = {
        "a": [],
        "b": ["a"],
        "c": ["a"],
        "d": ["b", "c"]
    }

    scheduler = noob_core.FastScheduler(dependencies)

    # Initially only 'a' should be ready
    ready = scheduler.get_ready_nodes()
    assert ready == ["a"]

    # Mark 'a' completed
    scheduler.mark_completed("a")

    # Now 'b' and 'c' should be ready
    ready = scheduler.get_ready_nodes()
    assert set(ready) == {"b", "c"}

    # Mark 'b' completed
    scheduler.mark_completed("b")
    ready = scheduler.get_ready_nodes()
    assert "c" in ready
    assert "d" not in ready  # d still waits for c

    # Mark 'c' completed
    scheduler.mark_completed("c")
    ready = scheduler.get_ready_nodes()
    assert ready == ["d"]

    # Mark 'd' completed
    scheduler.mark_completed("d")
    assert scheduler.is_complete()

    # Reset should allow reprocessing
    scheduler.reset()
    assert not scheduler.is_complete()
    ready = scheduler.get_ready_nodes()
    assert ready == ["a"]


def test_scheduler_performance():
    """Test that FastScheduler handles large graphs efficiently"""
    import noob_core
    import time

    # Create a chain of 1000 nodes
    dependencies = {}
    for i in range(1000):
        if i == 0:
            dependencies[f"node_{i}"] = []
        else:
            dependencies[f"node_{i}"] = [f"node_{i-1}"]

    start = time.time()
    scheduler = noob_core.FastScheduler(dependencies)

    # Process all nodes
    while not scheduler.is_complete():
        ready = scheduler.get_ready_nodes()
        if not ready:
            break
        for node in ready:
            scheduler.mark_completed(node)

    elapsed = time.time() - start

    assert scheduler.is_complete()
    assert elapsed < 2.0  # Should be fast (< 2s for 1000 nodes)
    print(f"✅ Processed 1000 nodes in {elapsed*1000:.2f}ms")


def test_scheduler_wide_graph():
    """Test FastScheduler with a wide dependency graph"""
    import noob_core

    # Create a graph with one root and many children
    dependencies = {"root": []}
    for i in range(100):
        dependencies[f"child_{i}"] = ["root"]

    scheduler = noob_core.FastScheduler(dependencies)

    # Root should be ready first
    ready = scheduler.get_ready_nodes()
    assert ready == ["root"]

    # After marking root done, all children should be ready
    scheduler.mark_completed("root")
    ready = scheduler.get_ready_nodes()
    assert len(ready) == 100
    assert all(f"child_{i}" in ready for i in range(100))


def test_rust_vs_python_scheduler_benchmark():
    """Compare Rust vs Python scheduler performance"""
    import noob_core
    from graphlib import TopologicalSorter
    import time

    # Create a complex diamond pattern: 1 root -> 10 middle -> 1 sink
    dependencies = {"root": []}
    for i in range(10):
        dependencies[f"middle_{i}"] = ["root"]
    dependencies["sink"] = [f"middle_{i}" for i in range(10)]

    # Benchmark Rust scheduler
    rust_times = []
    for _ in range(100):
        start = time.time()
        scheduler = noob_core.FastScheduler(dependencies)
        while not scheduler.is_complete():
            ready = scheduler.get_ready_nodes()
            for node in ready:
                scheduler.mark_completed(node)
        rust_times.append(time.time() - start)

    # Benchmark Python scheduler
    python_times = []
    for _ in range(100):
        start = time.time()
        ts = TopologicalSorter(dependencies)
        ts.prepare()
        while ts.is_active():
            ready = ts.get_ready()
            for node in ready:
                ts.done(node)
        python_times.append(time.time() - start)

    rust_avg = sum(rust_times) / len(rust_times) * 1000  # ms
    python_avg = sum(python_times) / len(python_times) * 1000  # ms
    speedup = python_avg / rust_avg

    print(f"\n🦀 Rust:   {rust_avg:.3f}ms")
    print(f"🐍 Python: {python_avg:.3f}ms")
    print(f"⚡ Speedup: {speedup:.1f}x")

    assert rust_avg < python_avg, "Rust should be faster than Python"
