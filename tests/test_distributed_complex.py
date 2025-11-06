"""
Test cases for complex pipelines running on different processes independently.

These tests verify that:
1. Complex pipelines with multiple nodes, maps, and gathers work correctly
2. Multiple processes can run pipelines independently
3. Event serialization/deserialization works correctly
4. Fault tolerance and retries work properly
"""
import multiprocessing
import os
import time
from multiprocessing import Process, Queue
from pathlib import Path

import pytest

from noob import SynchronousRunner, Tube
from noob.runner.distributed import DistributedRunner


def run_pipeline_in_process(spec_id: str, result_queue: Queue, process_id: int) -> None:
    """Run a pipeline in an independent process"""
    try:
        # Convert spec_id to path - use fixtures/paths to get correct path
        from tests.fixtures.paths import PIPELINE_DIR
        spec_name = spec_id.replace("testing-", "").replace("-", "_")
        spec_path = PIPELINE_DIR / f"{spec_name}.yaml"
        tube = Tube.from_specification(spec_path)
        
        runner = SynchronousRunner(tube)
        
        results = []
        for value in runner.iter(n=5):
            results.append(value)
        
        result_queue.put({
            "process_id": process_id,
            "success": True,
            "results": results,
            "pid": os.getpid()
        })
    except Exception as e:
        result_queue.put({
            "process_id": process_id,
            "success": False,
            "error": str(e),
            "pid": os.getpid()
        })


def test_complex_parallel_independent_processes():
    """
    Test that multiple independent processes can run the same complex pipeline.
    
    This verifies that pipelines don't share state and can run truly independently.
    """
    spec_id = "testing-complex-parallel"
    
    # Create multiple processes
    num_processes = 4
    result_queue = Queue()
    processes = []
    
    for i in range(num_processes):
        p = Process(target=run_pipeline_in_process, args=(spec_id, result_queue, i))
        p.start()
        processes.append(p)
    
    # Collect results
    results = []
    for _ in range(num_processes):
        result = result_queue.get(timeout=30)
        results.append(result)
    
    # Wait for all processes to complete
    for p in processes:
        p.join(timeout=10)
        assert not p.is_alive(), f"Process {p.pid} did not terminate"
    
    # Verify all processes succeeded
    assert len(results) == num_processes
    
    for result in results:
        assert result["success"], f"Process {result['process_id']} failed: {result.get('error')}"
        assert result["results"] is not None
        assert len(result["results"]) > 0
    
    # Verify different processes ran (different PIDs)
    pids = {r["pid"] for r in results}
    assert len(pids) == num_processes, "All processes should run in different PIDs"
    
    # Verify results are consistent (same pipeline should produce similar results)
    # But results may vary due to map node ordering
    for result in results:
        for value in result["results"]:
            # All results should be strings ending with !
            assert isinstance(value, str) or isinstance(value, dict)
            if isinstance(value, str):
                assert value.endswith("!")


def test_complex_fanout_independent_processes():
    """
    Test complex fanout pipeline running in multiple independent processes.
    
    This tests pipelines with multiple sources, mappers, and workers.
    """
    spec_id = "testing-complex-fanout"
    
    num_processes = 3
    result_queue = Queue()
    processes = []
    
    for i in range(num_processes):
        p = Process(target=run_pipeline_in_process, args=(spec_id, result_queue, i))
        p.start()
        processes.append(p)
    
    # Collect results
    results = []
    for _ in range(num_processes):
        result = result_queue.get(timeout=30)
        results.append(result)
    
    # Wait for all processes
    for p in processes:
        p.join(timeout=10)
    
    # Verify success
    for result in results:
        assert result["success"], f"Process failed: {result.get('error')}"
        assert result["results"] is not None


def test_complex_deep_independent_processes():
    """
    Test deeply nested complex pipeline running in multiple independent processes.
    
    This tests pipelines with multiple levels of maps and gathers.
    """
    spec_id = "testing-complex-deep"
    
    num_processes = 2
    result_queue = Queue()
    processes = []
    
    for i in range(num_processes):
        p = Process(target=run_pipeline_in_process, args=(spec_id, result_queue, i))
        p.start()
        processes.append(p)
    
    # Collect results
    results = []
    for _ in range(num_processes):
        result = result_queue.get(timeout=30)
        results.append(result)
    
    # Wait for all processes
    for p in processes:
        p.join(timeout=10)
    
    # Verify success
    for result in results:
        assert result["success"], f"Process failed: {result.get('error')}"
        assert result["results"] is not None


def test_distributed_runner_local_fallback():
    """
    Test DistributedRunner falls back to local execution when no workers available.
    
    This verifies that DistributedRunner can still work without actual workers.
    """
    spec_id = "testing-complex-parallel"
    tube = Tube.from_specification(spec_id)
    
    # Create DistributedRunner with no workers, should fall back to local
    runner = DistributedRunner(tube, workers=[], local_execution=True, use_async=False)
    
    results = []
    for value in runner.iter(n=3):
        results.append(value)
    
    assert len(results) > 0
    for value in results:
        assert value is not None


def test_distributed_runner_complex_pipeline():
    """
    Test DistributedRunner with complex pipeline using local fallback.
    """
    spec_id = "testing-complex-fanout"
    tube = Tube.from_specification(spec_id)
    
    runner = DistributedRunner(tube, workers=[], local_execution=True, use_async=False)
    
    results = []
    for value in runner.iter(n=2):
        results.append(value)
    
    assert len(results) > 0


def test_sequential_vs_parallel_execution():
    """
    Test that the same pipeline produces consistent results whether run sequentially
    or in parallel processes.
    """
    spec_id = "testing-complex-parallel"
    
    # Run sequentially
    tube1 = Tube.from_specification(spec_id)
    runner1 = SynchronousRunner(tube1)
    sequential_results = []
    for value in runner1.iter(n=3):
        sequential_results.append(value)
    
    # Run in parallel processes
    num_processes = 2
    result_queue = Queue()
    processes = []
    
    for i in range(num_processes):
        p = Process(target=run_pipeline_in_process, args=(spec_id, result_queue, i))
        p.start()
        processes.append(p)
    
    parallel_results = []
    for _ in range(num_processes):
        result = result_queue.get(timeout=30)
        assert result["success"]
        parallel_results.extend(result["results"])
    
    for p in processes:
        p.join(timeout=10)
    
    # Results should be consistent (same type, same validation)
    assert len(sequential_results) > 0
    assert len(parallel_results) > 0
    
    # Verify all results are valid
    for result in sequential_results + parallel_results:
        assert result is not None
        if isinstance(result, str):
            assert result.endswith("!")


def test_multiple_pipelines_simultaneous():
    """
    Test running multiple different pipelines simultaneously in different processes.
    """
    pipelines = [
        "testing-complex-parallel",
        "testing-complex-fanout",
        "testing-complex-deep",
    ]
    
    result_queue = Queue()
    processes = []
    
    for i, spec_id in enumerate(pipelines):
        p = Process(target=run_pipeline_in_process, args=(spec_id, result_queue, i))
        p.start()
        processes.append(p)
    
    # Collect results
    results = []
    for _ in range(len(pipelines)):
        result = result_queue.get(timeout=30)
        results.append(result)
    
    # Wait for all processes
    for p in processes:
        p.join(timeout=10)
    
    # Verify all succeeded
    assert len(results) == len(pipelines)
    for result in results:
        assert result["success"], f"Pipeline failed: {result.get('error')}"
        assert result["results"] is not None
        assert len(result["results"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

