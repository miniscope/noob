#!/usr/bin/env python
"""
NOOB Distributed Execution Demo

This script demonstrates the various distributed runners and their capabilities.
Run this to see the power of distributed graph processing!
"""

import time
from pathlib import Path

from noob import Tube, SynchronousRunner


def demo_synchronous():
    """Baseline: Single-threaded execution"""
    print("\n" + "="*80)
    print("1. SynchronousRunner - Baseline Single-Threaded Execution")
    print("="*80)

    # Find a test pipeline
    pipeline_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "pipelines"
    pipeline = pipeline_dir / "complex_parallel.yaml"

    if not pipeline.exists():
        print(f"⚠️  Pipeline not found: {pipeline}")
        return

    tube = Tube.from_specification(str(pipeline))
    runner = SynchronousRunner(tube)

    print(f"📊 Processing 5 epochs...")
    start = time.time()
    results = runner.run(n=5)
    duration = time.time() - start

    print(f"✅ Completed in {duration:.2f}s")
    print(f"📈 Results: {len(results) if results else 0} epochs processed")


def demo_multiprocess():
    """MultiProcessRunner: True parallel execution"""
    print("\n" + "="*80)
    print("2. MultiProcessRunner - Parallel Execution Across CPU Cores")
    print("="*80)

    try:
        from noob.runner import MultiProcessRunner
    except ImportError:
        print("⚠️  MultiProcessRunner not available (missing dependencies)")
        return

    pipeline_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "pipelines"
    pipeline = pipeline_dir / "complex_parallel.yaml"

    if not pipeline.exists():
        print(f"⚠️  Pipeline not found: {pipeline}")
        return

    tube = Tube.from_specification(str(pipeline))

    import multiprocessing
    cores = multiprocessing.cpu_count()
    print(f"🖥️  Available CPU cores: {cores}")

    runner = MultiProcessRunner(tube, max_workers=min(4, cores))

    print(f"📊 Processing 5 epochs across {min(4, cores)} workers...")
    start = time.time()
    results = runner.run(n=5)
    duration = time.time() - start

    print(f"✅ Completed in {duration:.2f}s")
    print(f"📈 Results: {len(results) if results else 0} epochs processed")
    print(f"⚡ Speedup: Significant on multi-core systems!")


def demo_task_queue():
    """TaskQueue: Self-contained distributed coordination"""
    print("\n" + "="*80)
    print("3. TaskQueue - Self-Contained Distributed Coordination (SQLite)")
    print("="*80)

    try:
        from noob.runner import TaskQueue, TaskPriority, TaskStatus
    except ImportError:
        print("⚠️  TaskQueue not available")
        return

    # Create in-memory queue
    queue = TaskQueue(persistent=False)

    print("📝 Submitting tasks with different priorities...")

    # Submit tasks
    task_ids = []
    for i in range(10):
        priority = TaskPriority.CRITICAL if i < 2 else TaskPriority.NORMAL
        task_id = queue.submit_task(
            node_id=f"node_{i}",
            epoch=1,
            args=[i],
            priority=priority
        )
        task_ids.append(task_id)
        print(f"  ✓ Task {i}: {task_id[:8]}... (priority: {priority.name})")

    # Claim and process tasks
    print("\n🔄 Processing tasks...")
    worker_id = "demo-worker"
    processed = 0

    while True:
        task = queue.claim_task(worker_id)
        if not task:
            break

        print(f"  ⚙️  Worker processing: {task.node_id} (priority: {task.priority})")
        queue.start_task(task.task_id)

        # Simulate processing
        time.sleep(0.1)

        queue.complete_task(task.task_id, result={"processed": True})
        processed += 1

    print(f"\n✅ Processed {processed} tasks")

    # Show statistics
    stats = queue.get_queue_stats()
    print("\n📊 Queue Statistics:")
    for status, count in stats.items():
        if count > 0:
            print(f"  {status}: {count}")

    queue.shutdown()


def demo_distributed_runner():
    """DistributedRunner: HTTP-based cluster execution"""
    print("\n" + "="*80)
    print("4. DistributedRunner - HTTP-Based Cluster Execution")
    print("="*80)

    try:
        from noob.runner import DistributedRunner, LoadBalancingStrategy
    except ImportError:
        print("⚠️  DistributedRunner not available (missing httpx)")
        return

    pipeline_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "pipelines"
    pipeline = pipeline_dir / "complex_parallel.yaml"

    if not pipeline.exists():
        print(f"⚠️  Pipeline not found: {pipeline}")
        return

    tube = Tube.from_specification(str(pipeline))

    print("🌐 Running with local fallback (no actual workers)")
    print("💡 Start workers with: python -m noob.runner.worker_server --port 8000")

    runner = DistributedRunner(
        tube,
        workers=[],  # No workers - will use local fallback
        local_execution=True,
        load_balancing=LoadBalancingStrategy.LEAST_LOADED,
        use_async=False  # Simpler for demo
    )

    print(f"📊 Processing 3 epochs...")
    start = time.time()
    results = runner.run(n=3)
    duration = time.time() - start

    print(f"✅ Completed in {duration:.2f}s (local fallback)")
    print(f"📈 Results: {len(results) if results else 0} epochs processed")

    runner.deinit()


def demo_worker_server_info():
    """Show how to start worker servers"""
    print("\n" + "="*80)
    print("5. WorkerServer - Microservice Execution Endpoint")
    print("="*80)

    print("""
🖥️  To start a worker server, run:

    python -m noob.runner.worker_server \\
        --host 0.0.0.0 \\
        --port 8000 \\
        --tube pipeline.yaml \\
        --max-tasks 10

📦 Or with Docker:

    docker run -p 8000:8000 noob-worker \\
        --tube /app/pipeline.yaml

🎯 With affinity tags (for GPU workers):

    python -m noob.runner.worker_server \\
        --host 0.0.0.0 \\
        --port 8000 \\
        --tags gpu,cuda \\
        --tube pipeline.yaml

🌐 API Endpoints:
    GET  /health          - Health check
    GET  /metrics         - Worker metrics
    POST /execute         - Execute task
    POST /load_tube       - Load tube specification
    POST /cancel/{task_id}- Cancel task
    """)


def main():
    """Run all demos"""
    print("\n" + "🚀"*40)
    print("NOOB DISTRIBUTED EXECUTION FRAMEWORK DEMO")
    print("🚀"*40)

    # Run demos
    demo_synchronous()
    demo_multiprocess()
    demo_task_queue()
    demo_distributed_runner()
    demo_worker_server_info()

    print("\n" + "="*80)
    print("✨ Demo Complete!")
    print("="*80)
    print("""
📚 For more information:
    - DISTRIBUTED_EXECUTION.md - Full documentation
    - DISTRIBUTED_SUMMARY.md    - Implementation summary
    - rust/README.md            - Rust extensions guide

🚀 Get started with distributed execution:

    from noob.runner import MultiProcessRunner
    runner = MultiProcessRunner(tube, max_workers=8)
    results = runner.run(n=100)

⚡ For extreme performance, install Rust extensions:

    cd rust/noob_core
    maturin build --release
    pip install target/wheels/*.whl

💬 Questions? Check the docs or open an issue!
    """)


if __name__ == "__main__":
    main()
