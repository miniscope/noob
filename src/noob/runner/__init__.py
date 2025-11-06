"""
NOOB Distributed Execution Framework

High-performance, zero-dependency distributed computing for graph processing pipelines.

Available Runners:
- SynchronousRunner: Single-threaded baseline execution
- MultiProcessRunner: Parallel execution across CPU cores (bypasses GIL)
- DistributedRunner: HTTP-based cluster execution with advanced load balancing
- QueuedRunner: Enterprise-grade distributed coordination with ACID guarantees
- WorkerServer: Microservice-based task execution endpoint

Features:
- True parallelism via multiprocessing
- Intelligent load balancing (round-robin, least-loaded, fastest-response)
- Self-contained task queue with SQLite backend (no Redis/RabbitMQ needed!)
- Circuit breakers and automatic failover
- Task priorities and deadlines
- Worker affinity and specialized routing
- ACID transaction safety
- Comprehensive metrics and monitoring
"""

from noob.runner.base import TubeRunner
from noob.runner.sync import SynchronousRunner

# Core runners - always available
__all__ = ["SynchronousRunner", "TubeRunner"]

# Advanced parallel execution (requires multiprocessing)
try:
    from noob.runner.multiprocess import MultiProcessRunner, SharedMemoryRunner
    __all__.extend(["MultiProcessRunner", "SharedMemoryRunner"])
except ImportError as e:
    MultiProcessRunner = None
    SharedMemoryRunner = None

# HTTP-based distributed execution (requires httpx)
try:
    from noob.runner.distributed import (
        DistributedRunner,
        LoadBalancingStrategy,
        WorkerConfig,
        WorkerStatus,
    )
    __all__.extend([
        "DistributedRunner",
        "LoadBalancingStrategy",
        "WorkerConfig",
        "WorkerStatus",
    ])
except ImportError as e:
    DistributedRunner = None
    LoadBalancingStrategy = None
    WorkerConfig = None
    WorkerStatus = None

# Queue-based distributed execution (SQLite-backed, zero external deps!)
try:
    from noob.runner.task_queue import TaskQueue, TaskStatus, TaskPriority, Task
    from noob.runner.queued import QueuedRunner
    __all__.extend([
        "QueuedRunner",
        "TaskQueue",
        "TaskStatus",
        "TaskPriority",
        "Task",
    ])
except ImportError as e:
    QueuedRunner = None
    TaskQueue = None
    TaskStatus = None
    TaskPriority = None
    Task = None

# Worker server for distributed execution (requires FastAPI)
try:
    from noob.runner.worker_server import WorkerServer, create_worker_server
    __all__.extend(["WorkerServer", "create_worker_server"])
except ImportError as e:
    WorkerServer = None
    create_worker_server = None
