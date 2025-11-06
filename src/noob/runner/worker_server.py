"""
Worker Server for Distributed Execution

This module provides a production-ready HTTP server that can execute
node processing tasks for distributed graph execution. The server can:
- Execute arbitrary node tasks with serialized inputs
- Handle JSON and pickle serialization
- Support task cancellation and timeouts
- Provide health checks and metrics
- Handle multiple concurrent tasks
"""

from __future__ import annotations

import asyncio
import json
import pickle
import signal
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from logging import Logger
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from noob import init_logger
from noob.event import Event
from noob.node import Node
from noob.store import EventStore
from noob.tube import Tube
from noob.utils import resolve_python_identifier


class TaskRequest(BaseModel):
    """Request model for task execution"""
    task_id: str | None = None
    node_id: str
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    epoch: int
    serialization: str = "json"  # json or pickle
    timeout: float | None = None


class TaskResponse(BaseModel):
    """Response model for task execution"""
    task_id: str
    node_id: str
    epoch: int
    success: bool
    events: list[dict] | None = None
    error: str | None = None
    execution_time: float
    worker_id: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    worker_id: str
    uptime: float
    tasks_completed: int
    tasks_failed: int
    tasks_active: int
    memory_mb: float | None = None


class WorkerMetrics(BaseModel):
    """Worker performance metrics"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_active: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    peak_memory_mb: float = 0.0


@dataclass
class WorkerServer:
    """
    HTTP server that executes node tasks for distributed execution.

    This server can load a Tube specification and execute individual
    node tasks on demand, returning serialized events.

    Features:
    - FastAPI-based REST API
    - Async task execution with timeouts
    - Health monitoring
    - Metrics collection
    - Graceful shutdown
    - Task cancellation
    """

    host: str = "0.0.0.0"
    port: int = 8000
    worker_id: str = field(default_factory=lambda: str(uuid4()))
    max_concurrent_tasks: int = 10
    default_timeout: float = 300.0  # 5 minutes

    _logger: Logger = field(default_factory=lambda: init_logger("worker.server"))
    _metrics: WorkerMetrics = field(default_factory=WorkerMetrics)
    _start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    _active_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    _tube: Tube | None = None
    _store: EventStore = field(default_factory=EventStore)
    _app: Any = None  # FastAPI app

    def __post_init__(self):
        """Initialize FastAPI app"""
        try:
            from fastapi import FastAPI, HTTPException, Request
            from fastapi.responses import JSONResponse
        except ImportError:
            raise ImportError(
                "FastAPI is required for WorkerServer. Install with: pip install fastapi uvicorn"
            )

        self._app = FastAPI(
            title="NOOB Worker Server",
            description="Distributed task execution worker for NOOB graph processing",
            version="1.0.0"
        )

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register all API endpoints"""
        from fastapi import HTTPException, Request
        from fastapi.responses import JSONResponse

        @self._app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return self._get_health_response()

        @self._app.get("/metrics")
        async def get_metrics():
            """Get worker metrics"""
            return self._metrics

        @self._app.post("/execute")
        async def execute_task(request: TaskRequest):
            """Execute a node task"""
            return await self._execute_task(request)

        @self._app.post("/load_tube")
        async def load_tube(request: Request):
            """Load a tube specification"""
            data = await request.json()
            return await self._load_tube(data)

        @self._app.post("/cancel/{task_id}")
        async def cancel_task(task_id: str):
            """Cancel a running task"""
            return await self._cancel_task(task_id)

        @self._app.get("/tasks")
        async def list_tasks():
            """List active tasks"""
            return {
                "active_tasks": list(self._active_tasks.keys()),
                "count": len(self._active_tasks)
            }

        @self._app.post("/shutdown")
        async def shutdown():
            """Graceful shutdown"""
            asyncio.create_task(self._shutdown())
            return {"status": "shutting_down"}

    def _get_health_response(self) -> HealthResponse:
        """Generate health check response"""
        uptime = (datetime.now(UTC) - self._start_time).total_seconds()

        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            memory_mb = None

        return HealthResponse(
            status="healthy",
            worker_id=self.worker_id,
            uptime=uptime,
            tasks_completed=self._metrics.tasks_completed,
            tasks_failed=self._metrics.tasks_failed,
            tasks_active=self._metrics.tasks_active,
            memory_mb=memory_mb
        )

    async def _load_tube(self, data: dict) -> dict:
        """Load a tube specification from serialized data"""
        try:
            # Load tube from specification
            if "yaml_path" in data:
                self._tube = Tube.from_specification(data["yaml_path"])
            elif "yaml_string" in data:
                import yaml
                spec = yaml.safe_load(data["yaml_string"])
                self._tube = Tube.from_specification(spec)
            elif "tube_pickle" in data:
                self._tube = pickle.loads(bytes.fromhex(data["tube_pickle"]))
            else:
                return {"success": False, "error": "No tube specification provided"}

            # Initialize nodes
            for node in self._tube.nodes.values():
                node.init()

            self._logger.info("Loaded tube with %d nodes", len(self._tube.nodes))
            return {
                "success": True,
                "nodes": list(self._tube.nodes.keys()),
                "node_count": len(self._tube.nodes)
            }
        except Exception as e:
            self._logger.error("Failed to load tube: %s", e)
            return {"success": False, "error": str(e)}

    async def _execute_task(self, request: TaskRequest) -> TaskResponse:
        """Execute a node task asynchronously"""
        task_id = request.task_id or str(uuid4())
        start_time = datetime.now(UTC)

        # Check concurrent task limit
        if len(self._active_tasks) >= self.max_concurrent_tasks:
            return TaskResponse(
                task_id=task_id,
                node_id=request.node_id,
                epoch=request.epoch,
                success=False,
                error="Worker at capacity",
                execution_time=0.0,
                worker_id=self.worker_id
            )

        # Update metrics
        self._metrics.tasks_active += 1

        try:
            # Deserialize arguments if needed
            args = request.args
            kwargs = request.kwargs

            if request.serialization == "pickle":
                args = [pickle.loads(bytes.fromhex(arg)) if isinstance(arg, str) else arg for arg in args]
                kwargs = {
                    k: pickle.loads(bytes.fromhex(v)) if isinstance(v, str) else v
                    for k, v in kwargs.items()
                }

            # Get the node
            if self._tube is None:
                raise ValueError("No tube loaded. Call /load_tube first")

            if request.node_id not in self._tube.nodes:
                raise ValueError(f"Node {request.node_id} not found in tube")

            node = self._tube.nodes[request.node_id]

            # Execute the task with timeout
            timeout = request.timeout or self.default_timeout

            async def execute():
                """Execute node processing"""
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    node.process,
                    *args,
                    **kwargs
                )
                return result

            # Create and track task
            task = asyncio.create_task(execute())
            self._active_tasks[task_id] = task

            try:
                value = await asyncio.wait_for(task, timeout=timeout)
            finally:
                if task_id in self._active_tasks:
                    del self._active_tasks[task_id]

            # Store result as events
            events = self._store.add(node.signals, value, request.node_id, request.epoch)

            # Serialize events
            serialized_events = [self._serialize_event(e, request.serialization) for e in events] if events else []

            # Calculate execution time
            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            # Update metrics
            self._metrics.tasks_completed += 1
            self._metrics.tasks_active -= 1
            self._metrics.total_execution_time += execution_time
            self._metrics.avg_execution_time = (
                self._metrics.total_execution_time / self._metrics.tasks_completed
            )

            return TaskResponse(
                task_id=task_id,
                node_id=request.node_id,
                epoch=request.epoch,
                success=True,
                events=serialized_events,
                execution_time=execution_time,
                worker_id=self.worker_id
            )

        except asyncio.TimeoutError:
            self._metrics.tasks_failed += 1
            self._metrics.tasks_active -= 1
            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            return TaskResponse(
                task_id=task_id,
                node_id=request.node_id,
                epoch=request.epoch,
                success=False,
                error=f"Task timeout after {timeout}s",
                execution_time=execution_time,
                worker_id=self.worker_id
            )

        except Exception as e:
            self._metrics.tasks_failed += 1
            self._metrics.tasks_active -= 1
            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            self._logger.error("Task execution failed: %s\n%s", e, traceback.format_exc())

            return TaskResponse(
                task_id=task_id,
                node_id=request.node_id,
                epoch=request.epoch,
                success=False,
                error=str(e),
                execution_time=execution_time,
                worker_id=self.worker_id
            )

    def _serialize_event(self, event: Event, serialization: str = "json") -> dict:
        """Serialize an event for network transmission"""
        value = event.get("value")

        if serialization == "json" and self._is_json_serializable(value):
            return {
                "id": event["id"],
                "timestamp": event["timestamp"].isoformat(),
                "node_id": event["node_id"],
                "signal": event["signal"],
                "epoch": event["epoch"],
                "value": value,
                "serialization": "json"
            }
        else:
            # Use pickle for complex objects
            return {
                "id": event["id"],
                "timestamp": event["timestamp"].isoformat(),
                "node_id": event["node_id"],
                "signal": event["signal"],
                "epoch": event["epoch"],
                "value": pickle.dumps(value).hex(),
                "serialization": "pickle"
            }

    def _is_json_serializable(self, obj: Any) -> bool:
        """Check if an object is JSON serializable"""
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False

    async def _cancel_task(self, task_id: str) -> dict:
        """Cancel a running task"""
        if task_id not in self._active_tasks:
            return {"success": False, "error": "Task not found"}

        task = self._active_tasks[task_id]
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        del self._active_tasks[task_id]
        self._metrics.tasks_active -= 1

        return {"success": True, "task_id": task_id}

    async def _shutdown(self):
        """Graceful shutdown"""
        self._logger.info("Shutting down worker server...")

        # Cancel all active tasks
        for task_id, task in self._active_tasks.items():
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)

        # Deinitialize nodes
        if self._tube:
            for node in self._tube.nodes.values():
                try:
                    node.deinit()
                except Exception as e:
                    self._logger.warning("Error deinitializing node: %s", e)

        self._logger.info("Worker server shut down")

    def run(self):
        """Start the worker server"""
        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "uvicorn is required to run WorkerServer. Install with: pip install uvicorn"
            )

        self._logger.info(
            "Starting worker server %s on %s:%d",
            self.worker_id,
            self.host,
            self.port
        )

        # Setup signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            self._logger.info("Received shutdown signal")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Run the server
        uvicorn.run(
            self._app,
            host=self.host,
            port=self.port,
            log_level="info"
        )

    async def run_async(self):
        """Run server asynchronously (for testing or embedding)"""
        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "uvicorn is required to run WorkerServer. Install with: pip install uvicorn"
            )

        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


def create_worker_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    tube_path: str | None = None,
    max_concurrent_tasks: int = 10
) -> WorkerServer:
    """
    Factory function to create a worker server.

    Args:
        host: Host to bind to
        port: Port to listen on
        tube_path: Optional path to tube YAML specification
        max_concurrent_tasks: Maximum concurrent tasks

    Returns:
        Configured WorkerServer instance
    """
    server = WorkerServer(
        host=host,
        port=port,
        max_concurrent_tasks=max_concurrent_tasks
    )

    # Load tube if path provided
    if tube_path:
        tube = Tube.from_specification(tube_path)
        asyncio.run(server._load_tube({"tube_pickle": pickle.dumps(tube).hex()}))

    return server


if __name__ == "__main__":
    """Command-line interface for running worker server"""
    import argparse

    parser = argparse.ArgumentParser(description="NOOB Worker Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--tube", help="Path to tube YAML specification")
    parser.add_argument("--max-tasks", type=int, default=10, help="Max concurrent tasks")
    parser.add_argument("--worker-id", help="Custom worker ID")

    args = parser.parse_args()

    server = WorkerServer(
        host=args.host,
        port=args.port,
        worker_id=args.worker_id or str(uuid4()),
        max_concurrent_tasks=args.max_tasks
    )

    if args.tube:
        tube = Tube.from_specification(args.tube)
        asyncio.run(server._load_tube({"tube_pickle": pickle.dumps(tube).hex()}))

    server.run()
