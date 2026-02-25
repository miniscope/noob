# ruff: noqa I001 - import order meaningful to avoid cycles
from typing import TYPE_CHECKING, Literal, overload

from noob.runner.base import TubeRunner

from noob.runner.asyncio import AsyncRunner
from noob.runner.sync import SynchronousRunner

if TYPE_CHECKING:
    from noob.runner.zmq import ZMQRunner


@overload
def get_runner(runner: Literal["sync"] = "sync") -> type[SynchronousRunner]: ...


@overload
def get_runner(runner: Literal["async"] = "async") -> type[AsyncRunner]: ...


@overload
def get_runner(runner: Literal["zmq"] = "zmq") -> type["ZMQRunner"]: ...


@overload
def get_runner(runner: Literal["sync", "async", "zmq"] = "sync") -> type[TubeRunner]: ...


def get_runner(runner: Literal["sync", "async", "zmq"] = "sync") -> type[TubeRunner]:
    """Get a runner by its short name"""
    if runner == "sync":
        return SynchronousRunner
    elif runner == "async":
        return AsyncRunner
    elif runner == "zmq":
        # importerror raised if deps not installed
        from noob.runner.zmq import ZMQRunner

        return ZMQRunner


__all__ = ["AsyncRunner", "SynchronousRunner", "TubeRunner", "get_runner"]
