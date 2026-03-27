# ruff: noqa I001 - import order meaningful to avoid cycles
import warnings
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Literal, overload

from noob.exceptions import EntrypointImportWarning
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


def get_runner(runner: str = "sync") -> type[TubeRunner]:
    """Get a runner by its short name"""
    if runner == "sync":
        return SynchronousRunner
    elif runner == "async":
        return AsyncRunner
    elif runner == "zmq":
        # importerror raised if deps not installed
        from noob.runner.zmq import ZMQRunner

        return ZMQRunner
    else:
        runner_cls = _get_entrypoint_runners().get(runner)
        if runner_cls is not None:
            return runner_cls
        else:
            raise KeyError(f"Unknown runner type: {runner}")


def _get_entrypoint_runners() -> dict[str, type[TubeRunner]]:
    """
    Get runners provided by package entrypoints like:

        [project.entry-points."noob.add_runners"]
        runners = "my_package.something:add_runners

    """
    runners = {}
    for ext in entry_points(group="noob.add_runners"):
        try:
            add_runners_fn = ext.load()
        except (ImportError, AttributeError):
            warnings.warn(
                f"Runner entrypoint {ext.name}, {ext.value} "
                f"could not be imported, or the function could not be found. Ignoring",
                EntrypointImportWarning,
                stacklevel=1,
            )
            continue
        try:
            runners.update(add_runners_fn())
        except Exception as e:
            # bare exception is fine here - we're calling external code and can't know.
            warnings.warn(
                f"Config source entrypoint {ext.name}, {ext.value} "
                f"threw an error, or returned an invalid list of paths, ignoring.\n{str(e)}",
                EntrypointImportWarning,
                stacklevel=1,
            )
    return runners


__all__ = ["AsyncRunner", "SynchronousRunner", "TubeRunner", "get_runner"]
