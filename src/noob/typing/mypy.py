from collections.abc import Callable

from mypy.plugin import MethodSigContext, Plugin
from mypy.types import FunctionLike

RUNNER_NAMES = (
    "noob.runner.sync.SynchronousRunner",
    "noob.runner.base.TubeRunner",
)
"""
Runner classes to support.
Not exactly sure how to reliably detect subclasses from within mypy yet,
so consider this a temporary hack pending figuring out the mypy machinery.
"""

_process_names = tuple(n + ".process" for n in RUNNER_NAMES)


class NoobPlugin(Plugin):
    """
    Support static type checking for noob pipelines :)

    TODO: type checking for...
    - runner `process` calls
    - node `process` chains
    - ...
    """

    def get_method_signature_hook(
        self, fullname: str
    ) -> Callable[[MethodSigContext], FunctionLike] | None:
        if fullname in _process_names:
            return get_runner_process_signature


def get_runner_process_signature(ctx: MethodSigContext) -> FunctionLike:
    raise NotImplementedError()


def plugin(version: str) -> type[NoobPlugin]:
    return NoobPlugin
