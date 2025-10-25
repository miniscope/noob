import warnings
from typing import TYPE_CHECKING, Any

from noob.exceptions import ExtraInputWarning
from noob.node.base import Node, Slot
from noob.types import ConfigSource, RunnerContext

if TYPE_CHECKING:
    from noob.runner import TubeRunner
    from noob.tube import Tube, TubeSpecification


class TubeNode(Node):
    """
    A node that contains another tube within it
    """

    tube: ConfigSource

    _tube: "Tube" = None
    _tube_spec: "TubeSpecification" = None
    _runner: "TubeRunner" = None

    @property
    def tube_spec(self) -> "TubeSpecification":
        from noob.tube import TubeSpecification

        if self._tube_spec is None:
            self._tube_spec = TubeSpecification.from_any(self.tube)
        return self._tube_spec

    def init(self, context: RunnerContext) -> None:
        from noob import SynchronousRunner, Tube

        with warnings.catch_warnings(action="ignore", category=ExtraInputWarning):
            self._tube = Tube.from_specification(
                self.tube, input={**context["tube"].input_collection.chain}
            )
        self._runner = SynchronousRunner(tube=self._tube)
        self._runner.init()

    def deinit(self) -> None:
        self._runner.deinit()

    def process(self, **kwargs: Any) -> Any:
        return self._runner.process(**kwargs)

    def _collect_slots(self) -> dict[str, Slot]:
        from noob.input import InputScope

        slots = {}
        for in_key, in_val in self.tube_spec.input.items():
            if in_val.scope == InputScope.process:
                slots[in_key] = Slot(name=in_key, annotation=Any)
        return slots
