import warnings
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Union

from noob.edge import Slot
from noob.exceptions import ExtraInputWarning
from noob.node.base import Node
from noob.node.spec import NodeSpecification
from noob.types import ConfigSource, Epoch, RunnerContext

if TYPE_CHECKING:
    from noob.runner import TubeRunner
    from noob.tube import Tube, TubeSpecification


class TubeNode(Node):
    """
    A node that contains another tube within it
    """

    tube: ConfigSource

    _tube: Union["Tube", None] = None
    _tube_spec: Union["TubeSpecification", None] = None
    _runner: Union["TubeRunner", None] = None

    @property
    def tube_spec(self) -> "TubeSpecification":
        from noob.tube import TubeSpecification

        if self._tube_spec is None:
            self._tube_spec = TubeSpecification.from_any(self.tube)
        return self._tube_spec

    # TODO: support dependency injected inits in plugin
    def init(self, context: RunnerContext) -> None:  # type: ignore[override]
        from noob import SynchronousRunner, Tube

        with warnings.catch_warnings(action="ignore", category=ExtraInputWarning):
            self._tube = Tube.from_specification(
                self.tube, input={**context["tube"].input_collection.chain}
            )
        self._runner = SynchronousRunner(tube=self._tube)
        self._runner.init()

    def deinit(self) -> None:
        if self._runner is not None:
            self._runner.deinit()

    def process(self, epoch: Epoch, **kwargs: Any) -> Any:
        if self._runner is None:
            raise RuntimeError(
                "TubeNode must be initialized within a Runner "
                "to receive the outer runner's context. "
                "It doesn't make sense to run a TubeNode on its own."
            )
        res = self._runner.process(**kwargs)
        if isinstance(res, dict):
            now = datetime.now(UTC)
            return [
                self._event_maker.new_event(
                    signal=key,
                    epoch=epoch,
                    value=value,
                    timestamp=now,
                )
                for key, value in res.items()
            ]
        else:
            return res

    @classmethod
    def get_slots(cls, spec: NodeSpecification | None = None) -> dict[str, Slot]:
        if spec is None:
            raise ValueError("Must pass a spec to get slots for a tube node")

        from noob.input import InputScope
        from noob.tube import TubeSpecification

        if not spec.params or "tube" not in spec.params:
            raise ValueError("Tube node specifications must have a `tube` in their params")
        tube_spec = TubeSpecification.from_any(spec.params["tube"])

        slots = {}
        for in_key, in_val in tube_spec.input.items():
            if in_val.scope == InputScope.process:
                slots[in_key] = Slot(name=in_key, annotation=Any)
        return slots
