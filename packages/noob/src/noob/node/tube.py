import warnings
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Union

from pydantic import ConfigDict

from noob.edge import Signal, Slot
from noob.event import MetaSignal
from noob.exceptions import ExtraInputWarning
from noob.node.base import Node
from noob.node.spec import NodeSpecification
from noob.types import ConfigSource, Epoch, RunnerContext

if TYPE_CHECKING:
    from noob.runner import TubeRunner
    from noob.tube import Tube, TubeSpecification


class TubeNode(Node):
    """
    A node that contains another tube within it.

    .. note::

        A nested tube may not return a scalar literal ``None`` from its return node,
        that is interpreted as a ``NoEvent``, as the return value of ``process``
        is ``None`` when no events are emitted.

        Wrap ``None``s in a dictionary return to disambiguate them.

        i.e. rather than::

            depends: node.value

        use::

            depends:
              - something: node.value

        If you have a need for returning scalar ``None``s from a nested tube,
        please raise an issue!

    """

    spec: NodeSpecification  # not optional for tube nodes
    tube: ConfigSource

    _tube: Union["Tube", None] = None
    _tube_spec: Union["TubeSpecification", None] = None
    _runner: Union["TubeRunner", None] = None

    model_config = ConfigDict(extra="allow")

    @property
    def tube_spec(self) -> "TubeSpecification":
        from noob.tube import TubeSpecification

        if self._tube_spec is None:
            self._tube_spec = TubeSpecification.from_any(self.tube)
        return self._tube_spec

    # TODO: support dependency injected inits in plugin
    def init(self, context: RunnerContext) -> None:  # type: ignore[override]
        from noob import SynchronousRunner, Tube

        spec_params = self.spec.params if self.spec.params else {}

        input_collection = context["input_collection"]
        input_params = input_collection.get_node_params(
            {k: v for k, v in spec_params.items() if k != "tube"}
        )
        if self.__pydantic_extra__:
            extra_params = {
                k: v for k, v in self.__pydantic_extra__.items() if k not in input_params
            }
        else:
            extra_params = {}

        with warnings.catch_warnings(action="ignore", category=ExtraInputWarning):
            self._tube = Tube.from_specification(
                self.tube, input={**input_collection.chain, **input_params, **extra_params}
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
        elif res is None:
            now = datetime.now(UTC)
            return [
                self._event_maker.new_event(
                    signal=key, epoch=epoch, value=MetaSignal.NoEvent, timestamp=now
                )
                for key in self.signals
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

    @classmethod
    def get_signals(cls, spec: NodeSpecification | None = None) -> dict[str, Signal]:
        """Forward signals from the return node"""
        if spec is None:
            raise ValueError("Must pass a spec to get slots for a tube node")

        from noob.tube import TubeSpecification

        if not spec.params or "tube" not in spec.params:
            raise ValueError("Tube node specifications must have a `tube` in their params")
        tube_spec = TubeSpecification.from_any(spec.params["tube"])

        signals = {}
        return_nodes = [n for n in tube_spec.nodes.values() if n.type_ == "return"]
        if not return_nodes:
            return {"value": Signal(name="value", annotation=Any)}
        else:
            return_node = return_nodes[0]
            if not return_node.depends or isinstance(return_node.depends, str):
                return {"value": Signal(name="value", annotation=Any)}

            for dep in return_node.depends:
                if isinstance(dep, str):
                    continue
                name = list(dep.keys())[0]
                signals[name] = Signal(name=name, annotation=Any)
            return signals
