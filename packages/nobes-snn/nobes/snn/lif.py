import time
from random import random
from typing import Annotated as A

from pydantic import Field

from noob import Name, Node, NodeSpecification
from noob.event import MetaSignal
from noob.node import Slot
from noob.node.base import all_slots_optional


class LIFNeuron(Node):
    """
    A leaky integrate and fire neuron!!!

    I strongly doubt this implementation is right, for example it doesn't exactly leak,
    but hey.
    """

    spec: NodeSpecification  # must have a spec to run LIFNeuron
    weights: dict[str, float] = Field(default_factory=dict)
    resistance: float = 10_000_000
    capacitance: float = 10
    v_membrane: float = -70e-3
    v_rest: float = -70e-3
    v_reset: float = -80e-3
    I_static: float = 0
    threshold: float = 10e-3
    spike_current: float = 10
    last_step: float = 0

    def init(self) -> None:
        for dep in self.spec.depends:
            key, val = next(iter(dep.items()))
            if key not in self.weights:
                self.weights[key] = random()

    def process(
        self, **kwargs: float
    ) -> tuple[A[float, Name("voltage")], A[float | MetaSignal, Name("spike")]]:
        # ignore the first step, nothing matters in the first step
        if self.last_step == 0:
            self.last_step = time.time()
            return self.v_membrane, MetaSignal.NoEvent

        current = self.I_static
        for k, v in kwargs.items():
            current += self.weights.get(k, 1) * v

        now = time.time()
        dt = now - self.last_step
        tau = self.resistance * self.capacitance
        self.last_step = now
        self.v_membrane += (
            ((self.v_rest - self.v_membrane) + (self.resistance * current)) * dt
        ) / tau

        spike = MetaSignal.NoEvent
        if self.v_membrane > self.threshold:
            self.v_membrane = self.v_reset
            spike = self.spike_current

        return self.v_membrane, spike

    def _collect_slots(self) -> dict[str, Slot]:
        return all_slots_optional(self.spec)
