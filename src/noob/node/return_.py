"""
Special Return sink that tube runners use to return values from :meth:`.TubeRunner.process`
"""

from collections import defaultdict
from typing import Any

from pydantic import PrivateAttr

from noob.event import MetaSignal
from noob.node.base import Node, Slot
from noob.types import Epoch, EventMap


class Return(Node):
    """
    Special sink node that returns values from a tube runner's `process` method
    """

    _args: tuple | None = None
    _kwargs: dict = PrivateAttr(default_factory=lambda: defaultdict(list))
    _seen_epochs: set[Epoch] = set()

    def process(self, *args: Any, __events: EventMap, **kwargs: Any) -> MetaSignal:
        """
        Store the incoming value to retrieve later with :meth:`.get`
        """
        if self._args is None:
            self._args = args
        else:
            self._args += args

        for key, val in kwargs.items():
            if __events[key]["epoch"] in self._seen_epochs:
                continue
            self._kwargs[key].append(val)

        for evt in __events.values():
            self._seen_epochs.add(evt["epoch"])

        return MetaSignal.NoEvent

    def get(self, keep: bool) -> Any | None:
        """
        Get the stored value from the process call, clearing it.
        """
        if self._kwargs:
            # flatten kwargs if only one received
            kwargs = {key: val[0] if len(val) == 1 else val for key, val in self._kwargs.items()}
        else:
            kwargs = {}

        try:
            # FIXME: what a nightmare - make all of these derive from the spec
            if self._args and self.spec is not None and isinstance(self.spec.depends, str):
                return self._args[0]
            elif self._args and kwargs:
                return self._args, kwargs
            elif self._args:
                return self._args
            elif kwargs:
                return kwargs
            else:
                return None
        finally:
            if not keep:
                self._args = None
                self._kwargs = defaultdict(list)
                self._seen_epochs = set()

    def _collect_slots(self) -> dict[str, Slot]:
        if self.spec is None or not self.spec.depends:
            raise ValueError("Return nodes must have a specification that defines what they return")
        if isinstance(self.spec.depends, str):
            return {}
        slots = {}
        for dep in self.spec.depends:
            if isinstance(dep, str):
                continue
            name = list(dep.keys())[0]
            slots[name] = Slot(name=name, annotation=Any)
        return slots
