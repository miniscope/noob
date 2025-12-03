"""
Special Return sink that tube runners use to return values from :meth:`.TubeRunner.process`
"""

from typing import Any

from noob.event import NOEVENT
from noob.node.base import Node, Slot


class Return(Node):
    """
    Special sink node that returns values from a tube runner's `process` method
    """

    _args: tuple | None = None
    _kwargs: dict | None = None

    def process(self, *args: Any, **kwargs: Any) -> object:
        """
        Store the incoming value to retrieve later with :meth:`.get`
        """
        if self._args is None:
            self._args = args
        else:
            self._args += args

        if self._kwargs is None:
            self._kwargs = kwargs
        else:
            self._kwargs.update(kwargs)

        return NOEVENT

    def get(self, keep: bool) -> Any | None:
        """
        Get the stored value from the process call, clearing it.
        """
        try:
            # FIXME: what a nightmare - make all of these derive from the spec
            if self._args and self.spec is not None and isinstance(self.spec.depends, str):
                return self._args[0]
            elif self._args and self._kwargs:
                return self._args, self._kwargs
            elif self._args:
                return self._args
            elif self._kwargs:
                return self._kwargs
            else:
                return None
        finally:
            if not keep:
                self._args = None
                self._kwargs = None

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
