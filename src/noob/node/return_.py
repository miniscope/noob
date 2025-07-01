"""
Special Return sink that tube runners use to return values from :meth:`.TubeRunner.process`
"""

from typing import Any

from noob.node.base import Node


class Return(Node):
    """
    Special sink node that returns values from a tube runner's `process` method
    """

    _value: dict | None = None

    def process(self, **kwargs: Any) -> None:
        """
        Store the incoming value to retrieve later with :meth:`.get`
        """
        if self._value is None:
            self._value = kwargs
        else:
            self._value.update(kwargs)

    def get(self) -> dict | None:
        """
        Get the stored value from the process call, clearing it.
        """
        try:
            return self._value
        finally:
            self._value = None
