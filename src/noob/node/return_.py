"""
Special Return sink that tube runners use to return values from :meth:`.TubeRunner.process`
"""

from typing import Any

from noob.node.base import Sink, TInput


class Return(Sink):
    """
    Special sink node that returns values from a tube runner's `process` method
    """

    input_type = Any

    _value: Any = None

    def process(self, **kwargs: Any) -> None:
        """
        Store the incoming value to retrieve later with :meth:`.get`
        """
        self._value = kwargs

    def get(self) -> dict[str, TInput] | None:
        """
        Get the stored value from the process call
        """
        val = self._value
        self._value = None
        return val

