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

    def process(self, value: TInput) -> None:
        """
        Store the incoming value to retrieve later with :meth:`.get`
        """
        self._value = value

    def get(self, keep: bool = False) -> dict[str, TInput] | None:
        """
        Get the stored value from the process call

        Args:
            keep (bool): If ``True``, keep the stored value, otherwise clear it, consume it
        """
        if self._value is None:
            return None
        else:
            val = {self.config["key"]: self._value}
            if not keep:
                self._value = None
            return val
