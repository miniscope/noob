"""
Special Return sink that tube runners use to return values from :meth:`.TubeRunner.process`
"""

import sys
from typing import Any

from noob.node.base import Sink, TInput

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class ReturnConfig(TypedDict):
    """
    Config for return nodes
    """

    key: str
    """The key to use in the returned dictionary"""


class Return(Sink):
    """
    Special sink node that returns values from a tube runner's `process` method
    """

    name = "return"
    input_type = Any

    config: ReturnConfig

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
