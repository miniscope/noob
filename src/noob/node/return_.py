"""
Special Return sink that tube runners use to return values from :meth:`.TubeRunner.process`
"""

from typing import Any

from noob.node.base import Node


class Return(Node):
    """
    Special sink node that returns values from a tube runner's `process` method
    """

    _args: tuple | None = None
    _kwargs: dict | None = None

    def process(self, *args: Any, **kwargs: Any) -> None:
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

    def get(self, keep: bool) -> Any | None:
        """
        Get the stored value from the process call, clearing it.
        """
        try:
            return self._args, self._kwargs
        finally:
            if not keep:
                self._args = None
                self._kwargs = None
