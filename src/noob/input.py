import re
from collections import ChainMap, defaultdict
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, Field, PrivateAttr

from noob.exceptions import InputMissingError
from noob.node.base import Edge
from noob.types import AbsoluteIdentifier, PythonIdentifier


class InputScope(StrEnum):
    """The scope that input must be provided in"""

    tube = "tube"
    process = "process"


class InputSpecification(BaseModel):
    """
    Specification of inputs to a noob tube.

    Inputs can be supplied at different times and frequencies,
    as specified by `scope`:

    - `tube`: When instantiating the tube
    - `process`: Per call to :meth:`.TubeRunner.process`

    `tube`-scoped inputs may be used in a node's `param` specification,
    and `process`-scoped inputs may be used as one of a node's `depends`.

    Inputs can be supplied at a "higher" scope and be accessed by lower scopes:
    e.g. input requested with a process scope can use input provided when instantiating the tube,
    if not provided to process but provided to the tube.
    """

    id: PythonIdentifier
    type_: AbsoluteIdentifier = Field(..., alias="type")
    scope: InputScope = InputScope.tube


class InputCollection(BaseModel):
    """
    A collection of input specifications used during runtime, split by scope,
    to validate presence of and to combine inputs.
    """

    INPUT_PATTERN: ClassVar[re.Pattern] = re.compile(r"input\.(?P<key>.*)")

    specs: dict[InputScope, dict[PythonIdentifier, InputSpecification]] = Field(
        default_factory=lambda: defaultdict(dict)
    )

    # store long-lived scope inputs
    _input: dict[InputScope, dict] = PrivateAttr(default_factory=lambda: defaultdict(dict))
    _chain: ChainMap | None = None

    @property
    def chain(self) -> ChainMap:
        """
        Make a chainmap of inputs at different scopes

        (for possible expansion of number of scopes, to e.g. a runner scope)
        """
        if self._chain is None:
            self._chain = ChainMap(self._input[InputScope.tube])
        return self._chain

    def get(self, key: str, input: dict | None = None) -> Any:
        """Get a value from the inputs at any scope, if present"""
        if input is None:
            input = {}
        return self.chain.new_child(input)[key]

    def get_node_params(self, params: dict) -> dict:
        """Get tube-scoped params specified as inputs needed when instantiating a node"""
        raise NotImplementedError()

    def collect(self, edges: list[Edge], input: dict) -> dict:
        raise NotImplementedError()

    def add_input(self, scope: InputScope, input: dict) -> None:
        """Add some scope's input to the input collection."""
        if scope == InputScope.process:
            raise ValueError("Can't store process-scoped input, since it is ephemeral")

        if isinstance(scope, str) and scope in InputScope.__members__:
            scope = getattr(InputScope, scope)

        if not isinstance(scope, InputScope):
            raise ValueError(f"Unknown scope: {scope}")

        new = {**self._input[scope], **input}
        self.validate_presence(scope, new)
        self._input[scope] = new
        self._chain = None

    def validate_presence(self, scope: InputScope, *args: dict) -> None:
        """Check that the required inputs are present in one of several input dicts"""
        chain = self.chain.new_child()
        for input in args:
            chain = chain.new_child(input)

        for spec in self.specs[scope].values():
            if spec.id not in chain:
                raise InputMissingError(
                    f"Requested input {spec.id} not present in inputs at scope {scope.value}"
                )
