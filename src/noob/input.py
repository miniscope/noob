import re
import warnings
from collections import ChainMap, defaultdict
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, Field, PrivateAttr

from noob.exceptions import ExtraInputWarning, InputMissingError
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
        for k, v in params.items():
            if not isinstance(v, str):
                continue
            if match := InputCollection.INPUT_PATTERN.fullmatch(v):
                input_key = match.groupdict()["key"]
                try:
                    params[k] = self.get(input_key)
                except KeyError as e:
                    raise InputMissingError(
                        f"Node params requested {input_key}, but not present in input"
                    ) from e
        return params

    def collect(self, edges: list[Edge], input: dict) -> dict:
        args = {}
        for edge in edges:
            if edge.source_node != "input":
                continue
            try:
                args[edge.target_slot] = self.get(edge.source_signal, input)
            except KeyError as e:
                raise InputMissingError(
                    f"Node depends on input {edge.source_signal}, "
                    "but not provided in any input scope"
                ) from e
        return args

    def add_input(self, scope: InputScope, input: dict) -> None:
        """Add some scope's input to the input collection."""
        if scope == InputScope.process:
            raise ValueError("Can't store process-scoped input, since it is ephemeral")

        if isinstance(scope, str) and scope in InputScope.__members__:
            scope = getattr(InputScope, scope)

        if not isinstance(scope, InputScope):
            raise ValueError(f"Unknown scope: {scope}")

        new = {**self._input[scope], **input}
        new = self.validate_input(scope, new)

        self._input[scope] = new
        self._chain = None

    def validate_input(self, scope: InputScope, input: dict) -> dict:
        """
        Check that the required inputs are present in one of several input dicts,
        and then filter to only specified input
        """
        if scope not in self.specs:
            # no input specs for this scope
            if input:
                warnings.warn(
                    f"Ignoring extra input for a scope `{scope.value}` "
                    "without any input specifications.",
                    ExtraInputWarning,
                    stacklevel=3,
                )
            return {}

        input = self.filter_input(scope, input)

        chain = self.chain.new_child(input)

        for spec in self.specs[scope].values():
            if spec.id not in chain:
                raise InputMissingError(
                    f"Requested input {spec.id} not present in inputs at scope {scope.value}"
                )
        return input

    def filter_input(self, scope: InputScope, input: dict) -> dict:
        """filter input to only specified keys, emitting an ExtraInput warning if found."""
        if scope not in self.specs:
            warnings.warn(
                f"Ignoring extra input for a scope `{scope.value}` "
                "without any input specifications.",
                ExtraInputWarning,
                stacklevel=3,
            )
            return {}

        filtered = {k: v for k, v in input.items() if k in self.specs[scope]}
        if len(input) > len(filtered):
            extra = set(input.keys()) - set(filtered.keys())
            warnings.warn(
                f"Ignoring extra input without a specification provided to scope "
                f"`{scope.value}`: {extra}",
                ExtraInputWarning,
                stacklevel=3,
            )
        return filtered
