from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from typing import (
    Callable,
    Any,
    Dict,
    Literal,
    List,
    get_type_hints,
)

import xarray as xr
from river import compose

from cala.streaming.composer.pipe_config import StreamingConfig
from cala.streaming.core import Parameters
from cala.streaming.core.distribution import Distributor
from cala.streaming.util.buffer import Buffer


@dataclass
class Runner:
    """Manages the execution of streaming calcium imaging analysis pipeline.

    This class orchestrates the preprocessing, initialization, and iterate steps
    of the calcium imaging analysis pipeline according to a provided configuration.
    """

    config: StreamingConfig
    """Configuration defining the pipeline structure and parameters."""
    _buffer: Buffer = field(init=False)
    """Internal frame buffer for multi-frame operations."""
    _state: Distributor = field(default_factory=lambda: Distributor())
    """Current state of the pipeline containing computed results."""
    execution_order: List[str] = None
    """Ordered list of initialization steps."""
    status: List[bool] = None
    """Completion status for each initialization step."""
    is_initialized: bool = False
    """Whether the pipeline initialization is complete."""

    def __post_init__(self):
        """Initialize the frame buffer after instance creation."""
        self._buffer = Buffer(
            buffer_size=10,
        )

    def preprocess(self, frame: xr.DataArray) -> xr.DataArray:
        """Execute preprocessing steps on a single frame.

        Args:
            frame: Input frame to preprocess.

        Returns:
            Dictionary containing preprocessed results.
        """
        execution_order = self._create_dependency_graph(self.config["preprocess"])

        pipeline = compose.Pipeline()

        for step in execution_order:
            transformer = self._build_transformer(process="preprocess", step=step)

            pipeline = pipeline | transformer

        pipeline.learn_one(x=frame)
        result = pipeline.transform_one(x=frame)

        return result

    def initialize(self, frame: xr.DataArray):
        """Initialize pipeline transformers in dependency order.

        Executes initialization steps that may require multiple frames. Steps are executed
        in topological order based on their dependencies.

        Args:
            frame: New frame to use for initialization.
        """
        self._buffer.add_frame(frame)

        if not self.execution_order or not self.status:
            self.execution_order = self._create_dependency_graph(
                self.config["initialization"]
            )
            self.status = [False] * len(self.execution_order)

        for idx, step in enumerate(self.execution_order):
            if self.status[idx]:
                continue

            n_frames = self.config["initialization"][step].get("n_frames", 1)
            if not self._buffer.is_ready(n_frames):
                break

            transformer = self._build_transformer(process="initialization", step=step)
            result = self._learn_transform(
                transformer=transformer, frame=self._buffer.get_latest(n_frames)
            )
            if result is not None:
                self.status[idx] = True

            result_type = get_type_hints(
                transformer.transform_one, include_extras=True
            )["return"]
            self._state.init(result, result_type)

        if all(self.status):
            self.is_initialized = True

    def iterate(self, frame: xr.DataArray):
        """Execute iterate steps on a single frame.

        Args:
            frame: Input frame to process for component iterate.
        """
        execution_order = self._create_dependency_graph(self.config["iteration"])

        # Execute transformers in order
        for step in execution_order:
            transformer = self._build_transformer(process="iteration", step=step)
            result = self._learn_transform(transformer=transformer, frame=frame)

            result_type = get_type_hints(
                transformer.transform_one, include_extras=True
            )["return"]

            self._state.update(result, result_type)

    def _build_transformer(
        self, process: Literal["preprocess", "initialization", "iteration"], step: str
    ):
        """Construct a transformer instance with configured parameters.

        Args:
            process: Type of process the transformer belongs to.
            step: Name of the configuration step.

        Returns:
            Configured transformer instance.
        """
        config = self.config[process][step]
        params = config.get("params", {})
        transformer = config["transformer"]

        param_class = next(
            (
                type_
                for type_ in transformer.__annotations__.values()
                if issubclass(type_, Parameters)
            ),
            None,
        )
        if param_class:
            param_obj = param_class(**params)
            transformer = transformer(param_obj)
        else:
            transformer = transformer()

        return transformer

    def _learn_transform(self, transformer, frame: xr.DataArray):
        """Execute learn and transform steps for a transformer.

        Args:
            transformer: Transformer instance to execute.
            frame: Input frame to process.

        Returns:
            Transformation results.
        """
        learn_injects = self._get_injects(self._state, transformer.learn_one)
        transform_injects = self._get_injects(self._state, transformer.transform_one)

        # Initialize and run transformer
        transformer.learn_one(frame=frame, **learn_injects)
        result = transformer.transform_one(**transform_injects)

        return result

    @staticmethod
    def _get_injects(state: Distributor, function: Callable) -> Dict[str, Any]:
        """Extract required dependencies from the current state based on function signature.

        Args:
            state: Current pipeline state containing all computed results.
            function: Function to get signature from.

        Returns:
            Dictionary mapping parameter names to matching state values.
        """
        matches = {}
        for param_name, param_type in get_type_hints(
            function, include_extras=True
        ).items():
            if param_name == "return":
                continue

            value = state.get(param_type)
            if value is not None:
                matches[param_name] = value

        return matches

    @staticmethod
    def _create_dependency_graph(steps: dict) -> list:
        """Create and validate a dependency graph for execution ordering.

        Args:
            steps: Dictionary of pipeline steps and their configurations.

        Returns:
            List of steps in topological order.

        Raises:
            ValueError: If dependencies contain cycles.
        """
        graph = {}
        for step in steps:
            graph[step] = set()

        for step, config in steps.items():
            if "requires" in config:
                graph[step] = set(config["requires"])

        # Create and prepare the sorter
        ts = TopologicalSorter(graph)
        return list(ts.static_order())
