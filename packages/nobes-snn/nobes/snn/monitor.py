import logging

from pydantic import PrivateAttr
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn
from rich.table import Column

from noob import Node, NodeSpecification
from noob.logging import init_logger
from noob.node import Slot
from noob.node.base import all_slots_optional


class Monitor(Node):
    """A monitor that can display numerical values on the CLI baby"""

    spec: NodeSpecification  # specs not optional for the monitor
    range: tuple[float, float] = (-70e-3, 10e-3)

    _console: Console | None = None
    _progress: Progress | None = None
    _tasks: dict[str, TaskID] = PrivateAttr(default_factory=dict)
    _logger: logging.Logger = init_logger("monitor")

    def process(self, **kwargs: float) -> None:
        for key, val in kwargs.items():
            if key not in self._tasks:
                continue
            self._progress.update(
                self._tasks[key], completed=val - self.range[0], mv=round(val * 1000, ndigits=2)
            )
        self._progress.refresh()

    def init(self) -> None:
        self._console = Console()
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, table_column=Column(ratio=4)),
            TextColumn("{task.fields[mv]} mV", table_column=Column(ratio=1)),
            auto_refresh=False,
            console=self._console,
        )
        for dep in self.spec.depends:
            key = list(dep.keys())[0]
            self._tasks[key] = self._progress.add_task(
                key, total=self.range[1] - self.range[0], mv=0
            )

        self._progress.start()

    def deinit(self) -> None:
        self._progress.stop()

    def _collect_slots(self) -> dict[str, Slot]:
        return all_slots_optional(self.spec)
