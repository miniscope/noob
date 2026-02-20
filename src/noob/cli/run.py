import json
import sys

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from noob import Tube
from noob.runner import AsyncRunner, SynchronousRunner, TubeRunner
from noob.runner.zmq import ZMQRunner

_runners: dict[str, type[TubeRunner]] = {
    "sync": SynchronousRunner,
    "async": AsyncRunner,
    "zmq": ZMQRunner,
}


@click.command("run")
@click.option("--tube", required=True)
@click.option("--runner", type=click.Choice(("sync", "async", "zmq")), default="sync")
@click.option("--n", "-n", type=click.INT)
@click.option("--format", type=click.Choice(("json",)))
def run(tube: str, runner: str = "sync", n: int | None = None, format: str | None = None) -> None:
    tube_id = tube
    tube = Tube.from_specification(tube)
    runner_cls = _runners[runner]
    runner = runner_cls(tube)
    if n is None:
        raise NotImplementedError("Just a demo!")
    else:

        console = Console(file=sys.stderr)
        progress = Progress(
            TextColumn(f"[bold green]{tube_id}[/bold green]"),
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            transient=True,
            console=console,
        )
    results = []
    with runner, progress:
        task = progress.add_task("Running", total=n)
        for result in runner.iter(n=n):
            results.append(result)
            progress.advance(task)

    if format == "json":
        click.echo(json.dumps(results))
