import json
import sys
from collections.abc import Generator
from typing import Literal as L

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from noob import Tube
from noob.runner import get_runner


@click.command("run")
@click.argument("tube")
@click.option("--runner", type=click.Choice(("sync", "async", "zmq")), default="sync")
@click.option("--n", "-n", type=click.INT)
@click.option(
    "--input-format",
    "-if",
    type=click.Choice(["json", "jsonl"]),
    default="json",
    help="""Format of process-scoped input data (piped from stdin). 
              json should be an array of inputs given to each process call,
              or a single json object if running a single epoch.
              jsonl should be one input object per line.
              """,
)
@click.option(
    "--output-format",
    "-of",
    type=click.Choice(("json", "jsonl")),
    default="json",
    help="""Output format (to stdout).
              json outputs results as a single array of results from all run epochs.
              jsonl emits each result separately as they are completed on newlines.
              """,
)
def run(
    tube: str,
    runner: L["sync", "async", "zmq"] = "sync",
    n: int | None = None,
    input_format: L["json", "jsonl"] = "json",
    output_format: L["json", "jsonl"] = "json",
) -> None:
    tube_id = tube
    tube_ = Tube.from_specification(tube)
    runner_cls = get_runner(runner)
    runner_ = runner_cls(tube_)

    assert tube_.spec is not None
    piped_input = not sys.stdin.isatty() and tube_.spec.input

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
    with runner_, progress:
        task = progress.add_task("Running", total=n)
        if piped_input:
            for input in _iter_stdin(input_format):
                result = runner_.process(**input)
                if output_format == "jsonl":
                    click.echo(json.dumps(result))
                else:
                    results.append(result)
                progress.advance(task)
        else:
            for result in runner_.iter(n=n):
                if output_format == "jsonl":
                    click.echo(json.dumps(result))
                else:
                    results.append(result)
                progress.advance(task)

    if output_format == "json":
        click.echo(json.dumps(results))


def _iter_stdin(format: L["json", "jsonl"] = "json") -> Generator[dict, None, None]:
    if format == "json":
        data = sys.stdin.read()
        data = json.loads(data)
        if isinstance(data, list):
            for line in data:
                yield line
        else:
            yield data
    elif format == "jsonl":
        for line in sys.stdin:
            yield json.loads(line)
    else:
        raise ValueError("Only json and jsonl input formats supported")
