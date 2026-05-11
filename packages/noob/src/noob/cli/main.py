try:
    import click
except ImportError as e:
    raise ImportError(
        "CLI Dependencies not installed! Install noob with the optional [cli] dependency group"
    ) from e


from noob.cli.run import run
from noob.cli.view import view


@click.group("noob")
def main() -> None:
    """
    The Noob CLI!
    """


main.add_command(run)
main.add_command(view)
