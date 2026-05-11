import click


@click.command()
@click.argument("tube_id", nargs=1)
def view(tube_id: str) -> None:
    """
    Launch a browser window to view a tube, live updating as it is edited.

    Requires noob to be installed with the optional [gui] dependency group!
    """
    from noob.gui.view import run_view

    run_view(tube_id)
