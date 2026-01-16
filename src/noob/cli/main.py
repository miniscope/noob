import click


from noob.cli.run import run
@click.group("noob")
def main() -> None:
    """
    The Noob CLI!
    """


main.add_command(run)