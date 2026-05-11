"""Build the JS components and bundle them in the package"""

from pathlib import Path
from subprocess import run


def pdm_build_hook_enabled(context):
    return context.target == "sdist"


def pdm_build_initialize(context):
    """Build the js components"""
    result = run(
        ["npm", "run", "build:package"], cwd=Path(__file__).parents[2] / "js", capture_output=True
    )
    if result.returncode != 0:
        raise RuntimeError("JS component build failed! \n" + result.stderr.decode("utf-8"))
