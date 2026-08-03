"""Build the JS components and bundle them in the package"""

import shutil
from pathlib import Path
from subprocess import run


def pdm_build_hook_enabled(context):
    return context.target == "sdist"


def pdm_build_initialize(context):
    """
    - Copy the readme
    - Build the js components

    See: https://github.com/pdm-project/pdm/issues/3824
    """
    root = Path(__file__).parents[2]
    package = Path(__file__).parent

    shutil.copy(root / "README.md", package / "README.md")

    result = run(["npm", "run", "build:package"], cwd=root / "js", capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "JS component build failed! \nstderr:\n"
            + result.stderr.decode("utf-8")
            + "\nstdout:\n"
            + result.stdout.decode("utf-8")
        )
