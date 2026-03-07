"""Move and copy files"""

import shutil
from pathlib import Path
from typing import Annotated as A

from noob import Name


def copy(
    source: Path, destination: Path
) -> tuple[A[Path, Name("source")], A[Path, Name("destination")]]:
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, destination)
    return source, destination


def move(
    source: Path, destination: Path
) -> tuple[A[Path, Name("source")], A[Path, Name("destination")]]:
    source, destination = copy(source, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(source, destination)
    return source, destination


def with_suffix(path: Path, suffix: str) -> A[Path, Name("path")]:
    return Path(path).with_suffix(suffix)


def relative_path(path: Path, relative: Path) -> A[Path, Name("path")]:
    path, relative = Path(path), Path(relative)
    new = path.parent / relative / path.name if path.is_file() else path / relative

    return new.resolve()
