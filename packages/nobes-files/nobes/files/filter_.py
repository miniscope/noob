import re
from pathlib import Path
from typing import Annotated as A
from typing import Literal as L

from noob import Name
from noob.event import MetaSignal


def filter_path(
    path: Path, extension: str | list[str] | None = None, pattern: str | None = None
) -> A[Path | L[MetaSignal.NoEvent], Name("path")]:
    path = Path(path)
    if extension is not None:
        if isinstance(extension, str):
            extension = [extension]
        if not any(path.name.endswith(ext) for ext in extension):
            return MetaSignal.NoEvent
    if pattern is not None and not re.match(pattern, str(path)):
        return MetaSignal.NoEvent
    return path
