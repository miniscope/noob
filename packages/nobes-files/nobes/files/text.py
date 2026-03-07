from pathlib import Path


def write_text(text: str, path: Path) -> None:
    Path(path).write_text(text)
