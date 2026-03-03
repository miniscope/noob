from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Annotated as A
from typing import Literal, TypeAlias

from watchfiles import Change
from watchfiles import watch as _watch

from noob import Name
from noob.event import MetaSignal

_ChangeStr: TypeAlias = Literal["added", "deleted", "modified"]


def watch(
    path: Path, change_types: Sequence[Change | _ChangeStr] | Change | _ChangeStr | None = None
) -> Generator[
    tuple[A[Path, Name("added")], A[Path, Name("deleted")], A[Path, Name("modified")]], None, None
]:
    """
    Watch a directory for changes, yielding a (Change, str) tuple for each change.

    Args:
        path (Path): Path to the directory to watch.
        changes (Sequence[Change | _ChangeStr] | Change | _ChangeStr | None): Optionally,
            filter to only the types of changes indicated ("added", "deleted", "modified").
    """
    if change_types and isinstance(change_types, str) or not isinstance(change_types, Sequence):
        change_types = {change_types}
    if change_types:
        change_types = {c if isinstance(c, Change) else Change[c] for c in change_types}

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Directory {path} does not exist!")

    for changes in _watch(path):
        for change in changes:
            change_type = change[0]
            changed_path = Path(change[1])
            if change_types and change_type not in change_types:
                continue

            if change_type == Change.added:
                yield changed_path, MetaSignal.NoEvent, MetaSignal.NoEvent
            elif change_type == Change.deleted:
                yield MetaSignal.NoEvent, changed_path, MetaSignal.NoEvent
            elif change_type == Change.modified:
                yield MetaSignal.NoEvent, MetaSignal.NoEvent, changed_path
            else:
                raise TypeError("Unknown change type")
