import sys
import time
from pathlib import Path
from threading import Event, Thread

import pytest
from watchfiles import Change

from nobes.files.watch import watch
from noob.event import MetaSignal


@pytest.mark.parametrize("change", [Change.added, Change.deleted, Change.modified])
def test_watch(tmp_path: Path, change: Change) -> None:
    """
    Watching a directory should yield changes, modifications, and removals as separate events.

    See the caveats in notify: https://docs.rs/notify/latest/notify

    on macOS, we can't observe modifications or deletions unless we own the files,
    or else we have to use polling which is hella slow.

    since we're not testing watchfile, but rather our wrapping of it,
    that's not our job.
    """
    stop_evt = Event()
    event_evt = Event()
    events = []

    def _iter_events() -> None:
        nonlocal events
        nonlocal event_evt
        iterator = watch(tmp_path, change_types=change, debounce=10)
        while not stop_evt.is_set():
            evt = next(iterator)
            events.append(evt)
            event_evt.set()

    iter_thread = Thread(target=_iter_events, daemon=True)
    iter_thread.start()
    try:
        time.sleep(0.01)
        # create a file!
        out_file = tmp_path / "out.txt"
        with open(out_file, "w") as f:
            f.write("hey")

        event_evt.wait(0.1)

        # we get two events here, one for the creation of the folder and the file
        if change == Change.added:
            assert len(events) == 2
            assert {events[0][0], events[1][0]} == {out_file.parent, out_file}

            assert all([e is MetaSignal.NoEvent for outer in events for e in outer[1:]])
        else:
            assert len(events) == 0

        # modify a file!
        with open(out_file, "a") as f:
            f.write("sup")

        event_evt.clear()
        event_evt.wait(0.1)

        if change == Change.modified:
            if sys.platform != "darwin":
                assert len(events) == 1
                assert events[0][2] == out_file
                assert events[0][0] is MetaSignal.NoEvent
                assert events[0][1] is MetaSignal.NoEvent
        else:
            assert len(events) == 0 if change == Change.deleted else 1

        # delete a file
        out_file.unlink()

        event_evt.clear()
        event_evt.wait(0.1)
        if change == Change.deleted:
            assert len(events) == 1
            assert events[0][1] == out_file
            assert events[0][0] is MetaSignal.NoEvent
            assert events[0][2] is MetaSignal.NoEvent
        else:
            if sys.platform != "darwin":
                assert len(events) == 2 if change == Change.modified else 1

    finally:
        stop_evt.set()
        time.sleep(0.1)
        stop_file = tmp_path / "stop.txt"
        with open(stop_file, "w") as f:
            f.write("hey")
        with open(stop_file, "a") as f:
            f.write("sup")
        stop_file.unlink()
