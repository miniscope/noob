"""
Logging factory and handlers
"""

import logging
import multiprocessing as mp
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.logging import RichHandler

from noob.config import LOG_LEVELS, config


def init_logger(
    name: str,
    log_dir: Path | None | Literal[False] = None,
    level: LOG_LEVELS | None = None,
    file_level: LOG_LEVELS | None = None,
    log_file_n: int | None = None,
    log_file_size: int | None = None,
    width: int | None = None,
) -> logging.Logger:
    """
    Make a logger.

    Log to a set of rotating files in the ``log_dir`` according to ``name`` ,
    as well as using the :class:`~rich.RichHandler` for pretty-formatted stdout logs.

    If this method is called from a process that isn't the root process,
    it will create new rich and file handlers in the root noob logger to avoid
    deadlocks from threading locks that are copied on forked processes.
    Since the handlers will be different across processes,
    to avoid file access conflicts, logging files will have the process's ``pid``
    appended (e.g. ``noob_12345.log`` )

    Args:
        name (str): Name of this logger. Ideally names are hierarchical
            and indicate what they are logging for, eg. ``noob.api.auth``
            and don't contain metadata like timestamps, etc. (which are in the logs)
        log_dir (:class:`pathlib.Path`): Directory to store file-based logs in. If ``None``,
            get from :class:`.Config`. If ``False`` , disable file logging.
        level (:class:`.LOG_LEVELS`): Level to use for stdout logging. If ``None`` ,
            get from :class:`.Config`
        file_level (:class:`.LOG_LEVELS`): Level to use for file-based logging.
             If ``None`` , get from :class:`.Config`
        log_file_n (int): Number of rotating file logs to use.
            If ``None`` , get from :class:`.Config`
        log_file_size (int): Maximum size of logfiles before rotation.
            If ``None`` , get from :class:`.Config`
        width (int, None): Explicitly set width of rich stdout console.
            If ``None`` , get from :class:`.Config`

    Returns:
        :class:`logging.Logger`
    """
    if log_dir is None:
        log_dir = config.logs.dir
    if level is None:
        level = (
            config.logs.level_stdout if config.logs.level_stdout is not None else config.logs.level
        )
    if file_level is None:
        file_level = (
            config.logs.level_file if config.logs.level_file is not None else config.logs.level
        )
    if log_file_n is None:
        log_file_n = config.logs.file_n
    if log_file_size is None:
        log_file_size = config.logs.file_size
    if width is None:
        width = config.logs.width

    # set our logger to the minimum of the levels so that it always handles at least that severity
    # even if one or the other handlers might not.
    min_level = min([getattr(logging, level), getattr(logging, file_level)])

    if not name.startswith("noob"):
        name = "noob." + name

    _init_root(
        stdout_level=level,
        file_level=file_level,
        log_dir=log_dir,
        log_file_n=log_file_n,
        log_file_size=log_file_size,
        width=width,
    )

    logger = logging.getLogger(name)
    logger.setLevel(min_level)

    return logger


def _init_root(
    stdout_level: LOG_LEVELS,
    file_level: LOG_LEVELS,
    log_dir: Path | Literal[False],
    log_file_n: int = 5,
    log_file_size: int = 2**22,
    width: int | None = None,
) -> None:
    root_logger = logging.getLogger("noob")

    # ensure each root logger has fresh handlers in subprocesses
    if mp.parent_process() is not None:
        current_pid = mp.current_process().pid
        file_name = f"noob_{current_pid}"
        rich_name = f"{file_name}_rich"
    else:
        file_name = "noob"
        rich_name = "noob_rich"

    root_logger.handlers = [h for h in root_logger.handlers if h.name in (rich_name, file_name)]

    file_handlers = [
        handler for handler in root_logger.handlers if isinstance(handler, RotatingFileHandler)
    ]
    stream_handlers = [
        handler for handler in root_logger.handlers if isinstance(handler, RichHandler)
    ]

    if log_dir is not False and not file_handlers:
        root_logger.addHandler(
            _file_handler(
                file_name,
                file_level,
                log_dir,
                log_file_n,
                log_file_size,
            )
        )
    else:
        for file_handler in file_handlers:
            file_handler.setLevel(file_level)

    if not stream_handlers:
        root_logger.addHandler(_rich_handler(stdout_level, name=rich_name, width=width))
    else:
        for stream_handler in stream_handlers:
            stream_handler.setLevel(stdout_level)

    # prevent propagation to the default root
    root_logger.propagate = False


def _file_handler(
    name: str,
    file_level: LOG_LEVELS,
    log_dir: Path,
    log_file_n: int = 5,
    log_file_size: int = 2**22,
) -> RotatingFileHandler:
    # See init_logger for arg docs

    filename = Path(log_dir) / ".".join([name, "log"])
    file_handler = RotatingFileHandler(
        str(filename), mode="a", maxBytes=log_file_size, backupCount=log_file_n
    )
    file_formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    return file_handler


def _rich_handler(
    level: LOG_LEVELS, name: str, width: int | None = None, **kwargs: Any
) -> RichHandler:
    console = _get_console()
    if width:
        console.width = width

    rich_handler = RichHandler(console=console, rich_tracebacks=True, markup=True, **kwargs)
    rich_handler.name = name
    rich_formatter = logging.Formatter(
        r"[bold green]\[%(name)s][/bold green] %(message)s",
        datefmt="[%y-%m-%dT%H:%M:%S]",
    )
    rich_handler.setFormatter(rich_formatter)
    rich_handler.setLevel(level)
    return rich_handler


_console_by_pid: dict[int | None, Console] = {}


def _get_console() -> Console:
    """get a console that was spawned in this process"""
    global _console_by_pid
    current_pid = mp.current_process().pid
    console = _console_by_pid.get(current_pid)
    if console is None:
        _console_by_pid[current_pid] = console = Console()
    return console
