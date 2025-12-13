from pathlib import Path

ENTRYPOINT_PATH = Path("/tmp/notreal/path")


def some_entrypoint_fn() -> list[Path]:
    """See test_config and config.get_entrypoint_sources"""
    return [ENTRYPOINT_PATH]
