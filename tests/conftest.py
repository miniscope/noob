import argparse
import platform
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

from .fixtures import *
from .fixtures.paths import CONFIG_DIR, PIPELINE_DIR, SPECIAL_DIR


@pytest.fixture(scope="session", autouse=True)
def patch_config_source(monkeypatch_session: MonkeyPatch) -> None:
    """Patch config sources so we don't accidentally use any user pipelines during testing."""
    from noob.config import add_config_source
    from noob.yaml import ConfigYAMLMixin

    current_sources = ConfigYAMLMixin.config_sources()
    original_method = ConfigYAMLMixin.config_sources

    for pth in (CONFIG_DIR, SPECIAL_DIR, PIPELINE_DIR):
        add_config_source(pth)

    def _config_sources(cls: type[ConfigYAMLMixin]) -> list[Path]:
        nonlocal current_sources, original_method
        return [p for p in original_method() if p not in current_sources]

    monkeypatch_session.setattr(ConfigYAMLMixin, "config_sources", classmethod(_config_sources))


@pytest.fixture(scope="session", autouse=True)
def patch_env_config(monkeypatch_session: MonkeyPatch) -> None:
    """Patch env settings, e.g. setting log levels and etc."""

    monkeypatch_session.setenv("NOOB_LOGS__LEVEL", "DEBUG")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # While zmq runner uses IPC, can't run on windows
    if platform.system() == "Windows":
        skip_zmq = pytest.mark.skip(reason="IPC not supported on windows")
        for item in items:
            if item.get_closest_marker("zmq_runner"):
                item.add_marker(skip_zmq)
