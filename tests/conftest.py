import argparse
import platform
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

from .fixtures import *
from .fixtures.paths import CONFIG_DIR, PIPELINE_DIR


@pytest.fixture(scope="session", autouse=True)
def patch_config_source(monkeypatch_session: MonkeyPatch) -> None:
    from noob.yaml import ConfigYAMLMixin

    current_sources = ConfigYAMLMixin.config_sources()

    def _config_sources(cls: type[ConfigYAMLMixin]) -> list[Path]:
        nonlocal current_sources
        return [CONFIG_DIR, PIPELINE_DIR, *current_sources]

    monkeypatch_session.setattr(ConfigYAMLMixin, "config_sources", classmethod(_config_sources))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # While zmq runner uses IPC, can't run on windows
    if platform.system() == "Windows":
        skip_zmq = pytest.mark.skip(reason="IPC not supported on windows")
        for item in items:
            if item.get_closest_marker("zmq_runner"):
                item.add_marker(skip_zmq)
