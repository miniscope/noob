from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

from .fixtures import *

DATA_DIR = Path(__file__).parent / "data"
CONFIG_DIR = DATA_DIR / "config"
MOCK_DIR = Path(__file__).parent / "mock"


@pytest.fixture(scope="session", autouse=True)
def patch_config_source(monkeypatch_session: MonkeyPatch) -> None:
    from noob.yaml import ConfigYAMLMixin

    current_sources = ConfigYAMLMixin.config_sources()

    def _config_sources(cls: type[ConfigYAMLMixin]) -> list[Path]:
        nonlocal current_sources
        return [CONFIG_DIR, *current_sources]

    monkeypatch_session.setattr(ConfigYAMLMixin, "config_sources", classmethod(_config_sources))
