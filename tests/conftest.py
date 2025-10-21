import datetime
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

from noob import testing as _testing

from .fixtures import *

DATA_DIR = Path(__file__).parent / "data"
CONFIG_DIR = DATA_DIR / "config"
PIPELINE_DIR = DATA_DIR / "pipelines"
MOCK_DIR = Path(__file__).parent / "mock"


@pytest.fixture(scope="session", autouse=True)
def patch_config_source(monkeypatch_session: MonkeyPatch) -> None:
    from noob.yaml import ConfigYAMLMixin

    current_sources = ConfigYAMLMixin.config_sources()

    def _config_sources(cls: type[ConfigYAMLMixin]) -> list[Path]:
        nonlocal current_sources
        return [CONFIG_DIR, PIPELINE_DIR, *current_sources]

    monkeypatch_session.setattr(ConfigYAMLMixin, "config_sources", classmethod(_config_sources))


MOCK_TIME = datetime.datetime(2000, 12, 25, 12, 0, 0)


@pytest.fixture(scope="session", autouse=True)
def patch_datetime_now(monkeypatch_session: MonkeyPatch) -> None:
    from datetime import tzinfo

    class _datetime(datetime.datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime.datetime:
            return MOCK_TIME

    monkeypatch_session.setattr(datetime, "datetime", _datetime)
