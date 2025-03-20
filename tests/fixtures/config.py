from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

import pytest
import tomli_w
import yaml
from _pytest.monkeypatch import MonkeyPatch

from noob.yaml import ConfigYAMLMixin, YamlDumper


@pytest.fixture()
def tmp_config_source(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    """
    Monkeypatch the config sources to include a temporary path
    """

    path = tmp_path / "configs"
    path.mkdir(exist_ok=True)
    current_sources = ConfigYAMLMixin.config_sources()

    def _config_sources(cls: type[ConfigYAMLMixin]) -> list[Path]:
        return [path, *current_sources]

    monkeypatch.setattr(ConfigYAMLMixin, "config_sources", classmethod(_config_sources))
    return path


@pytest.fixture()
def yaml_config(
    tmp_config_source: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> Callable[[str, dict, Path | None], Path]:
    out_file = tmp_config_source / "test_config.yaml"

    def _yaml_config(id: str, data: dict, path: Path | None = None) -> Path:
        if path is None:
            path = out_file
        else:
            path = Path(path)
            if not path.is_absolute():
                # put under tmp_path (rather than tmp_config_source)
                # in case putting a file outside the config dir is intentional.
                path = tmp_path / path

            if path.is_dir():
                path.mkdir(exist_ok=True, parents=True)
                path = path / "test_config.yaml"
            else:
                path.parent.mkdir(exist_ok=True, parents=True)

        data = {"id": id, **data}
        with open(path, "w") as yfile:
            yaml.dump(data, yfile)
        return path

    return _yaml_config


@pytest.fixture()
def tmp_cwd(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def set_env(monkeypatch: MonkeyPatch) -> Callable[[dict[str, Any]], None]:
    """
    Function fixture to set environment variables using a nested dict
    matching a GlobalConfig.model_dump()
    """

    def _set_env(config: dict[str, Any]) -> None:
        for key, value in _flatten(config).items():
            key = "NOOB_" + key.upper()
            monkeypatch.setenv(key, str(value))

    return _set_env


@pytest.fixture()
def set_dotenv(tmp_cwd: Path) -> Callable[[dict[str, Any]], Path]:
    """
    Function fixture to set config variables in a .env file
    """
    dotenv_path = tmp_cwd / ".env"

    def _set_dotenv(config: dict[str, Any]) -> Path:
        with open(dotenv_path, "w") as dfile:
            for key, value in _flatten(config).items():
                key = "NOOB_" + key.upper()
                dfile.write(f"{key}={value}\n")
        return dotenv_path

    return _set_dotenv


@pytest.fixture()
def set_pyproject(tmp_cwd: Path) -> Callable[[dict[str, Any]], Path]:
    """
    Function fixture to set config variables in a pyproject.toml file
    """
    toml_path = tmp_cwd / "pyproject.toml"

    def _set_pyproject(config: dict[str, Any]) -> Path:
        config = {"tool": {"noob": {"config": config}}}

        with open(toml_path, "wb") as tfile:
            tomli_w.dump(config, tfile)

        return toml_path

    return _set_pyproject


@pytest.fixture()
def set_local_yaml(tmp_cwd: Path) -> Callable[[dict[str, Any]], Path]:
    """
    Function fixture to set config variables in a noob_config.yaml file in the current directory
    """
    yaml_path = tmp_cwd / "noob_config.yaml"

    def _set_local_yaml(config: dict[str, Any]) -> Path:
        with open(yaml_path, "w") as yfile:
            yaml.dump(config, yfile, Dumper=YamlDumper)
        return yaml_path

    return _set_local_yaml


@pytest.fixture(
    params=[
        "set_env",
        "set_dotenv",
        "set_pyproject",
        "set_local_yaml",
    ]
)
def set_config(request: pytest.FixtureRequest) -> Callable[[dict[str, Any]], Path]:
    return request.getfixturevalue(request.param)


def _flatten(d: MutableMapping, parent_key: str = "", separator: str = "__") -> dict:
    """https://stackoverflow.com/a/6027615/13113166"""
    items = []
    for key, value in d.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(_flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)
