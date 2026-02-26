import os
import warnings
from importlib.metadata import entry_points
from pathlib import Path
from typing import Literal

from platformdirs import PlatformDirs
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from noob.exceptions import EntrypointImportWarning

_default_userdir = Path().home() / ".config" / "noob"
_dirs = PlatformDirs("noob", "noob")
LOG_LEVELS = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
_extra_sources = []
"""Extra sources for tube configs added by `add_sources`"""
_entrypoint_sources: list[Path] | None = None
"""Sources added by entrypoint functions. Initially `None`, populated on first load of a config"""


class LogConfig(BaseModel):
    """
    Configuration for logging
    """

    model_config = SettingsConfigDict(validate_default=True)

    level: LOG_LEVELS = "INFO"
    """
    Severity of log messages to process.
    """
    level_file: LOG_LEVELS | None = None
    """
    Severity for file-based logging. If unset, use ``level``
    """
    level_stdout: LOG_LEVELS | None = None
    """
    Severity for stream-based logging. If unset, use ``level``
    """
    dir: Path | Literal[False] = Path(_dirs.user_log_dir)
    """
    Directory where logs are stored.
    """
    file_n: int = 5
    """
    Number of log files to rotate through
    """
    file_size: int = 2**22  # roughly 4MB
    """
    Maximum size of log files (bytes)
    """
    width: int | None = None
    """
    Explicitly set width of rich stdout logs, leave as None for auto detection.
    """

    @field_validator("level", "level_file", "level_stdout", mode="before")
    @classmethod
    def uppercase_levels(cls, value: str | None = None) -> str | None:
        """
        Ensure log level strings are uppercased
        """
        if value is not None:
            value = value.upper()
        return value

    @field_validator("dir", mode="after")
    def create_dir(cls, value: Path | Literal[False]) -> Path | Literal[False]:
        if os.environ.get("READTHEDOCS", False) or value is False:
            return value
        value.mkdir(parents=True, exist_ok=True)
        return value

    @field_validator("dir", mode="after")
    def no_file_on_rtd(cls, value: Path | Literal[False]) -> Path | Literal[False]:
        """On readthedocs, don't log to file"""
        if os.environ.get("READTHEDOCS", False):
            return False
        return value


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="noob_",
        env_nested_delimiter="__",
        extra="ignore",
        nested_model_default_partial_update=True,
        yaml_file="noob_config.yaml",
        pyproject_toml_table_header=("tool", "noob", "config"),
        validate_default=True,
    )

    logs: LogConfig = LogConfig()
    user_dir: Path = Field(default=Path(_dirs.user_data_dir))
    tmp_dir: Path = Field(default=Path(_dirs.user_runtime_dir))
    config_dir: Path = Field(
        default=Path(_dirs.user_data_dir) / "config",
        description="Directory where config yaml files are stored",
    )

    @field_validator("user_dir", "config_dir", "tmp_dir", mode="after")
    def create_dir(cls, value: Path) -> Path:
        if os.environ.get("READTHEDOCS", False):
            return value
        value.mkdir(parents=True, exist_ok=True)
        return value

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Read config settings from, in order of priority from high to low, where
        high priorities override lower priorities:

        * in the arguments passed to the class constructor (not user configurable)
        * in environment variables like ``export NOOB_LOGS__DIR=~/``
        * in a ``.env`` file in the working directory
        * in a ``noob_config.yaml`` file in the working directory
        * in the ``tool.noob.config`` table in a ``pyproject.toml`` file
          in the working directory
        * the default values in the :class:`.Config` model

        """

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            PyprojectTomlConfigSettingsSource(settings_cls),
        )


def add_config_source(path: Path) -> None:
    """
    Add a directory as a source of tube configs when searching by tube id
    """
    global _extra_sources
    path = Path(path)
    _extra_sources.append(path)


def get_extra_sources() -> list[Path]:
    """
    Get the extra sources added by :func:`.add_config_source`

    (avoid importing the private module-level collection anywhere else,
    as it makes mutation weird and unpredictable)
    """
    global _extra_sources
    return _extra_sources


def get_entrypoint_sources() -> list[Path]:
    """
    Get additional config sources added by entrypoint functions.

    Packages that ship noob tubes can make those tubes available by adding an
    entrypoint function with a signature ``() -> list[Path]`` to their pyproject.toml
    like:

        [project.entry-points."noob.add_sources"]
        tubes = "my_package.something:add_sources"

    References:
        https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    """
    global _entrypoint_sources
    if _entrypoint_sources is None:
        _entrypoint_sources = []
        for ext in entry_points(group="noob.add_sources"):
            try:
                add_sources_fn = ext.load()
            except (ImportError, AttributeError):
                warnings.warn(
                    f"Config source entrypoint {ext.name}, {ext.value} "
                    f"could not be imported, or the function could not be found. Ignoring",
                    EntrypointImportWarning,
                    stacklevel=1,
                )
                continue
            try:
                _entrypoint_sources.extend([Path(p) for p in add_sources_fn()])
            except Exception as e:
                # bare exception is fine here - we're calling external code and can't know.
                warnings.warn(
                    f"Config source entrypoint {ext.name}, {ext.value} "
                    f"threw an error, or returned an invalid list of paths, ignoring.\n{str(e)}",
                    EntrypointImportWarning,
                    stacklevel=1,
                )
    return _entrypoint_sources


config = Config()
