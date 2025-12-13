import os
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

_default_userdir = Path().home() / ".config" / "noob"
_dirs = PlatformDirs("noob", "noob")
LOG_LEVELS = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


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
    dir: Path = Path(_dirs.user_log_dir)
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
    def create_dir(cls, value: Path) -> Path:
        if os.environ.get("READTHEDOCS", False):
            return value
        value.mkdir(parents=True, exist_ok=True)
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

    @field_validator("user_dir", mode="after")
    def set_rtd_dir(cls, value: Path) -> Path:
        """RM this after PR to add configs is merged"""
        if os.environ.get("READTHEDOCS", False):
            return Path("docs/assets/pipelines")
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


config = Config()
