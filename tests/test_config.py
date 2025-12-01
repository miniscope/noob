import os
import random
from pathlib import Path

import yaml

from noob.config import Config


def test_config(tmp_path):
    """
    Config should be able to make directories and set sensible defaults
    """
    config = Config(user_dir=tmp_path, logs={"dir": tmp_path / "log"})
    assert config.user_dir.exists()
    assert config.logs.dir.exists()


def test_set_config(set_config, tmp_path):
    """We should be able to set parameters from all available modalities"""
    file_n = int(random.randint(0, 100))
    user_dir = tmp_path / f"fake/dir/{random.randint(0, 100)}"

    set_config({"user_dir": str(user_dir), "logs": {"file_n": file_n}})

    config = Config()
    assert config.user_dir == user_dir
    assert config.logs.file_n == file_n


def test_config_from_environment(tmp_path, set_env):
    """
    Setting environmental variables should set the config, including recursive models
    """
    override_logdir = Path(tmp_path) / "fancylogdir"

    set_env({"user_dir": str(tmp_path), "logs": {"dir": str(override_logdir), "level": "error"}})

    config = Config()
    assert config.user_dir == Path(tmp_path)
    assert config.logs.dir == override_logdir
    assert config.logs.level == "error".upper()


def test_config_from_dotenv(tmp_path):
    """
    dotenv files should also set config

    this test can be more relaxed since its basically a repetition of previous
    """
    tmp_path.mkdir(exist_ok=True, parents=True)
    dotenv = tmp_path / ".env"
    with open(dotenv, "w") as denvfile:
        denvfile.write(f"NOOB_USER_DIR={str(tmp_path)}")

    config = Config(_env_file=dotenv, _env_file_encoding="utf-8")
    assert config.user_dir == Path(tmp_path)


def test_config_sources_overrides(set_env, set_dotenv, set_pyproject, set_local_yaml):
    """Test that the different config sources are overridden in the correct order"""
    set_pyproject({"logs": {"file_n": 2}})
    assert Config().logs.file_n == 2
    set_local_yaml({"logs": {"file_n": 3}})
    assert Config().logs.file_n == 3
    set_dotenv({"logs": {"file_n": 4}})
    assert Config().logs.file_n == 4
    set_env({"logs": {"file_n": 5}})
    assert Config().logs.file_n == 5
    assert Config(**{"logs": {"file_n": 6}}).logs.file_n == 6
