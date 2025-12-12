from importlib.metadata import version
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict
from ruamel.yaml import YAML

from noob.yaml import ConfigYAMLMixin, YAMLMixin, yaml_peek

yaml = YAML()


class NestedModel(BaseModel):
    d: int = 4
    e: str = "5"
    f: float = 5.5


class MyModel(ConfigYAMLMixin):
    noob_id: str = "my-config"
    a: int = 0
    b: str = "1"
    c: float = 2.2
    child: NestedModel = NestedModel()


class LoaderModel(ConfigYAMLMixin):
    """Model that just allows everything, only used to test write on load"""

    model_config = ConfigDict(extra="allow")


def test_yaml_mixin(tmp_path):
    """
    YAMLMixIn should give our models a from_yaml method to read from files
    """

    class MyModel(BaseModel, YAMLMixin):
        a_str: str
        a_int: int
        a_list: list[int]
        a_dict: dict[str, float]

    data = {"a_str": "string!", "a_int": 5, "a_list": [1, 2, 3], "a_dict": {"a": 1.1, "b": 2.5}}

    yaml_file = tmp_path / "temp.yaml"
    with open(yaml_file, "w") as yfile:
        yaml.dump(data, yfile)

    instance = MyModel.from_yaml(yaml_file)
    assert instance.model_dump() == data


@pytest.mark.parametrize(
    "id,path,valid",
    [
        ("default-path", None, True),
        ("nested-path", Path("configs/nested/path/config.yaml"), True),
        ("not-valid", Path("not_in_dir/config.yaml"), False),
    ],
)
def test_config_from_id(yaml_config, id, path, valid):
    """Configs can be looked up with the id field if they're within a config directory"""
    instance = MyModel(noob_id=id)
    yaml_config(id, instance.model_dump(), path)
    if valid:
        loaded = MyModel.from_id(id)
        assert loaded == instance
        assert loaded.child == instance.child
        assert isinstance(loaded.child, NestedModel)
    else:
        with pytest.raises(KeyError):
            MyModel.from_id(id)


def test_roundtrip_to_from_yaml(tmp_config_source):
    """Config models can roundtrip to and from yaml"""
    yaml_file = tmp_config_source / "test_config.yaml"

    instance = MyModel()
    instance.to_yaml(yaml_file)
    loaded = MyModel.from_yaml(yaml_file)
    assert loaded == instance
    assert loaded.child == instance.child
    assert isinstance(loaded.child, NestedModel)


@pytest.mark.parametrize(
    "src",
    [
        pytest.param(
            """
a: 9
b: "10\"""",
            id="missing",
        ),
        pytest.param(
            f"""
a: 9
noob_id: "my-config"
noob_model: "tests.test_yaml.MyModel"
noob_version: "{version("noob")}"
b: "10\"""",
            id="not-at-start",
        ),
        pytest.param(
            f"""
noob_version: "{version("noob")}"
noob_model: "tests.test_yaml.MyModel"
noob_id: "my-config"
a: 9
b: "10\"""",
            id="out-of-order",
        ),
    ],
)
def test_complete_header(tmp_config_source, src: str):
    """
    Config models saved without header information will have it filled in
    the source yaml they were loaded from
    """
    yaml_file = tmp_config_source / "test_config.yaml"

    with open(yaml_file, "w") as yfile:
        yfile.write(src)

    _ = MyModel.from_yaml(yaml_file)

    with open(yaml_file) as yfile:
        loaded = yaml.load(yfile)

    loaded_str = yaml_file.read_text()

    assert loaded["noob_version"] == version("noob")
    assert loaded["noob_id"] == "my-config"
    assert loaded["noob_model"] == MyModel._model_name()

    # the header should come at the top!
    lines = loaded_str.splitlines()
    for i, key in enumerate(("noob_id", "noob_model", "noob_version")):
        line_key = lines[i].split(":")[0].strip()
        assert line_key == key


@pytest.mark.parametrize(
    "key,expected,root,first",
    [
        ("key1", "val1", True, True),
        ("key1", "val1", False, True),
        ("key1", ["val1"], True, False),
        ("key1", ["val1", "val2"], False, False),
        ("key2", "val2", True, True),
        ("key3", False, True, True),
        ("key4", False, True, True),
        ("key4", "val4", False, True),
    ],
)
def test_peek_yaml(key, expected, root, first, yaml_config):
    yaml_file = yaml_config(
        "test", {"key1": "val1", "key2": "val2", "key3": {"key1": "val2", "key4": "val4"}}, None
    )

    if not expected:
        with pytest.raises(KeyError):
            _ = yaml_peek(key, yaml_file, root=root, first=first)
    else:
        assert yaml_peek(key, yaml_file, root=root, first=first) == expected


def test_yamlmixin_core_schema():
    """
    The __get_pydantic_core_schema__ method in the ConfigYamlMixin
    lets us use ids for keys everywhere
    """

    class B(ConfigYAMLMixin):
        value: list[int]

    class A(ConfigYAMLMixin):
        value: list[str]
        config: B

    class Container(BaseModel):
        model: A

    instance = Container(model="reference-a")
    assert isinstance(instance.model.config, B)
    assert instance.model.value == ["hey", "sup"]
    assert instance.model.config.value == [1, 2, 3]
