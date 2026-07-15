import json

from noob_core import Epoch
from pydantic import TypeAdapter


def test_epoch_from():
    """Epochs can be constructed from variant forms with pydantic"""
    adapter = TypeAdapter(Epoch)
    assert adapter.validate_python(0) == Epoch(0)
    assert adapter.validate_python([0, ("hey", 0)]) == Epoch(0) / ("hey", 0)
    assert adapter.validate_python([0, ["hey", 0]]) == Epoch(0) / ("hey", 0)


def test_epoch_json_roundtrip():
    """Epochs can be roundtripped to and from json"""
    adapter = TypeAdapter(Epoch)

    epoch = Epoch(0) / ("hey", 0) / ("sup", 0)

    as_json = adapter.dump_json(epoch)

    data = json.loads(as_json)
    assert data == [0, ["hey", 0], ["sup", 0]]

    assert adapter.validate_json(as_json) == epoch


def test_epoch_truediv():
    """
    Epochs can be created from truediv
    """
    expected = Epoch(root=5, path=[("hey", 0), ("sup", 50)])
    assert expected == Epoch(5) / ("hey", 0) / ("sup", 50)
