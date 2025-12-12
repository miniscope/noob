import builtins
import importlib
import sys
from typing import Any

from noob.types import AbsoluteIdentifier, AbsoluteIdentifierAdapter


def resolve_python_identifier(ref: AbsoluteIdentifier) -> Any:
    """
    Given some fully-qualified package.subpackage.Class identifier,
    return the referenced object, importing if needed.

    Returns a node from :data:`noob.node.SPECIAL_NODES` if match found
    """
    from noob.node import SPECIAL_NODES

    ref = AbsoluteIdentifierAdapter.validate_python(ref)
    if ref in SPECIAL_NODES:
        return SPECIAL_NODES[ref]
    elif "." not in ref:
        return getattr(builtins, ref)
    else:
        module_name, obj = ref.rsplit(".", 1)
        module = sys.modules.get(module_name, importlib.import_module(module_name))
        return getattr(module, obj)
