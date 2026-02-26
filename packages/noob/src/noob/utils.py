import builtins
import importlib
import sys
from collections.abc import Callable, Coroutine
from functools import partial
from typing import Any, TypeGuard

from noob.types import AbsoluteIdentifier, AbsoluteIdentifierAdapter

if sys.version_info < (3, 14):
    from asyncio import iscoroutinefunction
else:
    from inspect import iscoroutinefunction


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


def iscoroutinefunction_partial(f: Callable) -> TypeGuard[Callable[..., Coroutine]]:
    """
    Stolen from apscheduler, unwraps partials to test for coroutines

    References:
        https://github.com/agronholm/apscheduler/blob/f4df139771b7741f58f0eb456f091d3f659555c1/src/apscheduler/util.py#L444

    """
    while isinstance(f, partial):
        f = f.func

    # The asyncio version of iscoroutinefunction includes testing for @coroutine
    # decorations vs. the inspect version which does not.
    return iscoroutinefunction(f)
