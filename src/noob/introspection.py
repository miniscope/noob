"""
Utilities for working with python type annotations
"""

from types import NoneType, UnionType
from typing import Annotated, Any, Union, get_args, get_origin


def unwrap_optional(typ: type) -> type:
    if get_origin(typ) in (Union, UnionType):
        args = [unwrap_optional(arg) for arg in get_args(typ)]
        for arg in args:
            if arg is not None:
                return arg
    return typ


def unwrap_annotated(typ: type) -> type:
    if get_origin(typ) is Annotated:
        return get_args(typ)[0]
    return typ


def unwrap(typ: type) -> type:
    """
    Unwrap all 'extra' type wrappers to get to the actual type
    """
    typ = unwrap_annotated(typ)
    typ = unwrap_optional(typ)
    return typ


def is_union(dtype: Any) -> bool:
    """
    Check if a dtype is a union
    """
    if UnionType is None:
        return get_origin(dtype) is Union
    else:
        return get_origin(dtype) in (Union, UnionType)


def is_optional(dtype: Any) -> bool:
    dtype = unwrap_annotated(dtype)
    if not is_union(dtype):
        return False
    args = get_args(dtype)
    return None in args or NoneType in args
