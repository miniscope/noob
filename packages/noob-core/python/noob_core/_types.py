from __future__ import annotations

from typing import TypeAlias, TypedDict

_Signal: TypeAlias = tuple[str, str]
_Slot: TypeAlias = tuple[str, str]
_Item: TypeAlias = str | _Slot | _Signal


class SorterState(TypedDict):
    ready: set[_Item]
    out: set[_Item]
    done: set[_Item]
    disabled: set[_Item]
    ran: set[_Item]
    pending: set[_Item]
    npassedout: int
    nfinished: int
    info: dict[_Item, NodeRec]


class NodeRec(TypedDict):
    nqueue: int
    successors: set[_Item]
    predecessors: set[_Item]
    optional_predecessors: dict[_Slot, _Signal]
    optional_successors: set[_Slot]
