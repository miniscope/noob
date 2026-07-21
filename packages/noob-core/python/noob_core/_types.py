from typing import TypeAlias, TypedDict

_Item: TypeAlias = str | tuple[str, str]


class SorterState(TypedDict):
    ready: set[_Item]
    out: set[_Item]
    done: set[_Item]
    disabled: set[_Item]
    ran: set[_Item]
    pending: set[_Item]
    npassedout: int
    nfinished: int
