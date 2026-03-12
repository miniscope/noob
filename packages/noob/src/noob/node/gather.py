import uuid
from datetime import UTC, datetime
from multiprocessing import Lock
from multiprocessing.synchronize import Lock as LockType
from typing import Any, Generic, TypeVar, cast

from pydantic import PrivateAttr

from noob.event import Event, MetaSignal
from noob.node import Slot
from noob.node.base import Node
from noob.types import Epoch

_TInput = TypeVar("_TInput")


class Gather(Node, Generic[_TInput]):
    """
    Cardinality reduction.

    Given a node that emits >1 events, gather them into a single iterable.

    Two (mutually exclusive) modes:

    - gather a fixed number of events

    .. code-block:: yaml

        nodename:
          type: gather
          params:
            n: 5
          depends:
            - value: othernode.signal

    - gather events until a trigger is received

    .. code-block:: yaml

        nodename:
          type: gather
          depends:
            - value: othernode.signal_1
            - trigger: thirdnode.signal_2
    """

    n: int | None = None
    flatten: bool = False
    """
    If an individual gathered value is a sequence 
    (and thus the returned gathered value a sequence of sequences),
    flatten the sequences by 1 level.
    
    [['a', 'b'], ['c'], []] -> ['a', 'b', 'c'] 
    """
    _items: list[tuple[Epoch, _TInput]] = PrivateAttr(default_factory=list)
    _lock: LockType = PrivateAttr(default_factory=Lock)

    def process(
        self, value: _TInput, epoch: Epoch, trigger: Any | None = None, n: int | None = None
    ) -> Event[list[_TInput]] | MetaSignal:
        """Collect value in a list, emit if `n` is met or `trigger` is present"""
        if n is not None:
            self.n = n
        if trigger is not None and self.n is not None:
            raise ValueError("Cannot use trigger mode while `n` is set")
        with self._lock:
            self._items.append((epoch, value))
            if self._should_return(trigger):
                items = [item[1] for item in sorted(self._items, key=lambda i: i[0])]
                if self.flatten:
                    # can't figure out how to convince mypy that the inner type is a list
                    items = self._do_flatten(items)  # type: ignore[arg-type]

                try:
                    # collapse epoch if in a sub-epoch
                    ep = epoch.parent if len(epoch) > 1 else epoch
                    ep = cast(Epoch, ep)
                    return Event(
                        id=uuid.uuid4().int,
                        timestamp=datetime.now(UTC),
                        node_id=self.id,
                        signal="value",
                        epoch=ep,
                        value=items,
                    )
                finally:
                    # clear list after returning
                    self._items = []
            return MetaSignal.NoEvent

    def _collect_slots(self) -> dict[str, Slot]:
        slots = {
            "value": Slot(name="value", annotation=Any),
            "epoch": Slot(name="epoch", annotation=Epoch),
            "trigger": Slot(name="trigger", annotation=Any | None, required=False),
        }
        if (
            self.spec
            and self.spec.depends
            and any(
                next(iter(dep.keys())) == "n" for dep in self.spec.depends if isinstance(dep, dict)
            )
        ):
            slots["n"] = Slot(name="n", annotation=int | None, required=True)
        else:
            slots["n"] = Slot(name="n", annotation=int | None, required=False)
        return slots

    def _should_return(self, trigger: Any | None) -> bool:
        return (self.n is not None and len(self._items) >= self.n) or (
            self.n is None and trigger is not None
        )

    def _do_flatten(self, items: list[list[_TInput]]) -> list[_TInput]:
        flat = []
        for item in items:
            try:
                flat.extend(item)
            except TypeError as e:
                raise TypeError("Requested flatten, but error spreading the gathered value!") from e
        return flat
