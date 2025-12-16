# Events

An {class}`~noob.event.Event` is anything that gets returned from {class}`~noob.node.base.Node`,
wrapped in a container with some metadata.
The metadata include the event's ID, its name, when it happened, and where it happened. It is managed by
{class}`~noob.store.EventStore` {class}`~noob.runner.base.TubeRunner`, and
can be accessed with runner's {meth}`~noob.runner.base.TubeRunner.add_callback` method,
like the following:

```python
from noob import Tube, SynchronousRunner

tube = Tube.from_specification("...")
runner = SynchronousRunner(tube=tube)


def my_cb(): ...


runner.add_callback(my_cb)
```

## EventStore

The storage for all relevant events. When you're running a {class}`~noob.runner.sync.SynchronousRunner`, the
{class}`~noob.store.EventStore` is owned by the runner who manages all nodes, so this includes every event that gets
emitted by every node (unless it gets cleared.)

On the other hand, {class}`~noob.runner.zmq.ZMQRunner` does not manage a global {class}`~noob.store.EventStore`. Rather,
each {class}`~noob.runner.zmq.NodeRunner` manages events that are relevant to the single node it manages
(ref: {doc}`zmq`). Either way, the functionality remains the same:
preserving relevant events while there remains a node that may depend on it,
and collecting the values from within the events and returning them to the `runner`
so that a relevant node can use them.

## MetaEvent

Remember earlier when we said Events are things that get returned from nodes? We lied. Sort of. Tee hee.

We felt justifying in lying, because the other types of events are the "internal" events do not get exposed to the user.
We call these events {class}`~noob.event.MetaEvent`. These work by swapping the `signal` entry of the event with
{class}`~noob.event.MetaEventType`, which includes `NodeReady` and `EpochEnded`.
Their meanings are quite self-explanatory, where `NodeReady` signals that a node's dependencies have been satisfied
and the node is ready to process, and `EpochEnded` means... well, the given epoch (a full cycle of graph) has completed.
For debugging purposes, we do allow users to access some of these through runner callback. This behavior may change in
the future, however, so we do not recommend you depend on it.

## Optional Events

Sometimes, a node may decide not to emit anything. It accepted its inputs, processed, and nothing came of it.
Some examples of this include a {class}`~noob.nodes.gather.Gather` node that is has not gathered enough to emit its
collection, or a {class}`~noob.nodes.return_.Return` node.
Of course, a user-written node may also display this behavior. {class}`~noob.event.MetaSignal` comes in handy
in these circumstances.

Since we are under the assumption that a node can return quite literally _anything_,
we cannot flag this `NoEvent` behavior with anything that a user may use.
For example, if we decide to accept `None` as a signal for "nothing came out of the node," we end up reserving the
meaning for `None` and the user can no longer use `None` as a semantically meaningful output.

Let's assume the user wants to use the following function as a node.:

```python
from typing import TypeVar

T = TypeVar("T")


def maybe_first_element(things: list[T]) -> T:
    try:
        return things[0]  # what if the 0th element is None, like [None, 1, 2, ...] ??
    except IndexError:
        # I don't wanna fail. If things is empty, just skip.
        return None  # WRONG
```

Since we already decided to designate `None` as a flag to mean nothing came out of a node,
it became impossible for the try block to emit a meaningful `None`.
In this case, it becomes impossible for `first_element` to distinguish
"hey, the first element in this list is `None`." from "hey, there was nothing in `a`."

To circumvent this issue, we implemented a singleton `NoEvent` object. It can be used like below:

```python
from typing import TypeVar

from noob.event import MetaSignal

T = TypeVar("T")


def maybe_first_element(things: list[T]) -> T | MetaSignal:
    try:
        return things[0]  # it's ok if the 0th element is None!
    except IndexError:
        # If things is empty, just skip with NoEvent.
        return MetaSignal.NoEvent
```

This {class}`~noob.event.MetaSignal` enum class provides us with a few critical benefits:

1. It's a singleton, which means it cannot be confused with anything else. Even if a user made an identical StrEnum
   object in their own module, there will be no confusion.
2. It's serializable.
3. It allows us to sensibly annotate its type (as opposed to `NoEvent = object()`)