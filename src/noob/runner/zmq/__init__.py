"""
- Central command pub/sub
- each sub-runner has its own set of sockets for publishing and consuming events
- use the `node_id.signal` etc. as basically a feed address

.. todo::

    Currently only IPC is supported, and thus the zmq runner can't run across machines.
    Supporting TCP is WIP, it will require some degree of authentication
    among nodes to prevent arbitrary code execution,
    since we shouldn't count on users to properly firewall their runners.


.. todo::

    The socket spawning and event handling is awfully manual here.
    Leaving it as is because it's somewhat unlikely we'll need to generalize it,
    but otherwise it would be great to standardize socket names and have
    event handler decorators like:

        @on_router(MessageType.sometype)

"""

from importlib.util import find_spec

if find_spec("zmq") is None:
    raise ImportError(
        "Attempted to import zmq runner, but zmq deps are not installed. install with `noob[zmq]`",
    )

from noob.runner.zmq.command import CommandNode
from noob.runner.zmq.node import NodeRunner
from noob.runner.zmq.runner import ZMQRunner

__all__ = ["CommandNode", "NodeRunner", "ZMQRunner"]
