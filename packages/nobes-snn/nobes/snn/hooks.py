from nobes.snn.runner import ZMQFreeRunner
from noob.runner import TubeRunner


def add_runners() -> dict[str, type[TubeRunner]]:
    return {"zmq-freerun": ZMQFreeRunner}
