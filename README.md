# noob

[![PyPI - Version](https://img.shields.io/pypi/v/noob)](https://pypi.org/project/noob)
[![Readthedocs Status](https://app.readthedocs.org/projects/noob/badge/)](https://noob.readthedocs.io)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/noob)
![No AI - Organic Human Labor](https://raw.githubusercontent.com/miniscope/noob/main/assets/img/ai_badge.svg)

![Noob logo - It's noob! graph processing for noobs](https://raw.githubusercontent.com/miniscope/noob/refs/heads/main/assets/img/noob_logo.gif)

`noob` - a **streaming**, **event-driven**, **node-centric** graph processing library that scales from a single function call to comically large distributed pipelines!

---

## Introduction

See the [documentation for all noob related information](https://noob.readthedocs.io)

Most programs are structured as graphs![^graphs]
But most programs end up rewrite their own graph execution system, whether the developers know it or not.

Existing graph processing systems are specialized for "big data" on "managed systems,"
requiring a lot of custom framework-specific code and complex setups.

Noob is the glue package you've been looking for that lets you freely experiment with graph pipelines
that can fit in your pocket - 
use it within your package, to write a background service, a web crawler, and yes even a data analytics pipeline -
and then scale it up as needed.

Noob tubes require very little, if any noob-specific code, and its graph structure has a 1:1 mapping to python language features: 
every function is already a noob node, and every class is waiting to become one.

Say you have a function to yield frames from a webcam and another to display the frames

`mypackage/video.py`
```python
from typing import Generator
import cv2
import numpy as np

def webcam(index: int = 0) -> Generator[np.ndarray, None, None]:
    cap = cv2.VideoCapture(index)
    while cap.isOpened():
        ret, frame = cap.read()
        yield frame

def display(frame: np.ndarray) -> None:
    cv2.imshow('default', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
```

Then you can make the simplest graph you've ever seen (a line) like this:

`mypackage/webcam.yaml`
```yaml
noob_id: webcam-display

nodes:
  webcam:
    type: mypackage.video.webcam
  display: 
    type: mypackage.video.display
    depends:
      - frame: webcam.value
```

And run it like this

```python
from noob import Tube, SynchronousRunner

tube = Tube.from_specification("webcam-display")
with SynchronousRunner(tube) as runner:
    runner.run()
```

And then you can extend that tube as much as you want!
Say you want to be able to access the frames from within python, you can use a `return` node:

`mypackage/webcam-return.yaml`
```yaml
noob_id: webcam-display-and-return
extends: [webcam-display]

nodes:
  return:
    type: return
    depends: webcam.value
```

and then run it like this!

```python
from noob import Tube, SynchronousRunner

tube = Tube.from_specification("webcam-display")
with SynchronousRunner(tube) as runner:
    frame = runner.process()
    # do whatever you want with your frame! it's your frame!
    # and if it's not your frame, leave it alone!
```

Keep adding nodes until you have something fun like a live streaming webcam server that rotates the hue based on your mouse position
and send a discord message whenever your head touches the right side of the frame.
You write your code as a set of normal python functions and classes, 
and noob serves as the glue between them:
you get the best of both worlds - clean code that can be reused outside of noob in your script or package,
and an extensible, declarative processing graph that handles scheduling and dependency injection for you.

For more examples see [the examples](./examples)

## Installation

Install from pypi!

```bash
python -m pip install noob
```

The main package dependencies are extremely light, and additional functionality is provided through optional dependency groups

- `cli` - cli functionality with click! run tubes as unix program, piping input and output with standard streams
- `zmq` - network-based multiprocess runner using pyzmq!


## Features

- **Streaming** processing: designed for scientific applications and data acquisition, 
  work with single events in streams as data is acquired or iterate over batches of data
- **Declarative and Serializable** graph specification: simple yaml 1.1 pipeline specifications that can be easily distributed, extended,
  and interoperate across languages (noob in the browser coming soon....)
- **Decoupled Runners**: Graph structure is decoupled from runner logic - use the same pipeline with multiple runners,
  test things out with a simple synchronous runner, and deploy using a distributed ZMQ cluster.
- **Extremely Lightweight**: no external system services required! Running tubes is as easy as calling a function.
- **Compatible With Your Code**: You don't need to rewrite your entire package to use noob, and using noob doesn't lock you in.
  All noob-specific code is optional and is applied as type annotations.
- **Multiple signals and slots**: Nodes have a flexible input/output system that can couple input signals together and emit multiple independent streams
- **Cardinality Manipulation**: Handle mismatches between nodes that run at different speeds, couple code that works on collections with code that works on single objects
- **Recursive Tubes**: Structure your graph as a series of subgraphs and embed them within each other! 
  Combined with noobs runner system, this gives you fine-grained control over data locality and execution style. 
- **No Special Cases**: All functionality is generic across nodes - if you want to mess with how noob works,
  make weird time travel nodes, manipulate the graph while it runs, whatever, you can do that.

## Contributing

See [the contributing docs!](https://noob.readthedocs.io/en/latest/contributing.html)

---

> At first I thought this was software for processing graphs...
> 
> but now I realize it's for graph processing.
> 
> *-[plaidtron3000](https://jorts.horse/@plaidtron3000/115467534463581813)@[jorts.horse](https://jorts.horse/@plaidtron3000/115467538959685105)*

[^graphs]: all programs are actually graphs, but you know what i mean. 
