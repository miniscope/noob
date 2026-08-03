---
file_format: mystnb
mystnb:
    output_stderr: remove
    render_text_lexer: myst-ansi
    render_markdown_format: myst
myst:
    enable_extensions: ["colon_fence"]
---

# noob

whats up everyone

```{code-cell}
from noob import TubeClassicEdition
TubeClassicEdition()
``` 

```{toctree}
:maxdepth: 2
:caption: Usage:

usage/nodes
usage/tubes
usage/runners
usage/assets
usage/epochs
usage/events
usage/config
usage/zmq
```

```{toctree}
:maxdepth: 3
:caption: Developer:

api/index
api/noob-core/index
```

```{toctree}
:maxdepth: 2
:caption: Meta:

contributing
related_projects
changelog
```


