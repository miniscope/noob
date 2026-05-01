"""
View a noob tube!
"""

import contextlib
import threading
import time
import webbrowser
from collections.abc import AsyncGenerator
from pathlib import Path

from pydantic import ValidationError

from noob.tube import TubeSpecification

try:
    from litestar import Litestar, get, websocket_stream
except ImportError as e:
    raise ImportError(
        "GUI dependencies not installed, install noob with the [gui] dependency group"
    ) from e

import uvicorn
import watchfiles
from litestar.contrib.jinja import JinjaTemplateEngine
from litestar.logging import LoggingConfig
from litestar.response import Template
from litestar.static_files import create_static_files_router
from litestar.template.config import TemplateConfig


def _open_browser(
    url: str,
    delay: float = 1,
) -> None:
    time.sleep(delay)
    webbrowser.open(url, 2)


def make_view_app() -> Litestar:
    @get(path="/view/{tube_id: str}")
    async def view(tube_id: str) -> Template:
        spec_json = TubeSpecification.from_id(tube_id).model_dump_json()
        return Template(
            template_name="view.html.jinja2", context={"tube_id": tube_id, "spec": spec_json}
        )

    @websocket_stream("/spec/{tube_id: str}")
    async def stream_spec(tube_id: str) -> AsyncGenerator[int, None]:
        # yield the initial spec first, then reload whenever it changes
        tube_path = TubeSpecification.path_from_id(tube_id)
        with contextlib.suppress(ValidationError):
            yield TubeSpecification.from_yaml(tube_path).model_dump_json()

        watcher = watchfiles.awatch(tube_path)
        async for _ in watcher:
            # totally fine, the spec is malformed when typing in it sometimes!
            with contextlib.suppress(ValidationError):
                yield TubeSpecification.from_yaml(tube_path).model_dump_json()

    logging_config = LoggingConfig(
        root={"level": "INFO", "handlers": ["queue_listener"]},
        formatters={"standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}},
        log_exceptions="always",
    )

    app = Litestar(
        route_handlers=[
            view,
            stream_spec,
            create_static_files_router(
                path="/static", directories=[Path(__file__).parents[1] / "_js"]
            ),
        ],
        template_config=TemplateConfig(
            directory=Path(__file__).parent / "templates",
            engine=JinjaTemplateEngine,
        ),
        logging_config=logging_config,
    )
    return app


def run_view(tube_id: str) -> None:
    # pretty hacky, but don't see an "on loaded" callback on uvicorn
    url = f"http://127.0.0.1:8000/view/{tube_id}"
    open_browser = threading.Thread(target=lambda: _open_browser(url))
    open_browser.start()

    uvicorn.run(
        "noob.gui.view:make_view_app",
        factory=True,
    )
