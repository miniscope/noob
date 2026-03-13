import asyncio
from pathlib import Path
from typing import Annotated as A
from urllib.parse import urljoin

import httpx

from nobes.web.const import DEFAULT_USER_AGENT
from noob import Name
from noob.asset import Asset
from noob.event import MetaSignal


async def get_url(
    url: str, client: httpx.AsyncClient | None = None, request_params: dict | None = None
) -> tuple[
    A[httpx.Response | MetaSignal, Name("response")],
    A[Exception | MetaSignal, Name("error")],
    A[str, Name("url")],
]:
    if request_params is None:
        request_params = {}

    try:
        if client is None:
            with httpx.AsyncClient() as client:
                res = await client.get(url, **request_params)
        else:
            res = await client.get(url, **request_params)
    except Exception as e:
        return MetaSignal.NoEvent, e, url
    return res, MetaSignal.NoEvent, url


def response_path(response: httpx.Response, directory: Path) -> Path:
    url = response.url
    domain_part = url.host.replace(".", "_")
    path_parts = url.path.split("/")
    out_path = Path(directory, domain_part, *path_parts).with_suffix(".html")
    return out_path


def write_html(response: httpx.Response, directory: Path) -> None:
    """
    Write HTML to a file, using its URL to derive its filename beneath a directory.

    (if you have html and need to just write it to file, just use :func:`nobes.files.write_text` )
    """
    out_path = response_path(response, directory)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(response.text)


def write_binary(response: httpx.Response, directory: Path) -> None:
    """
    Write binary to a file, using its URL to derive its filename beneath a directory
    """
    out_path = response_path(response, directory)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(response.content)


def traverse_to(base: str, rel: str, fragments: bool = False) -> str | MetaSignal:
    """
    Given a base url and either another url, relative url, or url fragment,
    get the url that would be gotten if one were to click on that link

    Args:
        fragments (bool): If `True`, include fragments, otherwise NoEvent them
    """
    if rel.startswith("#") and not fragments:
        return MetaSignal.NoEvent
    return urljoin(base, rel)


class AsyncHttpxClient(Asset):
    """
    Reuse an HTTPX client!

    .. todo::

        Convert this to a contextmanager once we support contextmanager assets

    """

    obj: httpx.AsyncClient | None = None

    def init(self) -> None:
        if "headers" not in self.params:
            self.params["headers"] = {"User-Agent": DEFAULT_USER_AGENT}
        elif not self.params["headers"].get("User-Agent"):
            self.params["headers"]["User-Agent"] = DEFAULT_USER_AGENT

        self.obj = httpx.AsyncClient(**self.params)

    def deinit(self) -> None:
        if self.obj is None:
            return
        try:
            eventloop = asyncio.get_event_loop()
        except RuntimeError as e:
            raise RuntimeError(
                "AsyncHTTPXClient can only be used in the async runner "
                "until async assets are implemented!"
            ) from e
        eventloop.create_task(self.obj.aclose())
        self.obj = None
