"""
A URL queue to keep track of seen URLs during web scraping!

See examples/http-scrape-concise.yaml for example usage
"""

import sys
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Annotated as A
from urllib.parse import urldefrag, urljoin

from httpx import Response
from pydantic import Field, PrivateAttr

from nobes.web.parse import extract_tags
from noob import Asset, Epoch, Event, MetaSignal, Name, Node, NoEventable

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class QueueState(TypedDict):
    seen: set[str]
    todo: set[str]


class URLQueue(Asset):
    root_url: str

    obj: QueueState = Field(default_factory=lambda: QueueState(seen=set(), todo=set()))

    def init(self) -> None:
        self.obj["todo"].add(self.root_url)


def next_urls(queue: QueueState) -> tuple[A[set[str], Name("urls")], A[QueueState, Name("queue")]]:
    """Get the next urls to scrape from a URL queue!"""
    urls = queue["todo"].copy()
    queue["todo"] = set()
    return urls, queue


class update_urls(Node):
    filter_fragments: bool = True
    """If True, strip all fragments from urls before emitting or checking against the `seen` set."""
    gather: bool = True
    """
    If True, treat the node as a gather node, 
    where it collects urls and only emits them when all of a subepoch completes.
    See `http-scrape-concise` for an example.
    """

    stateful: bool = False

    _n: dict[Epoch, int] = PrivateAttr(default_factory=lambda: defaultdict(lambda: 0))
    _seen: dict[Epoch, int] = PrivateAttr(default_factory=lambda: defaultdict(lambda: 0))
    _states: dict[Epoch, QueueState] = PrivateAttr(
        default_factory=lambda: defaultdict(lambda: QueueState(seen=set(), todo=set()))
    )

    def process(
        self, response: Response, queue: QueueState, epoch: Epoch, n: int | None = None
    ) -> tuple[A[set[str], Name("urls")], A[NoEventable[QueueState], Name("queue")]]:
        if self.gather:
            return self._process_gather(response, queue, epoch, n)
        else:
            return self._process_single(response, queue)

    def _process_gather(
        self, response: Response, queue: QueueState, epoch: Epoch, n: int | None = None
    ) -> tuple[set[str], NoEventable[QueueState]]:
        parent = epoch.parent if epoch.parent else epoch
        if n is not None:
            self._n[parent] = n

        queue["seen"].add(str(response.url))
        queue["todo"].discard(str(response.url))
        self._states[parent]["seen"] |= queue["seen"]
        self._states[parent]["todo"].discard(str(response.url))

        hrefs = self._extract_urls(response)
        new_urls = set(hrefs) - self._states[parent]["seen"]
        queue["todo"] |= new_urls
        self._states[parent]["todo"] |= new_urls

        self._seen[parent] += 1
        if self._seen[parent] >= self._n[parent]:
            return new_urls, Event(
                id=uuid.uuid4().int,
                timestamp=datetime.now(UTC),
                node_id=self.spec.id,
                signal="queue",
                epoch=parent,
                value=self._states[parent],
            )
        else:
            return new_urls, MetaSignal.NoEvent

    def _process_single(self, response: Response, queue: QueueState) -> tuple[set[str], QueueState]:
        hrefs = self._extract_urls(response)
        queue["seen"].add(str(response.url))
        queue["todo"].discard(str(response.url))
        new_urls = set(hrefs) - queue["seen"]
        queue["todo"] |= new_urls
        return new_urls, queue

    def _extract_urls(self, response: Response) -> list[str]:
        hrefs = extract_tags(response, tag="a", attribute="href")
        # get absolute path relative to response url, ignoring fragments
        hrefs = [urljoin(str(response.url), h) for h in hrefs]
        if self.filter_fragments:
            hrefs = [urldefrag(h).url for h in hrefs]
        return hrefs
