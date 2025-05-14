from collections.abc import Generator
from itertools import count
from typing import Annotated as A

from noob import Name


def source(limit: int = 10) -> Generator[A[int, Name("index")], None, None]:
    counter = count()
    while val := next(counter) < limit:
        yield val


def multiply(left: int, right: int = 2) -> int:
    """
    Return value purposely unnamed,
    to be used as `{nodename}.value`
    """
    return left * right
