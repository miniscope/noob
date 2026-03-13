import httpx
from bs4 import BeautifulSoup, Tag


def extract_tags(
    page: str | httpx.Response, tag: str, inner: bool = False, attribute: str | None = None
) -> list[Tag] | list[str]:
    """
    Extract matching tags from an HTML page.

    Args:
        page (str | httpx.Response): Raw HTML for a page, or an httpx response.
            If an HTTPX response, only attempts to extract tags if response code is a 200
        tag (str): The tag to extract!
        inner (bool): If ``True`` , extract inner value of tag. Otherwise return the tag itself.


    Returns:
        list[bs4.Tag] | list[str]
    """
    if isinstance(page, httpx.Response):
        if page.status_code != 200:
            return []
        html = page.text
    else:
        html = page

    soup = BeautifulSoup(html, "lxml")
    tags = soup.find_all(tag)
    if inner:
        return [t.text for t in tags]
    elif attribute:
        return [t.attrs.get(attribute) for t in tags if t.attrs.get(attribute)]
    else:
        return tags
