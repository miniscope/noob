import pytest

from nobes.web import extract_tags

TEST_PAGE = """
<!DOCTYPE html> 
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Test Page!</title>
  <link rel="icon" href="favicon.png">
</head> 
<body>   
  <h1>Whats up my dogs</h1>
  <a href="https://example.com/">Some link</a>
  <p>Something else on the page idk</p> 
  <p>And a second thing</p>
</body> 
</html>
"""


@pytest.mark.parametrize(
    "params,expected",
    [
        ({"tag": "a"}, ['<a href="https://example.com/">Some link</a>']),
        ({"tag": "a", "attribute": "href"}, ["https://example.com/"]),
        ({"tag": "p", "inner": True}, ["Something else on the page idk", "And a second thing"]),
    ],
)
def test_extract_tags(params: dict, expected: list[str]):
    """We can extract tags or attributes from an html page!"""
    result = extract_tags(TEST_PAGE, **params)
    result = [str(result) for result in result]
    assert result == expected
