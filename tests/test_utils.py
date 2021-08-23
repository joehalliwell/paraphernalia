from typing import Sized

import pkg_resources
import pytest
from PIL import Image

from paraphernalia.utils import *


def test_cache_home():
    cache = cache_home()
    assert str(cache_home("FOO")) == "FOO"
    cache_home(cache)


@pytest.mark.parametrize(
    "args, expected",
    [
        (["hello world"], "hello-world"),
        (["it doesn't blend"], "it-doesnt-blend"),
        (["$£*^&£$+"], ""),
        (["numbers like 2 ok"], "numbers-like-2-ok"),
        (["hello", "world"], "hello_world"),
        (["ABC"], "abc"),
    ],
)
def test_slugify(args, expected):
    assert slugify(*args) == expected


def test_download_404():
    with pytest.raises(Exception) as excinfo:
        download("http://badurl.xyzzy/goo")
    assert "badurl" in excinfo.exconly()
