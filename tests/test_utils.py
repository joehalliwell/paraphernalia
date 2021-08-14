from typing import Sized

import pkg_resources
from PIL import Image

from paraphernalia.utils import *


def test_cache_home():
    cache = cache_home()
    assert str(cache_home("FOO")) == "FOO"
    cache_home(cache)
