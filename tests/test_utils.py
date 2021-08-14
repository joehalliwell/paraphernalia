from typing import Sized

import pkg_resources
from PIL import Image

from paraphernalia.utils import *


def test_cache_home():
    cache = cache_home()
    assert str(cache_home("FOO")) == "FOO"
    cache_home(cache)


# def test_upscale():
#     studio = Image.open(pkg_resources.resource_filename(__name__, "studio.jpg"))
#     studio_x2 = opencv_to_pil(upsample(pil_to_opencv(studio), scale=2))
#     studio_x2.save("/tmp/test.jpg")
#     assert studio_x2.size[0] == 2 * studio.size[0]
#     assert studio_x2.size[1] == 2 * studio.size[1]
