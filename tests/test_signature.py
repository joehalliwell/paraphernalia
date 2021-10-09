import pytest
from PIL import Image

from paraphernalia.signature import Signature


@pytest.mark.parametrize("imgfile", ["test.png", "test.jpg", "test.tif"])
def test_roundtrip(tmpdir, imgfile):
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    filename = str(tmpdir.join(imgfile))
    img.save(filename)

    creator = "Joe Halliwell"
    title = "Untitled #1"
    tags = ["procgen", "paraphernalia"]
    description = "A gesamkunstwerk"

    with Signature(filename) as sig:
        sig.creator = creator
        sig.title = title
        sig.tags = tags
        sig.description = description

    with Signature(filename) as sig:
        assert sig.creator == creator
        assert sig.creators == [creator]
        assert sig.title == title
        assert sig.tags == tags
        assert sig.description == description
