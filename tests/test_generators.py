import pytest
from PIL.Image import Image

from paraphernalia.torch.dall_e import DALL_E
from paraphernalia.torch.direct import Direct, DirectPalette
from paraphernalia.torch.siren import Siren
from paraphernalia.torch.taming import Taming
from tests import skipif_github_action


@pytest.fixture(
    scope="module",
    params=[
        DALL_E,
        Direct,
        DirectPalette,
        Siren,
        pytest.param(Taming, marks=skipif_github_action),
    ],
)
def generator(request):
    return request.param


def test_init(generator):
    img = generator(size=(16, 32)).generate_image()
    assert isinstance(img, Image)
    assert img.width == 16
    assert img.height == 32


def test_sketch(generator, studio):
    img = generator(start=studio).generate_image()
    assert isinstance(img, Image)
