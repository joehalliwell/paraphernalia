import pytest
from PIL.Image import Image

from paraphernalia.torch.dall_e import DALL_E
from paraphernalia.torch.direct import Direct, DirectPalette
from paraphernalia.torch.siren import Siren
from paraphernalia.torch.taming import Taming


@pytest.fixture(scope="module", params=[DALL_E, Direct, DirectPalette, Siren, Taming])
def generator(request):
    return request.param


def test_init(generator):
    img = generator().generate_image()
    assert isinstance(img, Image)


def test_sketch(generator, studio):
    img = generator(start=studio).generate_image()
    assert isinstance(img, Image)
