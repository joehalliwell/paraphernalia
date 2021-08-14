from PIL.Image import Image

from paraphernalia.torch.dall_e import DALL_E
from paraphernalia.torch.direct import Direct, DirectPalette
from paraphernalia.torch.siren import Siren
from paraphernalia.torch.taming import VQGAN_GUMBEL_F8, Taming


def test_dall_e():
    generator = DALL_E(latent=1)
    img = generator.generate_image()
    assert isinstance(img, Image)


def test_direct():
    generator = Direct()
    img = generator.generate_image()
    assert isinstance(img, Image)


def test_direct_palette():
    generator = DirectPalette()
    img = generator.generate_image()
    assert isinstance(img, Image)


def test_siren():
    generator = Siren()
    img = generator.generate_image()
    assert isinstance(img, Image)


def test_taming():
    generator = Taming(model_spec=VQGAN_GUMBEL_F8)
    generator.generate_image()
