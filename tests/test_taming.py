from paraphernalia.torch.taming import VQGAN_GUMBEL_F8, Taming


def test_init():
    generator = Taming(shape=(1, 256, 32, 32), model=VQGAN_GUMBEL_F8)
    generator.generate_image()
