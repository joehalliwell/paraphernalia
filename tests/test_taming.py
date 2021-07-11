from paraphernalia.torch.taming import VQGAN_GUMBEL_F8, Taming


def test_init():
    generator = Taming(model_spec=VQGAN_GUMBEL_F8)
    generator.generate_image()
