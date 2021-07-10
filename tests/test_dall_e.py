from paraphernalia.torch.dall_e import DALL_E


def test_init():
    dall_e = DALL_E(latent=1)
    dall_e.generate_image(0)
