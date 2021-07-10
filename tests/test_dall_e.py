from paraphernalia.torch.dall_e import DALL_E


def smoketest():
    dall_e = DALL_E(latent=4)
    dall_e.generate_image(0)
