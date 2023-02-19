from paraphernalia.torch.noise import perlin


def test_perlin():
    img = perlin(123, 456)
    assert img.shape == (456, 123)
