from paraphernalia.torch.noise import fractal, perlin


def test_perlin():
    img = perlin(123, 456)
    assert img.shape == (456, 123)


def test_fractal():
    img = fractal(456, 123)
    assert img.shape == (123, 456)
