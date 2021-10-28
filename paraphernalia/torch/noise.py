"""
Noise generating functions.
"""
from math import ceil, pi

import torch
from torch import Tensor


def perlin(
    width: int,
    height: int,
    frequency: float = 1.0,
    fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3,
    device=None,
) -> Tensor:
    """
    Generate 2d Perlin noise.

    Adapted from:

    - https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869
    - https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57
    - https://weber.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf

    Args:
        width (int): target width
        height (int): target height
        frequency (float, optional): Defaults to 1.0.
        fade: a fade function
        device ([type], optional): [description]. Defaults to None.

    Returns:
        Tensor: a (h, w) tensor
    """
    cell_size = int(min(width, height) // frequency)
    h = ceil(height / cell_size)
    w = ceil(width / cell_size)

    z = 2.0 * pi * torch.randn(h + 1, w + 1, 1, 1, device=device)
    gx = torch.sin(z)
    gy = torch.cos(z)
    xs = torch.linspace(0, 1, cell_size + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, cell_size + 1)[None, :-1].to(device)

    wx = 1 - fade(xs)
    wy = 1 - fade(ys)

    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))

    return (
        dots.permute(0, 2, 1, 3)
        .contiguous()
        .view(h * cell_size, w * cell_size)[0:height, 0:width]
    )


def fractal(
    width: int,
    height: int,
    frequency: float = 1.0,
    frequency_factor: float = 2.0,
    amplitude_factor: float = 0.8,
    octaves: int = 4,
    device=None,
) -> Tensor:
    """
    Fractal Perlin noise generator.

    Args:
        width (int): target width
        height (int): target height
        frequency (float, optional): [description]. Defaults to 1.0.
        frequency_factor (float, optional): [description]. Defaults to 2.0.
        amplitude_factor (float, optional): [description]. Defaults to 0.8.
        octaves (int, optional): Number of different scale to use. Defaults to 4.
        device ([type], optional): Defaults to None.

    Returns:
        Tensor: [description]
    """
    result = torch.zeros((height, width)).to(device)
    amplitude = 1.0
    for i in range(octaves):
        result += amplitude * perlin(width, height, frequency, device=device)
        frequency *= frequency_factor
        amplitude *= amplitude_factor

    a = result.min()
    b = result.max()

    if a == b:
        return torch.zeros(height, width)

    return (result - a) / (b - a)
