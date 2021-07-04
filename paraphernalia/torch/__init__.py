from typing import Optional, Tuple, Union

import torch
import torchvision.transforms as T
from torch import Tensor

from paraphernalia.utils import divide


def grid(steps: int, dimensions: Optional[int] = 2) -> Tensor:
    """
    TODO: Rename
    Generate a tensor of co-ordinates in the origin-centred hypercube of the
    specified dimension.

    Args:
        steps: Number of steps per side
        dimensions: The dimensionality of the hypercube. Defaults to 2.

    Returns:
        A (rank ``dimensions + 1``) tensor of the coordinates
    """
    axes = [torch.linspace(-1, 1, steps) for _ in range(dimensions)]
    grid = torch.stack(torch.meshgrid(*axes), dim=-1)
    return grid


def tile(img: Tensor, size: int) -> Tensor:
    """
    TODO: Remove
    Tile img with squares of side size. Any cut off at the edge is ignored.
    """
    b, c, h, w = img.shape
    img = T.functional.center_crop(img, (h // size * size, w // size * size))
    tiles = (
        img.unfold(1, 3, 3)
        .unfold(2, size, size)
        .unfold(3, size, size)
        .reshape(-1, c, size, size)
    )
    return tiles


def overtile(
    img: Tensor, tile_size: Union[int, Tuple[int, int]], overlap: float = 0.5
) -> Tensor:
    """
    TODO: Rename
    Generate an overlapping tiling that covers ``img``.

    Args:
        img: An image tensor (b, c, h, w)
        tile_size: The size of the tile, either a single int or a pair of them
        overlap: The *minimum* overlap as a fraction of tile size. Defaults to
            0.5, where two tiles cover every pixel except at the edges.

    Returns:
        Tensor: A batch of tiles of size ``tile_size`` covering img
    """

    b, c, h, w = img.shape

    if isinstance(tile_size, int):
        th = tile_size
        tw = tile_size
    else:
        th = int(tile_size[0])
        tw = int(tile_size[1])

    batch = []
    for top in divide(h, th, overlap * th):
        for left in divide(w, tw, overlap * tw):
            batch.append(T.functional.crop(img, int(top), int(left), th, tw))

    return torch.cat(batch, 0)
