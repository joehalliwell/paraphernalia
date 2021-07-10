from typing import List, Optional, Tuple, Union

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
) -> List[Tensor]:
    """
    TODO: Rename
    Generate an overlapping tiling that covers ``img``.

    Args:
        img: An image tensor (b, c, h, w)
        tile_size: The size of the tile, either a single int or a pair of them
        overlap: The *minimum* overlap as a fraction of tile size. Defaults to
            0.5, where two tiles cover every pixel except at the edges.

    Returns:
        List[Tensor]: A list of image batches of size ``tile_size`` covering img
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

    return batch


def regroup(img: List[Tensor]) -> Tensor:
    """
    Concatenate several image batches, regrouping them so that
    a single image is contiguous in the resulting batch.

    TODO: Is this part of torch under a different name?

    Args:
        img (List[Tensor]): A list of identically shaped image batches

    Returns:
        Tensor: A concatenation into a single image batch grouped
            so that each image in the source batches forms a contiguous block
            in the new batch
    """
    batch_size = img[0].shape[0]

    # If the batch size is 1, just concatenate
    if batch_size == 1:
        return torch.cat(img)

    # Otherwise shuffle things around
    img = torch.stack(img, 1)
    img = torch.flatten(img, start_dim=0, end_dim=1)
    return img


def cosine_similarity(a, b):
    """
    Compute the cosine similarity tensor.

    TODO: Explain restrictions

    Args:
        a (Tensor): (A, N) tensor
        b (Tensor): (B, N) tensor

    Returns:
        [Tensor]: (A, B) tensor of similarities
    """
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    result = torch.mm(a_norm, b_norm.transpose(0, 1))
    return result
