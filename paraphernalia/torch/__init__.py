import torch
import torchvision.transforms as T


def tile(img, size):
    """
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


def overtile(img, tile_size):
    """
    Generate an overlapping tiling that covers ``img``.

    Args:
        img (Tensor): An image tensor (b, c, h, w)
        tile_size (Union[int, Tuple[int,int]]): The size of the tile, either
            a single int or a pair

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

    nh = int(h // th) + 1
    nw = int(w // tw) + 1

    batch = []
    for top in [i * (h - th) / (nh - 1) for i in range(nh)]:
        for left in [i * (w - tw) / (nw - 1) for i in range(nw)]:
            batch.append(T.functional.crop(img, int(top), int(left), th, tw))

    return torch.cat(batch, 0)
