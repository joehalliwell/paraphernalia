import torchvision.transforms as T


def tile(img, size):
    """
    Tile img with squares of side size. Any cut off at the edge is ignored.
    """
    b, c, h, w = img.shape
    img = T.functional.center_crop(img, (h // size * size, w // size * size))
    tiles = (
        img.unfold(1, 3, 3)
        .unfold(2, img.shape[2], size)
        .unfold(3, img.shape[3], size)
        .reshape(-1, c, size, size)
    )
    return tiles
