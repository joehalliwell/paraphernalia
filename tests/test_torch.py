import pytest
import torch
from torch.tensor import Tensor

from paraphernalia.torch import (
    cosine_similarity,
    grid,
    make_random_resized_crop,
    one_hot_constant,
    one_hot_noise,
    one_hot_normalize,
    overtile,
    regroup,
)


def test_grid():
    t = grid(2, 2)
    assert t.shape == (2, 2, 2)
    assert torch.equal(t, Tensor([[[-1, -1], [-1, 1]], [[1, -1], [1, 1]]]))
    print(t)
    t = grid(4, 4, 4)
    assert t.shape == (4, 4, 4, 3)
    t = grid(4, 4, 4, 4)
    assert t.shape == (4, 4, 4, 4, 4)


def test_overtile():
    batch = grid(4, 4).permute(2, 0, 1).unsqueeze(0)
    assert batch.shape == (1, 2, 4, 4)

    # No overlap -- just a regular chessboard tiling
    tiles = overtile(batch, tile_size=2, overlap=0)
    tiles = torch.cat(tiles)
    assert tiles.shape == (4, 2, 2, 2)

    # 0.5 overlap
    tiles = overtile(batch, tile_size=2, overlap=0.5)
    tiles = torch.cat(tiles)
    assert tiles.shape == (9, 2, 2, 2)

    # One big tile
    tiles = overtile(batch, tile_size=4, overlap=0.5)
    tiles = torch.cat(tiles)
    assert tiles.shape == (1, 2, 4, 4)
    assert torch.equal(batch, tiles)

    # Overlap is too big
    with pytest.raises(ValueError):
        overtile(batch, tile_size=2, overlap=1.0)


def test_overtile_unusual_ratio():
    batch = grid(512, 512).permute(2, 0, 1).unsqueeze(0)
    assert batch.shape == (1, 2, 512, 512)

    tiles = overtile(batch, tile_size=224, overlap=0.1)
    assert len(tiles) == 9


def test_overtile_large_tile():
    batch = grid(512, 512).permute(2, 0, 1).unsqueeze(0)
    assert batch.shape == (1, 2, 512, 512)

    tiles = overtile(batch, tile_size=511)
    assert len(tiles) == 4


def test_regroup():
    img = torch.cat([torch.full((1, 3, 2, 2), i) for i in range(4)])

    # Check prior state
    assert img.shape == (4, 3, 2, 2)
    assert img[2, 0, 0, 0] == 2.0

    regrouped = regroup([img, img])  # 2x identity transformation
    assert regrouped.shape == (4 * 2, 3, 2, 2)
    assert regrouped[2, 0, 0, 0] == 1.0


def test_cosine_similarity():
    a = Tensor([1, 0, 0]).unsqueeze(0)
    b = Tensor([0, 0, 1]).unsqueeze(0)
    assert torch.equal(cosine_similarity(a, a), Tensor([[1.0]]))

    both = torch.cat([a, b])
    assert torch.equal(cosine_similarity(both, both), torch.eye(2))

    assert torch.equal(cosine_similarity(both, a), Tensor([[1], [0]]))


def test_random_resized_crop():
    """
    Trick to test as RandomResizedCrop falls back to a centre-crop.
    """
    # Square -> square
    t = make_random_resized_crop((400, 400), (100, 100))
    assert t.ratio == (1.0, 1.0)
    assert t.scale[0] <= (1.0 / 16.0)

    # Non-square -> square
    t = make_random_resized_crop((200, 100), (100, 100))
    assert t.ratio == (0.5, 0.5)
    assert t.scale[1] == 1.0

    # Target larger than source, should maybe generate a warning?
    t = make_random_resized_crop((100, 100), (200, 200))
    assert t.ratio == (1.0, 1.0)
    assert t.scale[1] >= 2.0


def test_one_hot_noise():
    shape = (1, 3, 128, 128)
    z = one_hot_noise(shape)
    assert z.shape == shape


def test_one_hot_constant():
    shape = (1, 3, 128, 128)
    z = one_hot_constant(shape, 0)
    assert z.shape == shape


def test_one_hot_normalize():
    shape = (1, 3, 1, 1)
    z = one_hot_constant(shape, 1)
    assert z.shape == shape

    z = one_hot_normalize(z)
    assert z.shape == shape

    assert z[0, 1, 0, 0] > z[0, 0, 0, 0]
    assert z[0, 1, 0, 0] > z[0, 2, 0, 0]
    assert z[0, 0, 0, 0] == z[0, 2, 0, 0]

    zp = torch.nn.functional.softmax(z)
    assert torch.all(zp > 0)
    assert torch.all(zp < 1)
