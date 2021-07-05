import pytest
import torch
from torch.tensor import Tensor

from paraphernalia.torch import grid, overtile, regroup


def test_grid():
    t = grid(2, 2)
    assert t.shape == (2, 2, 2)
    assert torch.equal(t, Tensor([[[-1, -1], [-1, 1]], [[1, -1], [1, 1]]]))
    print(t)
    t = grid(4, 2)
    assert t.shape == (4, 4, 2)


def test_overtile():
    batch = grid(4, 2).permute(2, 0, 1).unsqueeze(0)
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
    batch = grid(512, 2).permute(2, 0, 1).unsqueeze(0)
    assert batch.shape == (1, 2, 512, 512)

    tiles = overtile(batch, tile_size=224, overlap=0.1)
    assert len(tiles) == 9


def test_overtile_large_tile():
    # TODO: Test when the part size is almost whole
    # Currently resulting in 1 tile which is incorrect
    pass


def test_regroup():
    img = torch.cat([torch.full((1, 3, 2, 2), i) for i in range(4)])

    # Check prior state
    assert img.shape == (4, 3, 2, 2)
    assert img[2, 0, 0, 0] == 2.0

    regrouped = regroup([img, img])  # 2x identity transformation
    assert regrouped.shape == (4 * 2, 3, 2, 2)
    assert regrouped[2, 0, 0, 0] == 1.0
