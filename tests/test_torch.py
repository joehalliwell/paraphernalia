import pytest
import torch
from torch.tensor import Tensor

from paraphernalia.torch import grid, overtile


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
    assert tiles.shape == (4, 2, 2, 2)

    # 0.5 overlap
    tiles = overtile(batch, tile_size=2, overlap=0.5)
    assert tiles.shape == (9, 2, 2, 2)

    # One big tile
    tiles = overtile(batch, tile_size=4, overlap=0.5)
    assert tiles.shape == (1, 2, 4, 4)
    assert torch.equal(batch, tiles)

    # Overlap is too big
    with pytest.raises(ValueError):
        overtile(batch, tile_size=2, overlap=1.0)
