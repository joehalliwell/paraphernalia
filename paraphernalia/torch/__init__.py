"""
Utilities for working with PyTorch
"""

import gc as _gc
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch import Tensor
from torchvision.utils import make_grid

from paraphernalia.utils import divide


def grid(*steps: int) -> Tensor:
    """
    Generate a tensor of co-ordinates in the origin-centred hypercube of the
    specified dimension.

    Args:
        *steps: number of steps per dimension

    Returns:
        A (rank ``len(steps) + 1``) tensor of the coordinates. The co-ordinates
        themselves are in dimension -1.
    """
    if isinstance(steps, int):
        steps = (steps,)
    axes = [torch.linspace(-1, 1, s) for s in steps]
    grid = torch.stack(torch.meshgrid(*axes), dim=-1)
    return grid


def tile(img: Tensor, size: int) -> Tensor:
    """
    Tile img with squares of side size. Any cut off at the edge is ignored.
    TODO: Remove
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
        img (List[Tensor]): a list of identically shaped image batches

    Returns:
        Tensor: a concatenation into a single image batch grouped
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


def make_palette_grid(colors, size=128):
    """
    Create an image to preview a set colours, provided as an iterable of RGB
    tuples with each component in [0,1].
    """
    swatches = []
    swatches = torch.cat(
        [torch.Tensor(c).view(1, 3, 1, 1).repeat([1, 1, size, size]) for c in colors]
    )
    return T.functional.to_pil_image(make_grid(swatches))


def one_hot_noise(shape):
    """
    Generate a one-hot-encoded latent state suitable for use with a categorical
    variational decoder. This is a hard one-hot tensor. Use `one_hot_normalize()`
    if you want to soften.

    Args:
        shape (Tuple): desired shape (batch_size, num_classes, height, width)

    Returns:
        Tensor: one hot Tensor of dimension (batch_size, num_classes, height, width)
    """
    b, c, h, w = shape
    z = torch.nn.functional.one_hot(
        torch.randint(0, c, (b, h, w)),
        num_classes=c,
    )
    z = z.permute(0, 3, 1, 2)
    return z


def one_hot_constant(shape, index):
    """
    Generate a latent state using a constant value.

    Args:
        shape (Tuple): desired shape (batch_size, num_classes, height, width)

    Returns:
        Tensor: one hot Tensor of dimension (batch_size, num_classes, height, width)

    """
    b, c, h, w = shape
    if index < 0:
        raise ValueError("Index must be >= 0")
    if index >= c:
        raise ValueError(f"Index must be <= the provided number of classes ({c})")
    z = torch.nn.functional.one_hot(
        torch.full((b, h, w), index),
        num_classes=c,
    )
    z = z.permute(0, 3, 1, 2)
    return z


def one_hot_normalize(z, tau=0.001):
    """
    Normalized a log probability/one hot tensor, by locking in
    modes then converting a slightly noisy log probability.

    Args:
        shape (Tuple): desired shape (batch_size, num_classes, height, width)

    Returns:
        Tensor: one hot Tensor of dimension (batch_size, num_classes, height, width)

    """
    b, c, h, w = z.shape
    z = torch.argmax(z, axis=1)
    z = torch.nn.functional.one_hot(z, num_classes=c)
    z = z.permute(0, 3, 1, 2)
    z = torch.log(z + tau / c)
    return z


class ReplaceGrad(torch.autograd.Function):
    """
    Replace one function's gradient with another's.

    FIXME: I don't think this works.
    """

    @staticmethod
    def forward(ctx, x_forward, x_backward):
        """Replace the backward call"""
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        """Compute the gradient at this function"""
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    """
    Clamp an output but pass through gradients.
    """

    @staticmethod
    def forward(ctx, input: Tensor, min: float, max: float):
        """
        Clamp `input` between `min` and `max`
        """
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        """Compute the gradient at this function"""
        (input,) = ctx.saved_tensors
        return (
            grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0),
            None,
            None,
        )


clamp_with_grad = ClampWithGrad.apply


def free(device=None):
    """
    Compute free memory on the specified device.

    Args:
        device ([type], optional): the device to query

    Returns:
        Tuple: (total, used, free) in bytes
    """
    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    return (total, allocated, total - allocated)


def gc():
    """
    Trigger a Python/PyTorch garbage collection.
    """
    _gc.collect()
    torch.cuda.empty_cache()


def make_random_resized_crop(src_size, dest_size, scale=(0.08, 1.0), interpolation=2):
    """
    Returns a RandomResizedCrop transformation that will produce
    crops of `dest_size` from images of `src_size` with scale
    in the range indicated. Sizes are both torchvision style (h, w).
    """

    natural_ratio = src_size[1] / src_size[0] * dest_size[0] / dest_size[1]
    natural_scale = (dest_size[0] * dest_size[1]) / (src_size[0] * src_size[1])
    return T.RandomResizedCrop(
        dest_size,
        scale=(min(natural_scale, scale[0]), max(natural_scale, scale[1])),
        ratio=(natural_ratio, natural_ratio),
        interpolation=interpolation,
    )
