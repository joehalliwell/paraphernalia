"""
Base class and utility types for image generators.
"""
import logging
from abc import ABCMeta
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL.Image import Image
from torch import Tensor
from torchvision.utils import make_grid

from paraphernalia.torch import clamp_with_grad, one_hot_noise

_LOG = logging.getLogger(__name__)

SizeType = Union[int, Tuple[int, int]]


class Generator(nn.Module, metaclass=ABCMeta):
    """
    Base class for image generators.
    """

    def __init__(
        self,
        batch_size: int = 1,
        size: SizeType = 512,
        quantize: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        start=None,
    ):
        """
        Base class for (image) generators.

        Args:
            batch_size (int, optional): The number of images per batch. Defaults to 1.
            size (Union[int, Tuple[int, int]], optional): The size of the image
                either a (width, height) tuple or a single size for a square
                image. Defaults to 512.
            quantize (int, optional): Model-specific quantizing. Defaults to 1.
            device (Optional[Union[str, torch.device]], optional): The device
                name or device on which to run. Defaults to None.

        Raises:
            ValueError: if any parameter is invalid
        """
        super().__init__()

        if batch_size < 1:
            raise ValueError("batch_size must be >0")
        self.batch_size = batch_size

        if isinstance(size, int):
            size = (size, size)
        self.size = (size[0] // quantize * quantize, size[1] // quantize * quantize)
        if self.size != size:
            _LOG.warn(f"Size quantized from {size} to {self.size}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self._z = Tensor()  # Default latent is the empty tensor

    @property
    def width(self) -> int:
        """
        The width of the generated image
        """
        return self.size[0]

    @property
    def height(self) -> int:
        """
        The height of the generated image
        """
        return self.size[1]

    @property
    def z(self) -> Tensor:
        """
        The latent tensor associated with this model. Often but not always (b, c, h, w).
        """
        return self._z.detach().clone()

    @z.setter
    def z(self, z) -> None:
        """
        Set the latent tensor via the model's state_dict. Must be the same shape
        as the existing tensor.
        """
        sd = self.state_dict()
        shape = sd["_z"].shape
        if z.shape != shape:
            raise ValueError(f"Invalid size {z.shape}, should be {shape}")
        sd["_z"] = z
        self.load_state_dict(sd)

    def generate_image(self, index: Optional[int] = None, **kwargs) -> Image:
        """
        Convenience to generate a single PIL image (which may be a grid of
        images if `batch_size` > 1) within a `no_grad()` block.

        Args:
            index (int): Specify which image of a batch to generate

        Returns:
            Image: A generated image
        """
        with torch.no_grad():
            batch = self.forward(**kwargs)
            if index is not None:
                batch = batch[index].unsqueeze(0)
            return T.functional.to_pil_image(make_grid(batch, nrow=4, padding=10))
