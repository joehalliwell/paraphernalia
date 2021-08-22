import logging
from abc import ABCMeta
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.utils import make_grid

from paraphernalia.torch import clamp_with_grad, one_hot_noise

logger = logging.getLogger(__name__)

SizeType = Union[int, tuple[int, int]]


class Generator(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        batch_size: int = 1,
        size: SizeType = 512,
        quantize: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Base class for (image) generators.

        Args:
            batch_size (int, optional): The number of images per batch. Defaults to 1.
            size (Union[int, Tuple[int, int]], optional): The size of the image
                either a (width ,height) tuple or a single size for a square
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
            logger.warn(f"Size quantized from {size} to {self.size}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    def generate_image(self, index: Optional[int] = None, **kwargs):
        """
        Convenience to generate a single PIL image (which may be a grid of
        images if `batch_size` > 1) within a `no_grad()` block.

        Args:
            index (int): Specify which image of a batch to generate
        """
        with torch.no_grad():
            batch = self.forward(**kwargs)
            if index is not None:
                batch = batch[index].unsqueeze(0)
            return T.functional.to_pil_image(make_grid(batch, nrow=4, padding=10))
