import logging
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.utils import make_grid

from paraphernalia.torch import clamp_with_grad, one_hot_noise

logger = logging.getLogger(__name__)


class Generator(nn.Module, metaclass=ABCMeta):
    """
    Base class for (image) generators.

    TODO: Refactor for non-VAE generators.
    """

    def __init__(self, device: Optional[Union[str, torch.device]]):
        super(Generator, self).__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

    def generate_image(self, index: Optional[int] = None, **kwargs):
        """
        Convenience to generate a single PIL image (which may be a grid of
        images if batch_size > 1) within a no_grad block.

        Args:
            index (int): Specify which image of a batch to generate
        """
        with torch.no_grad():
            imgs = self.forward(**kwargs)
            if index is not None:
                imgs = imgs[index].unsqueeze(0)
            return T.functional.to_pil_image(make_grid(imgs, nrow=4, padding=10))


class Direct(Generator):
    """
    A direct generator i.e. a directly trainable RGB tensor.
    """

    def __init__(
        self,
        size: int = 128,
        start=None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(device)
        if start is not None:
            z = T.functional.to_tensor(start).unsqueeze(0)
            z = T.functional.resize(z, size=(size, size))
            z = (2.0 * z) - 1.0
        else:
            z = torch.randn((1, 3, size, size))

        z = z.to(self.device)
        self.z = torch.nn.Parameter(z)

    def forward(self):
        # z = self.z
        # z = torch.sigmoid(z)
        z = clamp_with_grad(self.z, -1.0, 1.0)
        z = (z + 1.0) / 2.0
        return z


class DirectPalette(Generator):
    """
    A palettized generator using gumbel sampling versus a provided palette.

    """

    def __init__(
        self,
        size: int = 256,
        start=None,
        colors=[(0.1, 0.1, 0.1), (0.6, 0.1, 0.1), (1.0, 0.1, 0.1), (0.9, 0.9, 0.9)],
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(device)
        self.tau = 1.0
        self.hard = True
        self.colors = torch.Tensor(colors).float().to(self.device)
        # z = one_hot_noise((1, len(colors), size, size)).float()
        # Create a one-hot representation of the first palette index
        z = torch.full((1, size, size), 0)
        z = torch.nn.functional.one_hot(z, len(colors)).float()
        z = z.permute(0, 3, 1, 2)
        z = torch.log(z + 0.1 / len(colors))
        z = z.to(self.device)
        print(z.shape)
        self.z = torch.nn.Parameter(z)

    def forward(self, tau=None, hard=None):
        if tau is None:
            tau = self.tau
        if hard is None:
            hard = self.hard
        sample = torch.nn.functional.gumbel_softmax(self.z, dim=1, tau=tau, hard=hard)
        img = torch.einsum("bchw,cs->bshw", sample, self.colors)
        return img
