"""
Generate images "directly" i.e. without a latent space.
"""

from typing import Optional, Union

import numpy as np
import PIL
import torch
import torchvision.transforms as T
from torch import Tensor

from paraphernalia.torch import clamp_with_grad, one_hot_noise, one_hot_normalize
from paraphernalia.torch.generator import Generator


class Direct(Generator):
    """
    A direct generator i.e. a directly trainable RGB tensor.
    """

    def __init__(self, start=None, scale=1, **kwargs):
        """
        Initialize a direct generator.

        Individual pixels in the latent space are upscaled via `scale`. If greater
        than the default of 1, this yields a pixel-art effect.

        Args:
            start ([type], optional): [description]. Defaults to None.
            scale (int, optional): Pixel size. Defaults to 1.
        """
        super().__init__(quantize=scale, **kwargs)
        h = self.height // scale
        w = self.width // scale
        if start is not None:
            z = T.functional.to_tensor(start).unsqueeze(0)
            z = T.functional.resize(z, size=(h, w))
            z = torch.log(z) - torch.log(1 - z)  # Inverse sigmoid
        else:
            z = 0.05 * torch.randn((self.batch_size, 3, h, w))

        z = z.to(self.device)
        self._z = torch.nn.Parameter(z)

    def forward(self):
        """
        Generate a batch of images.
        """
        img = torch.sigmoid(self._z)
        return T.functional.resize(
            img, size=(self.height, self.width), interpolation=PIL.Image.NEAREST
        )


class DirectPalette(Generator):
    """
    A palettized generator using gumbel sampling versus a provided palette.
    """

    def __init__(
        self,
        start=None,
        colors=[(0.1, 0.1, 0.1), (0.6, 0.1, 0.1), (1.0, 0.1, 0.1), (0.9, 0.9, 0.9)],
        scale=1,
        **kwargs,
    ):

        super().__init__(quantize=scale, **kwargs)

        if len(colors) > 256:
            raise ValueError("Palette must be <=256 colours")
        self.colors = torch.Tensor(colors).float().to(self.device)

        self.scale = scale
        self.tau = 1.0
        self.hard = True

        z = torch.full((1, self.height // scale, self.width // scale), 0)
        if start:
            z = self.encode(start)

        z = torch.nn.functional.one_hot(z, num_classes=len(colors)).float()
        z = z.permute(0, 3, 1, 2)
        z = torch.log(z + 0.001 / len(colors))
        z = z.to(self.device)
        self._z = torch.nn.Parameter(z)

    def forward(self, tau=None, hard=None):
        """
        Generate a batch of images.
        """
        if tau is None:
            tau = self.tau
        if hard is None:
            hard = self.hard
        sample = torch.nn.functional.gumbel_softmax(self._z, dim=1, tau=tau, hard=hard)
        img = torch.einsum("bchw,cs->bshw", sample, self.colors)
        return T.functional.resize(
            img, size=(self.height, self.width), interpolation=PIL.Image.NEAREST
        )

    def encode(self, img: Union[PIL.Image.Image, torch.Tensor]):
        """
        Encode an image or tensor.
        """

        img = PIL.ImageOps.pad(
            img, (self.width // self.scale, self.height // self.scale)
        )

        palette = PIL.Image.new("P", (1, 1))
        num_colors = len(self.colors)
        padded_colors = self.colors.cpu().numpy() * 255
        padded_colors = np.pad(padded_colors, [(0, 256 - num_colors), (0, 0)], "wrap")
        palette.putpalette(list(padded_colors.reshape(-1).astype("int")))
        quantized = img.quantize(colors=num_colors, palette=palette)
        z = torch.Tensor(np.mod(np.asarray(quantized), num_colors))
        z = z.long().unsqueeze(0)
        return z


class DirectTileset(Generator):
    """
    A generator using gumbel sampling versus a provided tile atlas.
    """

    def __init__(self, atlas: Optional[Tensor] = None, scale=1, z=None, **kwargs):
        """
        Initialize a discrete direct tileset generator.

        Args:
            atlas (Tensor): TODO

        Raises:
            ValueError: if the atlas is the wrong size
        """
        # This shouldn't really be optional, but it makes testing easier to have
        # no required arguments beyond what Generator needs.
        if atlas is None:
            atlas = torch.rand((16, 3, 16, 16))

        if atlas.shape[2] != atlas.shape[3]:
            raise ValueError(f"Tiles must be square, but atlas has shape {atlas.shape}")

        self.num_tiles = atlas.shape[0]
        self.tile_size = atlas.shape[2]
        self.scale = scale
        self.tau = 1.0
        self.hard = True

        super().__init__(quantize=self.tile_size * self.scale, **kwargs)

        atlas = atlas.reshape((-1, 3 * self.tile_size * self.tile_size))
        self.atlas = atlas.to(self.device)

        if z is None:
            z = one_hot_noise(
                (
                    self.batch_size,
                    self.num_tiles,
                    self.height // (self.tile_size * self.scale),
                    self.width // (self.tile_size * self.scale),
                )
            )

        z = one_hot_normalize(z)
        z = z.detach().clone().to(self.device)
        self._z = torch.nn.Parameter(z)

    def forward(self, tau=None, hard=None):
        """
        Generate a batch of images.
        """
        if tau is None:
            tau = self.tau
        if hard is None:
            hard = self.hard
        sample = torch.nn.functional.gumbel_softmax(self._z, dim=1, hard=hard, tau=tau)
        img = torch.einsum("bchw,cs->bshw", sample, self.atlas)
        img = torch.nn.functional.pixel_shuffle(img, self.tile_size)
        return T.functional.resize(
            img, size=(self.height, self.width), interpolation=PIL.Image.NEAREST
        )


# TODO: https://www.c64-wiki.com/wiki/Standard_Character_Mode
