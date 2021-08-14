from typing import Optional, Union

import numpy as np
import PIL
import torch
import torchvision.transforms as T

from paraphernalia.torch import clamp_with_grad
from paraphernalia.torch.generator import Generator


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
        size: int = 32,
        start=None,
        colors=[(0.1, 0.1, 0.1), (0.6, 0.1, 0.1), (1.0, 0.1, 0.1), (0.9, 0.9, 0.9)],
        device: Optional[Union[str, torch.device]] = None,
    ):

        if len(colors) > 256:
            raise ValueError("Palette must be <=256 colours")

        super().__init__(device)

        self.tau = 1.0
        self.hard = True
        self.size = (size * 16, size * 16)  # HACK
        self.colors = torch.Tensor(colors).float().to(self.device)
        # z = one_hot_noise((1, len(colors), size, size)).float()
        # Create a one-hot representation of the first palette index

        z = torch.full((1, size, size), 0)
        if start:
            z = self.encode(start)

        z = torch.nn.functional.one_hot(z, num_classes=len(colors)).float()
        z = z.permute(0, 3, 1, 2)
        z = torch.log(z + 0.001 / len(colors))
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

    def encode(self, img: Union[PIL.Image.Image, torch.Tensor]):
        """
        Encode an image or tensor.
        """

        img = PIL.ImageOps.pad(img, (self.size))
        # img = torch.unsqueeze(T.functional.to_tensor(img), 0)

        palette = PIL.Image.new("P", (1, 1))
        num_colors = len(self.colors)
        padded_colors = self.colors.cpu().numpy() * 255
        padded_colors = np.pad(padded_colors, [(0, 256 - num_colors), (0, 0)], "wrap")
        palette.putpalette(list(padded_colors.reshape(-1).astype("int")))
        quantized = img.quantize(colors=num_colors, palette=palette)
        z = torch.Tensor(np.mod(np.asarray(quantized), num_colors))
        z = z.long().unsqueeze(0)
        return z
