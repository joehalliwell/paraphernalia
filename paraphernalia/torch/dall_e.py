"""
Generate images with the discrete VAE component of DALL-E.
"""
from typing import Optional, Union

import dall_e
import PIL
import torch
import torchvision.transforms as T
from torch import Tensor

from paraphernalia.torch import one_hot_noise, one_hot_normalize
from paraphernalia.torch.generator import Generator
from paraphernalia.utils import download


class DALL_E(Generator):

    _NUM_CLASSES = 8192
    _SCALE = 8

    def __init__(self, tau: Optional[float] = 1.0, hard=False, start=None, **kwargs):
        """
        Image generator based on OpenAI's release of the discrete VAE component
        of DALL-E. Many parameters can be overridden via method arguments, so
        are best considered defaults.

        Args:
            start:
                Determines how to intitialize the hidden state.

        Attributes:
            tau (float):
                Gumbel softmax temperature parameter. Larger values make
                the underlying distribution more uniform.
            hard (bool):
                If true, then samples will be exactly one-hot
        """

        super().__init__(quantize=self._SCALE, **kwargs)

        self.tau = tau
        self.hard = hard

        self.decoder = dall_e.load_model(
            str(download("https://cdn.openai.com/dall-e/decoder.pkl")), self.device
        )

        # Initialize the state tensor
        if start is not None:
            z = self.encode(start)
            z = torch.cat([z.detach().clone() for _ in range(self.batch_size)])

        else:
            # Nice terrazzo style noise
            z = one_hot_noise(
                (
                    self.batch_size,
                    self._NUM_CLASSES,
                    self.height // self._SCALE,
                    self.width // self._SCALE,
                )
            )

        # Move to device and force to look like one-hot logits
        z = z.detach().clone()
        z = z.to(self.device)
        z = one_hot_normalize(z)
        self._z = torch.nn.Parameter(z)

    def forward(self, z=None, tau=None, hard=None) -> Tensor:
        """
        Generate a batch of images

        Returns:
            Tensor: An image batch tensor
        """
        if z is None:
            z = self._z
        if tau is None:
            tau = self.tau
        if hard is None:
            hard = self.hard

        samples = torch.nn.functional.gumbel_softmax(z, dim=1, tau=tau, hard=hard)

        buf = self.decoder(samples)
        buf = torch.sigmoid(buf[:, :3])
        buf = dall_e.unmap_pixels(buf.float())
        return buf

    def encode(self, img: Union[PIL.Image.Image, torch.Tensor]):
        """
        Encode an image or tensor.
        """
        if isinstance(img, PIL.Image.Image):
            img = PIL.ImageOps.pad(img, (self.width, self.height))
            img = torch.unsqueeze(T.functional.to_tensor(img), 0)

        with torch.no_grad():
            encoder = dall_e.load_model(
                str(download("https://cdn.openai.com/dall-e/encoder.pkl")), self.device
            )
            img = img.to(self.device)
            img = dall_e.map_pixels(img)
            z = encoder(img)

        return z.detach().clone()
