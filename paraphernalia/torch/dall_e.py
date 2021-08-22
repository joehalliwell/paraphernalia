from typing import Optional, Union

import dall_e
import PIL
import torch
import torchvision.transforms as T
from torch import Tensor

from paraphernalia.torch import one_hot_noise
from paraphernalia.torch.generator import Generator
from paraphernalia.utils import download


class DALL_E(Generator):
    """
    Image generator based on OpenAI's release of the discrete VAE component
    of DALL-E. Many parameters can be overridden via method arguments, so
    are best considered defaults.

    Args:
        batch_size (int):
            How many independent latent vectors to use in parallel. This has a
            huge impact on memory use.
        start ():
            Determines how to intitialize the hidden state.

    Attributes:
        tau (float):
            Gumbel softmax temperature parameter. Larger values make
            the underlying distribution more uniform.
        hard (bool):
            If true, then samples will be exactly one-hot
    """

    _NUM_CLASSES = 8192
    _SCALE = 8

    def __init__(
        self, tau: Optional[float] = 1.0, z=None, hard=False, start=None, **kwargs
    ):
        super().__init__(quantize=self._SCALE, **kwargs)

        self.tau = tau
        self.hard = hard

        self.decoder = dall_e.load_model(
            str(download("https://cdn.openai.com/dall-e/decoder.pkl")), self.device
        )

        # Initialize the state tensor
        # Option 1: From a provided tensor
        if z is not None:
            if start is not None:
                raise ValueError("If providing z, don't provide start image")
            if len(z.shape) == 3:
                z = z.unsqueeze(0)
            if len(z.shape) != 4:
                raise ValueError("z must be rank 4 (b, c, h, w)")
            z = torch.nn.functional.interpolate(
                z, size=(self.height // self._SCALE, self.width // self._SCALE)
            )
            # TODO: Handle batch size
            self.z = torch.nn.Parameter(z)
            return

        # Option 2: A provided PIL or Tensor image
        elif start is not None:
            z = self.encode(start)
            z = torch.cat([z.detach().clone() for _ in range(self.batch_size)])

        # Option 3: Random noise (this doesn't work very well)
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
        z = z.to(self.device)
        z = torch.argmax(z, axis=1)
        z = (
            torch.nn.functional.one_hot(z, num_classes=self._NUM_CLASSES)
            .permute(0, 3, 1, 2)
            .float()
        )
        z = torch.log(z + 0.001 / self._NUM_CLASSES)
        self.z = torch.nn.Parameter(z)

    def forward(self, z=None, tau=None, hard=None) -> Tensor:
        """
        Generate a batch of images.
        """
        if z is None:
            z = self.z
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
