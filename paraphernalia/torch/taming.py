"""
Generator based on Taming Transformers.

See also:
- https://colab.research.google.com/github/CompVis/taming-transformers/blob/master/scripts/reconstruction_usage.ipynb

"""

from dataclasses import dataclass
from typing import Union

import PIL
import torch
import torch.nn as nn
import torchvision.transforms as T
from omegaconf import OmegaConf
from taming.models.vqgan import GumbelVQ, VQModel
from torch.functional import Tensor

from paraphernalia.torch import clamp_with_grad
from paraphernalia.torch.generator import Generator
from paraphernalia.utils import cache_home, download


@dataclass
class TamingModel:
    name: str
    config_url: str
    checkpoint_url: str
    is_gumbel: bool
    scale: int


VQGAN_GUMBEL_F8 = TamingModel(
    "vqgan_gumbel_f8",
    "https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1",
    "https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fckpts%2Flast.ckpt&dl=1",
    True,
    8,
)

VQGAN_IMAGENET_F16_16384 = TamingModel(
    "vqgan_imagenet_f16_16384",
    "http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_16384.yaml",  # ImageNet 16384
    "http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_16384.ckpt",  # ImageNet 16384
    False,
    16,
)


class Taming(Generator):
    def __init__(
        self,
        model_spec: TamingModel = VQGAN_IMAGENET_F16_16384,
        start=None,
        batch_size=1,
        latent=32,
        device=None,
    ):
        super().__init__(device=device)

        self.batch_size = batch_size
        self.latent = latent
        self.model_spec = model_spec

        # TODO: Can we trade for a lighter dep?
        config = OmegaConf.load(
            download(model_spec.config_url, cache_home() / f"{model_spec.name}.yaml")
        )
        print(config)
        checkpoint = download(
            model_spec.checkpoint_url, cache_home() / f"{model_spec.name}.ckpt"
        )

        if model_spec.is_gumbel:
            model = GumbelVQ(**config.model.params)
        else:
            model = VQModel(**config.model.params)

        # Load checkpoint
        state = torch.load(checkpoint, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)

        # Disable training, ship to target device
        model.eval()
        model.to(self.device)
        self.model = model

        # Initialize z
        if start is None:
            z = torch.rand((batch_size, self.channels, latent, latent))
        else:
            z = self.encode(start)

        del model.encoder
        del model.loss

        z = z.detach()
        z = z.to(self.device)
        z = z.requires_grad_(True)
        self.z = nn.Parameter(z)

    def forward(self, z=None) -> Tensor:
        """
        Generate a batch of images

        Returns:
            Tensor: An image batch tensor
        """
        if z is None:
            z = self.z

        z = self.model.quantize(z)[0]
        z = self.model.decode(z)
        z = clamp_with_grad(z, -1.0, 1.0)
        z = (z + 1.0) / 2.0
        return z

    def encode(self, img: Union[PIL.Image.Image, torch.Tensor]):
        """
        Encode an image or tensor.
        """

        if isinstance(img, PIL.Image.Image):
            img = PIL.ImageOps.pad(
                img,
                (
                    self.latent * self.model_spec.scale,
                    self.latent * self.model_spec.scale,
                ),
            )
            img = torch.unsqueeze(T.functional.to_tensor(img), 0)

        img = img.to(self.device).mul(2.0).sub(1.0)
        return self.model.encode(img)[0]
