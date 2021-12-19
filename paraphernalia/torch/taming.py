"""
Generate images with `Taming Transformers <https://github.com/CompVis/taming-transformers>`_.

See also:

- <https://colab.research.google.com/github/CompVis/taming-transformers/blob/master/scripts/reconstruction_usage.ipynb>

"""  # noqa

from dataclasses import dataclass
from typing import Union

import PIL
import torch
import torch.nn as nn
import torchvision.transforms as T
from omegaconf import OmegaConf
from taming.models.vqgan import GumbelVQ, VQModel
from torch.functional import Tensor

from paraphernalia import settings
from paraphernalia.torch import clamp_with_grad
from paraphernalia.torch.generator import Generator
from paraphernalia.utils import download


@dataclass
class TamingModel:
    """
    Specification for a published Taming Transformers model.
    """

    name: str
    "Slug for this model e.g. vqgan_gumbel_f8"
    config_url: str
    "URL of the associated config.yaml"
    checkpoint_url: str
    "URL of the associated checkpoint file"
    is_gumbel: bool
    "True iff this is a discrete model"
    scale: int
    "Generated pixels per latent space pixel"


VQGAN_GUMBEL_F8 = TamingModel(
    "vqgan_gumbel_f8",
    "https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1",  # noqa
    "https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fckpts%2Flast.ckpt&dl=1",  # noqa
    True,
    8,
)

VQGAN_IMAGENET_F16_16384 = TamingModel(
    "vqgan_imagenet_f16_16384",
    "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1",  # noqa
    "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1",  # noqa
    False,
    16,
)


class Taming(Generator):
    """
    Image generator based on a Taming Transformers model.
    """

    def __init__(
        self, model_spec: TamingModel = VQGAN_IMAGENET_F16_16384, start=None, **kwargs
    ):
        """
        Args:
            model_spec (TamingModel, optional): Defaults to
                VQGAN_IMAGENET_F16_16384.
            start ([type], optional): Defaults to None.
        """
        super().__init__(quantize=model_spec.scale, **kwargs)

        self.channels = 256  # Always?
        self.model_spec = model_spec

        # TODO: Can we trade for a lighter dep?
        config = OmegaConf.load(
            download(
                model_spec.config_url, settings().cache_home / f"{model_spec.name}.yaml"
            )
        )
        # print(config)
        checkpoint = download(
            model_spec.checkpoint_url, settings().cache_home / f"{model_spec.name}.ckpt"
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
        # Freeze model weights
        for p in model.parameters():
            p.requires_grad = False
        self.model = model

        # Initialize z
        if start is None:
            z = torch.rand(
                (
                    self.batch_size,
                    self.channels,
                    self.height // model_spec.scale,
                    self.width // model_spec.scale,
                )
            )
        else:
            z = self.encode(start)

        del model.encoder
        del model.loss

        z = z.detach().clone()
        z = z.to(self.device)

        z = z.requires_grad_(True)
        self._z = nn.Parameter(z)

    def forward(self, z=None) -> Tensor:
        """
        Generate a batch of images

        Returns:
            Tensor: An image batch tensor
        """
        if z is None:
            z = self._z

        z = self.model.quantize(z)[0]
        z = self.model.decode(z)
        z = clamp_with_grad(z, -1.0, 1.0)
        z = (z + 1.0) / 2.0
        return z

    def encode(self, img: Union[PIL.Image.Image, Tensor]) -> Tensor:
        """
        Encode an image.
        """

        if isinstance(img, PIL.Image.Image):
            img = PIL.ImageOps.pad(
                img,
                (
                    self.width,
                    self.height,
                ),
            )
            img = torch.unsqueeze(T.functional.to_tensor(img), 0)

        img = img.to(self.device).mul(2.0).sub(1.0)
        return self.model.encode(img)[0]
