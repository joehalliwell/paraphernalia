"""
Generator based on Taming Transformers.

See also:
- https://colab.research.google.com/github/CompVis/taming-transformers/blob/master/scripts/reconstruction_usage.ipynb

"""

from dataclasses import dataclass
from typing import Union

import PIL
import torch
import torchvision.transforms as T
from _typeshed import NoneType
from omegaconf import OmegaConf
from taming.models.vqgan import GumbelVQ, VQModel
from torch.functional import Tensor

from paraphernalia.torch.generator import Generator
from paraphernalia.utils import cache_home, download


@dataclass
class TamingModel:
    name: str
    config_url: str
    checkpoint_url: str
    is_gumbel: bool


VQGAN_GUMBEL_F8 = TamingModel(
    "vqgan_gumbel_f8",
    "https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1",
    "https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fckpts%2Flast.ckpt&dl=1",
    True,
)


class Taming(Generator):
    def __init__(self, shape, model: TamingModel, device=None):
        super().__init__(shape, device=device)

        # TODO: Can we trade for a lighter dep?
        config = OmegaConf.load(
            download(model.config_url, cache_home() / f"{model.name}.yaml")
        )
        checkpoint = download(model.checkpoint_url, cache_home() / f"{model.name}.ckpt")

        if model.is_gumbel:
            model = GumbelVQ(**config.model.params)
        else:
            model = VQModel(**config.model.params)

        # Load checkpoint
        state = torch.load(checkpoint, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)

        # Disable training, ship to target device
        model.eval()
        model.to(self.device)
        self._model = model

    def forward(self, z=None) -> Tensor:
        """
        Generate a batch of images

        Returns:
            Tensor: An image batch tensor
        """
        if z is None:
            z = self.z
        buf = self._model.decode(z)
        buf = torch.clamp(buf, -1.0, 1.0)
        buf = (buf + 1.0) / 2.0
        return buf

    def encode(self, img: Union[PIL.Image.Image, torch.Tensor]):
        """
        Encode an image or tensor.
        """

        if isinstance(img, PIL.Image.Image):
            img = PIL.ImageOps.pad(img, (self.latent * 8, self.latent * 8))
            img = torch.unsqueeze(T.functional.to_tensor(img), 0)

        img = 2.0 * img - 1
        self._model.encode(img)
