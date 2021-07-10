import logging
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.utils import make_grid

from paraphernalia.torch import one_hot_noise

logger = logging.getLogger(__name__)


class Generator(nn.Module, metaclass=ABCMeta):
    """
    Base class for (image) generators.

    TODO: Refactor for non-VAE generators.
    """

    def __init__(self, device: Union[str, torch.device]):
        super(Generator, self).__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

    def generate_image(self, index: Optional[int] = None, **kwargs):
        """
        Convenience to generate a single PIL image within a no_grad block.

        Args:
            index (int):
        """
        with torch.no_grad():
            imgs = self.forward(**kwargs)
            if index is not None:
                imgs = imgs[index].unsqueeze(0)
            return T.functional.to_pil_image(make_grid(imgs, nrow=4, padding=10))
