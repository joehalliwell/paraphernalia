from abc import ABCMeta, abstractmethod
from typing import Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.utils import make_grid


class Generator(nn.Module, metaclass=ABCMeta):
    """
    Base class for (image) generators.
    """

    def __init__(self, device: Union[str, torch.device]):
        super(Generator, self).__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

    def generate_image(self, **kwargs):
        """
        Convenience to generate a single PIL image within a no_grad block.
        """
        with torch.no_grad():
            imgs = self.forward(**kwargs)
            return T.functional.to_pil_image(make_grid(imgs, nrow=4, padding=10))
