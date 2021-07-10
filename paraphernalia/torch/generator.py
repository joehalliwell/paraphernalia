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

    def __init__(self, shape, device: Union[str, torch.device]):
        super(Generator, self).__init__()

        self.shape = shape

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.z = None
        self.reset()

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

    def sample(self, z=None, tau=2.0, hard=True):
        """
        Sample a batch from the latent space.

        Args:
            z ([type], optional): [description]. Defaults to None.
            tau (float, optional): [description]. Defaults to 2.0.
            hard (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        if z is None:
            z = self.z
        if tau is None:
            tau = self.tau
        if hard is None:
            hard = self.hard

        samples = torch.nn.functional.gumbel_softmax(z, dim=1, tau=tau, hard=hard)
        return samples

    def reset(self, z=None, one_hot=True):
        """
        (Re)set the latent vector(s) associated with this generator.

        Args:
            z ([type], optional): [description]. Defaults to None.
            one_hot (bool, optional): [description]. Defaults to True.

        Raises:
            ValueError: [description]
            Exception: [description]

        Returns:
            [type]: [description]
        """
        if z is None:
            z = one_hot_noise(self.shape)
        z = z.detach()

        if len(z.shape) == 3:
            z = z.repeat(self.shape[0], 1, 1, 1)

        if len(z.shape) != 4:
            raise ValueError("z should have shape (batch_size, classes, height, width)")

        if z.shape[0] > self.shape[0]:
            logger.warn(f"Increasing batch size to {z.shape[0]}")

        # Force one hot
        if one_hot:
            z = torch.argmax(z, axis=1)
            z = (
                torch.nn.functional.one_hot(z, num_classes=self.shape[1])
                .permute(0, 3, 1, 2)
                .float()
            )
            # Add noise
            # TODO: is there any point to this?
            z = torch.log(z + 0.001 / self.shape[1])

        if self.z is not None:
            raise Exception("TODO: Support resetting z")

        z = z.to(self.device)
        self.z = nn.Parameter(z)
