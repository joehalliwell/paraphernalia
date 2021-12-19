"""
Generate images using a network with `sin` activation functions.

See https://vsitzmann.github.io/siren/
"""


import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
from PIL.Image import Image
from torch import Tensor

from paraphernalia.torch import grid
from paraphernalia.torch.generator import Generator


class Siren(Generator):
    """Siren image generator.

    Args:
        size: Target size (square for now)
        omega: Fudge factor/weight multiplier. High (around 30) is good for
            image fitting. Lower values seem better for CLIP-guided
            generation. Defaults to 5.0.
        features: [description]. Defaults to 64.
        hidden_layers (int, optional): [description]. Defaults to 8.
        device ([type], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        omega: Optional[float] = 5.0,
        features: Optional[int] = 64,
        hidden_layers: Optional[int] = 8,
        start: Optional[Image] = None,
        **kwargs
    ):
        if start is not None:
            warnings.warn("Initial image is not supported")

        super().__init__(**kwargs)

        self.omega = omega
        self.dimensions = 2
        self.grid = (
            grid(self.height, self.width)
            .detach()
            .view(-1, self.dimensions)
            .to(self.device)
        )

        submodules = nn.ModuleList()

        # Input layer
        layer = nn.Linear(self.dimensions, features)
        u = 1.0 / self.dimensions
        nn.init.uniform_(layer.weight, -u, u)
        submodules.append(layer)

        # Hidden layers
        for _ in range(hidden_layers):
            layer = nn.Linear(features, features)
            u = math.sqrt(6 / features) / self.omega
            nn.init.uniform_(layer.weight, -u, u)
            submodules.append(layer)

        # Output layer
        layer = nn.Linear(features, 3)
        nn.init.uniform_(layer.weight, -u, u)
        submodules.append(layer)

        self.submodules = submodules.to(self.device)

    def forward(self, size: int = None) -> Tensor:
        """
        Generate an image of the (optionally) specified size.

        Args:
            size ([type], optional): [description]. Defaults to None.

        Returns:
            Tensor: An image batch tensor
        """
        if size is None:
            size = self.size
        h, w, c = self.height, self.width, 3
        if size == self.size:
            x = self.grid
        else:
            x = (
                grid(self.height, self.width)
                .detach()
                .view(-1, self.dimensions)
                .to(self.device)
            )
        for i, module in enumerate(self.submodules):
            x = module(x)
            if i == len(self.submodules) - 1:
                x = torch.sigmoid(x)
            else:
                x = torch.sin(self.omega * x)
        img = x.view((1, h, w, c)).permute(0, 3, 1, 2)
        return img
