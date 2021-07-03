import math
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.utils import make_grid
from paraphernalia.torch.generator import Generator
from paraphernalia.utils import grid


class Siren(Generator):
    def __init__(self, size, omega=5.0, features=64, hidden_layers=8, device=None):
        """See https://vsitzmann.github.io/siren/

        Args:
            size ([type]): [description]
            omega (float, optional): [description]. Defaults to 5.0.
            features (int, optional): [description]. Defaults to 64.
            hidden_layers (int, optional): [description]. Defaults to 8.
            device ([type], optional): [description]. Defaults to None.
        """
        super(Siren, self).__init__(device)

        self.size = size
        self.omega = omega
        self.dimensions = 2
        self.grid = grid(self.size, dimensions=self.dimensions).detach().to(self.device)

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

    def forward(self, size=None):
        if size is None:
            size = self.size
        h, w, c = size, size, 3
        if size == self.size:
            x = self.grid
        else:
            x = grid(size, dimensions=self.dimensions).detach().to(self.device)
        for i, module in enumerate(self.submodules):
            x = module(x)
            if i == len(self.submodules) - 1:
                x = torch.sigmoid(x)
            else:
                x = torch.sin(self.omega * x)
        x = x.view((1, h, w, c)).permute(0, 3, 1, 2)
        return x
