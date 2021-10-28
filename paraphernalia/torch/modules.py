"""
A collection of utility PyTorch modules.
"""

import torch
import torch.nn as nn
from torch import Tensor

from paraphernalia.torch import cosine_similarity


class Constant(nn.Module):
    """
    Useful for testing?
    """

    def __init__(self, value: Tensor):
        super().__init__()
        self.value = value.detach().clone()

    def forward(self, x):
        return self.value


class WeightedSum(nn.Module):
    """
    More or less a weighted sum of named module outputs, but with special
    handling for negative weights.
    """

    def __init__(self, **components: nn.Module):
        """
        In order for the weighting to make sense, the components needs outputs
        with the same shape and meaning. For loss functions outputs should be
        in the range [0,1]
        """
        super().__init__()
        self.submodules = nn.ModuleDict(modules=components)
        self.weights = {name: 1.0 for name in components}
        self.total_weight = len(components)

    def set_weight(self, name: str, value: float):
        """
        Set the weight associated with a module

        Args:
            name (str): the name of the module
            value (float): the new weight
        """
        assert name in self.submodules, "Unknown name!"
        self.weights[name] = value
        self.total_weight = sum(self.weights[name] for name in self.weights)

    def forward(self, x: Tensor):
        """
        Compute the weighted loss
        """
        result = sum(
            m(x) * self.weights[n]
            for n, m in self.submodules.items()
            if self.weights[n] != 0  # No point running if weight is zero
        )
        bias = sum(abs(w) for w in self.weights.values() if w < 0)
        return (result + bias) / self.total_weight


class SimilarTo(nn.Module):
    """
    Cosine similarity test with mean pooling.
    """

    def __init__(self, targets: Tensor):
        """
        Args:
            targets: A tensor of dimension (targets, channels)
        """
        super().__init__()
        self.targets = targets.detach().clone()

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): A batch of vectors (batch, channels)
        """
        similarities = cosine_similarity(x, self.targets)
        return similarities.mean(dim=1)


class SimilarToAny(SimilarTo):
    """
    Cosine similarity test with max pooling.
    """

    def __init__(self, targets: Tensor):
        super().__init__(targets)

    def forward(self, x: Tensor):
        similarities = cosine_similarity(x, self.targets)
        return torch.max(similarities, dim=1)[0]
