"""
A collection of utility PyTorch modules.
"""
from typing import Any, List

from os import initgroups
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from paraphernalia.torch import cosine_similarity, regroup


class AdaptiveMultiLoss(nn.Module):
    """
    Automatic loss balancing.

    .. seealso::

        - https://arxiv.org/abs/1705.07115

    """

    def __init__(self, num_losses: int):
        """
        Args:
            components ([type]): [description]
        """
        super().__init__()
        if num_losses < 2:
            raise ValueError("Must provide more than two loss functions")
        self.weights = nn.parameter.Parameter(torch.zeros(num_losses))

    def forward(self, losses: Tensor) -> float:
        """
        Args:
            losses (Tensor): The losses to balance

        Returns:
            float: Combined loss
        """
        assert losses.shape == self.weights.shape
        precision = torch.exp(self.weights)
        result = precision * losses + self.weights
        return result.mean()


class Parallel(nn.Module):
    """
    A module that runs a number of submodule in parallel and collects their
    outputs into a list.
    """

    def __init__(self, components):
        """
        Args:
            components (List[nn.Module]): a list of submdodules to run in parallel
        """
        super().__init__()
        self.components = nn.ModuleList(components)

    def forward(self, *inputs) -> List:
        """
        Run each submodule on input, and accumulate the outputs into a list.

        Returns:
            List: the outputs of each module
        """
        outputs = [component(*inputs) for component in self.components]
        return outputs


class Constant(nn.Module):
    """
    A module that returns a constant value, ignoring any inputs.
    """

    def __init__(self, value: Tensor):
        super().__init__()
        self.value = value.detach().clone()

    def forward(self, *ignored):
        """
        Return the constant value, ignoring any inputs.

        Args:
            ignored: any number of inputs which will be ignored

        Returns:
            Tensor: the specific constant value
        """
        return self.value


class WeightedSum(nn.Module):
    """
    A weighted sum of named module outputs.
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

    def set_weight(self, name: str, weight: float):
        """
        Set the weight associated with a module

        Args:
            name (str): the name of the module
            value (float): the new weight
        """
        assert weight >= 0, "Weights must be positive"
        assert name in self.submodules, "Unknown name!"
        self.weights[name] = weight
        self.total_weight = sum(self.weights[name] for name in self.weights)

    def forward(self, x: Tensor):
        """
        Compute the weighted loss.
        """
        result = sum(
            m(x) * self.weights[n]
            for n, m in self.submodules.items()
            if self.weights[n] != 0  # No point running if weight is zero
        )
        return result / self.total_weight


class Not(nn.Module):
    """
    Negate a similarity module
    """

    def __init__(self, submodule: nn.Module):
        super().__init__()
        self.submodule = submodule

    def forward(self, x: Tensor):
        return 1.0 - self.submodule(x)


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

    def forward(self, candidates: Tensor):
        """
        Args:
            x (Tensor): A batch of vectors (batch, channels)
        """
        similarities = cosine_similarity(candidates, self.targets)
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


class Perceiver(torch.nn.Module):
    """
    A perceiver binds together:
    - A list of image transformations/augmentations
    - An evaluator that maps vectors to quality scores (canonically between 0 and 1)

    """

    def __init__(
        self,
        transforms: List[torch.nn.Module],
        evaluator: torch.nn.Module,
    ) -> None:
        """
        Initialize a perceiver.

        Args:
            transforms ([type]): a list of transforms to apply
            evaluator ([type]): an evaluator to determine how good images are
        """
        super().__init__()
        self.transforms = transforms
        self.evaluator = evaluator

    def forward(self: Tensor, imgs: Tensor) -> Tensor:
        """
        Evaluate an image batch, by applying transforms, encoding the resulting views
        and running the evaluator on them.

        Evaluation results are averaged over the transforms, to produce a single
        evaluation per image.
        """
        # Save the original batch size
        batch_size = imgs.shape[0]

        # Encode images
        transformed = regroup([transform(imgs) for transform in self.transforms])

        # Per-transform quality
        quality = self.evaluator(transformed)

        # Per-image mean similarity
        mean_quality = [
            chunk.mean() for chunk in torch.chunk(quality, chunks=batch_size)
        ]
        assert len(mean_quality) == batch_size
        return torch.stack(mean_quality)
