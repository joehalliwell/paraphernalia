"""Random number generator utilities."""

import logging
from typing import Any

_LOG = logging.getLogger(__name__)


_seed = None


def get_seed() -> int:
    """
    Returns:
        int: the last seed passed to :func:`set_seed`
    """
    return _seed


def set_seed(seed: Any) -> int:

    """
    Reset all known random number generators to use the provided seed.
    Currently:

    - `random.seed()`
    - `numpy.random.seed()`
    - `torch.manual_seed()`
    - `torch.cuda.manual_seed_all()`

    .. note::
        - Provided seeds are hashed before use. This allows you to pass in
          e.g. a string.

    Args:
        seed (Any): The seed to use

    """
    global _seed
    _seed = abs(hash(seed)) % (2 ** 32 - 1)
    _LOG.info(f"Setting global random seed to {_seed}")

    import random

    random.seed(_seed)

    # Numpy
    try:
        import numpy

        numpy.random.seed(_seed)
    except ImportError:
        pass

    # Torch
    try:
        import torch

        torch.manual_seed(_seed)
        torch.cuda.manual_seed_all(_seed)
    except ImportError:
        pass

    return _seed
