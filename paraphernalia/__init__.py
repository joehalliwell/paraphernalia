"""
Paraphernalia is a collection of tools for making digital art.
"""
import os

# TODO: Shift to poetry-version-plugin, once that's bedded in?
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)


def setup():
    """
    Setup the library. Not very useful or much used currently.
    """
    # Logging
    # Check CUDA and GPU -- maybe upgrade CUDA?
    # Default project?
    if running_in_colab():
        setup_colab()


def setup_colab():
    """
    Standard setup for Colaboratory. Mounts Google drive under `/content/drive`
    """
    from google.colab import drive

    from paraphernalia.utils import data_home

    drive.mount("/content/drive")
    data_home("/content/drive/MyDrive/Paraphernalia")


def running_in_colab():
    """
    True if running in Colaboratory.

    See:
    - https://stackoverflow.com/questions/53581278/test-if-notebook-is-running-on-google-colab

    Returns:
        bool: True if running in Colaboratory
    """
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return "google.colab" in str(get_ipython())


def running_in_github_action():
    """
    True if running as Github Action.

    Returns:
        bool: True if running in a Github action
    """
    return os.environ.get("GITHUB_ACTIONS", False)


def running_in_jupyter():
    """
    True if and only if running within Jupyter

    Returns:
        bool: True iff running in Jupyter
    """
    try:
        from IPython import get_ipython
    except:
        return False

    ip = get_ipython()
    if not ip:
        return False

    elif "IPKernelApp" not in get_ipython().config:  # pragma: no cover
        raise ImportError("console")
        return False

    elif "VSCODE_PID" in os.environ:  # pragma: no cover
        raise ImportError("vscode")
        return False

    else:  # pragma: no cover
        return True


def seed(seed):
    """
    Set all known random number generators with the specified value. Currently:
    - `random.seed()`
    - `torch.manual_seed()`
    """
    import random

    random.seed(seed)

    # Numpy
    try:
        import numpy

        numpy.random.seed(seed)
    except:
        pass

    # Torch
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass
