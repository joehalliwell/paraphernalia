"""
Paraphernalia is a collection of tools for making digital art.
"""
import logging
import os
import subprocess
import sys
from pathlib import Path
from time import time
from typing import Any, Optional

import xdg

# TODO: Shift to poetry-version-plugin, once that's bedded in?
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)

_LOG = logging.getLogger(__name__)
_BANNER = f"""Welcome to...

                           .-.                      .-.  _
                           : :                      : : :_;
 .---. .--. .--. .--. .---.: `-. .--..--.,-.,-..--. : : .-..--.
 : .; ' .; ;: ..' .; ;: .; : .. ' '_.: ..: ,. ' .; ;: :_: ' .; ;
 : ._.`.__,_:_; `.__,_: ._.:_;:_`.__.:_; :_;:_`.__,_`.__:_`.__,_;
 : :                  : :
 :_;                  :_;                               v{__version__}


"""


def setup() -> None:
    """
    Set up the library for interactive use by:

    - Configuring logging
    - Printing a vanity banner and some system information
    - (If running in Colaboratory) calling :func:`setup_colab`
    """
    setup_logging()

    _LOG.info(_BANNER)
    python_version = sys.version.replace("\n", " ")
    _LOG.info(f"Python: {python_version}")
    _LOG.info(f"   GPU: {get_gpu_name()} (CUDA: {get_cuda_version()})")
    _LOG.info(f"  Seed: {seed()}")

    if running_in_colab():
        setup_colab()


def setup_logging() -> None:
    """Basic logging setup"""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s : %(message)s",
        level=logging.INFO,
        datefmt="%X",
    )


def setup_colab():  # pragma: no cover
    """
    Standard setup for Colaboratory:

    - Ensures Google drive is mounted under `/content/drive`
    - Configures :func:`data_home` to use it
    - Adds the ``data_table`` and and ``tensorboard`` extensions
    """
    # Mount drive and use it
    from google.colab import drive

    drive.mount("/content/drive")
    data_home("/content/drive/MyDrive/Paraphernalia")

    # Load extensions
    from IPython import get_ipython

    get_ipython().magic("load_ext google.colab.data_table")
    get_ipython().magic("load_ext tensorboard")


def seed(seed: Optional[Any] = None) -> int:

    """
    Get the current random seed i.e. the last seed that was set.

    Alternatively, when a seed is provided, set all known random number generators
    using it. Currently:

    - `random.seed()`
    - `numpy.random.seed()`
    - `torch.manual_seed()`
    - `torch.cuda.manual_seed_all()`

    .. note::

        - On load this module sets the random seed to the current time in seconds
          since the epoch.
        - Provided seeds are hashed before use. This allows you to pass in e.g. a string.

    Args:
        seed (Optional[Any]): The seed. Defaults to None.

    Returns:
        int: The current random seed
    """
    global _seed

    if seed is not None:
        _seed = hash(seed)
        _LOG.info(f"Setting global random seed to {_seed}")

        import random

        random.seed(_seed)

        # Numpy
        try:
            import numpy

            numpy.random.seed(_seed)
        except:
            pass

        # Torch
        try:
            import torch

            torch.manual_seed(_seed)
            torch.cuda.manual_seed_all(_seed)
        except:
            pass

    return _seed


_seed = None
seed(int(time()))


def cache_home(cache_home: Optional[str] = None) -> Path:
    """
    Get the cache home for paraphernalia ensuring it exists.
    Defaults to $XDG_CACHE_HOME/paraphernalia

    Args:
        cache_home (Optional[str]): If present sets the cache home. Defaults to None.

    Returns:
        Path: path for the cache home

    """
    global _cache_home
    if cache_home is not None:
        _LOG.info(f"Setting cache home to {cache_home}")
        _cache_home = Path(cache_home)
    os.makedirs(_cache_home, exist_ok=True)
    return _cache_home


_cache_home = None
cache_home(xdg.xdg_cache_home() / "paraphernalia")


def data_home(data_home: Optional[str] = None) -> Path:
    """
    Get the data directory for paraphernalia ensuring it exists.
    Defaults to $XDG_DATA_HOME/paraphernalia

    Args:
        data_home (Optional[str]): If present sets the data home. Defaults to None.

    Returns:
        Path: path for the data home
    """
    global _data_home
    if data_home is not None:
        _LOG.info(f"Setting data home to {data_home}")
        _data_home = Path(data_home)
    os.makedirs(_data_home, exist_ok=True)
    return _data_home


_data_home = None
data_home(xdg.xdg_data_home() / "paraphernalia")


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


def running_in_github_action() -> bool:
    """
    True if running as Github Action.

    Returns:
        bool: True if running in a Github action
    """
    return bool(os.environ.get("GITHUB_ACTIONS", False))


def running_in_jupyter() -> bool:
    """
    True if running within Jupyter, and *not*:

    - A random python program
    - A console-based IPython shell
    - VSCode's interactive mode

    Returns:
        bool: True iff running in Jupyter
    """
    try:
        from IPython import get_ipython
    except:
        return False

    ip = get_ipython()
    # No species of iPython
    if not ip:
        return False

    # Console
    elif "IPKernelApp" not in get_ipython().config:  # pragma: no cover
        return False

    # VS Code
    elif "VSCODE_PID" in os.environ:  # pragma: no cover
        return False

    else:  # pragma: no cover
        return True


def get_cuda_version() -> Optional[str]:
    """
    Returns:
        the CUDA/nvcc version string e.g. 10.0, or None
    """
    try:
        return [
            s
            for s in subprocess.check_output(["nvcc", "--version"])
            .decode("UTF-8")
            .split(", ")
            if s.startswith("release")
        ][0].split(" ")[-1]
    except:
        return None


def get_gpu_name() -> Optional[str]:
    """
    Returns:
        the name of the GPU if available, or None.
    """
    try:
        import torch

        return torch.cuda.get_device_name()
    except:
        return None
