"""
Paraphernalia is a collection of tools for making digital art.
"""
import logging
import os
from pathlib import Path

import xdg

# TODO: Shift to poetry-version-plugin, once that's bedded in?
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)

_LOG = logging.getLogger(__name__)

_BANNER = f"""
                           .-.                      .-.  _
                           : :                      : : :_;
 .---. .--. .--. .--. .---.: `-. .--..--.,-.,-..--. : : .-..--.
 : .; ' .; ;: ..' .; ;: .; : .. ' '_.: ..: ,. ' .; ;: :_: ' .; ;
 : ._.`.__,_:_; `.__,_: ._.:_;:_`.__.:_; :_;:_`.__,_`.__:_`.__,_;
 : :                  : :
 :_;                  :_;                               v{__version__}

"""


def setup():
    """
    Setup the library.
    """
    print(_BANNER)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s : %(message)s",
        level=logging.INFO,
        datefmt="%X",
    )
    if running_in_colab():
        setup_colab()


def cache_home(cache_home: str = None) -> Path:
    """
    Get the cache home for paraphernalia ensuring it exists.
    Defaults to $XDG_CACHE_HOME/paraphernalia
    """
    global _CACHE_HOME
    if cache_home is not None:
        _LOG.info(f"Setting cache home to {_CACHE_HOME}")
        _CACHE_HOME = Path(cache_home)
    os.makedirs(_CACHE_HOME, exist_ok=True)
    return _CACHE_HOME


_CACHE_HOME = None
cache_home(xdg.xdg_cache_home() / "paraphernalia")


def data_home(data_home: str = None) -> Path:
    """
    Get the data directory for paraphernalia ensuring it exists.
    Defaults to $XDG_DATA_HOME/paraphernalia

    Args:
        data_home (str, optional): If present sets the data home. Defaults to None.

    Returns:
        Path: path for the data home
    """
    global _DATA_HOME
    if data_home:
        _LOG.info(f"Setting data home to {_DATA_HOME}")
        _DATA_HOME = Path(data_home)
    os.makedirs(_DATA_HOME, exist_ok=True)
    return _DATA_HOME


_DATA_HOME = None
data_home(xdg.xdg_data_home() / "paraphernalia")


def setup_colab():
    """
    Standard setup for Colaboratory. Mounts Google drive under `/content/drive`
    """
    from google.colab import drive

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
    - `numpy.random.seed()`
    - `torch.manual_seed()`
    - `torch.cuda.manual_seed_all()`
    """
    import random

    _LOG.info(f"Setting global random seed to {seed}")
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
