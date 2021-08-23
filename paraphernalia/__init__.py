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
    Setup the library. Not really useful or used currently.
    """
    # Logging
    # Check CUDA and GPU -- maybe upgrade CUDA?
    # Default project?
    if running_in_colab():
        setup_colab()


def setup_colab():
    """
    Standard setup for Colaboratory.
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
        bool: True if running in Colaboratory
    """
    return os.environ.get("GITHUB_ACTIONS", False)
