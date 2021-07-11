# TODO: Shift to poetry-version-plugin, once that's bedded in?
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)


def setup():
    # Logging
    # Check CUDA and GPU -- maybe upgrade CUDA?
    # Default project?
    if running_in_colab():
        setup_colab()


def setup_colab():
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
        [type]: [description]
    """
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return "google.colab" in str(get_ipython())
