# TODO: Shift to poetry-version-plugin, once that's bedded in?
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)


def setup():
    # Check CUDA and GPU -- maybe upgrade CUDA?
    # Setup Google Drive (if running in Colaboratory)
    # Default project?
    pass


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
