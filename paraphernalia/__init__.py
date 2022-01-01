"""Paraphernalia is a collection of tools for making digital art."""
import logging
import os
import subprocess
import sys
from typing import Optional

from paraphernalia._project import Project, project
from paraphernalia._random import get_seed, set_seed
from paraphernalia._settings import Settings, settings

# TODO: Shift to poetry-version-plugin, once that's bedded in?
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata  # type: ignore


_LOG = logging.getLogger(__name__)

__version__ = importlib_metadata.version(__name__)

_BANNER = f"""
                           .-.                      .-.  _
                           : :                      : : :_;
 .---. .--. .--. .--. .---.: `-. .--..--.,-.,-..--. : : .-..--.
 : .; ' .; ;: ..' .; ;: .; : .. ' '_.: ..: ,. ' .; ;: :_: ' .; ;
 : ._.`.__,_:_; `.__,_: ._.:_;:_`.__.:_; :_;:_`.__,_`.__:_`.__,_;
 : :                  : :
 :_;                  :_;                                  v{__version__}


"""


__all__ = [
    "setup",
    "setup_logging",
    "get_seed",
    "set_seed",
    "settings",
    "project",
    "Settings",
    "Project",
]


def setup() -> None:
    """
    Set up the library for interactive use by:

    - Configuring logging
    - Printing a vanity banner and some system information
    - (If running in Colaboratory) calling :func:`setup_colab`
    """
    setup_logging()

    if running_in_colab():
        setup_colab()

    setup_banner()


def setup_logging(use_rich: Optional[bool] = None) -> None:
    """
    Setup basic logging.

    Args:
        use_rich (bool, optional): use the pretty rich log handler if available.
        Defaults to value in settings.
    """
    handlers = None
    fmt = "%(asctime)s %(levelname)s %(name)s : %(message)s"

    use_rich = settings().use_rich if use_rich is None else use_rich

    if use_rich:
        try:
            import rich
            from rich.highlighter import NullHighlighter
            from rich.logging import RichHandler

            rich.get_console().width = 120  # Default is 93, which is too small
            fmt = "%(message)s"
            handlers = [
                RichHandler(highlighter=NullHighlighter(), rich_tracebacks=True)
            ]
        except ImportError:
            pass

    logging.basicConfig(
        level=logging.INFO, format=fmt, datefmt="[%X]", handlers=handlers
    )


def setup_banner():
    """Log a banner and some system information."""

    _LOG.info(_BANNER)
    python_version = sys.version.replace("\n", " ")
    _LOG.info(f"  Python: {python_version}")
    _LOG.info(f"     GPU: {get_gpu_name()} (CUDA: {get_cuda_version()})")
    _LOG.info(f"    Seed: {get_seed()}")
    _LOG.info(f" Creator: {settings().creator}")
    _LOG.info(f"Projects: {settings().project_home}")


def setup_colab():  # pragma: no cover
    """
    Standard setup for Colaboratory:

    - Ensures Google drive is mounted under `/content/drive`
    - Configures :func:`project_home` to use it
    - Adds the ``data_table`` and and ``tensorboard`` extensions
    """
    # Mount drive and use it
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
    settings().project_home = "/content/drive/MyDrive/Paraphernalia"

    # Load extensions
    from IPython import get_ipython

    get_ipython().magic("load_ext google.colab.data_table")
    get_ipython().magic("load_ext tensorboard")


def running_in_colab():
    """
    True if running in Colaboratory.

    See:

    - https://stackoverflow.com/questions/53581278/test-if-notebook-is-running-on-google-colab

    Returns:
        bool: True if running in Colaboratory
    """  # noqa
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
    except ImportError:
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
    except Exception:
        return None


def get_gpu_name() -> Optional[str]:
    """
    Returns:
        the name of the GPU if available, or None.
    """
    try:
        import torch

        return torch.cuda.get_device_name()
    except (ImportError, RuntimeError):
        return None


if settings().auto_setup:
    setup()
