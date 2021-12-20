"""Global settings."""
import logging
from pathlib import Path
from time import time
from typing import Set

from pydantic import BaseSettings, Field, validator
from xdg import XDG_CACHE_HOME, XDG_CONFIG_HOME, XDG_DATA_HOME

from paraphernalia._random import set_seed
from paraphernalia.utils import ensure_dir_exists

_LOG = logging.getLogger(__name__)


_settings = None


def settings(reload=False) -> "Settings":
    """
    Get the global settings object. Configuration is read from the environment
    and (optionally) a `paraphernalia.env` file in the user's configuration
    directory.

    Example:

    >>> import paraphernalia as pa
    >>> pa.settings().project_home = "Projects"

    Args:
        reload (bool, optional): Force a reload. Defaults to False.

    Returns:
        Settings: the global settings
    """
    global _settings
    if reload or _settings is None:
        _settings = Settings()
    return _settings


class Settings(BaseSettings):
    """
    Global settings object.

    Access via :func:`paraphernalia.settings`.
    """

    auto_setup: bool = Field(default=False, allow_mutation=False)
    """If true, run :func:`paraphernalia.setup` on load. Defaults to false."""

    use_rich: bool = Field(default=True, allow_mutation=False)
    """If true, use the rich console handling library where possible.
    Defaults to true."""

    initial_seed: int = Field(default_factory=lambda: int(time()), allow_mutation=False)
    """The initial random number generator seed. Defaults to the number of
    complete seconds since the epoch."""

    cache_home: Path = XDG_CACHE_HOME / "paraphernalia"
    """A writeable directory for cacheing files e.g. model artifacts."""

    project_home: Path = XDG_DATA_HOME / "paraphernalia"
    """A writeable directory for project outputs."""

    # Project defaults
    creator: str = "Anonymous"
    """Default creator for projects."""

    tags: Set[str] = {"paraphernalia"}
    """Default tag set for projects."""

    rights: str = "All rights reserved"
    """Default license for projects."""

    _ensure_dir_exists = validator("cache_home", "project_home", allow_reuse=True)(
        ensure_dir_exists
    )

    _set_inital_seed = validator("initial_seed", allow_reuse=True)(set_seed)

    class Config:  # pragma: no cover
        env_prefix = "pa_"
        env_file = XDG_CONFIG_HOME / "paraphernalia.env"
        env_file_encoding = "utf-8"
        allow_mutation = True
        validate_assignment = True
