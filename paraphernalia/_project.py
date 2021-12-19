"""
Tools for managing projects/metadata.
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Set

from pydantic import BaseModel, Field, validator

from paraphernalia._settings import settings
from paraphernalia.utils import ensure_dir_exists, set_seed, slugify

_LOG = logging.getLogger(__name__)


class Project(BaseModel):
    """
    Data model for projects. Access the current project via :func:`paraphernalia.project`.
    """

    title: str
    """Title for the project e.g. "The Mona Lisa". Required."""

    creator: str = Field(default_factory=lambda: settings().creator)
    """The creator of the project. Defaults to the value provided by :func:`settings`."""

    tags: Set[str] = Field(default_factory=lambda: settings().tags)
    """A set of tags for the project. Defaults to the value provided by :func:`settings`."""

    rights: str = Field(default_factory=lambda: settings().rights)
    """The license for project outputs. Defaults to the value provided by :func:`settings`."""

    created: datetime = Field(default_factory=datetime.now)
    seed: int = Field(default_factory=lambda: settings().seed)

    slug: str = Field(default_factory=lambda: None)
    """A short, filesystem-safe name for the project. Defaults to the date and
    slugified title if not specified."""

    path: Path = Field(default_factory=lambda: None)
    """Directory in which to store project outputs. Defaults to a directory
    called `slug` in the project home provided by :func:`settings`"""

    def __init__(self, activate=True, **data: Any) -> None:
        """
        Example:

        >>> from paraphernalia import project(), Project
        >>> Project(title="The Mona Lisa") # Also sets this to be the active project
        >>> project().title
        The Mona Lisa

        The activate keyword can be used to prevent a project from being activated
        on construction:

        >>> p2 = Project(title="The Mona Lisa Mk II", activate=False)
        >>> project().title # Unchanged because the new project hasn't been activated
        The Mona Lisa

        It can be activated later:

        >>> p2.activate()
        >>> project().title
        The Mona Lisa Mk II

        Args:
            activate (bool, optional): If false don't activate. Defaults to True.
        """
        super().__init__(**data)
        if activate:
            self.activate()

    @validator("slug", always=True, pre=True)
    def _slug(cls, v, values):
        return v.strip() if v else slugify(values["created"].date(), values["title"])

    @validator("path", always=True, pre=True)
    def _path(cls, v: Path, values):
        return v or settings().project_home / values["slug"]

    def activate(self):
        """
        Make this the current active project.

        As a side effect this:

        * Sets the global random seed
        * Ensures that project directories exist and are writable
        """
        global _active
        _LOG.info(f"Activating '{self.title}' by '{self.creator}'")
        set_seed(self.seed)
        ensure_dir_exists(self.path)
        _active = self

    class Config:  # pragma: no cover
        allow_mutation = False
        validate_assignment = True


_active = None


def project() -> Optional[Project]:
    """
    Get the current project.

    Returns:
        Optional[Project]: the current project if set.
    """
    return _active
