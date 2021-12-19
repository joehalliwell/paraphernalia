"""
Miscellaneous utility functions.
"""
import logging
import math
import os
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse

from tqdm import tqdm

_LOG = logging.getLogger(__name__)


def divide(whole: int, part: int, min_overlap: int = 0) -> List[int]:
    """
    Divide ``whole`` into several ``part``-sized chunks which overlap by
    at least ``min_overlap``.

    Args:
        whole (int): The total to subdivide
        part (int): The size of the chunk
        min_overlap (int, optional): The minimum overlap between chunks.
            Defaults to 0 i.e. chunks won't overlap unless required.

    Returns:
        List[int]: A list of chunk offset
    """
    if part > whole:
        # Log something?
        return []

    if min_overlap >= part:
        raise ValueError(
            f"Overlap must be strictly smaller than part ({min_overlap} >= {part})"
        )

    parts = math.ceil((whole - min_overlap) / (part - min_overlap))
    stride = (whole - part) / (parts - 1) if parts > 1 else 1
    return [int(i * stride) for i in range(parts)]


def step_down(steps, iterations):
    """
    Step down generator

    .. deprecated:: future
        This will be removed

    .. note::
        - Add value checks
        - Think about how to do this kind of thing more generically

    Args:
        steps: the number of plateaus
        iterations: the total number of iterations over which to step down from
            1.0 to 0.0
    """
    if steps <= 0:
        raise ValueError("Steps must be >= 0")
    if iterations <= 0:
        raise ValueError("Iteration must be >= 0")

    i = iterations
    while True:
        i -= 1
        yield max(0, int(i / iterations * steps) / (steps - 1))


_FORBIDDEN = re.compile(r"[^a-z0-9_-]+")


def slugify(*bits) -> str:
    """
    Make a lower-case alphanumeric representation of the arguments by stripping
    other characters and replacing spaces with hyphens.
    """
    # Single item
    if len(bits) == 1:
        bit = bits[0]
        if isinstance(bit, datetime):
            bit = bit.strftime("%Y-%m-%d_%Hh%M")
        if not isinstance(bit, str) and isinstance(bit, Iterable):
            return slugify(*bit)
        return _FORBIDDEN.sub("-", str(bit).lower())

    # Multiple items
    return "_".join(slugify(bit) for bit in bits)


def ensure_dir_exists(path: Path) -> Path:
    if not path.exists():
        _LOG.info(f"Creating {path}")
        os.makedirs(path, exist_ok=True)

    if path.is_dir() and os.access(path, os.R_OK | os.W_OK):
        return path

    raise Exception(f"{path} already exists or is not writable")


def download(url: str, target: Path = None, overwrite: bool = False) -> Path:
    """
    Download ``url`` to local disk and return the Path to which it was written.

    Args:
        url (str): the URL to fetch.
        target (Path, optional): The target path. Defaults to None.
        overwrite (bool, optional): If true, overwrite the target path.
            Defaults to False.

    Raises:
        Exception: if the download target is a directory

    Returns:
        Path: the file that was written
    """
    if target is None:
        # Defer import to allow this module to be used freely
        from paraphernalia import settings

        name = urlparse(url).path
        name = os.path.basename(name)
        target = settings().cache_home / name
    if target.is_dir():
        raise Exception(f"Download target '{target}' is a directory")
    if not target.exists() or overwrite:
        _download(url, target)
    else:
        print(f"Using cached {target}")
    return target


def _download(url: str, target: Path):
    """
    Helper method used by `download()`

    Args:
        url (str): [description]
        target (Path): [description]
    """
    desc = os.path.basename(target)

    class _DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    try:
        with _DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=desc
        ) as t:
            urllib.request.urlretrieve(url, filename=target, reporthook=t.update_to)
    except Exception as exc:
        raise Exception(f"Failed to download {url}") from exc
