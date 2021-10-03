"""
Collection of utility functions.

TODO: Move to module init?
"""
import logging
import math
import os
import re
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse

import xdg
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_cuda_version():
    """
    Return the CUDA version string e.g. 10.0
    """
    return [
        s
        for s in subprocess.check_output(["nvcc", "--version"])
        .decode("UTF-8")
        .split(", ")
        if s.startswith("release")
    ][0].split(" ")[-1]


def divide(whole: int, part: int, min_overlap: int = 0) -> List[int]:
    """
    Divide ``whole`` into several ``part``-sized chunks which overlap by
    at least ``min_overlap``.

    Args:
        whole (int): The length the subdivide
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
    Step down generator.

    TODO:
    - Add value checks
    - Think about how to do this kind of thing more generically (sin, saw etc.)

    steps: the number of plateaus
    iterations: the total number of iterations over which to step down from 1.0 to 0.0
    """
    if steps <= 0:
        raise ValueError("Steps must be >= 0")
    if iterations <= 0:
        raise ValueError("Iteration must be >= 0")

    i = iterations
    while True:
        i -= 1
        yield max(0, int(i / iterations * steps) / (steps - 1))


def cache_home(cache_home: str = None) -> Path:
    """
    Get the cache home for paraphernalia ensuring it exists.
    Defaults to $XDG_CACHE_HOME/paraphernalia
    """
    global _CACHE_HOME
    if cache_home is not None:
        _CACHE_HOME = Path(cache_home)
    logger.info(f"Setting cache home to {_CACHE_HOME}")
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
        _DATA_HOME = Path(data_home)
    logger.info(f"Setting data home to {_DATA_HOME}")
    os.makedirs(_DATA_HOME, exist_ok=True)
    return _DATA_HOME


_DATA_HOME = None
data_home(xdg.xdg_data_home() / "paraphernalia")


_FORBIDDEN = re.compile(r"[^a-z0-9_-]+")


def slugify(*bits):
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


def download(url: str, target: Path = None, overwrite: bool = False) -> Path:
    """
    Download ``url`` to local disk and return the Path to which it was written.

    Args:
        url (str): the URL to fetch.
        target (Path, optional): The target path. Defaults to None.
        overwrite (bool, optional): If true, overwrite the target path.
            Defaults to False.

    Raises:
        Exception: [description]

    Returns:
        Path: the file that was written
    """
    if target is None:
        name = urlparse(url).path
        name = os.path.basename(name)
        target = cache_home() / name
    if target.is_dir():
        raise Exception(f"Download target '{target}' is a directory")
    if not target.exists() or overwrite:
        _download(url, target)
    else:
        print(f"Using cached {target}")
    return target


def _download(url, target):
    """
    Helper method used by `download()`

    Args:
        url ([type]): [description]
        target ([type]): [description]
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
