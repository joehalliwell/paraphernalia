import os
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import xdg
from tqdm import tqdm


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


def step_down(steps, iterations):
    """
    Step down generator.

    TODO:
    - Add value checks
    - Think about how to do this kind of think more generically (sin, saw etc.)

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


def cache_home():
    """
    Get the cache home for paraphernalia ensuring it exists.
    Defaults to $XDG_CACHE_HOME/paraphernalia
    """
    cache = xdg.xdg_cache_home() / "paraphernalia"
    os.makedirs(cache, exist_ok=True)
    return cache


def download(url, target=None, overwrite=False):
    """
    Download ``url`` to local disk and return the Path to which it was written.
    """
    if target is None:
        name = urlparse(url).path
        name = os.path.basename(name)
        target = cache_home() / name
    if not target.exists() or overwrite:
        _download(url, target)
    else:
        print(f"Using cached {target}")
    return target


def _download(url, target):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=target, reporthook=t.update_to)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
