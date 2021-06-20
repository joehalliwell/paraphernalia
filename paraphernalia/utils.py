import os
import xdg
from pathlib import Path
import urllib.request
from urllib.parse import urlparse

from tqdm import tqdm


def cache_home():
    """
    Get the cache home for paraphernalia ensuring it exists.
    Defaults to $XDG_CACHE_HOME/paraphernalia
    """
    cache = xdg.xdg_cache_home() / "paraphernalia"
    os.makedirs(cache, exist_ok=True)
    return cache


def download(url, target=None, overwrite=False):
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
