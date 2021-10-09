"""
Sign and tag images.

For DC/XMP interactions see https://www.exiv2.org/tags-xmp-dc.html
"""

from pathlib import Path
from typing import List, Optional, Set, Union

import click
import libxmp
from libxmp.consts import XMP_NS_DC as DC


class Signature:
    """
    Context manager for working with Dublin Core XMP metadata on files.
    """

    def __init__(self, path: str) -> None:
        """
        Create a new Signature for a file

        Args:
            path: The file with which to work

        Raises:
            ValueError: if the path does not exist
        """
        if not Path(path).exists():
            raise ValueError(f"'{path}' does not exist")
        self._path = str(path)

    def __enter__(self):
        self._xmpfile = libxmp.XMPFiles(file_path=self._path, open_forupdate=True)

        self._xmp = self._xmpfile.get_xmp()
        if self._xmp is None:
            self._xmp = libxmp.core.XMPMeta()

        assert self._xmpfile.can_put_xmp(self._xmp)
        return self

    def __exit__(self, type, value, traceback):
        self._xmpfile.put_xmp(self._xmp)
        self._xmpfile.close_file()

    @property
    def creator(self):
        "The creator of this object"
        return self.creators[0] if self.creators else None

    @creator.setter
    def creator(self, value: str):
        self.creators = [value]


def _make_seq_property(name):
    """
    Make a property wrapping an XMP array.
    """

    def _getter(self):
        if not self._xmp.does_property_exist(DC, name):
            return []
        num_items = self._xmp.count_array_items(DC, name)
        tags = [self._xmp.get_array_item(DC, name, i) for i in range(1, num_items + 1)]
        return tags

    def _setter(self, value):
        self._xmp.delete_property(DC, name)
        if isinstance(value, str):
            raise ValueError("Should be a list")
        for tag in value:
            self._xmp.append_array_item(
                DC,
                name,
                tag,
                array_options={
                    "prop_array_is_ordered": True,
                    "prop_value_is_array": True,
                },
            )

    def _deleter(self):
        self._xmp.delete_property(DC, name)

    return property(_getter, _setter, _deleter)


def _make_lang_property(name, generic_lang="", specific_lang="en"):
    """
    Make a Python property wrapping an XMP alt language attribute.
    """

    def _getter(self):
        if not self._xmp.does_property_exist(DC, name):
            return None
        return self._xmp.get_localized_text(DC, name, generic_lang, specific_lang)

    def _setter(self, value):
        self._xmp.set_localized_text(DC, name, generic_lang, specific_lang, value)

    def _deleter(self):
        self._xmp.delete_property(DC, name)

    return property(_getter, _setter, _deleter)


Signature.creators = _make_seq_property("creator")
Signature.tags = _make_seq_property("subject")

Signature.title = _make_lang_property("title")
Signature.description = _make_lang_property("description")
Signature.rights = _make_lang_property("rights")


@click.command()
@click.argument("target", type=click.Path(exists=True, readable=True, writable=True))
@click.option("--creator", "-c", "creators", multiple=True)
@click.option("--title", "-t", nargs=1)
@click.option("--tag", "-tg", "tags", multiple=True)
@click.option("--description", "-d", nargs=1)
@click.option("--rights", "-r", nargs=1)
def sign(
    target,
    creators: Optional[List[str]] = None,
    title: Optional[str] = None,
    tags: Optional[List[str]] = None,
    description: Optional[str] = None,
    rights: Optional[str] = None,
):
    """
    Toy command to sign a file.
    """
    with Signature(target) as sig:
        if creators:
            sig.creators = creators
        if title:
            sig.title = title
        if tags:
            sig.tags = tags
        if description:
            sig.description = tags
        if rights:
            sig.rights = rights

    with Signature(target) as sig:
        print(f"Title: {sig.title}")
        print(f"Creator: {', '.join(sig.creators)}")
        print(f"Tags: {', '.join(sig.tags)}")
        print(f"Description: {sig.description}")
