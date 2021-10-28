"""
Sign and tag images.
"""

import warnings
from pathlib import Path
from typing import List, Optional

import click

# libxmp raises an error on import if libexempi is not available
try:
    import libxmp
    from libxmp.consts import XMP_NS_DC as DC
except Exception as e:
    warnings.warn(f"Could not import libxmp: {e}")


class XMP:
    """
    Context manager for working with XMP metadata on files, with convenience
    accessors for some Dublin Core fields.

    For DC/XMP interactions see https://www.exiv2.org/tags-xmp-dc.html

    Example:

    >>> with XMP("test.png") as xmp:
    ...     xmp.title = "Mona Lisa"
    ...     xmp.creator = "Leonardo"

    >>> with XMP("test.png") as xmp:
    ...     print(xmp.creator)
    Leonardo

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

    def clear(self) -> None:
        """
        Clear all metadata.
        """
        self.creators = []
        self.title = ""
        self.tags = []
        self.description = ""
        self.rights = ""

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
    def creator(self) -> str:
        "The creator of this object"
        return self.creators[0] if self.creators else None

    @creator.setter
    def creator(self, value: str) -> None:
        self.creators = [value]


def _make_seq_property(name):
    """
    Make a property wrapping an XMP array.
    """

    def _getter(self) -> List[str]:
        if not self._xmp.does_property_exist(DC, name):
            return []
        num_items = self._xmp.count_array_items(DC, name)
        tags = [self._xmp.get_array_item(DC, name, i) for i in range(1, num_items + 1)]
        return tags

    def _setter(self, value: List[str]) -> None:
        self._xmp.delete_property(DC, name)
        if isinstance(value, str):
            raise ValueError(f"Value for {name} should be a list")
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

    def _deleter(self) -> None:
        self._xmp.delete_property(DC, name)

    return property(_getter, _setter, _deleter, doc=f"The {name}")


def _make_lang_property(name, generic_lang="", specific_lang="en"):
    """
    Make a Python property wrapping an XMP alt language attribute.
    """

    def _getter(self) -> str:
        if not self._xmp.does_property_exist(DC, name):
            return None
        return self._xmp.get_localized_text(DC, name, generic_lang, specific_lang)

    def _setter(self, value: str) -> None:
        self._xmp.set_localized_text(DC, name, generic_lang, specific_lang, value)

    def _deleter(self) -> None:
        self._xmp.delete_property(DC, name)

    return property(_getter, _setter, _deleter, doc=f"The {name}")


XMP.creators = _make_seq_property("creator")
XMP.tags = _make_seq_property("subject")
XMP.title = _make_lang_property("title")
XMP.description = _make_lang_property("description")
XMP.rights = _make_lang_property("rights")


@click.command()
@click.argument("target", type=click.Path(exists=True, readable=True, writable=True))
@click.option(
    "--clear/--no-clear",
    default=False,
    help="Clear metadata.",
    show_default=True,
)
@click.option("--creator", "-c", "creators", multiple=True, help="Set the creator(s).")
@click.option("--title", nargs=1, help="Set the title.")
@click.option("--tag", "-t", "tags", multiple=True, help="Add one or more tags.")
@click.option("--description", "-d", nargs=1, help="Set the description.")
@click.option("--rights", "-r", nargs=1, help="Set the statement of rights.")
def sign(
    target,
    clear,
    creators: Optional[List[str]] = None,
    title: Optional[str] = None,
    tags: Optional[List[str]] = None,
    description: Optional[str] = None,
    rights: Optional[str] = None,
):
    """
    Manipulate basic XMP/Dublin Core metadata in PNG, JPG and TIFF images.
    """
    with XMP(target) as sig:
        if clear:
            sig.clear()

        if creators:
            sig.creators = creators
        if title:
            sig.title = title
        if tags:
            sig.tags = tags
        if description:
            sig.description = description
        if rights:
            sig.rights = rights

    with XMP(target) as sig:
        print(f"Title: {sig.title}")
        print(f"Creator: {', '.join(sig.creators) if creators else 'None'}")
        print(f"Tags: {', '.join(sig.tags) if tags else 'None'}")
        print(f"Description: {sig.description}")
