"""
Colour and palette utilities.
"""
from enum import Enum
from typing import List, Tuple

from PIL import ImageColor


class Palette(Enum):
    """
    A palette of colours.
    """

    def rgb(self) -> Tuple:
        """
        Returns:
            Tuple: an (r, g, b) tuple where each component is 0-255
        """
        return ImageColor.getcolor(self.value, "RGB")

    def unit_rgb(self) -> Tuple:
        """
        Returns:
            Tuple: and (r, g, b) tuple where each component is 0-1
        """
        c = self.rgb()
        return (c[0] / 255, c[1] / 255, c[2] / 255)

    @classmethod
    def as_rgb(cls) -> List[Tuple]:
        """Convert this palette to a list of (r, g, b) tuples where each component is 0-255"""
        return [c.rgb() for c in cls]

    @classmethod
    def as_unit_rgb(cls) -> List[Tuple]:
        """Convert this palette to a list of (r, g, b) tuples where each component is 0-1"""
        return [c.unit_rgb() for c in cls]


class BW(Palette):
    """
    A simple black and white palette.
    """

    BLACK = "#000000"
    WHITE = "#FFFFFF"


class C64(Palette):
    """
    The Commodore 64 palette. Taken from https://www.c64-wiki.com/wiki/Color
    """

    BLACK = "#000000"
    WHITE = "#FFFFFF"
    RED = "#880000"
    CYAN = "#AAFFEE"
    PURPLE = "#CC44CC"
    GREEN = "#00CC55"
    BLUE = "#0000AA"
    YELLOW = "#EEEE77"
    ORANGE = "#DD8855"
    BROWN = "#664400"
    LIGHT_RED = "#FF7777"
    GREY_1 = "#333333"
    GREY_2 = "#777777"
    LIGHT_GREEN = "#AAFF66"
    LIGHT_BLUE = "#0088FF"
    GREY_3 = "#BBBBBB"


class ZX_SPECTRUM(Palette):
    """
    The Sinclair ZX Spectrum palette. Taken from https://en.wikipedia.org/wiki/ZX_Spectrum_graphic_modes
    """

    BLACK = "#000000"
    BLUE = "#0000EE"
    RED = "#EE0000"
    MAGENTA = "#EE00EE"
    GREEN = "#00EE00"
    CYAN = "#00EEEE"
    YELLOW = "#EEEE00"
    WHITE = "#EEEEEE"
    BRIGHT_BLACK = "#000000"
    BRIGHT_BLUE = "#0000FF"
    BRIGHT_RED = "#FF0000"
    BRIGHT_MAGENTA = "#FF00FF"
    BRIGHT_GREEN = "#00FF00"
    BRIGHT_CYAN = "#00FFFF"
    BRIGHT_YELLOW = "#FFFF00"
    BRIGHT_WHITE = "#FFFFFF"
