import os
import random

import numpy as np

from util.fonty import Glyph, Font, Bitmap

HERE = os.path.dirname(__file__)

def text_at(image, text, ox, oy, colour):
    """Draw text on the given image, at location ox, oy, with the specified
    colour."""

    fnt = Font(os.path.join(HERE, 'fonts', 'UbuntuMono-R.ttf'), 24)

    ftext = fnt.render_text(text)

    for y in range(ftext.height):
        for x in range(ftext.width):
            if ftext.pixels[y * ftext.width + x]:
                try:
                    image[ox + y, oy + x] = colour
                except IndexError:
                    pass

def draw_cross(annot, x, y, c):
    """Draw a cross centered at x, y on the given array. c is the colour
    which should be a single value for grayscale images or an array for colour
    images."""

    try:
        for xo in np.arange(-4, 5, 1):
            annot[x+xo, y] = c
        for yo in np.arange(-4, 5, 1):
            annot[x,y+yo] = c
    except IndexError:
        pass

def random_rgb():
    c1 = random.randint(127, 255) 
    c2 = random.randint(0, 127) 
    c3 = random.randint(0, 255) 

    #l = [c1, c2, c3]

    return tuple(random.sample([c1, c2, c3], 3))
