import re
import os
import random

import numpy as np
from libtiff import TIFF

def random_rgb():
    c1 = random.randint(127, 255) 
    c2 = random.randint(0, 127) 
    c3 = random.randint(0, 255) 

    #l = [c1, c2, c3]

    return tuple(random.sample([c1, c2, c3], 3))

def normalise2D(input_image):
    """Normalise a single channel 8 bit image (as numpy array)."""

    norm_factor = 10. / np.median(input_image)

    return input_image * norm_factor

def read_2D_image(filename):
    """Read 2D image from filename and return it as a numpy array"""

    f_tiff = TIFF.open(filename)

    im_array = f_tiff.read_image()

    return im_array

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def get_fpaths(directory, channel_index):
    """Yield the file paths for a particular channel."""
    fpaths = []
    for fname in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, fname)):
            continue
        name, ext = fname.split('.')
        channel = name.split('_')[-2]
        if channel == 'C{}'.format(channel_index):
            fpaths.append(os.path.join(directory, fname))
    return sorted_nicely(fpaths)

def read_stack_array(input_dir, channel=0):
    """Given an input directory, read all of the images corresponding to the
    given channel. Convert each image into a 2D numpy array, then join them
    together to give a 3D array."""

    stack_files = get_fpaths(input_dir, channel)
    all_image_arrays = [read_2D_image(f) for f in stack_files]
    stack_array = np.dstack(all_image_arrays)

    return stack_array

def stack_as_uint8(sa):
    """Convert the given stack to uint8 format."""

    if sa.dtype == np.uint8:
        return sa.copy()

    smax = np.max(sa)
    smin = np.min(sa)

    conversion = 255.0 / (smax - smin)

    sconv = (conversion * (sa - smin)).astype(np.uint8)

    return sconv
