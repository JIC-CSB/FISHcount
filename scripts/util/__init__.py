import os
import errno

import numpy as np
import scipy.misc

from jicimagelib.io import FileBackend, AutoName
from jicimagelib.image import DataManager

HERE = os.path.dirname(__file__)
UNPACK = os.path.join(HERE, '..', '..', 'data', 'jic_backend')

def grayscale_to_rgb(image_array):
    """Given a grayscale image array, return a colour version, setting each of
    the RGB channels to the original value."""

    return np.dstack(3 * [image_array])

def safe_mkdir(dir_path):

    try:
        os.makedirs(dir_path)
    except OSError, e:
        if e.errno != errno.EEXIST:
            print "Error creating directory %s" % dir_path
            sys.exit(2)

def unpack_data(confocal_file):
    """Unpack the file and return an image collection object."""
    safe_mkdir(UNPACK)

    backend = FileBackend(UNPACK)
    data_manager = DataManager(backend)

    data_manager.load(confocal_file)

    image_collection = data_manager[0]

    return image_collection

def imsave_with_outdir(fname, im):
    """Save images to the specified output directory."""
    fpath = os.path.join(AutoName.directory, fname)
    scipy.misc.imsave(fpath, im)

def human_to_computer_index(i):
    """Convert one-based index to zero-based."""
    return i-1
