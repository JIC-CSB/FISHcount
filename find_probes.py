"""Load image stack, locate RNA probes and generate an annotated image
with probe locations."""

import os
import re
import random
import argparse

import scipy.misc
import scipy.ndimage
import numpy as np
from libtiff import TIFF

import skimage
from skimage.feature import blob_log, peak_local_max
from skimage.morphology import square

def random_rgb():
    c1 = random.randint(127, 255) 
    c2 = random.randint(0, 127) 
    c3 = random.randint(0, 255) 

    #l = [c1, c2, c3]

    return tuple(random.sample([c1, c2, c3], 3))


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

def summary(stack_array):
    """Print summary statistics for a stack_array"""

    print "Shape: {}".format(stack_array.shape)
    print "Max: {}, Min: {}".format(np.max(stack_array), np.min(stack_array))

def stack_as_uint8(sa):
    """Convert the given stack to uint8 format."""

    smax = np.max(sa)
    smin = np.min(sa)

    conversion = 255.0 / (smax - smin)

    sconv = (conversion * (sa - smin)).astype(np.uint8)

    return sconv

def stack_as_float64(sa):
    """Convert the given stack to float64 format."""

    smax = np.max(sa)
    smin = np.min(sa)

    conversion = 1 / (smax - smin)

    sconv = (conversion * (sa - smin)).astype(np.float64)

    return sconv

def max_intensity_projection(sa):
    """Return max intensity projection for stack array."""

    iz_max = np.argmax(sa, 2)

    xmax, ymax, _ = sa.shape

    projection = np.zeros((xmax, ymax), dtype=sa.dtype)

    for x in range(xmax):
        for y in range(ymax):
            projection[x, y] = sa[x, y, iz_max[x, y]]

    return projection

def find_edge_image(stack_array):
    """Find projected image with edges."""

    projection = max_intensity_projection(stack_array)

    #scipy.misc.imsave('projection.png', projection)

    gauss = skimage.filter.gaussian_filter(projection, 0.4)

    edges = skimage.filter.sobel(gauss)

    return edges

def find_probe_coordinates(stack_array):
    """Given an input stack (as a 3D numpy array of uint8 values), find
    the locations of fluorescing RNA probes. Returns a list of 2D coordinate
    pairs, each of which represents a located probe cluster."""

    edges = find_edge_image(stack_array)

    scipy.misc.imsave('edges.png', edges)

    thresh = edges > skimage.filter.threshold_otsu(edges)

    scipy.misc.imsave('thresh.png', thresh)

    hough_radii = np.arange(1, 3, 1)
    hough_res = skimage.transform.hough_circle(thresh, hough_radii)

    hmax0 = hough_res[0].max()
    probe_locs = np.where(hough_res[0] > 0.5 * hmax0)

    hmax1 = hough_res[1].max()
    probe_locs1 = np.where(hough_res[1] > 0.3 * hmax1)

    probe_locs2 = np.where((hough_res[0] > 0.5 * hmax0) & (hough_res[1] > 0.3 * hmax1))
    hough_data = hough_res[0] + hough_res[1]
    loc_array = peak_local_max(hough_data, min_distance=5, threshold_rel=0.5)

    return loc_array

def generate_annotated_image(input_dir):

    stack_array = read_stack_array(input_dir)

    converted_stack_array = stack_as_uint8(stack_array)

    loc_array = find_probe_coordinates(converted_stack_array)

    edges = find_edge_image(stack_array)

    thresh = edges > skimage.filter.threshold_otsu(edges)

    xdim, ydim = thresh.shape

    annot_array = np.zeros((xdim, ydim, 3), dtype=np.uint8)

    annot_array[:,:] = [255, 255, 255]

    annot_array[np.where(thresh)] = [0, 0, 0]

    def draw_cross(x, y, c):
        for xo in np.arange(-4, 5, 1):
            annot_array[x+xo, y] = c
        for yo in np.arange(-4, 5, 1):
            annot_array[x,y+yo] = c

    for pair in loc_array:
        x, y = pair
        c = random_rgb()
        draw_cross(x, y, c)
    
    scipy.misc.imsave('annotated.png', annot_array)


def main():
    parser = argparse.ArgumentParser(__doc__)
    
    parser.add_argument('input_dir', help="Input directory.")

    args = parser.parse_args()

    generate_annotated_image(args.input_dir)

if __name__ == "__main__":
    main()
