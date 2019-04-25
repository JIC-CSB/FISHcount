"""Segment the given z-stacks using the nuclear markers as seeds and the
cellular background autofluorescence to represent the cell."""

import argparse

import protoimg
from protoimg.stack import Stack, normalise_stack, equalize_stack
from protoimg.transform import (
    make_named_transform,
    ImageArray,
    dilate_simple,
    component_centroids,
    remove_small_objects,
    threshold_otsu,
    gaussian_filter,
    find_edges,
    min_intensity_projection,
    max_intensity_projection,
    find_connected_components,
    close_holes,
    watershed_with_seeds,
    equalize_adaptive
)

from scipy.ndimage.measurements import watershed_ift

import skimage
from skimage.exposure import equalize_adapthist
import numpy as np
from skimage.morphology import watershed, erosion, disk, dilation
from skimage.segmentation import random_walker
import scipy.misc

from fonty import Glyph, Font, Bitmap

from find_probe_locs import find_probe_locations

def RGB_from_single_channel(array):
    return np.dstack(3 * [array])

def generate_segmentation_seeds(nuclear_stack):
    """Given the nuclear fluorescence channel, find markers representing the
    locations of those nuclei so that they can be used to seed a segmentation.
    """

    normed_stack = normalise_stack(nuclear_stack)
    max_nuclear_proj = max_intensity_projection(normed_stack)
    eq_proj = equalize_adaptive(max_nuclear_proj, n_tiles=16, name='equalized_nuclear_proj')
    gauss = gaussian_filter(eq_proj, sigma=3)
    edges = find_edges(gauss, name='seed_edges')
    thresh = threshold_otsu(edges, mult=1)
    nosmall = remove_small_objects(thresh, min_size=500)
    #dilated = dilate_simple(nosmall)
    connected_components = find_connected_components(nosmall, background=0, name='conn_seeds')
    seeds = component_centroids(connected_components, name='seed_centroids')

    return seeds


@make_named_transform('filter_segmentation')
def filter_segmentation(image_array, min_size=None):
    """Filter the given segmentation, removing objects smaller than the minimum
    size. If min_size is none, remove everything smaller than 10% of the mean
    size."""

    filtered_ia = np.copy(image_array)

    background = 0

    # Produce a dictionary of id : list of coordinates for each id representing
    # a region in the image
    by_coords = {index : zip(*np.where(image_array==index))
                 for index in np.unique(image_array)}

    if min_size is None:
        mean_size = np.mean(map(len, by_coords.values()))
        min_size = int(0.1 * mean_size)

    for index, coords in by_coords.items():
        if len(coords) < min_size:
            filtered_ia[tuple(zip(*coords))] = background

    return filtered_ia


def find_segmented_regions(seeds, autof_stack, imsave):

    min_autof_proj = min_intensity_projection(autof_stack)
    equal_autof = equalize_adaptive(min_autof_proj, name='equal_autof')
    smoothed_autof = gaussian_filter(equal_autof, sigma=5, name='smooth_autof')
    edge_autof = find_edges(smoothed_autof, name='edge_autof')
    thresh_autof = threshold_otsu(smoothed_autof, mult=0.6, name='thresh_autof')

    # ndfeed = skimage.img_as_uint(edge_autof.image_array & thresh_autof)
    # imsave('ndfeed.png', ndfeed)
    # altseg = watershed_ift(ndfeed, seeds.image_array)
    # imsave('altseg.png', altseg)

    #segmentation = watershed_with_seeds(smoothed_autof, ImageArray(altseg, 'atseg'),
    segmentation = watershed_with_seeds(smoothed_autof, seeds,
                               mask_image=thresh_autof)

    # my_maker = make_named_transform('hughbert')
    # my_filter = my_maker(filter_segmentation)
    # filtered_segmentation = my_filter(segmentation)
    filtered_segmentation = filter_segmentation(segmentation)

    re_watershed = watershed_with_seeds(smoothed_autof, filtered_segmentation,
                                mask_image=thresh_autof, name='re_watershed')

    return re_watershed
    
def segmentation_from_stacks(nuclear_stack, autof_stack, imsave):
    """Return the segmentation from the given stack."""

    seeds = generate_segmentation_seeds(nuclear_stack)

    imsave('seeds.png', seeds.image_array)

    segmentation = find_segmented_regions(seeds, autof_stack, imsave)

    return segmentation

def load_stack_and_segment(path, imsave):
    """Load a stack from the given path and segment it."""

    autof_stack = Stack.from_path(path, channel=2)
    nuclear_stack = Stack.from_path(path, channel=2)

    return segmentation_from_stacks(nuclear_stack, autof_stack, imsave)

def segmentation_border_image(segmentation, index, width=1):

    isolated_region = np.zeros(segmentation.image_array.shape, dtype=np.uint8)

    isolated_region[np.where(segmentation.image_array == index)] = 255

    selem = disk(width)
    border = isolated_region - erosion(isolated_region, selem)

    return border

def test_segmentation_from_stack(stack_path, imsave):
    segmentation = load_stack_and_segment(stack_path, imsave)

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stack_path', help="Path to stack files.")
    args = parser.parse_args()

    test_segmentation_from_stack(args.stack_path, imsave=scipy.misc.imsave)

if __name__ == "__main__":
    main()
