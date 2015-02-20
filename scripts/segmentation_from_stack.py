"""Segment the given z-stacks using the nuclear markers as seeds and the
cellular background autofluorescence to represent the cell."""

import argparse

import protoimg
from protoimg.stack import Stack, normalise_stack, equalize_stack
from protoimg.transform import (
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

def component_find_centroid(connected_components, index):
    loc = np.mean(np.where(connected_components.image_array == index), axis=1)

    x, y = map(int, loc)

    return x, y
    
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

def find_segmented_regions(seeds, autof_stack):

    min_autof_proj = min_intensity_projection(autof_stack)
    equal_autof = equalize_adaptive(min_autof_proj, 'equal_autof')
    smoothed_autof = gaussian_filter(equal_autof, sigma=5, name='smooth_autof')
    edge_autof = find_edges(smoothed_autof, name='edge_autof')
    thresh_autof = threshold_otsu(smoothed_autof, mult=0.6, name='thresh_autof')

    # ndfeed = skimage.img_as_uint(edge_autof.image_array & thresh_autof)
    # scipy.misc.imsave('ndfeed.png', ndfeed)
    # altseg = watershed_ift(ndfeed, seeds.image_array)
    # scipy.misc.imsave('altseg.png', altseg)

    #segmentation = watershed_with_seeds(smoothed_autof, ImageArray(altseg, 'atseg'),
    segmentation = watershed_with_seeds(smoothed_autof, seeds,
                               mask_image=thresh_autof)

    return segmentation
    
def segmentation_from_stacks(nuclear_stack, autof_stack):
    """Return the segmentation from the given stack."""

    seeds = generate_segmentation_seeds(nuclear_stack)

    # Seed hacking
    # next_seed = 1 + max(np.unique(seeds.image_array))
    # seeds.image_array[136, 468] = next_seed
    # seeds.image_array[430, 430] = 1 + next_seed
    # seeds.image_array[500, 400] = 2 + next_seed
    scipy.misc.imsave('seeds.png', seeds.image_array)

    segmentation = find_segmented_regions(seeds, autof_stack)

    return segmentation

def load_stack_and_segment(path):
    """Load a stack from the given path and segment it."""

    autof_stack = Stack.from_path(path, channel=2)
    nuclear_stack = Stack.from_path(path, channel=2)

    return segmentation_from_stacks(nuclear_stack, autof_stack)

def segmentation_border_image(segmentation, index, width=1):

    isolated_region = np.zeros(segmentation.image_array.shape, dtype=np.uint8)

    isolated_region[np.where(segmentation.image_array == index)] = 255

    selem = disk(width)
    border = isolated_region - erosion(isolated_region, selem)

    return border

def text_at(image, text, ox, oy, colour):
    fnt = Font('scripts/fonts/UbuntuMono-R.ttf', 24)

    ftext = fnt.render_text(text)

    for y in range(ftext.height):
        for x in range(ftext.width):
            if ftext.pixels[y * ftext.width + x]:
                try:
                    image[ox + y, oy + x] = colour
                except IndexError:
                    pass

def component_find_centroid(connected_components, index):
    loc = np.mean(np.where(connected_components.image_array == index), axis=1)

    x, y = map(int, loc)

    return x, y

def generate_annotated_image(stack_path, segmentation):

    stack = Stack.from_path(stack_path)
    simple_proj = max_intensity_projection(stack, name='simple_proj')
    norm_stack = normalise_stack(stack)

    annot_proj = max_intensity_projection(norm_stack, name='annot_proj')

    eqproj = equalize_adapthist(annot_proj.image_array)
    scipy.misc.imsave('eqproj.png', eqproj)

    zero_pad = np.zeros(eqproj.shape, eqproj.dtype)
    red_image = np.dstack([eqproj, zero_pad, zero_pad])

    scipy.misc.imsave('pretty_proj.png', red_image)

    protoimg.autosave = False

    white16 = 255 << 8, 255 << 8, 255 << 8
    probe_locs = find_probe_locations(stack_path)
    real_ids = set(np.unique(segmentation.image_array)) - set([0])
    for index in real_ids:
        border = segmentation_border_image(segmentation, index)
        red_image[np.where(border == 255)] = 255 << 8, 255 << 8, 255 << 8
        seg_area = set(zip(*np.where(segmentation.image_array == index)))
        selected_probes = set(probe_locs) & seg_area
        n_probes = len(selected_probes)
        ox, oy = component_find_centroid(segmentation, index)
        text_at(red_image, str(n_probes), ox, oy, white16)

    # sel_array = np.zeros(segmentation.image_array.shape, dtype=np.uint8)
    # for x, y in selected_probes:
    #     sel_array[x, y] = 255
    # scipy.misc.imsave('sel_array.png', sel_array)
        

    # for loc in probe_locs:
    #     if loc in np.where

    scipy.misc.imsave('proj_with_borders.png', red_image)


def test_segmentation_from_stack(stack_path):
    segmentation = load_stack_and_segment(stack_path)

    #generate_annotated_image(stack_path, segmentation)

    #seg_edges = find_edges(segmentation, name='seg_edges')

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stack_path', help="Path to stack files.")
    args = parser.parse_args()

    test_segmentation_from_stack(args.stack_path)

if __name__ == "__main__":
    main()
