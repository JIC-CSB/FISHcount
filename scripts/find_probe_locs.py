"""Function(s) to find RNA probe locations, given a stack."""

import argparse

import skimage.measure
from skimage.feature import blob_log, peak_local_max, match_template
import skimage.transform
from skimage.morphology import disk
import numpy as np
import scipy.misc

from protoimg.imgutil import random_rgb
from protoimg.stack import Stack, normalise_stack
from protoimg.transform import (
    ImageArray,
    find_connected_components,
    component_centroids,
    max_intensity_projection, find_edges, gaussian_filter, threshold_otsu,
    equalize_adaptive
)

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

def grayscale_to_rgb(image_array):
    """Given a grayscale image array, return a colour version, setting each of
    the RGB channels to the original value."""

    return np.dstack(3 * [image_array])

def make_stage1_template():
    """Make a template for initial matching. This is an annulus."""

    template = disk(3)
    template[3, 3] = 0

    return template

def find_best_template(edges, imsave):
    """Find the best exemplar of a probe in the image. We use template matching
    with a hollow disk (annulus), and return the best match in the image."""

    template = make_stage1_template()
    stage1_match = match_template(edges.image_array, template, pad_input=True)
    imsave('stage1_match.png', stage1_match)
    cmax = np.max(stage1_match)
    px, py = zip(*np.where(stage1_match == cmax))[0]

    tr = 4
    better_template = edges.image_array[px-tr:px+tr,py-tr:py+tr]
    imsave('better_template.png', better_template)

    return better_template

def generate_probe_loc_image(norm_projection, probe_locs, imsave):
    """Generate an annotated image showing the probe locations as crosses."""

    probe_loc_image = grayscale_to_rgb(norm_projection.image_array)
    for coords in probe_locs:
        x, y = coords
        c = random_rgb()
        draw_cross(probe_loc_image, x, y, c)
    
    imsave('probe_locations.png', probe_loc_image)

def find_probe_locations(stack_dir, imsave):
    """Find probe locations. Given a path, we construct a z stack from the first
    channel of the images in that path, and then find probes within that stack.
    Returns a list of coordinate pairs, representing x, y locations of probes.
    """

    zstack = Stack.from_path(stack_dir)
    # For comparative purposes (so we save the image)
    projection = max_intensity_projection(zstack)
    # Normalise each image in the stack
    norm_stack = normalise_stack(zstack)
    # Now take a maximum intensity projection
    norm_projection = max_intensity_projection(norm_stack, 'norm_projection')
    # Find edges should show the circle-like probes as annuli
    edges = find_edges(norm_projection)

    # Find a suitable template image for matching
    template = find_best_template(edges, imsave)

    match_result = match_template(edges.image_array, template, pad_input=True)
    imsave('stage2_match.png', match_result)

    # Set a threshold for matched locations

    match_thresh = 0.6

    print "t,c"
    for t in np.arange(0.1, 1, 0.05):
        print "{},{}".format(t, len(np.where(match_result > t)[0]))

    locs = np.where(match_result > match_thresh)
    annotated_edges = grayscale_to_rgb(edges.image_array)
    annotated_edges[locs] = edges.image_array.max(), 0, 0
    imsave('annotated_edges.png', annotated_edges)

    # Find the centroids of the locations where we think there's a probe
    cloc_array = match_result > match_thresh
    ia_locs = ImageArray(cloc_array, name='new_cloc')
    connected_components = find_connected_components(ia_locs)
    centroids = component_centroids(connected_components)
    probe_locs = zip(*np.where(centroids.image_array != 0))

    generate_probe_loc_image(norm_projection, probe_locs, imsave)

    return probe_locs

def hough_stuff(imsave):
    """Deprecated. Template matching works better."""
    thresh = threshold_otsu(edges, mult=1.5)

    hough_radii = np.arange(1, 3, 1)
    hough_res = skimage.transform.hough_circle(thresh.image_array, hough_radii)
    hough_data = hough_res[0] + hough_res[1]
    #loc_array = peak_local_max(hough_data, min_distance=5, threshold_rel=0.5)
    cloc_array = peak_local_max(hough_data, min_distance=5, threshold_rel=0.5, indices=False)
    imsave('cloc.png', cloc_array)
    connected_components, n_cc = skimage.measure.label(cloc_array, neighbors=8, return_num=True)

    labels = np.unique(connected_components)

    annot = np.zeros((512, 512, 3), dtype=np.uint8)
    annot[:,:] = 255, 255, 255
    annot[np.where(thresh.image_array)] = [0, 0, 0]

    def draw_cross(x, y, c):
        for xo in np.arange(-4, 5, 1):
            annot[x+xo, y] = c
        for yo in np.arange(-4, 5, 1):
            annot[x,y+yo] = c

    probe_locs = []
    for label in labels:
        coord_list = zip(*np.where(connected_components == label))
        probe_locs.append(coord_list[0])

    for coords in probe_locs:
        x, y = coords
        c = random_rgb()
        draw_cross(x, y, c)
    
    imsave('probe_locations.png', annot)
    #print '\n'.join(thresh.history)

    return probe_locs

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stack_dir', help='Directory containing individual 2D image files')
    args = parser.parse_args()

    find_probe_locations(args.stack_dir, imsave=scipy.misc.imsave)

if __name__ == "__main__":
    main()
