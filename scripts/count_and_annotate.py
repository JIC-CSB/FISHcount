"""Locate RNA FISH probes."""

import os
import errno
import random
import argparse

import numpy as np

from jicimagelib.io import FileBackend
from jicimagelib.image import DataManager, Image
from jicimagelib.transform import transformation

import scipy.misc
from skimage.morphology import disk, erosion
from skimage.feature import match_template

from fonty import Glyph, Font, Bitmap

from protoimg2.transform import (
    max_intensity_projection, 
    min_intensity_projection,
    normalise_stack,
    equalize_adaptive,
    gaussian_filter,
    find_edges,
    threshold_otsu,
    remove_small_objects,
    find_connected_components,
    component_centroids,
    watershed_with_seeds,
    filter_segmentation,
    component_find_centroid
)

from protoimg2.annotate import (
)

#from segmentation_from_stack import segmentation_from_stacks

HERE = os.path.dirname(__file__)
UNPACK = os.path.join(HERE, '..', 'data', 'jic_backend')

def text_at(image, text, ox, oy, colour):
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

def generate_segmentation_seeds(raw_z_stack):
    """Generate the seeds for segmentation from the z stack."""

    normed_stack = normalise_stack(raw_z_stack)
    max_projection = max_intensity_projection(normed_stack)
    eq_proj = equalize_adaptive(max_projection, n_tiles=16)
    gauss = gaussian_filter(eq_proj, sigma=3)
    edges = find_edges(gauss)
    thresh = threshold_otsu(edges, mult=1)
    nosmall = remove_small_objects(thresh, min_size=500)
    connected_components = find_connected_components(nosmall, background=0)
    seeds = component_centroids(connected_components)

    return seeds

def segment_image(image_collection):
    """Segment the image."""
    
    nuclear_z_stack = image_collection.zstack_array(c=2)
    seeds = generate_segmentation_seeds(nuclear_z_stack)

    # FIXME - this isn't actually the probe stack!
    probe_stack = image_collection.zstack_array(c=2)
    min_autof_proj = min_intensity_projection(probe_stack)
    equal_autof = equalize_adaptive(min_autof_proj)
    smoothed_autof = gaussian_filter(equal_autof, sigma=5)
    edge_autof = find_edges(smoothed_autof)
    thresh_autof = threshold_otsu(smoothed_autof, mult=0.6)

    segmentation = watershed_with_seeds(smoothed_autof, seeds,
                               mask_image=thresh_autof)

    filtered_segmentation = filter_segmentation(segmentation)

    re_watershed = watershed_with_seeds(smoothed_autof, filtered_segmentation,
                                mask_image=thresh_autof)

    return re_watershed

def make_stage1_template():
    """Make a template for initial matching. This is an annulus."""

    template = disk(3)
    template[3, 3] = 0

    return template

def find_best_template(edges):
    """Find the best exemplar of a probe in the image. We use template matching
    with a hollow disk (annulus), and return the best match in the image."""

    template = make_stage1_template()
    stage1_match = match_template(edges, template, pad_input=True)
    #imsave('stage1_match.png', stage1_match)
    cmax = np.max(stage1_match)
    px, py = zip(*np.where(stage1_match == cmax))[0]

    tr = 4
    better_template = edges[px-tr:px+tr,py-tr:py+tr]
    #imsave('better_template.png', better_template)

    return better_template

def generate_probe_loc_image(norm_projection, probe_locs, imsave):
    """Generate an annotated image showing the probe locations as crosses."""

    probe_loc_image = grayscale_to_rgb(norm_projection)
    for coords in probe_locs:
        x, y = coords
        c = random_rgb()
        draw_cross(probe_loc_image, x, y, c)
    
    imsave('probe_locations.png', probe_loc_image)

def find_probe_locations(raw_z_stack):

    normed_stack = normalise_stack(raw_z_stack)
    norm_projection = max_intensity_projection(normed_stack)
    edges = find_edges(norm_projection)
    exemplar = find_best_template(edges)
    match_result = match_template(edges, exemplar, pad_input=True)
    scipy.misc.imsave('match_result.png', match_result)

    match_thresh = 0.6

    locs = np.where(match_result > match_thresh)
    annotated_edges = grayscale_to_rgb(edges)
    annotated_edges[locs] = edges.max(), 0, 0
    scipy.misc.imsave('annotated_edges.png', annotated_edges)

    cloc_array = match_result > match_thresh
    scipy.misc.imsave('cloc_array.png', cloc_array)
    connected_components = find_connected_components(cloc_array)
    centroids = component_centroids(connected_components)

    probe_locs = zip(*np.where(centroids != 0))

    generate_probe_loc_image(norm_projection, probe_locs, scipy.misc.imsave)

    return probe_locs

def segmentation_border_image(segmentation, index, width=1):

    isolated_region = np.zeros(segmentation.shape, dtype=np.uint8)

    isolated_region[np.where(segmentation == index)] = 255

    selem = disk(width)
    border = isolated_region - erosion(isolated_region, selem)

    return border

def generate_annotated_image(segmentation, probe_locs, stack, imsave):

    norm_stack = normalise_stack(stack)
    annot_proj = max_intensity_projection(norm_stack, name='annot_proj')

    eqproj = equalize_adaptive(annot_proj)
    imsave('eqproj.png', eqproj)

    zero_pad = np.zeros(eqproj.shape, eqproj.dtype)
    red_image = np.dstack([eqproj, zero_pad, zero_pad])

    if imsave:
        imsave('pretty_proj.png', red_image)

    white16 = 255 << 8, 255 << 8, 255 << 8
    white8 = 255, 255, 255
    real_ids = set(np.unique(segmentation)) - set([0])

    for index in real_ids:
        border = segmentation_border_image(segmentation, index)
        red_image[np.where(border == 255)] = white8
        seg_area = set(zip(*np.where(segmentation == index)))
        selected_probes = set(probe_locs) & seg_area
        n_probes = len(selected_probes)
        ox, oy = component_find_centroid(segmentation, index)
        pixel_area = len(seg_area)
        text_at(red_image, str(n_probes), ox, oy, white8)
        text_at(red_image, str(pixel_area), ox+30, oy-20, white8)

    if imsave:
        imsave('annotated_projection.png', red_image)

def count_and_annotate(confocal_image, pchannel, imsave):
    """Find probe locations, segment the image and produce an annotated image
    showing probe counts per identified cell."""

    image_collection = unpack_data(confocal_image)

    segmentation = segment_image(image_collection)

    probe_stack = image_collection.zstack_array(c=pchannel)
    probe_locations = find_probe_locations(probe_stack)

    generate_annotated_image(segmentation, probe_locations, probe_stack, imsave)

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_image', help='Confocal image to analyse')
    parser.add_argument('output_dir', help='Path to output directory.')
    parser.add_argument('-p', '--probe_channel',
            default=1, type=int, help='Probe channel (default 1)')
    args = parser.parse_args()

    if args.probe_channel == 0:
        parser.error('Probe channel index is one-based; index zero is invalid.')
    pchannel = args.probe_channel - 1

    safe_mkdir(args.output_dir)

    def imsave_with_outdir(fname, im):
        """Save images to the specified output directory."""
        fpath = os.path.join(args.output_dir, fname)
        scipy.misc.imsave(fpath, im)

    count_and_annotate(args.confocal_image, pchannel, imsave_with_outdir)

if __name__ == "__main__":
    main()
