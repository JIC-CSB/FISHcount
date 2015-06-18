"""Locate RNA FISH probes."""

import os
import errno
import random
import argparse

import numpy as np

import scipy.misc
from skimage.morphology import disk, erosion
from skimage.feature import match_template

from jicimagelib.io import FileBackend, AutoName
from jicimagelib.image import DataManager, Image
from jicimagelib.transform import (
    max_intensity_projection, 
    min_intensity_projection,
    smooth_gaussian,
    remove_small_objects,
)


from protoimg2.transform import (
    scale_median_stack,
    convert_to_uint8,
    equalize_adaptive,
    find_edges,
    threshold_otsu,
    find_connected_components,
    component_centroids,
    watershed_with_seeds,
    filter_segmentation,
    component_find_centroid
)

from protoimg2.annotate import (
    text_at,
    draw_cross,
    random_rgb
)

#from segmentation_from_stack import segmentation_from_stacks

HERE = os.path.dirname(__file__)
UNPACK = os.path.join(HERE, '..', 'data', 'jic_backend')

PROBE_RADIUS = 3


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

    normed_stack = scale_median_stack(raw_z_stack)
    max_projection = max_intensity_projection(normed_stack)
    compressed = convert_to_uint8(max_projection)
    eq_proj = equalize_adaptive(compressed, n_tiles=16)
    gauss = smooth_gaussian(eq_proj, sigma=3)
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
    compressed = convert_to_uint8(min_autof_proj)
    equal_autof = equalize_adaptive(compressed)
    smoothed_autof = smooth_gaussian(equal_autof, sigma=5)
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

    template = disk(PROBE_RADIUS)
    template[PROBE_RADIUS, PROBE_RADIUS] = 0

    return template

def find_best_template(edges):
    """Find the best exemplar of a probe in the image. We use template matching
    with a hollow disk (annulus), and return the best match in the image."""

    template = make_stage1_template()
    stage1_match = match_template(edges, template, pad_input=True)
    #imsave('stage1_match.png', stage1_match)
    cmax = np.max(stage1_match)
    px, py = zip(*np.where(stage1_match == cmax))[0]

    better_template = edges[px-PROBE_RADIUS:px+PROBE_RADIUS+1,py-PROBE_RADIUS:py+PROBE_RADIUS+1]
    #imsave('better_template.png', better_template)

    return better_template

def generate_probe_loc_image(norm_projection, probe_locs, channel_id, imsave):
    """Generate an annotated image showing the probe locations as crosses."""

    probe_loc_image = grayscale_to_rgb(norm_projection)
    for coords in probe_locs:
        x, y = coords
        c = random_rgb()
        draw_cross(probe_loc_image, x, y, c)
    
    imsave('probe_locations_channel_{}.png'.format(channel_id+1), probe_loc_image)

def calculate_probe_intensities(norm_projection, probe_locs, channel_id):
    """Calculate the probe intensities."""
    circle = disk(PROBE_RADIUS)
    fname= 'intensities_channel_{}.csv'.format(channel_id+1)
    fpath = os.path.join(AutoName.directory, fname)
    with open(fpath, 'w') as fh:
        fh.write('"x","y","max_intensity","sum_intensity"\n')
        for x, y in probe_locs:
            pixels = norm_projection[x-PROBE_RADIUS:x+PROBE_RADIUS+1, y-PROBE_RADIUS:y+PROBE_RADIUS+1]
            max_intensity = np.max(pixels * circle)
            sum_intensity = np.sum(pixels * circle)
            fh.write('{},{},{},{}\n'.format(x, y, max_intensity, sum_intensity))

def find_probe_locations(raw_z_stack, channel_id, imsave):

    normed_stack = scale_median_stack(raw_z_stack)
    norm_projection = max_intensity_projection(normed_stack)
    edges = find_edges(norm_projection)
    exemplar = find_best_template(edges)
    match_result = match_template(edges, exemplar, pad_input=True)
    imsave('match_result_channel_{}.png'.format(channel_id+1), match_result)

    if channel_id == 0:
        match_thresh = 0.6
    if channel_id == 1:
        match_thresh = 0.8

    locs = np.where(match_result > match_thresh)
    annotated_edges = grayscale_to_rgb(edges)
    annotated_edges[locs] = edges.max(), 0, 0
    imsave('annotated_edges_channel_{}.png'.format(channel_id+1), annotated_edges)

    cloc_array = match_result > match_thresh
    imsave('cloc_array_channel_{}.png'.format(channel_id+1), cloc_array)
    connected_components = find_connected_components(cloc_array)
    centroids = component_centroids(connected_components)

    probe_locs = zip(*np.where(centroids != 0))

    generate_probe_loc_image(norm_projection, probe_locs, channel_id, imsave)

    calculate_probe_intensities(norm_projection, probe_locs, channel_id)

    return probe_locs

def segmentation_border_image(segmentation, index, width=1):

    isolated_region = np.zeros(segmentation.shape, dtype=np.uint8)

    isolated_region[np.where(segmentation == index)] = 255

    selem = disk(width)
    border = isolated_region - erosion(isolated_region, selem)

    return border

def generate_annotated_channel(segmentation, probe_locs, stack, imsave):
    norm_stack = scale_median_stack(stack)
    annot_proj = max_intensity_projection(norm_stack)

    compressed = convert_to_uint8( annot_proj)
    eqproj = equalize_adaptive(compressed)
    eqproj_uint8 = convert_to_uint8(eqproj)

    zero_pad = np.zeros(eqproj_uint8.shape, eqproj_uint8.dtype)
    red_image = np.dstack([eqproj_uint8, zero_pad, zero_pad])

    white16 = 255 << 8, 255 << 8, 255 << 8
    white8 = 255, 255, 255
    real_ids = set(np.unique(segmentation)) - set([0])

    for index in real_ids:
        border = segmentation_border_image(segmentation, index)
        red_image[np.where(border == 255)] = white8
        seg_area = set(zip(*np.where(segmentation == index)))

        probe_counts = []
        selected_probes = set(probe_locs) & seg_area
        n_probes = len(selected_probes)
        probe_counts.append(n_probes)

        ox, oy = component_find_centroid(segmentation, index)
        pixel_area = len(seg_area)

        probe_count_string = '/'.join(str(pc) for pc in probe_counts)
        text_at(red_image, probe_count_string, ox, oy, white8)

        text_at(red_image, str(pixel_area), ox+30, oy-20, white8)

    return red_image
    
def generate_annotated_image(segmentation, probe_loc_sets, stacks, imsave):
    annotated_image = None
    for i in range(len(stacks)):
        tmp = generate_annotated_channel(segmentation, 
            probe_loc_sets[i], stacks[i], imsave)
        if annotated_image is None:
            annotated_image = tmp
        else:
            annotated_image = np.concatenate((annotated_image, tmp), axis=1)
    
    if imsave:
        imsave('annotated_projection.png', annotated_image)

def count_and_annotate(confocal_image, pchannels, imsave):
    """Find probe locations, segment the image and produce an annotated image
    showing probe counts per identified cell. Probe locations are found for
    each channel in the list pchannnels."""

    image_collection = unpack_data(confocal_image)

    segmentation = segment_image(image_collection)

    probe_stacks = [image_collection.zstack_array(c=pc) for pc in pchannels]
    probe_location_sets = [find_probe_locations(ps, i, imsave)
        for i, ps in enumerate(probe_stacks)]

    generate_annotated_image(segmentation, probe_location_sets, 
                             probe_stacks, imsave)

def parse_probe_channels(probe_channels_as_string):
    """Parse the command line input to specify which probe channels should be
    analysed."""

    probe_channel_list = probe_channels_as_string.split(',')

    probe_channel_int_list = map(int, probe_channel_list)

    def subtract1(input_int):
        return input_int - 1

    return map(subtract1, probe_channel_int_list)

def test_parse_probe_channels():

    example_input = "1,2"
    parsed_input = parse_probe_channels(example_input)
    assert(parsed_input == [0, 1])

    example_input = "1"
    parsed_input = parse_probe_channels(example_input)
    assert(parsed_input == [0])

def main():
    
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_image', help='Confocal image to analyse')
    parser.add_argument('output_dir', help='Path to output directory.')
    parser.add_argument('-p', '--probe_channels',
            default='1', help='Probe channels, comma separated (default 1)')
    args = parser.parse_args()

    pchannels = parse_probe_channels(args.probe_channels)

    if any(c<0 for c in pchannels):
        parser.error('Probe channel index is one-based; index zero is invalid.')

    safe_mkdir(args.output_dir)

    AutoName.directory = args.output_dir
    def imsave_with_outdir(fname, im):
        """Save images to the specified output directory."""
        fpath = os.path.join(args.output_dir, fname)
        scipy.misc.imsave(fpath, im)

    count_and_annotate(args.confocal_image, pchannels, imsave_with_outdir)

if __name__ == "__main__":
    main()
