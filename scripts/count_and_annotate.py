"""Locate RNA FISH probes."""

import os
import argparse

import numpy as np

import scipy.misc
from skimage.morphology import disk, erosion
from skimage.feature import match_template

from jicimagelib.io import AutoName
from jicimagelib.transform import (
    max_intensity_projection, 
    equalize_adaptive_clahe,
)

from util import (
    grayscale_to_rgb,
    safe_mkdir,
    unpack_data,
    parse_probe_channels,
)

from util.transform import (
    scale_median_stack,
    convert_to_uint8,
    find_edges,
    find_connected_components,
    component_centroids,
    component_find_centroid
)

from util.annotate import (
    text_at,
    draw_cross,
    random_rgb
)

from segment import segment_image

#from segmentation_from_stack import segmentation_from_stacks

PROBE_RADIUS = 3


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

    eqproj = equalize_adaptive_clahe(annot_proj)
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

    segmentation = segment_image(image_collection, channel=2)

    probe_stacks = [image_collection.zstack_array(c=pc) for pc in pchannels]
    probe_location_sets = [find_probe_locations(ps, i, imsave)
        for i, ps in enumerate(probe_stacks)]

    generate_annotated_image(segmentation, probe_location_sets, 
                             probe_stacks, imsave)

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
