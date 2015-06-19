"""Find probe locations."""

import argparse

import numpy as np

from skimage.morphology import disk#, erosion
from skimage.feature import match_template

from jicimagelib.io import AutoName
from jicimagelib.transform import (
    max_intensity_projection, 
)

from util import (
    safe_mkdir,
    unpack_data,
    imsave_with_outdir,
    grayscale_to_rgb,
    human_to_computer_index,
)

from util.transform import (
    scale_median_stack,
    find_edges,
    find_connected_components,
    component_centroids,
)

from util.annotate import (
    random_rgb,
    draw_cross,
)

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

def find_probe_locations(image_collection, channel_id, match_thresh, imsave):
    """Return probe locations as list of x,y coordinates."""

    raw_z_stack = image_collection.zstack_array(c=channel_id)
    normed_stack = scale_median_stack(raw_z_stack)
    norm_projection = max_intensity_projection(normed_stack)
    edges = find_edges(norm_projection)
    exemplar = find_best_template(edges)
    match_result = match_template(edges, exemplar, pad_input=True)
    imsave('match_result_channel_{}.png'.format(channel_id+1), match_result)

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

#   calculate_probe_intensities(norm_projection, probe_locs, channel_id)

    return probe_locs


def main():
    
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_image', help='Confocal image to analyse')
    parser.add_argument('output_dir', help='Path to output directory.')
    parser.add_argument('-t', '--threshold', type=float, default=0.6,
        help="Threshold for spot detection (default 0.6)")
    parser.add_argument('-c', '--channel',
        type=int,
        default=1,
        help='Channel to identify spots in (default=1)')
    args = parser.parse_args()

    pchannel = human_to_computer_index(args.channel)

    safe_mkdir(args.output_dir)
    AutoName.directory = args.output_dir

    image_collection = unpack_data(args.confocal_image)
    probe_locations = find_probe_locations(image_collection,
        pchannel,
        args.threshold,
        imsave_with_outdir)

if __name__ == "__main__":
    main()
