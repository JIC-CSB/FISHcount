"""Segment image into cells based on background fluorescence."""

import argparse

from jicimagelib.io import AutoName
from jicimagelib.transform import (
    max_intensity_projection, 
    equalize_adaptive_clahe,
    smooth_gaussian,
    threshold_otsu,
    remove_small_objects,
    min_intensity_projection,
)

from util import (
    unpack_data,
    safe_mkdir,
    human_to_computer_index,
)

from util.transform import (
    scale_median_stack,
    find_edges,
    find_connected_components,
    component_centroids,
    watershed_with_seeds,
    filter_segmentation,
)

def generate_segmentation_seeds(raw_z_stack):
    """Generate the seeds for segmentation from the z stack."""

    normed_stack = scale_median_stack(raw_z_stack)
    max_projection = max_intensity_projection(normed_stack)
    eq_proj = equalize_adaptive_clahe(max_projection, ntiles=16)
    gauss = smooth_gaussian(eq_proj, sigma=3)
    edges = find_edges(gauss)
    thresh = threshold_otsu(edges, multiplier=1)
    nosmall = remove_small_objects(thresh, min_size=500)
    connected_components = find_connected_components(nosmall, background=None)
    seeds = component_centroids(connected_components)

    return seeds

def segment_image(image_collection, nuclear_channel):
    """Segment the image."""
    
    nuclear_stack = image_collection.zstack_array(c=nuclear_channel)
    seeds = generate_segmentation_seeds(nuclear_stack)

    min_autof_proj = min_intensity_projection(nuclear_stack)
    equal_autof = equalize_adaptive_clahe(min_autof_proj)
    smoothed_autof = smooth_gaussian(equal_autof, sigma=5)
    edge_autof = find_edges(smoothed_autof)
    thresh_autof = threshold_otsu(smoothed_autof, multiplier=0.6)

    segmentation = watershed_with_seeds(smoothed_autof, seeds,
                               mask_image=thresh_autof)

    filtered_segmentation = filter_segmentation(segmentation)

    re_watershed = watershed_with_seeds(smoothed_autof, filtered_segmentation,
                                mask_image=thresh_autof)

    return re_watershed

def main():
    
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_image', help='Confocal image to analyse')
    parser.add_argument('output_dir', help='Path to output directory.')
    parser.add_argument('-n', '--nuclear_channel',
            type=int,
            default=3,
            help='Used to segment image into cells (default=3)')
    args = parser.parse_args()

    nchannel = human_to_computer_index(args.nuclear_channel)

    safe_mkdir(args.output_dir)
    AutoName.directory = args.output_dir

    image_collection = unpack_data(args.confocal_image)
    segmentation = segment_image(image_collection, nchannel)

if __name__ == "__main__":
    main()
