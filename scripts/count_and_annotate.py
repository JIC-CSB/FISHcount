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
    safe_mkdir,
    unpack_data,
    imsave_with_outdir,
    human_to_computer_index
)

from util.transform import (
    scale_median_stack,
    convert_to_uint8,
    component_find_centroid
)

from util.annotate import text_at

from segment import segment_image
from find_probe_locations import find_probe_locations

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
        selected_probes = set([p.astuple() for p in probe_locs]) & seg_area
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

def count_and_annotate(confocal_image, nuclear_channel, pchannels, thresholds, imsave):
    """Find probe locations, segment the image and produce an annotated image
    showing probe counts per identified cell. Probe locations are found for
    each channel in the list pchannnels."""

    image_collection = unpack_data(confocal_image)

    segmentation = segment_image(image_collection, nuclear_channel=nuclear_channel)

    probe_stacks = [image_collection.zstack_array(c=pc) for pc in pchannels]
    probe_location_sets = [find_probe_locations(image_collection, i, t, imsave)
        for i, (ps, t) in enumerate(zip(probe_stacks, thresholds))]

    generate_annotated_image(segmentation, probe_location_sets, 
                             probe_stacks, imsave)

def main():
    
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_image', help='Confocal image to analyse')
    parser.add_argument('output_dir', help='Path to output directory.')
    parser.add_argument('-o', '--only_rna_probe_channel',
            default=False,
            action="store_true",
            help='Only find probes in the RNA channel')
    parser.add_argument('-r', '--rna_probe_channel',
            type=int, default=1, help='RNA probe channel (default 1)')
    parser.add_argument('-u', '--unspliced_probe_channel',
            type=int, default=2, help='RNA probe channel (default 2)')
    parser.add_argument('-n', '--nuclear_channel',
            type=int, default=3, help='Nuclear channel (default 2)')
    parser.add_argument('-t', '--rna_probe_channel_threshold',
            type=float, default=0.6,
            help='RNA probe channel threshold (default 0.6)')
    parser.add_argument('-s', '--unspliced_probe_channel_threshold',
            type=float, default=0.8,
            help='Unspliced probe channel threshold (default 0.8)')
    args = parser.parse_args()

    nchannel = human_to_computer_index(args.nuclear_channel)
    pchannels = map(human_to_computer_index, [args.rna_probe_channel,
        args.unspliced_probe_channel])
    thresholds = [args.rna_probe_channel_threshold,
        args.unspliced_probe_channel_threshold]

    if args.only_rna_probe_channel:
        pchannels = pchannels[0:1]
        thresholds = thresholds[0:1]

    if any(c<0 for c in pchannels):
        parser.error('Probe channel index is one-based; index zero is invalid.')

    safe_mkdir(args.output_dir)

    AutoName.directory = args.output_dir

    count_and_annotate(args.confocal_image, nchannel, pchannels, thresholds,
        imsave_with_outdir)

if __name__ == "__main__":
    main()
