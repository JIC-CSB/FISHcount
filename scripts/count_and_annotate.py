"""Locate RNA FISH probes."""

import os
import errno
import argparse

import numpy as np

from jicimagelib.io import FileBackend
from jicimagelib.image import DataManager, Image
from jicimagelib.transform import transformation

import scipy.misc

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
    filter_segmentation
)

#from segmentation_from_stack import segmentation_from_stacks

HERE = os.path.dirname(__file__)
UNPACK = os.path.join(HERE, '..', 'data', 'jic_backend')

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


def count_and_annotate(confocal_image, imsave):

    image_collection = unpack_data(confocal_image)
    
    nuclear_z_stack = image_collection.zstack_array(c=2)
    seeds = generate_segmentation_seeds(nuclear_z_stack)

    probe_stack = image_collection.zstack_array(c=1)
    min_autof_proj = min_intensity_projection(probe_stack)
    scipy.misc.imsave('mymin.png', min_autof_proj)
    equal_autof = equalize_adaptive(min_autof_proj)
    smoothed_autof = gaussian_filter(equal_autof, sigma=5)
    edge_autof = find_edges(smoothed_autof)
    thresh_autof = threshold_otsu(smoothed_autof, mult=0.6)

    segmentation = watershed_with_seeds(smoothed_autof, seeds,
                               mask_image=thresh_autof)

    filtered_segmentation = filter_segmentation(segmentation)

    re_watershed = watershed_with_seeds(smoothed_autof, filtered_segmentation,
                                mask_image=thresh_autof)

    #segmentation_from_stacks(raw_z_stack, raw_z_stack, imsave)

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

    count_and_annotate(args.confocal_image, imsave_with_outdir)

if __name__ == "__main__":
    main()
