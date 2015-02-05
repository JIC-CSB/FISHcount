"""Script to detect and count spots in cells."""

import os
import os.path

import argparse

from libtiff import TIFF

import numpy as np
from scipy import ndimage

from skimage.feature import peak_local_max
from skimage.filter import threshold_otsu, sobel
from skimage.morphology import (
    disk,
    remove_small_objects,
    binary_closing,
    binary_erosion,
    watershed,
)


import matplotlib.pyplot as plt

INPUT_DIR = "/localscratch/olssont/flc_single_mol_analysis/"

def get_fpaths(directory, channel_index):
    """Yield the file paths for a particular channel."""
    fpaths = []
    for fname in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, fname)):
            continue
        name, ext = fname.split('.')
        channel = name.split('_')[-1]
        if channel == 'C{}'.format(channel_index):
            fpaths.append(os.path.join(directory, fname))
    return fpaths

def get_image(fpath):
    """Return image array."""
    tif = TIFF.open(fpath, mode='r')
    im_array = tif.read_image()
    return im_array

def get_average_image_from_stack(stack):
    """Return average image from a stack of images."""
    ar = np.sum(stack, axis=2)
    return ar / stack.shape[2]

def get_average_image_from_fpaths(fpaths):
    """Return average image from a list of image file paths."""
    shape = get_image(fpaths[0]).shape
    ar = np.zeros(shape, dtype=float)
    for fpath in fpaths:
        ar = ar + get_image(fpath)
    return ar / len(fpaths)

def get_stack(fpaths):
    """Return 3D array from a list of image file paths."""
    shape = get_image(fpaths[0]).shape
    shape_3d = shape[0], shape[1], len(fpaths)
    ar = np.zeros(shape_3d, dtype=float)
    for i, fpath in enumerate(fpaths):
        ar[:,:,i] = get_image(fpath)
    return ar

def get_masked_stack(stack, mask):
    """Return masked stack."""
    ar = np.zeros(stack.shape, dtype=float)
    for i in range(stack.shape[2]):
        ar[:,:,i] = stack[:,:,i] * mask
    return ar

def get_local_maxima(im_array, indices, min_distance=50, threshold_rel=0.5):
    """Return the local maxima."""
    return peak_local_max(im_array,
                          indices=indices,
                          min_distance=min_distance,
                          threshold_rel=threshold_rel)
    

def get_markers(im_array):
    """Return markers for the watershed algorithm."""
    local_maxima = get_local_maxima(im_array, indices=False)
    markers = ndimage.label(local_maxima)[0]
    return markers

def get_thresholded_image(im_array):
    """Return thresholded image."""
    thresh = threshold_otsu(im_array)
    return im_array > thresh
    
def get_rois(im_array):
    """Return the regions of interest."""
    markers = get_markers(im_array)
    salem = disk(3)
    im = get_thresholded_image(im_array)
    im = remove_small_objects(im, min_size=50)
    im = binary_closing(im, salem)
    im = watershed(-im_array, markers, mask=im)
    return im

def get_mask_outline(mask):
    """Return mask outline."""
    outline =  sobel(mask) != 0
    outline = outline * 1.0
    outline[outline == 0] = np.nan
    return outline

def get_blobs(im_stack, sigma):
    """Return enhanced blobs from Gaussian of Laplace transformation."""
    stack = ndimage.filters.gaussian_laplace(im_stack, sigma)
    return stack

def analyse_image(directory):
    output_dir = os.path.join(directory, 'analysis')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    # Segment into cells.
    blue_fpaths = get_fpaths(directory, 2)
    blue_average_im = get_average_image_from_fpaths(blue_fpaths)
    blue_rois = get_rois(blue_average_im)
    cell_mask = np.array(blue_rois, dtype=bool)
    mask_outline = get_mask_outline(blue_rois)

    # Display the average blue channel.
    plt.subplot('221')
    plt.imshow(blue_average_im, cmap=plt.cm.Blues)
    plt.title('Blue channel average z-stack projection.', fontsize=10)

    # Display the segmentation from the average blue channel.
    plt.subplot('222')
    plt.imshow(blue_average_im, cmap=plt.cm.Blues)
    plt.imshow(mask_outline, cmap=plt.cm.gray)
    plt.title('Segmentation from average blue channel.', fontsize=10)

    # Find fluorescent blobs.
    green_fpaths = get_fpaths(directory, 0)
    green_average_im = get_average_image_from_fpaths(green_fpaths)
    green_stack = get_stack(green_fpaths)
    green_blobs = get_blobs(green_stack, sigma=3)
    green_blobs_in_cells = get_masked_stack(green_blobs, cell_mask)
    rna_molecules = get_local_maxima(-green_blobs_in_cells,
                                     indices=True,
                                     min_distance=5,
                                     threshold_rel=0.3)

    # Display average green channel.
    plt.subplot('223')
    plt.imshow(green_average_im, cmap=plt.cm.Greens)
    plt.title('Green channel average z-stack projection.', fontsize=10)

    # Display the RNA molecules identified.
    ax = plt.subplot('224')
    plt.imshow(green_average_im, cmap=plt.cm.Greens)
    plt.imshow(mask_outline, cmap=plt.cm.gray)
    ax.autoscale(False)
    plt.plot(rna_molecules[:,1], rna_molecules[:,0],
             marker='o',
             markeredgecolor='red',
             fillstyle='none',
             linestyle='none',
    )
    plt.title('Number of RNA molecules: {}'.format(len(rna_molecules)), fontsize=10)
    plt.savefig(os.path.join(output_dir, 'summary.png'))

    # Write out the stack to a tmp directory.
    green_stack_in_cells = get_masked_stack(green_stack, cell_mask)
    stack_dir = os.path.join(output_dir, 'stack')
    if not os.path.isdir(stack_dir):
        os.mkdir(stack_dir)
    for i in range(green_stack_in_cells.shape[2]):
        im = green_stack_in_cells[:,:,i]
        rna_mols_in_plane = []
        for x in rna_molecules:
            if x[2] == i:
                rna_mols_in_plane.append(x)
        num_rna_mols_in_plane = len(rna_mols_in_plane)
        ax = plt.subplot('111')
        plt.axis('off')
        plt.imshow(im, cmap=plt.cm.Greens)
        ax.autoscale(False)
        plt.plot(rna_molecules[:,1], rna_molecules[:,0],
                 marker='o',
                 markeredgecolor='purple',
                 fillstyle='none',
                 linestyle='none',
        )
        if num_rna_mols_in_plane > 0:
            coords = np.array(rna_mols_in_plane)
            plt.plot(coords[:,1], coords[:,0],
                     marker='o',
                     markeredgecolor='red',
                     fillstyle='none',
                     linestyle='none',
            )
        plt.savefig(os.path.join(stack_dir, '{:04}.tif'.format(i)),
                    dpi=300,
                    bbox_inches='tight')
    
def do_all(input_dir):
    """Do analysis on all subdirs in an input directory."""
    for d in os.listdir(input_dir):
        image_input_dir = os.path.join(input_dir, d)
        if os.path.isdir(image_input_dir):
            print('Working on {}...'.format(image_input_dir))
            analyse_image(image_input_dir)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir')
    args = parser.parse_args()
    do_all(args.input_dir)


