"""Script to detect and count spots in cells."""

import os
import os.path

import argparse

import re

from libtiff import TIFF

import numpy as np
from scipy import ndimage

from skimage.color import gray2rgb
from skimage.feature import peak_local_max
from skimage.filter import threshold_otsu, sobel
from skimage.draw import circle_perimeter
from skimage.morphology import (
    disk,
    remove_small_objects,
    binary_closing,
    binary_erosion,
    watershed,
)


import matplotlib.pyplot as plt

INPUT_DIR = "/localscratch/olssont/flc_single_mol_analysis/"

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

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
    return sorted_nicely(fpaths)

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
    ar = ar / len(fpaths)
    return np.array(ar, dtype=np.uint16)

def get_stack(fpaths):
    """Return 3D array from a list of image file paths."""
    shape = get_image(fpaths[0]).shape
    shape_3d = shape[0], shape[1], len(fpaths)
    ar = np.zeros(shape_3d, dtype=np.uint16)
    for i, fpath in enumerate(fpaths):
        ar[:,:,i] = get_image(fpath)
    return ar

def get_masked_stack(stack, mask):
    """Return masked stack."""
    ar = np.zeros(stack.shape, dtype=np.uint16)
    for i in range(stack.shape[2]):
        ar[:,:,i] = stack[:,:,i] * mask
    return ar

def get_local_maxima(im_array, min_distance, threshold_rel):
    """Return the local maxima."""
    img =  peak_local_max(im_array,
                          indices=False,
                          min_distance=min_distance,
                          threshold_rel=threshold_rel)
    coords =  peak_local_max(im_array,
                             indices=True,
                             min_distance=min_distance,
                             threshold_rel=threshold_rel)
    return img, coords
    

def get_markers(im_array, min_distance, threshold_rel):
    """Return markers for the watershed algorithm."""
    img, coords = get_local_maxima(im_array, min_distance, threshold_rel)
    markers = ndimage.label(img)[0]
    return markers, coords

def get_thresholded_image(im_array):
    """Return thresholded image."""
    thresh = threshold_otsu(im_array)
    return im_array > thresh
    
def get_rois(im_array, markers):
    """Return the regions of interest."""
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
    return ndimage.filters.gaussian_laplace(im_stack, sigma)

def save_summary_image(output_dir, blue_average_im, green_average_im, nuclei_coords, mask_outline, rna_molecules):
    """Save a summary image."""

    # Display the average blue channel.
    plt.subplot('221')
    plt.imshow(blue_average_im, cmap=plt.cm.Blues)
    plt.plot(nuclei_coords[:,1], nuclei_coords[:,0], 'r.')
    plt.title('Blue channel average z-stack projection.', fontsize=10)

    # Display the segmentation from the average blue channel.
    plt.subplot('222')
    plt.imshow(blue_average_im, cmap=plt.cm.Blues)
    plt.imshow(mask_outline, cmap=plt.cm.gray)
    plt.title('Segmentation from average blue channel.', fontsize=10)

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

def save_augmented_rna_stack(output_dir, green_stack, green_blobs, cell_mask, rna_molecules):
    """Save stack of green channel with detection highlighted."""
    def get_circle_im(shape, coordinates, radius=3):
        im = np.zeros(shape, dtype=bool)
        for x, y, z in coordinates:
            rr, cc = circle_perimeter(x, y, radius)
            im[rr, cc] = True
        return im
        
    def scale_uint16_to_uint8(uint16):
        uint16_max = 65535.0
        uint8_max = 255.0
        scale = uint8_max / uint16_max
        scaled =  np.array(uint16*scale, dtype=float)
        scaled[ scaled > 255 ] = 255
        return np.array(scaled, dtype=np.uint8)

    def get_augmented_rgb(im, all_circles, in_plane_circles):
        rgb = gray2rgb(im)
        rgb = scale_uint16_to_uint8(rgb)
        red = rgb[:,:,0]
        green = rgb[:,:,1]
        blue = rgb[:,:,2]

        # Add the circles
        red[all_circles] = 255
        green[all_circles] = 0
        blue[all_circles] = 0
        blue[in_plane_circles] = 255
        return red, green,blue

    def append_images(im1, im2):
        rows1 = im1.shape[0]
        rows2 = im2.shape[0]
        return np.concatenate((im1, im2), axis=1)

    green_stack_in_cells = get_masked_stack(green_stack, cell_mask)
    green_blobs_in_cells = get_masked_stack(green_blobs, cell_mask)
    all_circles = get_circle_im(cell_mask.shape, rna_molecules)

    stack_dir = os.path.join(output_dir, 'stack')
    if not os.path.isdir(stack_dir):
        os.mkdir(stack_dir)

    for i in range(green_stack_in_cells.shape[2]):
        org_im = green_stack_in_cells[:,:,i]
        blob_im = green_blobs_in_cells[:,:,i]
        rna_mols_in_plane = []
        for x in rna_molecules:
            if x[2] == i:
                rna_mols_in_plane.append(x)
        in_plane_circles = get_circle_im(cell_mask.shape, rna_mols_in_plane)

        # Create rgb channels.
        r1, g1, b1 = get_augmented_rgb(org_im, all_circles, in_plane_circles)
        r2, g2, b2 = get_augmented_rgb(blob_im, all_circles, in_plane_circles)
        red = append_images(r1, r2)
        green = append_images(g1, g2)
        blue = append_images(b1, b2)
        
        # Write the augmented rgb tif.
        out_fn = os.path.join(stack_dir, '{:04}.tif'.format(i))
        tif = TIFF.open(out_fn, 'w')
        rgb_stack = np.array([red, green, blue])
        tif.write_image(rgb_stack, write_rgb=True)

def analyse_image(directory):
    print('Working on {}...'.format(directory))
    output_dir = os.path.join(directory, 'analysis')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Create lists of file paths.
    green_fpaths = get_fpaths(directory, 0)
    blue_fpaths = get_fpaths(directory, 2)
    
    # Create z-projected average images.
    blue_average_im = get_average_image_from_fpaths(blue_fpaths)
    green_average_im = get_average_image_from_fpaths(green_fpaths)

    # Create markers for the segmentation.
    nuclei_loc_im, nuclei_coords = get_markers(blue_average_im,
                                           min_distance=50,
                                           threshold_rel=0.5)

    # Segment into cells.
    rois = get_rois(green_average_im, nuclei_loc_im)
    cell_mask = np.array(rois, dtype=bool)
    mask_outline = get_mask_outline(rois)

    # Find fluorescent blobs.
    green_stack = get_stack(green_fpaths)
    green_blobs = get_blobs(green_stack, sigma=0)
    green_blobs_in_cells = get_masked_stack(green_blobs, cell_mask)
    _, rna_molecules = get_local_maxima(green_blobs_in_cells,
                                     min_distance=5,
                                     threshold_rel=0.5)


    # Save summary image.
    save_summary_image(output_dir,
                       blue_average_im,
                       green_average_im,
                       nuclei_coords,
                       mask_outline,
                       rna_molecules)

    # Save the augmented stack.
    save_augmented_rna_stack(output_dir,
                             green_stack,
                             green_blobs,
                             cell_mask,
                             rna_molecules)

def do_all(input_dir):
    """Do analysis on all subdirs in an input directory."""
    for d in os.listdir(input_dir):
        image_input_dir = os.path.join(input_dir, d)
        if os.path.isdir(image_input_dir):
            analyse_image(image_input_dir)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir')
    args = parser.parse_args()
#   do_all(args.input_dir)
    analyse_image(args.input_dir)


