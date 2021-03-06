"""Given the path to a series of channels and z-stacks, generate a 2D projection
annotated with a cell segmentation and counts of detected RNA probes enumerated
by region of the segmentation."""

import argparse
import os.path

import scipy.misc
import numpy as np

from skimage.morphology import disk, erosion

from fonty import Glyph, Font, Bitmap

from segmentation_from_stack import load_stack_and_segment
from find_probe_locs import find_probe_locations
from protoimg.stack import Stack, normalise_stack
from protoimg.transform import (
    max_intensity_projection,
    equalize_adapthist,
    component_find_centroid
    )

HERE = os.path.dirname(__file__)

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

def segmentation_border_image(segmentation, index, width=1):

    isolated_region = np.zeros(segmentation.image_array.shape, dtype=np.uint8)

    isolated_region[np.where(segmentation.image_array == index)] = 255

    selem = disk(width)
    border = isolated_region - erosion(isolated_region, selem)

    return border

def generate_annotated_image(segmentation, probe_locs, stack_path, imsave, pchannel):

    stack = Stack.from_path(stack_path, channel=pchannel)
    norm_stack = normalise_stack(stack)
    annot_proj = max_intensity_projection(norm_stack, name='annot_proj')

    eqproj = equalize_adapthist(annot_proj.image_array)
    imsave('eqproj.png', eqproj)

    zero_pad = np.zeros(eqproj.shape, eqproj.dtype)
    red_image = np.dstack([eqproj, zero_pad, zero_pad])

    if imsave:
        imsave('pretty_proj.png', red_image)

    white16 = 255 << 8, 255 << 8, 255 << 8
    real_ids = set(np.unique(segmentation.image_array)) - set([0])
    for index in real_ids:
        border = segmentation_border_image(segmentation, index)
        red_image[np.where(border == 255)] = 255 << 8, 255 << 8, 255 << 8
        seg_area = set(zip(*np.where(segmentation.image_array == index)))
        selected_probes = set(probe_locs) & seg_area
        n_probes = len(selected_probes)
        ox, oy = component_find_centroid(segmentation, index)
        text_at(red_image, str(n_probes), ox, oy, white16)

    if imsave:
        imsave('annotated_projection.png', red_image)

def segment_count_annotate(stack_path, imsave, pchannel):
    segmentation = load_stack_and_segment(stack_path, imsave)
    probe_locs = find_probe_locations(stack_path, imsave, pchannel)

    generate_annotated_image(segmentation, probe_locs, stack_path, imsave,
        pchannel)

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stack_path', help="Path to stack files.")
    parser.add_argument('output_dir', help="Path to output directory.")
    parser.add_argument('-p', '--probe_channel',
        default=1, type=int, help="Probe channel (default 1)")
    args = parser.parse_args()

    if args.probe_channel == 0:
        parser.error('Probe channel index is one-based; index zero is invalid.')
    pchannel = args.probe_channel - 1  # Make the index zero-based.

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    def imsave_with_outdir(fname, im):
        """Save images to the specified output directory."""
        fpath = os.path.join(args.output_dir, fname)
        scipy.misc.imsave(fpath, im)

    from protoimg import transform
    transform.imsave = imsave_with_outdir
    segment_count_annotate(args.stack_path,
        imsave=imsave_with_outdir, pchannel=pchannel)

if __name__ == "__main__":
    main()

