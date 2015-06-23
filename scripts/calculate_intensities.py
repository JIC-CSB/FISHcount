"""Calculate probe intensities."""

import argparse
import os.path

import numpy as np
from skimage.morphology import disk

from jicimagelib.io import AutoName

from util.transform import scale_median_stack, convert_to_uint8
from jicimagelib.transform import max_intensity_projection

from util import (
    safe_mkdir,
    unpack_data,
    imsave_with_outdir,
    human_to_computer_index,
    computer_to_human_index,
    grayscale_to_rgb,
)

from util.annotate import draw_cross

from find_probe_locations import find_probe_locations, PROBE_RADIUS

def custom_sort(probe_list, attribute, reverse=True):
    """Return probe list sorted by attribute."""
    l = sorted(probe_list, key=lambda p: getattr(p, attribute))
    if reverse:
        l.reverse()
    return l

def test_custom_sort():
    from jicimagelib.geometry import Point2D
    points = [Point2D(i,i) for i in range(3)]
    points[0].max_intensity = 5
    points[1].max_intensity = 3
    points[2].max_intensity = 10

    points = custom_sort(points, 'max_intensity')

    assert points[0].max_intensity == 10
    assert points[1].max_intensity == 5
    assert points[2].max_intensity == 3

def calculate_probe_intensities(projection, probe_list, channel_id):
    """Return list with probe intensities added to it."""
    circle = disk(PROBE_RADIUS)
    for probe in probe_list:
        x, y = probe.astuple()
        pixels = projection[x-PROBE_RADIUS:x+PROBE_RADIUS+1, y-PROBE_RADIUS:y+PROBE_RADIUS+1]
        probe.max_intensity = np.max(pixels * circle)
        probe.sum_intensity = np.sum(pixels * circle)
    return probe_list

def write_probe_instensity_csv_file(probe_list, fpath):
    """Write probe intensity csv file."""
    with open(fpath, 'w') as fh:
        fh.write('"x","y","max_intensity","sum_intensity"\n')
        for probe in probe_list:
            x, y = probe.astuple()
            fh.write('{},{},{},{}\n'.format(x, y, probe.max_intensity, probe.sum_intensity))

def generate_annotated_intensities_image(projection, probe_list, fraction, fname, imsave):
    """Generate image with top 10% of probes annotated."""
    projection = convert_to_uint8(projection)
    probe_loc_image = grayscale_to_rgb(projection)
    number_to_include = int(fraction * len(probe_list))
    for attribute, color, offset in ( ('max_intensity', (255,0,0), 0),
        ('sum_intensity', (0, 0, 255), 1) ):
        sorted_probes = custom_sort(probe_list, attribute)
        for i in range(number_to_include):
            x, y = sorted_probes[i]
            draw_cross(probe_loc_image, x+offset, y+offset, color)
    imsave(fname, probe_loc_image)

def calculate_intensities(image_collection, channel_id, match_thresh, fraction, imsave):
    """Return list of probes with intensity measurements."""

    probe_list = find_probe_locations(image_collection, channel_id,
        match_thresh, imsave)
    
    # IS THIS REALLY WHAT WE WANT TO BE CALCULATING INTENSITY VALUES FROM?
    raw_z_stack = image_collection.zstack_array(c=channel_id)
    normed_stack = scale_median_stack(raw_z_stack)
    norm_projection = max_intensity_projection(normed_stack)

    probe_list = calculate_probe_intensities(norm_projection, probe_list,
        channel_id)

    # Write csv results to csv file.
    fname= 'intensities_channel_{}.csv'.format(computer_to_human_index(channel_id))
    fpath = os.path.join(AutoName.directory, fname)
    write_probe_instensity_csv_file(probe_list, fpath)

    # Generate image annotated with top intensity probes.
    fname= 'annotated_intensities_channel_{}.png'.format(computer_to_human_index(channel_id))
    generate_annotated_intensities_image(norm_projection, probe_list, fraction, fname,
        imsave)

    return probe_list

def main():
    
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_image', help='Confocal image to analyse')
    parser.add_argument('output_dir', help='Path to output directory.')
    parser.add_argument('-t', '--threshold', type=float, default=0.6,
        help="Threshold for spot detection (default 0.6)")
    parser.add_argument('-f', '--fraction', type=float, default=0.1,
        help="Fraction to annotate on output image (default 0.1)")
    parser.add_argument('-c', '--channel',
        type=int,
        default=1,
        help='Channel to identify spots in (default=1)')
    args = parser.parse_args()

    pchannel = human_to_computer_index(args.channel)

    safe_mkdir(args.output_dir)
    AutoName.directory = args.output_dir

    image_collection = unpack_data(args.confocal_image)
    calculate_intensities(image_collection, pchannel, args.threshold,
        args.fraction, imsave_with_outdir)

if __name__ == "__main__":
    main()
