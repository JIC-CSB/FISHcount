"""Calculate probe intensities."""

import argparse
import os.path

import numpy as np
from skimage.morphology import disk

from jicimagelib.io import AutoName

from util.transform import scale_median_stack
from jicimagelib.transform import max_intensity_projection

from util import (
    safe_mkdir,
    unpack_data,
    imsave_with_outdir,
    human_to_computer_index,
    computer_to_human_index,
)

from find_probe_locations import find_probe_locations, PROBE_RADIUS

def calculate_probe_intensities(image_collection, probe_list, channel_id):
    """Return list with probe intensities added to it.
    
    Write out csv file with max_intensity and sum_intensity.
    """
    # IS THIS REALLY WHAT WE WANT TO BE CALCULATING INTENSITY VALUES FROM?
    raw_z_stack = image_collection.zstack_array(c=channel_id)
    normed_stack = scale_median_stack(raw_z_stack)
    norm_projection = max_intensity_projection(normed_stack)

    circle = disk(PROBE_RADIUS)

    # Add max_intensity and sum_intensity to each Probe instance in the list.
    for probe in probe_list:
        x, y = probe.astuple()
        pixels = norm_projection[x-PROBE_RADIUS:x+PROBE_RADIUS+1, y-PROBE_RADIUS:y+PROBE_RADIUS+1]
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

def calculate_intensities(image_collection, channel_id, match_thresh, imsave):
    """Return list of probes with intensity measurements."""
    probe_list = find_probe_locations(image_collection, channel_id,
        match_thresh, imsave)
    probe_list = calculate_probe_intensities(image_collection, probe_list,
        channel_id)

    # Write csv results to csv file.
    fname= 'intensities_channel_{}.csv'.format(computer_to_human_index(channel_id))
    fpath = os.path.join(AutoName.directory, fname)
    write_probe_instensity_csv_file(probe_list, fpath)

    return probe_list

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
    calculate_intensities(image_collection, pchannel, args.threshold,
        imsave_with_outdir)

if __name__ == "__main__":
    main()
