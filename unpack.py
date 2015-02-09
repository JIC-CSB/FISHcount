"""Read directory with LIF files, create appropriate directory structure in 
which to unpack and do so"""

import os
import errno
import shutil
import argparse
import subprocess

BFCONVERT = '/common/tools/bftools/bfconvert'

def mkdir_p(path):
    try:
        os.makedirs(path)   
    except OSError as exc:
        # FIXME
        if exc.errno == errno.EEXIST:
            pass
        else: raise

def parse_file_name(filename):
    name, ext = os.path.splitext(filename)

    return name.split('_')

def unpack(image_file, output_dir, output_format='.tif'):
    """Use bioformats to unpack the given confocal image file into a series
    of 2D images, for each series, channel and plane in the file.

    Inputs:

    image_file - the confocal image file
    output_dir - the directory in which output files will be created
    output_format - the format extension for output files
    """

    basename = os.path.basename(image_file)
    name, ext = os.path.splitext(basename)

    format_specifier = name + '_S%s_C%c_Z%z' + output_format

    unpack_cmd = [BFCONVERT]
    unpack_cmd += [image_file]
    unpack_cmd += [os.path.join(output_dir, format_specifier)]
     
    subprocess.call(unpack_cmd)

def setup_and_unpack(image_file, output_dir):
    """Create the output directory if it does not exist, then unpack the
    image file."""

    mkdir_p(output_dir)
    unpack(image_file, output_dir)

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('image_file', help="Path to file containing microscope image")
    parser.add_argument('data_root', help="Path to root directory to contain unpacked data")

    args = parser.parse_args()

    setup_and_unpack(args.image_file, args.data_root)

if __name__ == "__main__":
    main()
