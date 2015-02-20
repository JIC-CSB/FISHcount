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

def unpack_many(image_dir, output_dir_root):
    """Unpack multiple confocal images into a given directory, creating new
    directories for each."""

    image_files = os.listdir(image_dir)

    for f in image_files:
        name, ext = os.path.splitext(os.path.basename(f))
        image_path = os.path.join(image_dir, f)
        output_path = os.path.join(output_dir_root, name)
        unpack(image_path, output_path)

def unpack(image_file, output_dir, output_format='.tif'):
    """Use bioformats to unpack the given confocal image file into a series
    of 2D images, for each series, channel and plane in the file.

    Inputs:

    image_file - the confocal image file
    output_dir - the directory in which output files will be created. This
    directory will be created if it does not exist
    output_format - the format extension for output files
    """

    mkdir_p(output_dir)

    basename = os.path.basename(image_file)
    name, ext = os.path.splitext(basename)

    format_specifier = name + '_S%s_C%c_Z%z' + output_format

    unpack_cmd = [BFCONVERT]
    unpack_cmd += [image_file]
    unpack_cmd += [os.path.join(output_dir, format_specifier)]
     
    subprocess.call(unpack_cmd)

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('image_file', help="Path to file containing microscope image")
    parser.add_argument('data_root', help="Path to root directory to contain unpacked data")

    args = parser.parse_args()

    unpack(args.image_file, args.data_root)
    #unpack_many(args.image_file, args.data_root)

if __name__ == "__main__":
    main()
