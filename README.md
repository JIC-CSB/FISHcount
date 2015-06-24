# Counting single RNA molecule probes

This software is intended to assist with the enumeration of single molecule
RNA probes. It operates on microscope data files, eventually outputting an
annotated image showing a projection of the data with suggested cell boundaries
and probe counts for those cells.

## Installation notes

### Install BioFormats ``bftools`` version 4.4.4

**Warning does not work with BioFormats greater than version 4.4.4**

Create a directory for the BioFormats tools and download the Java programs into
the newly created directory.

```bash
mkdir -p ~/tools/bftools
cd ~/tools/bftools
wget http://downloads.openmicroscopy.org/bio-formats/4.4.4/loci_tools.jar
wget http://downloads.openmicroscopy.org/bio-formats/4.4.4/bftools.zip
unzip bftools.zip
```

Add the BioFormats tools directory to your ``$PATH`` environment variable.
(It may be worth adding the line below to your .bashrc file.)

```bash
export PATH=$PATH:~/tools/bftools
```

### Make sure that freetype is installed

#### Linux

On a yum based system like Fedora:

```
sudo yum install freetype
```

#### Mac

Ensure that you have Xcode installed, for example by running ``gcc`` in the
terminal. Alternatively, you can use the app store.

```bash
gcc
```

Install Homebrew.

```bash
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Install freetype using Homebrew.

```bash
brew install freetype
```

### Create a Python virtual environment to run the scripts from

Install ``virtualenv`` using ``easy_install`` (virtualenv allows you to create
a virtual Python environment).

```bash
sudo easy_install virtualenv
```

Create a virtual environment.

```bash
virtualenv env
```

Source the virtual environment (note the ``.`` at the start of the line).

```bash
. ./env/bin/activate
```

Install Python dependencies into the virtual environment.

```bash
pip install numpy
pip install pillow
pip install scipy
pip install scikit-image
pip install freetype-py
pip install jicimagelib
```


## Image analysis

**Ensure that the Python virtual environment has been sourced before trying to
run any scripts.**

```bash
. ./env/bin/activate
```


### Count spots and calculate cell areas

```bash
python scripts/count_and_annotate.py /path/to/microscopy.czi /path/to/output/dir
```

Note that the image of most interest is ``annotated_projection.png``.

For more help on the script use the ``-h`` flag.

```
python scripts/count_and_annotate.py -h
usage: Locate RNA FISH probes. [-h] [-o] [-r RNA_PROBE_CHANNEL]
                               [-u UNSPLICED_PROBE_CHANNEL]
                               [-n NUCLEAR_CHANNEL]
                               [-t RNA_PROBE_CHANNEL_THRESHOLD]
                               [-s UNSPLICED_PROBE_CHANNEL_THRESHOLD]
                               confocal_image output_dir

positional arguments:
  confocal_image        Confocal image to analyse
  output_dir            Path to output directory.

optional arguments:
  -h, --help            show this help message and exit
  -o, --only_rna_probe_channel
                        Only find probes in the RNA channel
  -r RNA_PROBE_CHANNEL, --rna_probe_channel RNA_PROBE_CHANNEL
                        RNA probe channel (default 1)
  -u UNSPLICED_PROBE_CHANNEL, --unspliced_probe_channel UNSPLICED_PROBE_CHANNEL
                        RNA probe channel (default 2)
  -n NUCLEAR_CHANNEL, --nuclear_channel NUCLEAR_CHANNEL
                        Nuclear channel (default 2)
  -t RNA_PROBE_CHANNEL_THRESHOLD, --rna_probe_channel_threshold RNA_PROBE_CHANNEL_THRESHOLD
                        RNA probe channel threshold (default 0.6)
  -s UNSPLICED_PROBE_CHANNEL_THRESHOLD, --unspliced_probe_channel_threshold UNSPLICED_PROBE_CHANNEL_THRESHOLD
                        Unspliced probe channel threshold (default 0.8)
```

### Calculate intensities

```bash
python scripts/calculate_intensities.py /path/to/microscoy.czi /path/to/output/dir
```

Note that the image of most interest is
``annotated_intensities_channel_1.png``, which with the default settings shows
the top 10% of max (red) and sum (blue) intensities.

All the intensities are in the file ``intensities_channel_1.csv``.

For more help use the ``-h`` flag.

```
$ python scripts/calculate_intensities.py -h
usage: Calculate probe intensities. [-h] [-t THRESHOLD] [-f FRACTION]
                                    [-c CHANNEL]
                                    confocal_image output_dir

positional arguments:
  confocal_image        Confocal image to analyse
  output_dir            Path to output directory.

optional arguments:
  -h, --help            show this help message and exit
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold for spot detection (default 0.6)
  -f FRACTION, --fraction FRACTION
                        Fraction to annotate on output image (default 0.1)
  -c CHANNEL, --channel CHANNEL
                        Channel to identify spots in (default=1)
```
