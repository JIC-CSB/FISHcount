# Counting single RNA molecule probes

This software is intended to assist with the enumeration of single molecule
RNA probes. It operates on microscope data files, eventually outputting an
annotated image showing a projection of the data with suggested cell boundaries
and probe counts for those cells.

## Installation notes for Mac

Ensure that you have Xcode installed, for example by running ``gcc`` in the
terminal. Alternatively, you can use the app store.

```bash
gcc
```

Install Homebrew.

```bash
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Install freetype and libtiff using Homebrew.

```bash
brew install freetype
brew install libtiff
```

Install ``virtualenv`` using ``easy_install`` (virtualenv allows you to create
a virtual Python environment).

```bash
sudo easy_install virtualenv
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
pip install six
pip install "scikit-image==0.10.1"
pip install freetype-py
pip install libtiff
```

### Install BioFormats ``bftools``

**If you are working with czi file you may need BioFormats less than version 4.4.4,
e.g. 4.4.2**

The Python image analysis code works on Tiff files. We therefore need to
convert the microscopy data using the BioFormats ``bfconvert`` tool.

The BioFormats tools require Java, which needs to be installed. Download the
installer from the Oracle website.
https://www.java.com/en/download/mac_download.jsp?locale-=en

Install the Java JDK. Download the installer form the Oracle website.
http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html

Install both of them.

Download the ``bftools.zip`` file from:

http://www.openmicroscopy.org/site/support/bio-formats5.0/users/comlinetools/

Place the downloaded ``bftools`` directory where you want it. For example:

```bash
mkdir ~/tools
mv ~/Downloads/bftools ~/tools/
```

Add the ``bftools`` directory to your ``$PATH`` environment variable.

```bash
export PATH=$PATH:~/tools/bftools
```

## Image analysis

### Download FISHCount software

Download the FISHcount software zip file from GitHub.

https://github.com/JIC-CSB/FISHcount/archive/master.zip

Go into the FISHCount directory (e.g. ``~/Downlaods/FISHcount``).

```
cd ~/Downloads/FISHcount
```


### Convert microscopy images to tiff files

Make sure that the ``bftools`` directory is in your path.

```bash
export PATH=$PATH:~/tools/bftools
```

Run the ``unpack.py`` script.

```bash
python scripts/protoimg/unpack.py lif_input_dir tiff_output_dir
```

Where ``lif_input_dir`` is the directory with the original lif files and
``tiff_output_dir`` is the directory to where the tiff files will be written.

### Do the image analysis

Make sure that the Python virtual environment has been sourced.

```bash
. ./env/bin/activate
```

Run the analysis on the image directory of interest and specify an output
directory to store any generated images.

```bash
python ./scripts/count_probes.py tiff_output_dir/image_of_interest_dir output_dir
```

Note that the file of most interest is ``output_dir/annotated_projection.png``.
