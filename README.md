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

Install freetype using Homebrew.

```bash
brew install freetype
```

Install ``virtualenv`` using ``easy_install`` (virtualenv allows you to create
a virtual Python environment).

```bash
sudo easy_install virtualenv
```

Source the virtual environment (note the ``.`` at the start of the line).

```bash
. ./env/bin/activate
```

Install Python dependencies into the virtual environment.

```bash
pip install numpy
pip install scikit-image
pip install scipy
pip install freetype-py
```
