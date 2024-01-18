DeepMap code for DL on historical Aerial Images
-----------------------------------------------

The `notebooks/` folder contains sample exploratory code to get familiar with libraries and APIs for GIS.
The `scripts/` folder contains the preprocessing and training code for the DeepMap model.

### Installation

`pip install .` should work, though due to latest `importlib.resources` not supporting directories,
you may need to install the package in editable mode (`pip install -e .`) to run scripts in the `scripts/` folder.

### Data

Sample data is included in the `deepmap.data` module. Larger data files are download on demand using the `deepmap.data.ondemand.get_file` function.