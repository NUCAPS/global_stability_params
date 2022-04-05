# Gridded stability parameters
Author: Rebekah Esmaili (rebekah@stcnet.com)

This code repo calculates many stability from NUCAPS for the previous day of data, grids profiles to 0.1 degrees, and consolidates them into a netCDF4 files/generates a preview image. This code is used for a "quick look" website, but can also be run offline.

This was developed as a test system to evaluate/compare satellite derived stability parameters.

## Programs
The following programs are included in this repository:

* downloadClassOrder.sh: downloads data routinely from NOAA/CLASS via a subscription. This is optional, you can also supply your own data. My current order only processes North America.
* demo_files.py: user provides a date, opens files inside of the *data* directory (input), applies the NUCAPS surface correction, and computes stability parameters using the SHARPpy package. SHARPpy calculations emulate those found in the AWIPS/NSHARP system. The output is saved to the *derived* directory.
* gridding.py: User provides a date, open each npz file in the *derived* directory (input), grids to 0.1 degrees and masks out area outside of swath. Sorts data into ascending/descending orbits. This program creates a netCDF4 file, which is saved to the *gridded* directory (output). This program also optionally creates plots, which are used on an STC quick look website.
