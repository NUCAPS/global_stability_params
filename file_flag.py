import xarray as xr
import netCDF4
import scipy.interpolate
import timeit
from os.path import basename

import sharppy
import sharppy.sharptab.profile as profile
import sharppy.sharptab.interp as interp
import sharppy.sharptab.winds as winds
import sharppy.sharptab.utils as utils
import sharppy.sharptab.params as params
import sharppy.sharptab.thermo as thermo
import sharppy.sharptab.fire as fire

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import glob
from io import StringIO
import sys

ddir="/home/rebekah/alberta/data/"
odir='/home/rebekah/alberta/qf/'
files =  glob.glob(ddir+'/*')

for i, FILE in enumerate(files[2019:]):
    start = timeit.default_timer()

    fname =  basename(FILE)
    oname = 'qf_'+fname.split('_')[3]

    lats=[]
    lons=[]
    times=[]
    ascend=[]
    qf=[]

    print(i, 'Now processing file: '+fname)
    nc = xr.open_dataset(FILE, decode_times=True)


    lat_fov = list(nc.Latitude.values)
    lon_fov = list(nc.Longitude.values)
    qf_fov = list(nc.Quality_Flag.values)
    ascend_fov = list(nc.Ascending_Descending.values)
    time_fov = list(nc.Time.values)

    lons.append(lon_fov)
    lats.append(lat_fov)
    times.append(time_fov)
    qf.append(qf_fov)
    ascend.append(ascend_fov)


    # for FOR in nc.Number_of_CrIS_FORs.values:
    #     tmp = nc.sel(Number_of_CrIS_FORs=FOR, drop=True)
    #
    #     lat_fov = tmp.Latitude.item()
    #     lon_fov = tmp.Longitude.item()
    #     qf_fov = tmp.Quality_Flag.item()
    #     ascend_fov = tmp.Ascending_Descending.item()
    #     time_fov = tmp.Time.values.item()
    #
    #     lons.append(lon_fov)
    #     lats.append(lat_fov)
    #     times.append(time_fov)
    #     qf.append(qf_fov)
    #     ascend.append(ascend_fov)

    np.savez(odir+oname, lat=lats, lon=lons, times=times, qf=qf, ascend=ascend)
    stop = timeit.default_timer()

    print('-- File Done! Time: ', (stop - start)/60, " mins")
