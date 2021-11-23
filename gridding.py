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

def restructurePoints(lats, lons):
    length = np.shape(lats)[0]
    points = np.zeros((length, 2))
    for i in range(length):
        points[i, 0] = lons[i]
        points[i, 1] = lats[i]
    return points

def generateMask(points, X, Y, threshold=1):
    mask = np.zeros((np.shape(X)))
    for point in points:
        lon = point[0]
        lat = point[1]
        distance = np.sqrt(np.power(X - lon, 2) + np.power(Y - lat, 2))

        mask[np.where(distance <= threshold)] = 1
    return mask

def horizontallyInter(points, variable, X, Y, mask):
    gridOut = scipy.interpolate.griddata(points, variable, (X, Y), method='nearest')
    gridOut[mask == 0] = np.nan
    return gridOut

start = timeit.default_timer()

ddir = 'derived/'
odir = 'gridded/'

files = glob.glob(ddir+"*")

lats = np.empty(0)
lons = np.empty(0)
haines = np.empty(0)
cape = np.empty(0)
times = np.empty(0)

for file in files:
    npzfile = np.load(file)
    lats = np.append(lats, npzfile['lat'])
    lons = np.append(lons, npzfile['lon'])
    haines = np.append(haines, npzfile['haines'])
    cape = np.append(cape, npzfile['cape'])
    times = np.append(times, npzfile['times'])

stop = timeit.default_timer()
print('Combining files done! Time: ', (stop - start)/60, " mins")


files = glob.glob("qf/*")

lats = np.empty(0)
lons = np.empty(0)
times = np.empty(0)
ascend = np.empty(0)
qf = np.empty(0)

for file in files:
    npzfile = np.load(file)
    lats = np.append(lats, npzfile['lat'])
    lons = np.append(lons, npzfile['lon'])
    ascend = np.append(ascend, npzfile['ascend'])
    qf = np.append(qf, npzfile['qf'])
    # times = np.append(times, npzfile['times'])

# -----------------------------------
# Gridding
#------------------------------------
start = timeit.default_timer()

points = restructurePoints(lats, lons)

print("Done restructure!")

nx = 720
ny = 360
nx_dim = 720j
ny_dim = 360j
dist = 1.0

xrout_dims =  (X.shape[0],X.shape[1], 2)

# Coverage for the global grid.
regionCoverage = [-179.9999999749438 , -89.9999999874719 , 179.9999999749438 , 89.9999999874719]
Y, X = np.mgrid[regionCoverage[3]:regionCoverage[1]:ny_dim, regionCoverage[0]:regionCoverage[2]:nx_dim]
print("Done making grids!")

# This takes ~ 32 mins/day
mask = generateMask(points, X, Y, threshold=dist)
print("Done masking!")

# Decending = 1, Ascending = 0
ascend_flag = (ascend==0)
descend_flag = (ascend==1)

points_d = restructurePoints(lats[descend_flag], lons[descend_flag])
points_a = restructurePoints(lats[ascend_flag], lons[ascend_flag])

mask_d = generateMask(points_d, X, Y, threshold=dist)
mask_a = generateMask(points_a, X, Y, threshold=dist)

# -----------
gridded_cape = np.empty(xrout_dims)
gridded_cape[:,:,0] = horizontallyInter(points_d, cape[descend_flag], X, Y, mask_d)
gridded_cape[:,:,1] = horizontallyInter(points_a, cape[ascend_flag], X, Y, mask_a)

gridded_haines = np.empty(xrout_dims)
gridded_haines[:,:,0] = horizontallyInter(points_d, haines[descend_flag], X, Y, mask_d)
gridded_haines[:,:,1] = horizontallyInter(points_a, haines[ascend_flag], X, Y, mask_a)

gridded_times = np.empty(xrout_dims)
gridded_times[:,:,0] = horizontallyInter(points_d, times[descend_flag], X, Y, mask_d)
gridded_times[:,:,1] = horizontallyInter(points_a, times[ascend_flag], X, Y, mask_a)

gridded_qf = np.empty(xrout_dims)
gridded_qf[:,:,0] = horizontallyInter(points_d, qf[descend_flag], X, Y, mask_d)
gridded_qf[:,:,1] = horizontallyInter(points_a, qf[ascend_flag], X, Y, mask_a)

# gridded_ascend = horizontallyInter(points, ascend, X, Y, mask)

ds = xr.Dataset(
    data_vars=dict(
        cape=(["x", "y","ascend_descend"], gridded_cape),
        haines=(["x", "y", "ascend_descend"], gridded_haines),
        time=(["x", "y", "ascend_descend"], gridded_times/1e9),
        quality_flag = (["x", "y", "ascend_descend"], gridded_qf)
        ),
    coords=dict(
        lon=(["x", "y"], X),
        lat=(["x", "y"], Y),
        ascend_descend=[0, 1]
        ),
    attrs=dict(description="Testing derived parameters")
  )

# Time
ds.time.attrs['units']='seconds since 1970-01-01 00:00'
ds.time.attrs['calendar'] = 'standard'
ds = xr.decode_cf(ds, decode_times=True)
# ds.time.dt.hour

#ascend_descend
ds.ascend_descend.attrs['long_name'] = '1=Descending, 0=Ascending'

# CAPE
ds.cape.attrs['units']='J kg-1'
ds.cape.attrs['long_name'] = 'Convective Available Potential Energy (CAPE)'
ds.cape.attrs['standard_name'] = 'atmosphere_convective_available_potential_energy_wrt_surface'
ds.cape.attrs['valid_range'] = [0,10000]

# Haines
ds.haines.attrs['units']='1'
ds.haines.attrs['long_name'] = 'Haines Index'
ds.haines.attrs['valid_range'] = [2, 6]

ds.to_netcdf(odir+'derived_gridded.nc')

# plt.figure()
# plt.imshow(gridded_cape, vmin=0, vmax=5000)
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(gridded_haines[:,:,1], vmin=0, vmax=6)
# plt.colorbar()
# plt.show()
#
# plt.figure()
# plt.imshow(mask)
# plt.colorbar()
# plt.show()

stop = timeit.default_timer()
print('Done! Time: ', (stop - start)/60, " mins")
