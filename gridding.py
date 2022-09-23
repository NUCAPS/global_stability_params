# /home/rebekah/miniconda3/envs/stabil/bin/python /home/rebekah/stability/gridding.py

import argparse

import xarray as xr
import netCDF4
import scipy.interpolate
import timeit
from datetime import datetime, timedelta
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
plt.rcParams.update({'font.size': 30})

# --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str, required=False)
parser.add_argument('--indir', type=str, required=False)
parser.add_argument('--outdir', type=str, required=False)
parser.add_argument('--casename', type=str, required=False)

args = parser.parse_args()

# Manual processing 'YYYYmmdd'
search_day = args.date
indir = args.indir
outdir = args.outdir
casename = args.casename

if (search_day == None):
    # Cronjob processing:
    search_day = datetime.now() - timedelta(days=1)
    search_day = search_day.strftime("%Y%m%d")

if (indir == None):
    indir = 'derived'

if (outdir == None):
    outdir = 'gridded'

ddir = '/home/rebekah/stability/'+indir+'/'
odir = '/home/rebekah/stability/'+outdir+'/'
# plotdir = '/home/rebekah/stability/plots/'
plotdir = '/mnt/stcnucapsnet/plots/'

if (casename != None):
    ddir=ddir+casename+'/'
    odir=odir+casename+'/'

print("Gridding.py: Processing ", search_day, ddir)
# ------------------------

start = timeit.default_timer()

metadata_file = "/home/rebekah/stability/metadata.csv"
metadata = pd.read_csv(metadata_file, sep=',', header=0)

files = glob.glob(ddir+"derived_*"+search_day+"*.npz")

if len(files) == 0:
    print("Gridding.py: No files found! End program.")
    quit()

# ------------------------
def createGrid():
    nx = 720
    ny = 360
    nx_dim = 720j
    ny_dim = 360j

    # Coverage for the global grid.
    regionCoverage = [-179.9999999749438 , -89.9999999874719 , 179.9999999749438 , 89.9999999874719]
    Y, X = np.mgrid[regionCoverage[3]:regionCoverage[1]:ny_dim, regionCoverage[0]:regionCoverage[2]:nx_dim]
    xrout_dims =  (X.shape[0],X.shape[1], 2)

    return Y, X, xrout_dims

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

def gridValues(xrout_dims, var, var_name, points_d, points_a, descend_flag, ascend_flag, mask_d, mask_a):
    if var_name == 'times':
        var=var/1e9

    gridded_vals = np.full(xrout_dims, fill_value=np.nan)

    if len(var[descend_flag]) > 0:
        gridded_vals[:,:,0] = horizontallyInter(points_d, var[descend_flag], X, Y, mask_d)


    if len(var[ascend_flag]) > 0:
        gridded_vals[:,:,1] = horizontallyInter(points_a, var[ascend_flag], X, Y, mask_a)

    dict_item = { var_name : (["x", "y", "ascend_descend"], gridded_vals) }

    return dict_item

def generateGlobalAtrrs(search_day):
    # Global attributes for netCDF file
    date_created=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
    time_coverage=search_day[0:4]+'-'+search_day[4:6]+'-'+search_day[6:8]
    time_coverage_start=time_coverage+"T00:00:00Z"
    time_coverage_end=time_coverage+"T23:59:59Z"

    global_attrs = {
    'description' : "Derived parameters computed from NUCAPS EDR v3r0 from NOAA-20",
    'Conventions' : "CF-1.5",
    'Metadata_Conventions' : "CF-1.5, Unidata Datasset Discovery v1.0",
    'institution' : "Science and Technology Corp.",
    'creator_name' : "Rebekah Esmaili",
    'creator_email' : "rebekah@stcnet.com",
    'platform_name' : "J01",
    'date_created' : date_created,
    'time_coverage_start' : time_coverage_start,
    'time_coverage_end' : time_coverage_end
    }

    return global_attrs

def generateEncodingAttrs(var_names):
    dict={}
    for var_name in var_names:
        dict.update( { var_name : {"zlib": True, "complevel": 9} })

    return dict

def generatePlot(ds, var_name, search_day):
    vmin = ds[var_name].cbar_range[0]
    vmax = ds[var_name].cbar_range[1]
    cmap = ds[var_name].cmap

    time_coverage=search_day[0:4]+'-'+search_day[4:6]+'-'+search_day[6:8]

    subplot_kws=dict(projection=ccrs.PlateCarree(), transform=ccrs.PlateCarree())
    cbar_kwargs={"extend": "both", "orientation" : "vertical", "shrink" : .75, "cmap" : cmap}

    p = ds[var_name].plot(x="lon", y="lat", row="ascend_descend", figsize=[20,20], subplot_kws=subplot_kws, cbar_kwargs=cbar_kwargs, vmin=vmin, vmax=vmax)

    for i, ax in enumerate(np.flip(p.axes.flat)):
        ax.set_extent([-170, -20, 0, 70])
        ax.coastlines('50m')
        if i == 0:
            ax.set_title(time_coverage + " (ascending)")
        else:
            ax.set_title(time_coverage + " (descending)")

    plt.savefig(plotdir + search_day + '_' + var_name + '.png')
    plt.close()
# -----------------------------

lats = np.empty(0)
lons = np.empty(0)
ascend = np.empty(0)

for file in files:
    npzfile = np.load(file)
    lats = np.append(lats, npzfile['lat'])
    lons = np.append(lons, npzfile['lon'])
    ascend = np.append(ascend, npzfile['ascend'])

# Get var names
all_var_names = npzfile.files

# drop dim variables
all_var_names.remove('lat')
all_var_names.remove('lon')
all_var_names.remove('ascend')
# all_var_names.remove('times')

print('Gridding.py: Done combining files!')

# -----------------------------------
# Gridding
#------------------------------------
start = timeit.default_timer()

Y, X, xrout_dims = createGrid()

# Decending = 1, Ascending = 0
ascend_flag = (ascend==0)
descend_flag = (ascend==1)

points_d = restructurePoints(lats[descend_flag], lons[descend_flag])
points_a = restructurePoints(lats[ascend_flag], lons[ascend_flag])

print("Gridding.py: Done making grids!")

# This takes ~ 32 mins/day
mask_d = generateMask(points_d, X, Y, threshold=1.0)
mask_a = generateMask(points_a, X, Y, threshold=1.0)

print("Gridding.py: Done making masks!")

data_vars = {}
for i, var_name in enumerate(all_var_names):
    print ('--'+var_name, i)
    var = np.empty(0)
    for file in files:
        npzfile = np.load(file)
        var = np.append(var, npzfile[var_name])

    # temporary bug fix --------
    if len(var) !=  len(descend_flag):
        continue

    dict_item = gridValues(xrout_dims, var, var_name, points_d, points_a, descend_flag, ascend_flag, mask_d, mask_a)
    data_vars.update(dict_item)

print("Gridding.py: Done gridding variables!")

# Global attributes for netCDF file
global_attrs = generateGlobalAtrrs(search_day)


ds = xr.Dataset(
    data_vars=data_vars,
    coords=dict(
        lon=(["x", "y"], X),
        lat=(["x", "y"], Y),
        ascend_descend=[0, 1]
        ),
    attrs=global_attrs
  )

# Add time/coord attrs
ds.times.attrs['units'] = 'seconds since 1970-01-01 00:00'
ds.times.attrs['standard_name'] = 'time'
ds.times.attrs['calendar'] = 'standard'

ds.lat.attrs['units'] = 'degree_north'
ds.lat.attrs['long_name'] = 'Latitude'
ds.lat.attrs['standard_name'] = 'latitude'
ds.lat.attrs['valid_range'] = [-90.0, 90.0]

ds.lon.attrs['units'] = 'degree_east'
ds.lon.attrs['long_name'] = 'Longitude'
ds.lon.attrs['standard_name'] = 'longitude'
ds.lon.attrs['valid_range'] = [-180.0, 180.0]

ds.ascend_descend.attrs['long_name'] = '0=Descending, 1=Ascending'

ds = xr.decode_cf(ds, decode_times=True)

# Add variable level attrs
# Sneakilly adding plot generator call here...

var_names = list(data_vars.keys())

for i, var_name in enumerate(list(metadata.var_name)):
# for i, var_name in enumerate(var_names):
    # temporary bug fix (try statement) --------
    try:
        ds[var_name].attrs['long_name']=metadata.long_name[i]
        ds[var_name].attrs['standard_name']=metadata.standard_name[i]
        ds[var_name].attrs['units']=metadata.units[i]
        ds[var_name].attrs['cbar_range']=[metadata.cbar_range_min[i],metadata.cbar_range_max[i]]
        ds[var_name].attrs['cmap']=metadata.cmap[i]
        # ds[var_name].attrs['_FillValue'] = np.nan

        generatePlot(ds, var_name, search_day)
    except:
        continue

encodeAttrs = generateEncodingAttrs(all_var_names)

ds.to_netcdf(odir+'derived_gridded_'+search_day+'.nc', format='netCDF4', encoding=encodeAttrs)

# ds.to_netcdf(odir+'derived_gridded_'+search_day+'.nc', format='netCDF4')

# Test plot
# data_vars.keys()
# test_var = data_vars['qf'][1][:,:,1]

stop = timeit.default_timer()
print('Gridding.py: Done! Time: ', (stop - start)/60, " mins"),
