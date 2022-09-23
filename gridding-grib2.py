# /home/rebekah/miniconda3/envs/stabil/bin/python /home/rebekah/stability/gridding.py
# test

import argparse
import xarray as xr
import netCDF4
import math
import scipy.interpolate
import timeit
import time
from datetime import datetime
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from herbie.archive import Herbie
import cartopy.feature as cfeature
from pyresample.geometry import GridDefinition, AreaDefinition, SwathDefinition
from pyresample import kd_tree
import numpy.ma as ma

plt.rcParams.update({'font.size': 22})
hrrr_lons = [-134.09547973426314, -60.91719277183779]
hrrr_lats = [21.13812300000003, 52.61565330680793]

# --------
indir = 'hrrr'
outdir = 'gridded-hrrr'
search_day = '2022-07-04'
search_time = '19:00'
fcst_time = 0
#-------

odir = outdir+'/'
plotdir = 'plots/'

# print("Gridding.py: Processing ", search_day, search_time)

def matchup_spatial(latitude, longitude, site_lat, site_lon, 
        radius_km=50.0, closest_only=True):
    ''' This function calcualtes the distance between a list of retrieval coordinates and
    and a point observation. It returns all matches that are within a given radius or the
    closest point.
    '''
    # Find index for pixels within radius_km around ground site  
    distance_matrix = np.full(latitude.shape,  6378.0)

    # Calculate the distance in degrees
    dist_deg = np.sqrt((np.array(latitude)-site_lat)**2
        +(np.array(longitude)-site_lon)**2)
    close_pts = (dist_deg < 1.0)
    
    # Replace angle distance with km distance
    distance_matrix[close_pts] = haversine(site_lat, site_lon, 
        latitude[close_pts], longitude[close_pts])
    keep_index = (distance_matrix < radius_km)
    
    # Return a single (closest) value
    if closest_only:
       if len(keep_index[keep_index==True]) > 0:
           keep_index = (distance_matrix == distance_matrix.min())

    return keep_index

def haversine(deglat1, deglon1, deglat2, deglon2):
    lat1=deglat1*np.pi/180.0
    lat2=deglat2*np.pi/180.0
    
    long1=deglon1*np.pi/180.0
    long2=deglon2*np.pi/180.0
      
    a = np.sin(0.5 * (lat2 - lat1))
    b = np.sin(0.5 * (long2 - long1))
    
    dist = 12742.0 * np.arcsin(np.sqrt(a * a + np.cos(lat1) * np.cos(lat2) * b * b))
    
    return dist

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

# -----------------------------
def get_hrrr(search_datetime, fcst_time=0, var="CAPE"):
    # syntax: H = Herbie("2022-06-07 16:00", model="hrrr", product="sfc", fxx=0)
    # ds.longitude, ds.latitude, ds.cape.values[2,:,:,0]
    # https://blaylockbk.github.io/Herbie/_build/html/user_guide/notebooks/tutorial.html?highlight=product

    H = Herbie(search_datetime, model="hrrr", product="sfc", fxx=fcst_time)
    # np.unique(H.read_idx().variable)
    ds = H.xarray(var)

    mask = (ds.longitude > 180)
    ds.longitude.values[mask] = ds.longitude.values[mask]-360

    return ds

def create_test_plot(x, y, z, oname):
    """ This code will create a CONUS-based test plot for gridded data only in the plot directory specified in the top of the code """
    lat_0 = 22.74
    lon_0 = -132.23
    lat_1 = 53.42
    lon_1 = -61.21
    
    img_extent = [lon_0, lon_1, lat_0, lat_1]

    plt.figure(figsize=[16,16])
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    p = ax.pcolormesh(x, y, z, transform=ccrs.PlateCarree(), vmin=0, vmax=10000)
    plt.colorbar(p, orientation="horizontal", pad=0.05)
    ax.set_extent(img_extent)
    plt.savefig(plotdir + oname + '.png')
    plt.close()

def grid_hrrr(ds)
    # Swath definition
    Y, X, xrout_dims = createGrid()
    grid_def = GridDefinition(lons=X, lats=Y)

    lats = ds.latitude.values
    lons = ds.longitude.values-360
    swath_def = SwathDefinition(lons=lons, lats=lats)

    data = ds.cape.values[2,:,:,0]
    result = kd_tree.resample_nearest(swath_def, data, grid_def, radius_of_influence=50000, epsilon=0.5, fill_value=np.nan)
    return result

def unique_yyyymmdd(ddir):
    all_files = glob.glob(ddir+"derived*.npz")
    all_files_dt = [file.split('_s')[1][0:8] for file in all_files]
    return np.unique(all_files_dt)


def extract_file_date(fname):
    # format needs to be "2022-06-07 16:00"
    yr = fname[0:4]
    mon = fname[4:6]
    day = fname[6:8]
    hr = fname[8:10]
    min = fname[10:12]

    min_int = math.floor(int(min)/15)*15
    # if min_int >= 60:
    #     min_int = 45

    # if min_int < 0:
    #     min_int = 0

    min = str(min_int).zfill(2)

    return yr + '-' + mon + '-' + day + ' ' + hr + ':' + min

def epoch2datetime(etime):
    etime_seconds = etime/1000.0
    dt_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(etime_seconds))

    return dt_str

# metadata_file = "/home/rebekah/stability/metadata.csv"
# metadata = pd.read_csv(metadata_file, sep=',', header=0)

search_day = '20220704'
ddir = 'derived/'

search_days = unique_yyyymmdd(ddir)

for search_day in search_days:

    files = sorted(glob.glob(ddir+"derived_*"+search_day+"*.npz"))
    search_datetimes = [extract_file_date(file.split('_s')[1][0:12]) for file in files]

    df = pd.DataFrame(columns=['nucaps_lat', 'nucaps_lon', 'nucaps_cape', 'hrrr_lat', 'hrrr_lon', 'hrrr_cape', 'hrrr_time'])
    search_datetime_prev = ''
    for ii, file in enumerate(files):

        print(ii, '/', len(files), " Processing: ", file)
        search_datetime = search_datetimes[ii]

        lats = np.empty(0)
        lons = np.empty(0)
        cape = np.empty(0)

        npzfile = np.load(file)
        lats = np.append(lats, npzfile['lat'])
        lons = np.append(lons, npzfile['lon'])
        capes = np.append(cape, npzfile['sfccape'])

        for i in range(0, len(lats)):
            lat = lats[i]
            lon = lons[i]
            cape = capes[i]

            # Check if point in HRRR range
            in_bounds = (lon >= hrrr_lons[0]) & (lon <= hrrr_lons[1]) & (lat >= hrrr_lats[0]) & (lat <= hrrr_lats[1])

            if ~in_bounds:
                continue

            # Only open file if it's not already open
            if (search_datetime_prev != search_datetime):
                print(search_datetime_prev, search_datetime)
                ds = get_hrrr(search_datetime)
                search_datetime_prev = search_datetime

            # Search for closest pixel, if it exists
            index = matchup_spatial(ds.latitude.values, ds.longitude.values, lat, lon, closest_only=True)

            if np.sum(index) > 0:
                data = pd.DataFrame([{'hrrr_time': search_datetime, 'nucaps_lat' : lat, 'nucaps_lon' : lon, 'nucaps_cape' : cape, 'hrrr_lat': ds.latitude.values[index].item(), 'hrrr_lon' : ds.longitude.values[index].item(), 'hrrr_cape' : ds.cape.values[2,:,:,0][index].item() }])

                df = pd.concat([df, data], ignore_index=True)

    df.to_csv('hrrr_matches/'+search_day+'-cape.csv')