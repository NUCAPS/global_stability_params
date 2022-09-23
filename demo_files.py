# /home/rebekah/miniconda3/envs/stabil/bin/python /home/rebekah/stability/demo_files.py

import xarray as xr
import netCDF4
import scipy.interpolate
import timeit
from os.path import basename, isfile
from os import remove
from datetime import datetime, timedelta

import argparse

import sharppy
import sharppy.sharptab.profile as profile
import sharppy.sharptab.params as params
import sharppy.sharptab.thermo as thermo
import sharppy.sharptab.fire as fire

import numpy as np
import glob
import sys

# --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str, required=False)
parser.add_argument('--indir', type=str, required=False)
parser.add_argument('--outdir', type=str, required=False)
parser.add_argument('--casename', type=str, required=False)

args = parser.parse_args()

# Manual processing: format YYYYmmdd
# search_day = '20210917'
search_day = args.date
indir = args.indir
outdir = args.outdir
casename = args.casename

if (search_day == None):
    # Cronjob processing:
    search_day = datetime.now() - timedelta(days=1)
    search_day = search_day.strftime("%Y%m%d")

if (indir == None):
    indir = 'data'

if (outdir == None):
    outdir = 'derived'

ddir="/home/rebekah/stability/"+indir+"/"
odir="/home/rebekah/stability/"+outdir+"/"

if (casename != None):
    ddir=ddir+casename+'/'
    odir=odir+casename+'/'

# --------------------

print("demo_files.py: Processing: ", search_day, " in ", ddir )

# NOTE: One day took ~ 3 hours (5s/file)

Rd = 287.
g = 9.81
p_std = 1013.25
navog = 6.02214199e+23  # NIST Avogadro's number (molecules/mole)
mw_d = 28.9644  # gm/mole dry air
mw_wv = 18.0151  # gm/mole water
g_std = 980.664  # acceleration of gravity, cm/s^2
cdair = 1000.0 * p_std * navog / (mw_d * g_std)
wv_eps = mw_wv / mw_d
FILL_VAL = np.nan
# Filler wind fields
WDIR = np.repeat(0, 100)
WSPD = np.repeat(0, 100)

# Find the surface within a sounding
def findSurface(pres, surfpres):
    diff = np.abs((pres - surfpres))
    mindiff = np.min(diff)
    clev = np.where(diff == mindiff)[0][0]
    surflev = clev
    if surfpres < pres[clev]:
        surflev = clev
    if surfpres > pres[clev] and mindiff >= 5.0:
        surflev = clev
    if surfpres > pres[clev] and mindiff > 5.0:
        surflev = clev + 1
    return surflev

# Find BOTLEV & BLMULT
def get_botlev_blmult(plev, psurf, nobs):
    botlev = np.zeros((nobs), dtype=float)
    blmult = np.zeros((nobs), dtype=float)

    for i in range(nobs):
        surflev = findSurface(plev, psurf[i])
        num = psurf[i] - plev[surflev - 1]
        denom = plev[surflev] - plev[surflev - 1]
        blmult[i] = num / denom
        botlev[i] = surflev
    return blmult, botlev

# Calculate surface water vapor column density using BLMULT
def calc_wvcd_sfc(botlev, blmult, nobs, p_layer, wvcd):
    nlev_100 = np.shape(p_layer)[0]
    wvcd_sfc = np.zeros((nobs), dtype=float)

    for i in range(nobs):
        if botlev[i] < nlev_100 - 1:
            sfc = int(botlev[i])
        if botlev[i] == nlev_100 - 1:
            sfc = nlev_100 - 1

        wvcd_sfc[i] = wvcd[i, int(sfc)] * blmult[i]
    return wvcd_sfc

# Calculate surface temperature using BLMULT
def calc_Tsfc(botlev, blmult, nobs, plev, temperature):
    nlev_100 = np.shape(plev)[0]
    tsfc = np.zeros((nobs), dtype=float)

    for i in range(nobs):
        if botlev[i] < nlev_100 - 1:
            sfc = int(botlev[i])
        if botlev[i] == nlev_100 - 1:
            sfc = nlev_100 - 1

        t_diff = temperature[i, int(sfc)] - temperature[i, int(sfc) - 1]
        tsfc[i] = temperature[i, int(sfc) - 1] + blmult[i] * t_diff
    return tsfc

# Insert pressure into final lists.
def insert_surface_pressure(botlev, nobs, plev, psurf):
    nlev_100 = np.shape(plev)[0]
    blmult_P_ALL = []

    for i in range(nobs):
        sfc = botlev[i]
        # Add surface values
        blmult_P_footprint = np.append(plev[0:int(sfc)], psurf[i])

        # Append each footprint to the larger array
        blmult_P_ALL.append(blmult_P_footprint)

    # Convert to arrays
    blmult_P_ALL = np.asarray(blmult_P_ALL, dtype="object")

    return blmult_P_ALL

# Insert surface temperature into final lists.
def insert_surface_temperature(botlev, nobs, plev, tsfc, temperature):
    nlev_100 = np.shape(plev)[0]
    blmult_T_ALL = []

    for i in range(nobs):
        sfc = botlev[i]
        if sfc + 1 == nlev_100 - 1:
            temperature[i, sfc + 1] = FILL_VAL
        if sfc + 1 > nlev_100 - 1:
            temperature[i, sfc + 1:nlev_100 - 1] = FILL_VAL

        lev_T = np.zeros(nlev_100, dtype=float)

        for j in range(int(sfc)):
            lev_T[j] = temperature[i, j]

        # Add surface values
        blmult_T_footprint = np.append(lev_T[0:int(sfc)], tsfc[i])

        # Append each footprint to the larger array
        blmult_T_ALL.append(blmult_T_footprint)

    # Convert to arrays
    blmult_T_ALL = np.asarray(blmult_T_ALL, dtype="object")

    return blmult_T_ALL

# Insert surface values into final lists.
def insert_surface_water_vapor(botlev, nobs, plev, wvcd_sfc, wvcd):
    nlev_100 = np.shape(plev)[0]
    blmult_wvcd_ALL = []

    for i in range(nobs):
        sfc = botlev[i]
        if sfc + 1 == nlev_100 - 1:
            wvcd[i, sfc + 1] = FILL_VAL
        if sfc + 1 > nlev_100 - 1:
            wvcd[i, sfc + 1:nlev_100 - 1] = FILL_VAL

        lev_wvcd = np.zeros(nlev_100, dtype=float)
        lev_wvcd[0] = wvcd[i, 0]

        for j in range(1, int(sfc)):
            lev_wvcd[j] = (wvcd[i,j-1] + wvcd[i,j]) / 2

        # Add surface values
        blmult_wvcd_footprint = np.append(lev_wvcd[0:int(sfc)], wvcd_sfc[i])

        # Append each footprint to the larger array
        blmult_wvcd_ALL.append(blmult_wvcd_footprint)

    # Convert to arrays
    blmult_wvcd_ALL = np.asarray(blmult_wvcd_ALL, dtype="object")

    return blmult_wvcd_ALL

# Convert water vapor column density to mixing ratio
def convert_cd2mr(nobs, blmult_wvcd_ALL, blmult_P_ALL, psurf, botlev):
    wvmr = []

    for i in range(nobs):
        nlev_NEW = np.shape(blmult_P_ALL[i])[0]
        wvmr_footprint = np.zeros((nlev_NEW), dtype=float)
        wvcd_footprint = blmult_wvcd_ALL[i]

        deltap = np.zeros((nlev_NEW), dtype=float)
        pres = blmult_P_ALL[i]
        deltap[0] = pres[0]
        deltap[1:nlev_NEW] = pres[1:nlev_NEW] - pres[0:nlev_NEW-1]

        for j in range(nlev_NEW):
            wvmr_footprint[j] = wvcd_footprint[j] / ((cdair * deltap[j] / p_std) / wv_eps)

        # Append footprint array into larger array
        wvmr.append(wvmr_footprint)

    # Convert larger array to numpy array
    wvmr = np.asarray(wvmr, dtype="object")
    return wvmr

# Calculate virtual temperature
def calc_virtual_temperature(nobs, blmult_P_ALL, wvmr, blmult_T_ALL):
    tv = []

    for i in range(nobs):
        nlev_NEW = np.shape(blmult_P_ALL[i])[0]
        tv_footprint = np.zeros((nlev_NEW), dtype=float)
        T_footprint = blmult_T_ALL[i]
        wvmr_footprint = wvmr[i]

        for j in range(nlev_NEW):
            tv_footprint[j] = T_footprint[j] * (1 + 0.608 * wvmr_footprint[j])

        # Append footprint array into larger array
        tv.append(tv_footprint)

    # Convert larger array to numpy array
    tv = np.asarray(tv, dtype="object")
    return tv

# Calculate mean sea level pressure
def calc_mslp(nobs, topographygraphy, tsfc, psurf):
    mslp = np.zeros((nobs), dtype=float)

    height_const = np.multiply(0.0065, topographygraphy)
    fraction = np.divide(height_const, np.add(tsfc, height_const))
    mslp = np.multiply(psurf, np.power((1 - fraction), -5.257))
    return mslp

# Calculate geopotential Height
def calc_geopotential_height(nobs, blmult_P_ALL, mslp, tv):
    z = []

    for i in range(nobs):
        nlev_NEW = np.shape(blmult_P_ALL[i])[0]
        z_footprint = np.zeros((nlev_NEW), dtype=float)
        mtv = np.zeros((nlev_NEW), dtype=float)
        tv_footprint = tv[i]
        tvsfc = tv_footprint[nlev_NEW - 1]
        P_footprint = blmult_P_ALL[i]

        for j in range(nlev_NEW):
            mtv[j] = (tvsfc + tv_footprint[j]) / 2
            z_footprint[j] = ((Rd * mtv[j]) / g) * np.log(mslp[i] / P_footprint[j])

        # Append footprint array into larger array
        z.append(z_footprint)

    # Convert larger array to numpy array
    z = np.asarray(z, dtype="object")
    return z

# Convert mixing ratio to dew point
def calc_dewpoint(nobs, blmult_P_ALL, blmult_T_ALL, wvmr):
    dew_point = []

    for i in range(nobs):
        nlev_NEW = np.shape(blmult_P_ALL[i])[0]
        RH_footprint = np.zeros((nlev_NEW), dtype=float)
        dew_point_footprint = np.zeros((nlev_NEW), dtype=float)
        T_footprint = blmult_T_ALL[i]
        P_footprint = blmult_P_ALL[i]
        wvmr_footprint = wvmr[i]

        for j in range(nlev_NEW):
            # Substitute mixing ratio (MR) for specific humidity (SHx) since they are approximately equal.
            RH_footprint[j] = find_RH(P_footprint[j], T_footprint[j], wvmr_footprint[j])
            dew_point_footprint[j] = find_dewpoint(T_footprint[j], RH_footprint[j])

        # Append footprint array into larger array
        dew_point.append(dew_point_footprint)

    # Convert larger array to numpy array
    dew_point = np.asarray(dew_point, dtype="object")
    return dew_point

# Find relative humidity for single profile.
def find_RH(P, TEMP_K, SHx):
    "Calculate Relative Humidity(0 to 100) from Pressure(hPa), Temp(K), and Specific Humidity."
    a = 22.05565
    b = 0.0091379024
    c = 6106.396
    epsilonx1k = 622.0

    shxDenom = SHx * 0.378
    shxDenom += epsilonx1k

    tDenom = -b*TEMP_K
    tDenom += a
    tDenom -= c/TEMP_K

    RH = P * SHx
    RH /= shxDenom
    RH /= np.exp(tDenom)

    RH = RH * 1000
    return RH

# Find dewpoint for single profile.
def find_dewpoint(TEMP_K, RH):
    "Calculate dewpoint(C) from temperature(K) and relative humidity(0 to 100)."
    b=0.0091379024*TEMP_K
    b += 6106.396/TEMP_K
    b -= np.log(RH/100)
    val = b*b
    val -= 223.1986
    val = np.sqrt(val)
    DpT = b-val
    DpT /= 0.0182758048
    DpT = DpT - 273.15
    return DpT

def calc_instability(pcl):
    cape_fov = pcl.bplus
    cin_fov = pcl.bminus
    lclhght_fov = pcl.lclhght

    return cape_fov, cin_fov, lclhght_fov

def calc_wetbulb(pcl):
    wm10c_fov = pcl.wm10c
    wm20c_fov = pcl.wm20c
    wm30c_fov = pcl.wm30c

    return wm10c_fov, wm20c_fov, wm30c_fov

def calc_heightC(pcl):
    hghtm10c_fov = pcl.hghtm10c
    hghtm20c_fov = pcl.hghtm20c
    hghtm30c_fov = pcl.hghtm30c

    return hghtm10c_fov, hghtm20c_fov, hghtm30c_fov

# def clean_files(FILE):
#     nc = xr.open_dataset(FILE, decode_times=False)
#
#     lats = nc.Latitude.values
#     lons = nc.Longitude.values
#
#     # (lats[i] > 70) | (lats[i] < -60):
#     lat_mean = lats.mean()
#     lon_mean = lons.mean()
#     if (lat_mean > 75) | (lat_mean < 0):
#         remove(FILE)
#         return
#     if (lon_mean < -180) | (lon_mean > -20):
#         remove(FILE)
#         return
#
# check = [clean_files(file) for file in allfiles]

# -----------------------------------------
start_all = timeit.default_timer()

allfiles =  glob.glob(ddir+'/*'+search_day+'*.nc')
tot_files = len(allfiles)

# Check if file already processed
files=[]
for file in allfiles:
    check_for_file = odir+'derived_'+file.split('_')[3]+'.npz'
    if isfile(check_for_file):
        continue
    else:
        files.append(file)

# files = ['/home/rebekah/stability/NUCAPS-EDR_v2r0_npp_s20190430185959_e20190430190557_c20190430190557.nc']

if len(files) == 0:
    print("demo_files.py: No files found!")

for i, FILE in enumerate(files):
    start = timeit.default_timer()

    fname =  basename(FILE)

    # To address minor differences in var/filenames
    sat = 'noaa'
    idt = 3 # loc of date in fname
    if ('M01' in fname) or ('M02' in fname):
        sat = 'iasi'
        idt = 4 # loc of date in fname

    oname = 'derived_'+fname.split('_')[idt]

    # mid-level stability parameters
    k_index=[]
    t_totals=[]
    lapserate_700_500=[]
    lapserate_850_500=[]
    haines=[]

    # sfc, most unstable (mu), and mixed layer (ml) parameters
    sfccape=[]
    mucape=[]
    mlcape=[]

    sfccin=[]
    mucin=[]
    mlcin=[]

    sfchghtm10c=[]
    sfchghtm20c=[]
    sfchghtm30c=[]

    sfclclhght=[]
    mulclhght=[]
    mllclhght=[]

    sfcwm10c=[]
    sfcwm20c=[]
    sfcwm30c=[]

    precip_water=[]

    print("demo_files.py: ", i, '/', tot_files, 'Now processing file: ', fname)
    s1 = timeit.default_timer()

    nc = xr.open_dataset(FILE, decode_times=False)

    lats = nc.Latitude.values
    lons = nc.Longitude.values
    qf = nc.Quality_Flag.values
    ascend = nc.Ascending_Descending.values
    times = nc.Time.values

    temperature = np.array(nc.Temperature)
    try:
        wvcd = np.array(nc.H2O)
    except:
        wvcd = np.array(nc.H2O_LCD)
    p_layer = np.array(nc.Effective_Pressure[0, :])
    plev = np.array(nc.Pressure[0, :])
    psurf = np.array(nc.Surface_Pressure)
    nobs = len(nc.Latitude)
    topography = np.array(nc.Topography)

    blmult, botlev = get_botlev_blmult(plev, psurf, nobs)

    # Find wvcd and temperature at surface using BLMULT
    wvcd_sfc = calc_wvcd_sfc(botlev, blmult, nobs, p_layer, wvcd)
    tsfc = calc_Tsfc(botlev, blmult, nobs, plev, temperature)

    # Insert surface values into temperature, water vapor and pressure arrays.
    blmult_P_ALL = insert_surface_pressure(botlev, nobs, plev, psurf)
    blmult_T_ALL = insert_surface_temperature(botlev, nobs, plev, tsfc, temperature)
    blmult_wvcd_ALL = insert_surface_water_vapor(botlev, nobs, plev, wvcd_sfc, wvcd)

    # Derive the other variables
    wvmr = convert_cd2mr(nobs, blmult_wvcd_ALL, blmult_P_ALL, psurf, botlev)
    tv = calc_virtual_temperature(nobs, blmult_P_ALL, wvmr, blmult_T_ALL)
    mslp = calc_mslp(nobs, topography, tsfc, psurf)
    z = calc_geopotential_height(nobs, blmult_P_ALL, mslp, tv)
    dew_point = calc_dewpoint(nobs, blmult_P_ALL, blmult_T_ALL, wvmr)
    #######################################

    # Convert temperature to Celsius
    for x in range(len(blmult_T_ALL)):
        blmult_T_ALL[x] = blmult_T_ALL[x] - 273.15


    if sat == 'iasi':
        for_vals = nc.Number_of_Dice.values
    else:
        for_vals = nc.Number_of_CrIS_FORs.values

    for i, FOR in enumerate(for_vals):
        # print("Processing FOR", i)

        # To speed up processing, skip the poles, skip partial granules
        if (lats[i] > 70) | (lats[i] < -60):
            k_index.append(FILL_VAL)
            t_totals.append(FILL_VAL)
            lapserate_700_500.append(FILL_VAL)
            lapserate_850_500.append(FILL_VAL)
            haines.append(FILL_VAL)
            #sfc
            sfccape.append(FILL_VAL)
            sfccin.append(FILL_VAL)
            sfchghtm10c.append(FILL_VAL)
            sfchghtm20c.append(FILL_VAL)
            sfchghtm30c.append(FILL_VAL)
            sfclclhght.append(FILL_VAL)
            sfcwm10c.append(FILL_VAL)
            sfcwm20c.append(FILL_VAL)
            sfcwm30c.append(FILL_VAL)
            #mu
            mucape.append(FILL_VAL)
            mucin.append(FILL_VAL)
            mulclhght.append(FILL_VAL)

            #ml
            mlcape.append(FILL_VAL)
            mlcin.append(FILL_VAL)

            mllclhght.append(FILL_VAL)

            # tpw
            precip_water.append(FILL_VAL)
            continue

        # Apply BLMULT
        temps = blmult_T_ALL[FOR]
        dewPoint = dew_point[FOR]
        press = blmult_P_ALL[FOR]
        Z = z[FOR]

        HGHT = np.flip(Z)
        TEMP = np.flip(temps)
        LEVEL = np.flip(press)
        DWPT = np.flip(dewPoint)
        mask = (DWPT > TEMP)
        DWPT[mask] = TEMP[mask]

        # Create profile
        prof = profile.create_profile(profile='default', pres=LEVEL, hght=HGHT, tmpc=TEMP, dwpc=DWPT, wspd=WSPD[:len(TEMP)], wdir=WDIR[:len(TEMP)], FILL_VAL=FILL_VAL, strictQC=False)

        # Calculate mid-level stability parameters
        k_index_fov = params.k_index(prof)
        t_totals_fov = params.t_totals(prof)

        precip_water_fov = params.precip_water(prof)

        lapserate_700_500_fov = params.lapse_rate(prof, 700., 500., pres=True)
        lapserate_850_500_fov = params.lapse_rate(prof, 850., 500., pres=True)

        # Haines
        topo = topography[i]
        if topo > 914:
            haines_fov = fire.haines_high(prof)
        elif topo < 305:
            haines_fov = fire.haines_low(prof)
        else:
            haines_fov = fire.haines_mid(prof)

        k_index.append(k_index_fov)
        t_totals.append(t_totals_fov)
        lapserate_700_500.append(lapserate_700_500_fov)
        lapserate_850_500.append(lapserate_850_500_fov)
        precip_water.append(precip_water_fov)
        haines.append(haines_fov)

        # Calculate sfc, most unstable (mu), and mixed layer (ml) parcels
        # Slowest step: 0.20817320235073566 s
        sfcpcl = params.parcelx(prof, flag=1)
        mupcl = params.parcelx(prof, flag=3)
        mlpcl = params.parcelx(prof, flag=4)

        #sfc
        cape_fov, cin_fov, lclhght_fov = calc_instability(sfcpcl)

        sfccape.append(cape_fov)
        sfccin.append(cin_fov)
        sfclclhght.append(lclhght_fov)

        hghtm10c_fov, hghtm20c_fov, hghtm30c_fov = calc_heightC(sfcpcl)

        sfchghtm10c.append(hghtm10c_fov)
        sfchghtm20c.append(hghtm20c_fov)
        sfchghtm30c.append(hghtm30c_fov)

        wm10c_fov, wm20c_fov, wm30c_fov = calc_wetbulb(sfcpcl)

        sfcwm10c.append(wm10c_fov)
        sfcwm20c.append(wm20c_fov)
        sfcwm30c.append(wm30c_fov)

        #mu
        cape_fov, cin_fov, lclhght_fov = calc_instability(mupcl)

        mucape.append(cape_fov)
        mucin.append(cin_fov)
        mulclhght.append(lclhght_fov)

        #ml
        cape_fov, cin_fov, lclhght_fov = calc_instability(mlpcl)

        mlcape.append(cape_fov)
        mlcin.append(cin_fov)
        mllclhght.append(lclhght_fov)

    s2 = timeit.default_timer()
    print (s2-s1, "seconds")

    np.savez(odir+oname,
    lat=lats, lon=lons, times=times, qf=qf, ascend=ascend,
    k_index=k_index, t_totals=t_totals, haines=haines,
    lapserate_700_500=lapserate_700_500, lapserate_850_500=lapserate_850_500,
    sfccape=sfccape, mucape=mucape, mlcape=mlcape,
    sfccin=sfccin, mucin=mucin, mlcin=mlcin,
    sfclclhght=sfclclhght, mulclhght=mulclhght, mllclhght=mllclhght,
    sfchghtm10c=sfchghtm10c, sfchghtm20c=sfchghtm20c, sfchghtm30c=sfchghtm30c,
    sfcwm10c=sfcwm10c, sfcwm20c=sfcwm20c, sfcwm30c=sfcwm30c,
    precip_water=precip_water
    )

stop = timeit.default_timer()
print('"demo_files.py: "All done! Time: ', (stop - start_all)/60, " mins")
