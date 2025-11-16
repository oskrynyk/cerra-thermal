"""
ERA5 vs CERRA Comparison Analysis

This script compares thermal comfort calculations from ERA5 and CERRA reanalysis datasets.
It loads radiation, wind, and temperature/humidity data from both sources and computes
differences in thermal indices (MRT, UTCI) and atmospheric variables.

The comparison helps evaluate the performance of high-resolution CERRA data against
the well-established ERA5 reanalysis for thermal comfort applications.

Configuration:
    Modify src/config.py to change data paths and filenames.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import thermofeel
import numpy as np
from netCDF4 import Dataset
import dask.array as da
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime

from src.config import (
    DATA_DIR,
    ERA5_DATA_DIR,
    OUTPUT_DIR,
    RADIATION_FILE,
    WIND_FILE,
    TEMPERATURE_HUMIDITY_FILE,
)

print(f"Main data directory: {DATA_DIR}")
print(f"ERA5 data directory: {ERA5_DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Open the radiation dataset
e1 = xr.open_dataset(str(ERA5_DATA_DIR / RADIATION_FILE))
# Note: Uncomment and modify time selection as needed for specific analysis periods
# e1 = e1.sel(time='2017-01')

# Open the wind speed and humidity datasets
e2 = xr.open_dataset(str(ERA5_DATA_DIR / WIND_FILE))
# e2 = e2.sel(time='2017-01')

e3 = xr.open_dataset(str(ERA5_DATA_DIR / TEMPERATURE_HUMIDITY_FILE))
# e3 = e3.sel(time='2017-01')

# Open the CERRA radiation dataset
f1 = xr.open_dataset(str(DATA_DIR / RADIATION_FILE))
# f1 = f1.sel(time='2017-01')
f1 = f1.assign_coords(longitude=(((f1.longitude + 180) % 360) - 180))

# Open the CERRA wind speed and humidity datasets
f2 = xr.open_dataset(str(DATA_DIR / WIND_FILE))
# f2 = f2.sel(time='2017-01')
f2 = f2.assign_coords(longitude=(((f2.longitude + 180) % 360) - 180))

f3 = xr.open_dataset(str(DATA_DIR / TEMPERATURE_HUMIDITY_FILE))
# f3 = f3.sel(time='2017-01')
f3 = f3.assign_coords(longitude=(((f3.longitude + 180) % 360) - 180))
lon_mg, lat_mg = xr.broadcast(f3.longitude, f3.latitude)


#ssrd
era = e1.sel(time=e1.time.dt.hour == 12).ssrd / 3600
cerra = f1.sel(time=f1.time.dt.hour == 12).ssrd / 10800

coords = {"latitude": f1.latitude, "longitude": f1.longitude}
era_interp = era.interp(coords=coords, method="nearest")

diff = cerra - era_interp
ssrd = diff.mean(dim='time')


#ssr
era = e1.sel(time=e1.time.dt.hour == 12).ssr / 3600
cerra = f1.sel(time=f1.time.dt.hour == 12).ssr / 10800

coords = {"latitude": f1.latitude, "longitude": f1.longitude}
era_interp = era.interp(coords=coords, method="nearest")

diff = cerra - era_interp
ssr = diff.mean(dim='time')

#strd
era = e1.sel(time=e1.time.dt.hour == 12).strd / 3600
cerra = f1.sel(time=f1.time.dt.hour == 12).strd / 10800

coords = {"latitude": f1.latitude, "longitude": f1.longitude}
era_interp = era.interp(coords=coords, method="nearest")

diff = cerra - era_interp
strd = diff.mean(dim='time')

#strr
era = e1.sel(time=e1.time.dt.hour == 12).str / 3600
cerra = f1.sel(time=f1.time.dt.hour == 12).str / 10800

coords = {"latitude": f1.latitude, "longitude": f1.longitude}
era_interp = era.interp(coords=coords, method="nearest")

diff = cerra - era_interp
strr = diff.mean(dim='time')

#fdir
era = e1.sel(time=e1.time.dt.hour == 12).fdir / 3600
cerra = f1.sel(time=f1.time.dt.hour == 12).tidirswrf / 10800

coords = {"latitude": f1.latitude, "longitude": f1.longitude}
era_interp = era.interp(coords=coords, method="nearest")

diff = cerra - era_interp
fdir = diff.mean(dim='time')


#WIND SPEED
era = np.sqrt(e2.sel(time=e2.time.dt.hour == 12).u10 ** 2 + e2.sel(time=e2.time.dt.hour == 12).v10 ** 2)
cerra = f2.sel(time=f2.time.dt.hour == 12).si10

coords = {"latitude": f2.latitude, "longitude": f2.longitude}
era_interp = era.interp(coords=coords, method="nearest")

diff = cerra - era_interp
ws10 = diff.mean(dim='time')

#t2m temperature
era = e3.sel(time=e3.time.dt.hour == 12).t2m

cerra = f3.sel(time=f3.time.dt.hour == 12).t2m

coords = {"latitude": f3.latitude, "longitude": f3.longitude}
era_interp = era.interp(coords=coords, method="nearest")

diff = cerra - era_interp
t2m = diff.mean(dim='time')


#Relative Humidity (%)
era_t2m = e3.sel(time=e3.time.dt.hour == 12).t2m
era_d2m = e3.sel(time=e3.time.dt.hour == 12).d2m
era = xr.DataArray(np.zeros_like(era_d2m), dims=era_d2m.dims, coords=era_d2m.coords)
for timestamp in np.unique(era_d2m.time.values):
    t2m = era_t2m.sel(time=timestamp)
    d2m = era_d2m.sel(time=timestamp)
    rh_ts = thermofeel.calculate_relative_humidity_percent(t2k=t2m.values, tdk=d2m.values)
    era.loc[{'time': timestamp}] = rh_ts
    print('Timestamp: {}'.format(timestamp))

cerra = f3.sel(time=f3.time.dt.hour == 12).r2

coords = {"latitude": f3.latitude, "longitude": f3.longitude}
era_interp = era.interp(coords=coords, method="nearest")

diff = cerra - era_interp
rh = diff.mean(dim='time')


#dsrp and MRT
era = e1.sel(time=e1.time.dt.hour == 12) / 3600
# lon_mg_era, lat_mg_era = xr.broadcast(era.longitude, era.latitude)
lon_mg_era, lat_mg_era = np.meshgrid(era.longitude.values, era.latitude.values)

era_windspeed = np.sqrt(e2.sel(time=e2.time.dt.hour == 12).u10 ** 2 + e2.sel(time=e2.time.dt.hour == 12).v10 ** 2)

era_cosszainstant = xr.DataArray(np.zeros_like(era.fdir), dims=era.fdir.dims, coords=era.fdir.coords)
era_dsrp = xr.DataArray(np.zeros_like(era.fdir), dims=era.fdir.dims, coords=era.fdir.coords)
era_mrt = xr.DataArray(np.zeros_like(era.fdir), dims=era.fdir.dims, coords=era.fdir.coords)
era_rh = xr.DataArray(np.zeros_like(era.fdir), dims=era.fdir.dims, coords=era.fdir.coords)
era_utci = xr.DataArray(np.zeros_like(era.fdir), dims=era.fdir.dims, coords=era.fdir.coords)
for timestamp in np.unique(era.fdir.time.values):
    cosszainstant = thermofeel.calculate_cos_solar_zenith_angle(lon=lon_mg_era, lat=lat_mg_era,
                                                                y=timestamp.astype('datetime64[Y]').astype(int) + 1970,
                                                                m=timestamp.astype('datetime64[M]').astype(int) % 12 + 1,
                                                                d=timestamp.astype('datetime64[D]').astype(int) % 31 + 1,
                                                                h=timestamp.astype('datetime64[h]').astype(int) % 24)
    era_cosszainstant.loc[{'time': timestamp}] = cosszainstant

    fdir = era.fdir.sel(time=timestamp)  # Total sky direct solar radiation at surface [W m-2]
    ssrd = era.ssrd.sel(time=timestamp)  # Surface solar radiation downwards [W/m^-2]
    ssr = era.ssr.sel(time=timestamp)  # Surface net solar radiation [W/m^-2]
    strd = era.strd.sel(time=timestamp)  # Surface thermal radiation downwards [W/m^-2]
    strr = era.str.sel(time=timestamp)  # Surface net thermal radiation [W/m^-2]

    approx_dsrp = thermofeel.approximate_dsrp(fdir=fdir.values, cossza=cosszainstant)
    era_dsrp.loc[{'time': timestamp}] = approx_dsrp

    mrt = thermofeel.calculate_mean_radiant_temperature(cossza=cosszainstant,
                                                        dsrp=approx_dsrp,
                                                        fdir=fdir.values,
                                                        ssrd=ssrd.values,
                                                        ssr=ssr.values,
                                                        strd=strd.values,
                                                        strr=strr.values) # calculate MRT in K
    mrt_c = mrt - 273.15  # convert MRT to °C
    era_mrt.loc[{'time': timestamp}] = mrt_c
    mrt = mrt.clip(min=243.14999999999998, max=303.15) #-30 and +70 in Celsius

    windspeed = era_windspeed.sel(time=timestamp)
    windspeed = windspeed.clip(min=0.5, max=17)

    t2m = e3.sel(time=e3.time.dt.hour == 12).t2m.sel(time=timestamp)
    d2m = e3.sel(time=e3.time.dt.hour == 12).d2m.sel(time=timestamp)
    rh_ts = thermofeel.calculate_relative_humidity_percent(t2k=t2m.values, tdk=d2m.values)
    era_rh.loc[{'time': timestamp}] = rh_ts

    t2m = t2m.clip(min=223.14999999999998, max=323.15) #-50 and +50 in Celsius

    utci = thermofeel.calculate_utci(t2_k=t2m.values, va_ms=windspeed.values, mrt_k=mrt, td_k=d2m.values) # Calculate UTCI with a 6th order polynomial approximation
    era_utci.loc[{'time': timestamp}] = utci
    print('Timestamp: {}'.format(timestamp))


cerra = f3.sel(time=f3.time.dt.hour == 12)
lon_mg, lat_mg = xr.broadcast(cerra.longitude, cerra.latitude)

cerra_cosszainstant = xr.DataArray(np.zeros_like(cerra.t2m), dims=cerra.t2m.dims, coords=cerra.t2m.coords)
cerra_dsrp = xr.DataArray(np.zeros_like(cerra.t2m), dims=cerra.t2m.dims, coords=cerra.t2m.coords)

cerra_rh = f3.sel(time=f3.time.dt.hour == 12).r2

cerra_mrt = xr.DataArray(np.zeros_like(cerra.t2m), dims=cerra.t2m.dims, coords=cerra.t2m.coords)
cerra_utci = xr.DataArray(np.zeros_like(cerra.t2m), dims=cerra.t2m.dims, coords=cerra.t2m.coords)
for timestamp in np.unique(cerra.time.values):
    # Convert timestamp to a UTC datetime object
    # dt = datetime.datetime.utcfromtimestamp(timestamp.astype(int) * 1e-9)
    # # Calculate the cosine of the solar zenith angle for this timestamp
    # cosszainstant = thermofeel.calculate_cos_solar_zenith_angle(lat=lat_mg.values, lon=lon_mg.values,
    #                                                             y=dt.year,
    #                                                             m=dt.month,
    #                                                             d=dt.day,
    #                                                             h=dt.hour)
    # Calculate the cosine of the solar zenith angle for this timestamp with original dtype
    cosszainstant = thermofeel.calculate_cos_solar_zenith_angle(lon=lon_mg.values, lat=lat_mg.values,
                                                                y=timestamp.astype('datetime64[Y]').astype(int) + 1970,
                                                                m=timestamp.astype('datetime64[M]').astype(int) % 12 + 1,
                                                                d=timestamp.astype('datetime64[D]').astype(int) % 31 + 1,
                                                                h=timestamp.astype('datetime64[h]').astype(int) % 24)
    cerra_cosszainstant.loc[{'time': timestamp}] = cosszainstant

    fdir = f1.tidirswrf.sel(time=timestamp) / 10800 # Time-integrated surface direct short wave radiation flux - counterpart of the Total sky direct solar radiation at surface [W m-2]
    ssrd = f1.ssrd.sel(time=timestamp) / 10800 # Surface solar radiation downwards [W/m^-2]
    ssr = f1.ssr.sel(time=timestamp) / 10800 # Surface net solar radiation [W/m^-2]
    strd = f1.strd.sel(time=timestamp) / 10800 # Surface thermal radiation downwards [W/m^-2]
    strr = f1.str.sel(time=timestamp) / 10800 # Surface net thermal radiation [W/m^-2]

    approx_dsrp = thermofeel.approximate_dsrp(fdir=fdir.values, cossza=cosszainstant)  # approximation of the Direct radiation from the Sun [W/m^-2]
    cerra_dsrp.loc[{'time': timestamp}] = approx_dsrp

    mrt = thermofeel.calculate_mean_radiant_temperature(cossza=cosszainstant,
                                                        dsrp=approx_dsrp,
                                                        fdir=fdir.values,
                                                        ssrd=ssrd.values,
                                                        ssr=ssr.values,
                                                        strd=strd.values,
                                                        strr=strr.values) # calculate MRT in K
    mrt_c = mrt - 273.15 # convert MRT to °C
    cerra_mrt.loc[{'time': timestamp}] = mrt_c
    mrt_c = mrt_c.clip(min=-30, max=70)

    windspeed = f2.si10.sel(time=timestamp)
    windspeed = windspeed.clip(min=0.5, max=17)

    rh = f3.r2.sel(time=timestamp)
    t2k = f3.t2m.sel(time=timestamp)
    t2m = f3.t2m.sel(time=timestamp) - 273.15
    t2m = t2m.clip(min=-50, max=50)

    rh_kPa = thermofeel.calculate_saturation_vapour_pressure(tk=t2k.values) * rh.values / 1000.0 # Calculate rh in kPa
    utci = thermofeel.calculate_utci_polynomial(t2m=t2m.values,
                                                mrt=mrt_c,
                                                va=windspeed.values,
                                                rh=rh_kPa) # Calculate UTCI with a 6th order polynomial approximation
    cerra_utci.loc[{'time': timestamp}] = utci
    print('Timestamp: {}'.format(timestamp))


cerra_cosszainstant = xr.DataArray(np.zeros_like(cerra.t2m), dims=cerra.t2m.dims, coords=cerra.t2m.coords)
cerra_dsrp = xr.DataArray(np.zeros_like(cerra.t2m), dims=cerra.t2m.dims, coords=cerra.t2m.coords)

cerra_rh = f3.sel(time=f3.time.dt.hour == 12).r2

cerra_mrt = xr.DataArray(np.zeros_like(cerra.t2m), dims=cerra.t2m.dims, coords=cerra.t2m.coords)
cerra_utci = xr.DataArray(np.zeros_like(cerra.t2m), dims=cerra.t2m.dims, coords=cerra.t2m.coords)


#cosszainstant
coords = {"latitude": f3.latitude, "longitude": f3.longitude}
era_interp = era_cosszainstant.interp(coords=coords, method="nearest")

diff = cerra_cosszainstant - era_interp
cossza = diff.mean(dim='time')

#dsrp
coords = {"latitude": f3.latitude, "longitude": f3.longitude}
era_interp = era_dsrp.interp(coords=coords, method="nearest")

diff = cerra_dsrp - era_interp
dsrp = diff.mean(dim='time')


#mrt
coords = {"latitude": f3.latitude, "longitude": f3.longitude}
era_interp = era_mrt.interp(coords=coords, method="nearest")

diff = cerra_mrt - era_interp
mrt = diff.mean(dim='time')

#Relative Humidity (%)
cerra_rh = f3.sel(time=f3.time.dt.hour == 12).r2

coords = {"latitude": f3.latitude, "longitude": f3.longitude}
era_interp = era_rh.interp(coords=coords, method="nearest")

diff = cerra_rh - era_interp
rh = diff.mean(dim='time')

#utci
coords = {"latitude": f3.latitude, "longitude": f3.longitude}
era_interp = era_utci.interp(coords=coords, method="nearest")

diff = cerra_utci - era_interp
utci = diff.mean(dim='time')





# Define the Lambert Conformal projection with parameters according to CERRA dataset
cerra_proj = ccrs.LambertConformal(
    central_longitude=8,
    central_latitude=50,
    standard_parallels=(50, 50),
    globe=ccrs.Globe(ellipse='sphere', semimajor_axis=6371229)
)

datasets = [ssrd, ssr, strd, strr, fdir, cossza, dsrp, mrt, t2m, ws10, rh, utci]
labels = ['ssrd (W/m\u00b2)', 'ssr (W/m\u00b2)', 'strd (W/m\u00b2)', 'strr (W/m\u00b2)', 'fdir (W/m\u00b2)', 'cossza', 'dsrp (W/m\u00b2)',
          'mrt (°C)', 't2m (°C)', 'ws10 (m/s)', 'rh (%)', 'utci (°C)']
paths = ['ssrd', 'ssr', 'strd', 'strr', 'fdir', 'cossza', 'dsrp', 'mrt', 't2m', 'ws10', 'rh', 'utci']

datasets = [mrt, utci]
labels = ['mrt (°C)', 'utci (°C)']
paths = ['mrt', 'utci']

# plot
for dataset, label, path in zip(datasets, labels, paths):
    fig = plt.figure(figsize=(5, 5), edgecolor=None, constrained_layout=True)
    ax = plt.axes(projection=cerra_proj)
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.BORDERS, linestyle='--', edgecolor='black')
    abs_val = np.abs(dataset.values).max()
    filled_c = plt.pcolormesh(lon_mg.values, lat_mg.values, dataset.values, transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=-abs_val, vmax=abs_val)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    ax.margins(0, x=None, y=None, tight=True)
    cbar = fig.colorbar(filled_c, ax=ax, orientation="horizontal", pad=0.02)
    cbar.set_label('2017 Jul ERA5 - CERRA mean diff. ' + ''.join(label))
    fig.savefig('output_fig_comparison/mean_diff_jul-' + path, dpi=600)
    plt.close(fig)