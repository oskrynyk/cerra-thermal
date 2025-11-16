"""
Thermal Forecast Testing

This module tests CERRA thermal forecasts and creates interpolated datasets
from 3-hour to 1-hour steps for forecast analysis and validation.

Warning: Experimental code for forecast evaluation.
"""

import sys
from pathlib import Path

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

from src.config import DATA_DIR, OUTPUT_DIR

print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Load forecast radiation datasets
# Update these filenames to match your actual forecast data files
# Common naming: radiation_forecast_lead1.nc, radiation_forecast_lead2.nc, etc.
rad1 = xr.open_dataset(str(DATA_DIR / "radiation_forecast_lead1.nc"))
rad1 = rad1.assign_coords(longitude=(((rad1.longitude + 180) % 360) - 180))

rad2 = xr.open_dataset(str(DATA_DIR / "radiation_forecast_lead2.nc"))
rad2 = rad2.assign_coords(longitude=(((rad2.longitude + 180) % 360) - 180))

rad3 = xr.open_dataset(str(DATA_DIR / "radiation_forecast_lead3.nc"))
rad3 = rad3.assign_coords(longitude=(((rad3.longitude + 180) % 360) - 180))

rad3 = rad3 - rad2
rad2 = rad2 - rad1

rad2['time'] = rad2['time'] + np.timedelta64(1, 'h')
rad2 = rad2.assign_coords(step=np.timedelta64(3600000000000, 'ns'))
rad2 = rad2.assign_coords(valid_time=rad2['time'] + rad2['step'])

rad3['time'] = rad3['time'] + np.timedelta64(2, 'h')
rad3 = rad3.assign_coords(step=np.timedelta64(3600000000000, 'ns'))
rad3 = rad3.assign_coords(valid_time=rad3['time'] + rad3['step'])

rad = xr.concat(objs=[rad1, rad2, rad3], dim='time', data_vars='all', coords='different', compat='broadcast_equals')
rad = rad.sortby('time')
rad = rad / 3600

rad['ssrг'] = rad.ssrd - rad.ssr
rad['stru'] = rad.strd - rad.str
global_mean = rad.mean(dim=('y', 'x'))
plt.figure(figsize=(10, 6))
for var_name in global_mean.data_vars:
    plt.plot(global_mean['time'], global_mean[var_name], label=var_name)
plt.title('Global Mean Radiation Variables')
plt.xlabel('Time')
plt.ylabel('Global Mean')
plt.legend()
plt.grid()
plt.show()

rad = rad.sel(time='2017-07-01')

def approximate_dsrp_revised(fdir, cossza):
    """
    Helper function to approximate dsrp from fdir and cossza
    Note that this introduces large errors as cossza approaches zero.
    Only use this approximation if dsrp is not available in your dataset.
    :param fdir: is Total sky direct solar radiation at surface [W m-2]
    :param cossza: is cosine of solar zenith angle [dimentionless]
    returns direct radiation from the Sun [W m-2]
    """
    # filter statement for solar zenith angle to avoid division by zero.
    csza_filter1 = np.where((cossza > 0.001))
    dsrp = 0  # for cossza <= 0.01, equals to fdir
    dsrp[csza_filter1] = dsrp[csza_filter1] / cossza[csza_filter1]
    return dsrp

def approximate_dsrp_revised(fdir, cossza):
    dsrp = xr.where(cossza > 0.001, fdir / cossza, 0)
    return dsrp

from xclim.core.units import (
    amount2rate,
    convert_units_to,
    declare_units,
    flux2rate,
    rate2flux,
    units,
    units2pint,
)
from xclim.core.utils import Quantified
from xclim.indices.helpers import (
    _gather_lat,
    _gather_lon,
    cosine_of_solar_zenith_angle,
    day_lengths,
    distance_from_sun,
    extraterrestrial_solar_radiation,
    solar_declination,
    time_correction_for_solar_angle,
    wind_speed_height_conversion,
)

"W m-2"

rad.longitude.attrs["units"] = "degrees"

dates = rad.tidirswrf.time
lat = _gather_lat(rad.tidirswrf)
lon = _gather_lon(rad.tidirswrf)
dec = solar_declination(dates)

csza_i = cosine_of_solar_zenith_angle(dates, dec, lat, lon=lon, stat="average", sunlit=False)

csza_s = cosine_of_solar_zenith_angle(dates, dec, lat, lon=lon, stat="average", sunlit=True)

i_star = xr.where(csza_s > 0.001, rad.tidirswrf / csza_s, 0)

cerra_mrt = xr.DataArray(np.zeros_like(rad.ssr), dims=rad.ssr.dims, coords=rad.ssr.coords)

for timestamp in np.unique(rad.time.values):
    # cossza_integrated = thermofeel.calculate_cos_solar_zenith_angle_integrated(lon=rad.longitude.values, lat=rad.latitude.values,
    #                                                                            y=timestamp.astype('datetime64[Y]').astype(int) + 1970,
    #                                                                            m=timestamp.astype('datetime64[M]').astype(int) % 12 + 1,
    #                                                                            d=timestamp.astype('datetime64[D]').astype(int) % 31 + 1,
    #                                                                            h=timestamp.astype('datetime64[h]').astype(int) % 24,
    #                                                                            tbegin=0,
    #                                                                            tend=1)
    cossza_integrated = csza_i.sel(time=timestamp).values
    # cerra_time = '2017-07-01T14'
    # era_time = '2017-07-01T15'

    cerra_time = timestamp
    cerra_time = str(cerra_time)[:-16]
    era_time = timestamp + np.timedelta64(1, 'h')
    era_time = str(era_time)[:-16]

    # fdir = xr.open_dataset("radiation_forecast_era.nc")['fdir'].sel(time=era_time) / 3600 # Total sky direct solar radiation at surface [W m-2]
    # coords = {"longitude": rad.longitude, "latitude": rad.latitude}
    # fdir_interp = fdir.interp(coords=coords, method="nearest")
    fdir_interp = rad.tidirswrf.sel(time=cerra_time)

    # dsrp_approx = approximate_dsrp_revised(fdir=fdir_interp.values, cossza=cossza_integrated)
    dsrp_approx = i_star.sel(time=timestamp).values

    ssr = rad.ssr.sel(time=cerra_time)  # Surface net solar radiation [W m-2]
    ssrd = rad.ssrd.sel(time=cerra_time) # Surface solar radiation downwards [W m-2]
    strr = rad.str.sel(time=cerra_time) # Surface net thermal radiation [W m-2]
    strd = rad.strd.sel(time=cerra_time) # Surface thermal radiation downwards [W m-2]
    mrt = thermofeel.calculate_mean_radiant_temperature(cossza=cossza_integrated,
                                                        fdir=fdir_interp.values,
                                                        dsrp=dsrp_approx,
                                                        ssr=ssr.values,
                                                        ssrd=ssrd.values,
                                                        strr=strr.values,
                                                        strd=strd.values)
    mrt = mrt - 273.15
    cerra_mrt.loc[{'time': timestamp}] = mrt
    print('Timestamp: {}'.format(timestamp))

    # mrt diff
    path = 'era5/'
    file_pattern = 'ECMWF_mrt_*.nc'
    era_mrt = xr.open_mfdataset(f'{path}/{file_pattern}', combine='by_coords')['mrt'] - 273.15
    era_mrt = era_mrt.rename({'lon': 'longitude', 'lat': 'latitude'})
    era_mrt = era_mrt.sel(time=era_time)
    coords = {"latitude": rad.latitude, "longitude": rad.longitude}
    era_interp = era_mrt.interp(coords=coords, method="nearest")

    cerra_xr = xr.DataArray(np.zeros_like(ssr), dims=ssr.dims, coords=ssr.coords)
    cerra_xr.data = mrt  # <--################################################################

    diff = cerra_xr - era_interp

    # Define the Lambert Conformal projection with parameters according to CERRA dataset
    cerra_proj = ccrs.LambertConformal(
        central_longitude=8,
        central_latitude=50,
        standard_parallels=(50, 50),
        globe=ccrs.Globe(ellipse='sphere', semimajor_axis=6371229)
    )
    # plot
    fig = plt.figure(figsize=(5, 5), edgecolor=None, constrained_layout=True)
    ax = plt.axes(projection=cerra_proj)
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.BORDERS, linestyle='--', edgecolor='black')
    abs_val = np.abs(diff.values).max()
    filled_c = plt.pcolormesh(lon_mg.values, lat_mg.values, diff.values, transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=-abs_val, vmax=abs_val) #, vmin=-abs_val, vmax=abs_val
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    ax.margins(0, x=None, y=None, tight=True)
    cbar = fig.colorbar(filled_c, ax=ax, orientation="horizontal", pad=0.08)
    cbar.set_label(era_time) #(°C) (W/m\u00b2)
    fig.savefig('output_hourly/mrt_diff_cerra-era_' + era_time + '.png', dpi=600)
    plt.close()




rad = xr.open_dataset("radiation_forecast_3.nc")
rad = rad.assign_coords(longitude=(((rad.longitude + 180) % 360) - 180))
lon_mg, lat_mg = xr.broadcast(rad.longitude, rad.latitude)
cossza_integrated = thermofeel.calculate_cos_solar_zenith_angle_integrated(lon=lon_mg.values, lat=lat_mg.values,
                                                                          y=2017,
                                                                          m=7,
                                                                          d=1,
                                                                          h=12,
                                                                          tbegin=0,
                                                                          tend=3)
cerra_time = '2017-07-01T12'
era_time = '2017-07-01T15'

fdir_interp = rad.tidirswrf.sel(time=cerra_time) / 10800 #3600, 7200, 10800
dsrp_approx = thermofeel.approximate_dsrp(fdir=fdir_interp.values, cossza=cossza_integrated)

ssr = rad.ssr.sel(time=cerra_time) / 10800 # Surface net solar radiation [W m-2]
ssrd = rad.ssrd.sel(time=cerra_time) / 10800 # Surface solar radiation downwards [W m-2]
strr = rad.str.sel(time=cerra_time) / 10800 # Surface net thermal radiation [W m-2]
strd = rad.strd.sel(time=cerra_time) / 10800 # Surface thermal radiation downwards [W m-2]
mrt = thermofeel.calculate_mean_radiant_temperature(cossza=cossza_integrated,
                                                    fdir=fdir_interp.values,
                                                    dsrp=dsrp_approx,
                                                    ssr=ssr.values,
                                                    ssrd=ssrd.values,
                                                    strr=strr.values,
                                                    strd=strd.values)
mrt = mrt - 273.15

# mrt diff
path = 'era5/'
file_pattern = 'ECMWF_mrt_*.nc'
era_mrt = xr.open_mfdataset(f'{path}/{file_pattern}', combine='by_coords')['mrt'] - 273.15
era_mrt = era_mrt.rename({'lon': 'longitude', 'lat': 'latitude'})
era_mrt = era_mrt.sel(time=era_time)
coords = {"latitude": rad.latitude, "longitude": rad.longitude}
era_interp = era_mrt.interp(coords=coords, method="nearest")

cerra_xr = xr.DataArray(np.zeros_like(ssr), dims=ssr.dims, coords=ssr.coords)
cerra_xr.data = mrt  # <--################################################################

diff = cerra_xr - era_interp


# Define the Lambert Conformal projection with parameters according to CERRA dataset
cerra_proj = ccrs.LambertConformal(
    central_longitude=8,
    central_latitude=50,
    standard_parallels=(50, 50),
    globe=ccrs.Globe(ellipse='sphere', semimajor_axis=6371229)
)
# plot
fig = plt.figure(figsize=(5, 5), edgecolor=None, constrained_layout=True)
ax = plt.axes(projection=cerra_proj)
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.BORDERS, linestyle='--', edgecolor='black')
abs_val = np.abs(diff.values).max()
filled_c = plt.pcolormesh(lon_mg.values, lat_mg.values, diff.values, transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=-abs_val, vmax=abs_val) #, vmin=-abs_val, vmax=abs_val
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.margins(0, x=None, y=None, tight=True)
cbar = fig.colorbar(filled_c, ax=ax, orientation="horizontal", pad=0.08)
cbar.set_label(era_time) #(°C) (W/m\u00b2)
fig.savefig('output_hourly/tidirswrf_forecast/mrt_diff_cerra-era_' + era_time, dpi=600)
plt.close()