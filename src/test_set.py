"""
Test set for thermal comfort calculations.

This module contains experimental code and testing routines for validating
thermal indices calculations against different datasets and methodologies.

Warning: This is test/experimental code and may not be production-ready.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import xarray as xr
import numpy as np
import thermofeel
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from src.config import (
    DATA_DIR,
    ERA5_DATA_DIR,
    RADIATION_FILE,
    WIND_FILE,
    TEMPERATURE_HUMIDITY_FILE,
)

# Load CERRA radiation data
f = xr.open_dataset(str(DATA_DIR / RADIATION_FILE))
f = f.assign_coords(longitude=(((f.longitude + 180) % 360) - 180))
# Note: Modify time selection as needed for analysis
# f = f.sel(time='2017-07-01T03') /10800

# Load ERA5 radiation data
e = xr.open_dataset(str(ERA5_DATA_DIR / RADIATION_FILE))
# e = e.sel(time='2017-07-01T06') /3600

lon_mg, lat_mg = xr.broadcast(f.longitude, f.latitude)
cosszainstant = thermofeel.calculate_cos_solar_zenith_angle(lon=lon_mg.values, lat=lat_mg.values,
                                                            y=2017,
                                                            m=7,
                                                            d=1,
                                                            h=6)

fdir = f.tidirswrf  # Time-integrated surface direct short wave radiation flux - counterpart of the Total sky direct solar radiation at surface [W m-2]


approx_dsrp = thermofeel.approximate_dsrp(fdir=fdir.values, cossza=cosszainstant)

ssr = f.ssr  # Surface net solar radiation [W/m^-2]
ssrd = f.ssrd  # Surface solar radiation downwards [W/m^-2]
strr = f.str  # Surface net thermal radiation [W/m^-2]
strd = f.strd  # Surface thermal radiation downwards [W/m^-2]
mrt = thermofeel.calculate_mean_radiant_temperature(cossza=cosszainstant,
                                                    fdir=fdir.values,
                                                    dsrp=approx_dsrp,
                                                    ssr=ssr.values,
                                                    ssrd=ssrd.values,
                                                    strr=strr.values,
                                                    strd=strd.values)

#ERA5
era_lon_mg, era_lat_mg = np.meshgrid(e.longitude.values, e.latitude.values)
era_cosszainstant = thermofeel.calculate_cos_solar_zenith_angle(lon=era_lon_mg, lat=era_lat_mg,
                                                                y=2017,
                                                                m=7,
                                                                d=1,
                                                                h=12)

cossza_integrated = thermofeel.calculate_cos_solar_zenith_angle_integrated(lon=era_lon_mg, lat=era_lat_mg,
                                                                          y=2017,
                                                                          m=7,
                                                                          d=1,
                                                                          h=6,
                                                                          tbegin=0,
                                                                          tend=6)

era_fdir = e.fdir  # Total sky direct solar radiation at surface [W m-2]
era_approx_dsrp = thermofeel.approximate_dsrp(fdir=era_fdir.values, cossza=era_cosszainstant)

era_ssr = e.ssr  # Surface net solar radiation [W/m^-2]
era_ssrd = e.ssrd  # Surface solar radiation downwards [W/m^-2]
era_strr = e.str  # Surface net thermal radiation [W/m^-2]
era_strd = e.strd  # Surface thermal radiation downwards [W/m^-2]
era_mrt = thermofeel.calculate_mean_radiant_temperature(cossza=era_cosszainstant,
                                                        fdir=era_fdir.values,
                                                        dsrp=era_approx_dsrp,
                                                        ssr=era_ssr.values,
                                                        ssrd=era_ssrd.values,
                                                        strr=era_strr.values,
                                                        strd=era_strd.values)

era_xr = xr.DataArray(np.zeros_like(era_fdir), dims=era_fdir.dims, coords=era_fdir.coords)
era_xr.data = era_approx_dsrp       # <--################################################################
coords = {"latitude": cerra_xr.latitude, "longitude": cerra_xr.longitude}
era_interp = era_xr.interp(coords=coords, method="nearest")

path = 'era5/'
file_pattern = 'ECMWF_mrt_*.nc'
era_mrt = xr.open_mfdataset(f'{path}/{file_pattern}', combine='by_coords')['mrt']
era_mrt = era_mrt.rename({'lon': 'longitude', 'lat': 'latitude'})
era_mrt = era_mrt.sel(time='2017-07-01T12')
coords = {"latitude": f.latitude, "longitude": f.longitude}
era_interp = era_mrt.interp(coords=coords, method="nearest")


cerra_xr = xr.DataArray(np.zeros_like(fdir), dims=fdir.dims, coords=fdir.coords)
cerra_xr.data = approx_dsrp           # <--################################################################


diff = cerra_xr - era_interp




# Define the Lambert Conformal projection with parameters according to CERRA dataset
cerra_proj = ccrs.LambertConformal(
    central_longitude=8,
    central_latitude=50,
    standard_parallels=(50, 50),
    globe=ccrs.Globe(ellipse='sphere', semimajor_axis=6371229)
)
coords = {"latitude": cerra_xr.latitude, "longitude": cerra_xr.longitude}
era_interp = era_fdir.interp(coords=coords, method="nearest")


lon_mg, lat_mg = xr.broadcast(f.longitude, f.latitude)
cossza_integrated = thermofeel.calculate_cos_solar_zenith_angle_integrated(lat=lat_mg.values, lon=lon_mg.values,
                                                                          y=2017,
                                                                          m=7,
                                                                          d=1,
                                                                          h=3,
                                                                          tbegin=3,
                                                                          tend=6)

fdir = f.tidirswrf  # Time-integrated surface direct short wave radiation flux - counterpart of the Total sky direct solar radiation at surface [W m-2]
approx_dsrp = thermofeel.approximate_dsrp(fdir=fdir.values, cossza=cossza_integrated)

ssr = f.ssr  # Surface net solar radiation [W/m^-2]
ssrd = f.ssrd  # Surface solar radiation downwards [W/m^-2]
strr = f.str  # Surface net thermal radiation [W/m^-2]
strd = f.strd  # Surface thermal radiation downwards [W/m^-2]
mrt = thermofeel.calculate_mean_radiant_temperature(cossza=cossza_integrated,
                                                    fdir=fdir.values,
                                                    dsrp=approx_dsrp,
                                                    ssr=ssr.values,
                                                    ssrd=ssrd.values,
                                                    strr=strr.values,
                                                    strd=strd.values)
mrt = mrt - 273.15

# plot
fig = plt.figure(figsize=(5, 5), edgecolor=None, constrained_layout=True)
ax = plt.axes(projection=cerra_proj)
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.BORDERS, linestyle='--', edgecolor='black')
# abs_val = np.abs(diff.values).max()
filled_c = plt.pcolormesh(lon_mg.values, lat_mg.values, cossza_integrated, transform=ccrs.PlateCarree(), cmap="RdYlBu_r") #, vmin=-abs_val, vmax=abs_val
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.margins(0, x=None, y=None, tight=True)
cbar = fig.colorbar(filled_c, ax=ax, orientation="horizontal", pad=0.08)
cbar.set_label('2020-07-01T06 (°C)') #(°C) (W/m\u00b2)
fig.savefig('output_fig_mrt_test/diff/2020-07-01T12-cossza_integrated3-6_3', dpi=600)
plt.close()




def approximate_dsrp_1(fdir, cossza):
    """
    Helper function to approximate dsrp from fdir and cossza
    Note that this introduces large errors as cossza approaches zero.
    Only use this approximation if dsrp is not available in your dataset.
    :param fdir: is Total sky direct solar radiation at surface [W m-2]
    :param cossza: is cosine of solar zenith angle [dimentionless]
    returns direct radiation from the Sun [W m-2]
    """
    # filter statement for solar zenith angle to avoid division by zero.
    csza_filter1 = np.where((cossza > 0.1))
    dsrp = fdir  # for cossza <= 0.01, equals to fdir
    dsrp[csza_filter1] = dsrp[csza_filter1] / cossza[csza_filter1]
    return dsrp

approx_dsrp = approximate_dsrp_1(fdir=fdir.values, cossza=cosszainstant)












mrt_xr = xr.DataArray(np.zeros_like(f.ssr), dims=f.ssr.dims, coords=f.ssr.coords)
cosszainstant_xr = xr.DataArray(np.zeros_like(f.ssr), dims=f.ssr.dims, coords=f.ssr.coords)
dsrp_appx_xr = xr.DataArray(np.zeros_like(f.ssr), dims=f.ssr.dims, coords=f.ssr.coords)
for timestamp in np.unique(f.time.values):
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
    cosszainstant_xr.loc[{'time': timestamp}] = cosszainstant

    ssrd = f.ssrd.sel(time=timestamp) / 10800 # Surface solar radiation downwards [W/m^-2]
    ssr = f.ssr.sel(time=timestamp) / 10800 # Surface net solar radiation [W/m^-2]
    strd = f.strd.sel(time=timestamp) / 10800 # Surface thermal radiation downwards [W/m^-2]
    fdir = f.tidirswrf.sel(time=timestamp) / 10800 # Time-integrated surface direct short wave radiation flux - counterpart of the Total sky direct solar radiation at surface [W m-2]
    strr = f.str.sel(time=timestamp) / 10800 # Surface net thermal radiation [W/m^-2]

    # ssrc = f_test.ssrc.sel(time=timestamp) / 10800
    # strc = f_test.strc.sel(time=timestamp) / 10800

    # testing fdir
    dsrp_appx = thermofeel.approximate_dsrp(fdir=fdir.values, cossza=cosszainstant) # approximation of the Direct radiation from the Sun [W/m^-2]
    dsrp_appx_xr.loc[{'time': timestamp}] = dsrp_appx

    mrt = thermofeel.calculate_mean_radiant_temperature(ssrd=ssrd.values, ssr=ssr.values, dsrp=dsrp_appx,
                                                        strd=strd.values, fdir=fdir.values, strr=strr.values,
                                                        cossza=cosszainstant) # calculate MRT in K

    mrt_c = mrt - 273.15 # convert MRT to °C
    mrt_xr.loc[{'time': timestamp}] = mrt_c
    print('Timestamp: {}'.format(timestamp))