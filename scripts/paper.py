"""
Publication Figures Generation

This script generates publication-quality figures for thermal comfort analysis.
It creates maps and plots showing thermal stress indices (MRT, UTCI, NET) and
their spatial variation across the study domain.

Configuration:
    Modify src/config.py to change data paths and output locations.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import thermofeel
import xclim

from src.config import DATA_DIR, OUTPUT_DIR, PLOTS_DIR

print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Load radiation data
# Universal naming: For seasonal analysis, specify which dataset to load
# rad = xr.open_dataset(DATA_DIR / "radiation_seasonal.nc")  # Universal seasonal dataset
# For comparative analysis across different periods:
rad = xr.open_dataset(str(DATA_DIR / "radiation.nc"))

print("Radiation dataset loaded successfully")

# rad = rad.sel(time='2017-01-01')
# rad = rad.sel(time='2017-07-01')
# rad['time'] = rad['time'] - np.timedelta64(30, 'm')
# rad['diffuse'] = rad.ssrd - rad.tidirswrf
# rad.diffuse.attrs["units"] = "W m-2"

# rad = rad.drop_vars(['step', 'surface', 'valid_time'])
# rad.longitude.attrs["units"] = "degrees"
# dates = rad.tidirswrf.time
# lat = _gather_lat(rad.tidirswrf)
# lon = _gather_lon(rad.tidirswrf)
# dec = solar_declination(dates)
# csza_i = cosine_of_solar_zenith_angle(dates, dec, lat, lon=lon, stat="instant", sunlit=False)

# csza_sunlit = cosine_of_solar_zenith_angle(dates, dec, lat, lon=lon, stat="average", sunlit=True).astype('float32')
# # csza_s['time'] = csza_s['time'] + np.timedelta64(30, 'm')
# csza_integ = xr.DataArray(np.zeros_like(rad.tidirswrf), dims=rad.tidirswrf.dims, coords=rad.tidirswrf.coords)
# for timestamp in np.unique(rad.time.values):
#     cossza_integrated = thermofeel.calculate_cos_solar_zenith_angle_integrated(lon=rad.longitude.values, lat=rad.latitude.values,
#                                                                                y=timestamp.astype('datetime64[Y]').astype(int) + 1970,
#                                                                                m=timestamp.astype('datetime64[M]').astype(int) % 12 + 1,
#                                                                                d=timestamp.astype('datetime64[D]').astype(int) % 31 + 1,
#                                                                                h=timestamp.astype('datetime64[h]').astype(int) % 24,
#                                                                                tbegin=0,
#                                                                                tend=1)
#     csza_s_ecmwf.loc[{'time': timestamp}] = cossza_integrated
#     print('Timestamp: {}'.format(timestamp))

# start_time = time.time()
# #
# end_time = time.time()
# elapsed_time = end_time - start_time
# print("Elapsed time:", elapsed_time, "seconds")

# import numpy as np
# csza_s_integrated = cos_solar_zenith_angle_integrated(
#     datetime.datetime(2017, 1, 1),
#     datetime.datetime(2017, 1, 31),
#     rad.latitude.values,
#     rad.longitude.values,
#     intervals_per_hour=1,
#     integration_order=3
# )
#
# begin_date = datetime.datetime(2017, 1, 1)
# end_date = datetime.datetime(2017, 1, 31)
# cos_sza_values = []
# current_date = begin_date
# while current_date <= end_date:
#     print("Processing timestamp:", current_date)
#     cos_sza_hour = cos_solar_zenith_angle_integrated(
#         current_date, current_date + datetime.timedelta(hours=1),
#         rad.latitude, rad.longitude
#     )
#     cos_sza_values.append(cos_sza_hour)
#     current_date += datetime.timedelta(hours=1)
# cos_sza_array = xr.concat(cos_sza_values, dim='time')
# print(cos_sza_array)

csza_sunlit.to_netcdf('paper/csza_sunlit_jan.nc')
csza_integ.to_netcdf('paper/csza_integ_jan.nc')

csza_sunlit.to_netcdf('paper/csza_sunlit_jul.nc')
# csza_sunlit.to_netcdf('paper/csza_sunlit_jul_float64.nc')
csza_integ.to_netcdf('paper/csza_integ_jul.nc')

csza_sunlit = xr.open_dataarray('paper/csza_sunlit_jan.nc')
csza_integ = xr.open_dataarray('paper/csza_integ_jan.nc')

csza_sunlit = xr.open_dataarray('paper/csza_sunlit_jul.nc')
# csza_sunlit = xr.open_dataarray('paper/csza_sunlit_jul_float64.nc')
csza_integ = xr.open_dataarray('paper/csza_integ_jul.nc')

cerra_mrt_sunlit = mean_radiant_temperature(rsds=rad.ssrd, rsus=rad.ssru, rlds=rad.strd, rlus=rad.stru, csza=csza_sunlit, csza_thr=0.001, rsds_direct=rad.tidirswrf)-273.15
cerra_mrt_integ = mean_radiant_temperature(rsds=rad.ssrd, rsus=rad.ssru, rlds=rad.strd, rlus=rad.stru, csza=csza_integ, csza_thr=0.05, rsds_direct=rad.tidirswrf)-273.15
cerra_mrt_sunlit_apx = mean_radiant_temperature(rsds=rad.ssrd, rsus=rad.ssru, rlds=rad.strd, rlus=rad.stru, csza=csza_sunlit, csza_thr=0.001).astype('float32')-273.15 # rsds_direct is not given
cerra_mrt_integ_apx = mean_radiant_temperature(rsds=rad.ssrd, rsus=rad.ssru, rlds=rad.strd, rlus=rad.stru, csza=csza_integ, csza_thr=0.1).astype('float32')-273.15 # rsds_direct is not given

cerra_mrt_sunlit.to_netcdf('paper/mrt_jan_sunlit_001.nc')
cerra_mrt_integ.to_netcdf('paper/mrt_jan_integ.nc')
cerra_mrt_sunlit_apx.to_netcdf('paper/mrt_jan_sunlit_apx_001.nc')
cerra_mrt_integ_apx.to_netcdf('paper/mrt_jan_integ_apx_1.nc')

cerra_mrt_sunlit.to_netcdf('paper/mrt_jul_sunlit_001.nc')
cerra_mrt_integ.to_netcdf('paper/mrt_jul_integ.nc')
cerra_mrt_sunlit_apx.to_netcdf('paper/mrt_jul_sunlit_apx_001.nc')
cerra_mrt_integ_apx.to_netcdf('paper/mrt_jul_integ_apx_1.nc')

# cerra_mrt_sunlit = xr.open_dataarray('paper/mrt_jan_sunlit.nc')
cerra_mrt_sunlit = xr.open_dataarray('paper/mrt_jan_sunlit_001.nc')
cerra_mrt_integ = xr.open_dataarray('paper/mrt_jan_integ.nc')
# cerra_mrt_sunlit_apx = xr.open_dataarray('paper/mrt_jan_sunlit_apx.nc')
cerra_mrt_sunlit_apx = xr.open_dataarray('paper/mrt_jan_sunlit_apx_001.nc')
# cerra_mrt_integ_apx = xr.open_dataarray('paper/mrt_jan_integ_apx.nc')
cerra_mrt_integ_apx = xr.open_dataarray('paper/mrt_jan_integ_apx_1.nc')

# cerra_mrt_sunlit = xr.open_dataarray('paper/mrt_jul_sunlit.nc')
cerra_mrt_sunlit = xr.open_dataarray('paper/mrt_jul_sunlit_001.nc')
cerra_mrt_integ = xr.open_dataarray('paper/mrt_jul_integ.nc')
# cerra_mrt_sunlit_apx = xr.open_dataarray('paper/mrt_jul_sunlit_apx.nc')
cerra_mrt_sunlit_apx = xr.open_dataarray('paper/mrt_jul_sunlit_apx_001.nc')
# cerra_mrt_integ_apx = xr.open_dataarray('paper/mrt_jul_integ_apx.nc')
cerra_mrt_integ_apx = xr.open_dataarray('paper/mrt_jul_integ_apx_1.nc')

path = 'era5/'
file_pattern = 'ECMWF_mrt_*.nc'
era_mrt = xr.open_mfdataset(f'{path}/{file_pattern}', combine='by_coords')['mrt']
era_mrt = era_mrt.sel(time='2017-07') - 273.15
era_mrt = era_mrt.rename({'lon': 'longitude', 'lat': 'latitude'})
coords = {"latitude": cerra_mrt_integ.latitude, "longitude": cerra_mrt_integ.longitude}
era_mrt_interp = era_mrt.interp(coords=coords, method="nearest")

# plt.figure(figsize=(4, 3))
# plt.hexbin(csza_integ.values.flatten(), era_mrt_interp.values.flatten(), gridsize=100, cmap='viridis', bins='log', mincnt=1)
# plt.xlabel('COSSZA integrated')  # - CERRA sunlit
# plt.ylabel('ERA5 MRT (°C)')
# plt.colorbar(label='Density (log scale)')
# plt.tight_layout()
# plt.savefig('paper/hexbin/hexbin_era5_csza_mrt_integ_jul.png', dpi=1000)
# plt.close()
#
# ds = xr.Dataset()
# ds['integ'] = cerra_mrt_integ
# thrs = []
# for thr in thrs:
#     plt.figure(figsize=(4, 3))
#     plt.hexbin(ds.integ.values.flatten(), ds[thr].values.flatten(), gridsize=100, cmap='viridis', bins='log', mincnt=1)
#     plt.xlabel('MRT (°C) - COSSZA integrated')
#     plt.ylabel('MRT (°C) - COSSZA sunlit')
#     plt.colorbar(label='Density (log scale)')
#     plt.tight_layout()
#     plt.savefig('paper/hexbin/hexbin_mrt_jul_05x01.png', dpi=1000)
#     plt.close()

correlation = xr.corr(cerra_mrt_integ, cerra_mrt_sunlit, dim='time')
correlation.plot()

cerra_mrt_integ_sel = cerra_mrt_integ.resample(time='3H').mean(dim='time')
cerra_mrt_sunlit_sel = cerra_mrt_sunlit.resample(time='3H').mean(dim='time')
integ_daily_max = cerra_mrt_integ_sel.resample(time='1D').max(dim='time')
sunlit_daily_max = cerra_mrt_sunlit_sel.resample(time='1D').max(dim='time')

integ_daily_max = cerra_mrt_integ.resample(time='1D').max(dim='time')
sunlit_daily_max = cerra_mrt_sunlit.resample(time='1D').max(dim='time')

bias = integ_daily_max - sunlit_daily_max
bias_mean = bias.mean(dim='time')
bias_mean.plot()

bias = cerra_mrt_integ - cerra_mrt_sunlit
bias_sel = bias.sel(time=bias['time.hour'] == 11) #bias.time.dt.hour == 11
bias_mean = bias.mean(dim='time')


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True, sharex=True, sharey=True)
hb1 = ax1.hexbin(ds.sunlit_cossza.values.flatten(), ds.era.values.flatten(), gridsize=100, cmap='viridis', bins='log', mincnt=1)
hb2 = ax2.hexbin(ds.integ_cossza.values.flatten(), ds.integ.values.flatten(), gridsize=100, cmap='viridis', bins='log', mincnt=1)
hb3 = ax3.hexbin(ds.sunlit_cossza.values.flatten(), ds.sunlit_apx.values.flatten(), gridsize=100, cmap='viridis', bins='log', mincnt=1)
hb4 = ax4.hexbin(ds.sunlit_cossza.values.flatten(), ds.sunlit.values.flatten(), gridsize=100, cmap='viridis', bins='log', mincnt=1)

# Set labels
ax1.set_xlabel('sunlit average COSSZA')
ax1.set_ylabel('ERA5 MRT (°C)')
ax2.set_xlabel('quadrature COSSZA')
ax2.set_ylabel('CERRA MRT (°C)')
ax3.set_xlabel('sunlit average COSSZA')
ax3.set_ylabel('CERRA MRT (°C, calc. fdir)')
ax4.set_xlabel('sunlit average COSSZA')
ax4.set_ylabel('CERRA MRT (°C)')

for ax in (ax1, ax2, ax3, ax4):
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.yaxis.set_tick_params(labelleft=True)

plt.colorbar(hb3, ax=(ax1, ax2, ax3, ax4), orientation="horizontal", label='Density (log scale)', pad=0.01)

plt.savefig('paper/hexbin/!hexbin_subplots3.png', dpi=1000)
plt.close()





ds = xr.Dataset()
ds['era'] = era_mrt_interp
ds['integ'] = cerra_mrt_integ
ds['sunlit'] = cerra_mrt_sunlit
ds['integ_apx'] = cerra_mrt_integ_apx
ds['sunlit_apx'] = cerra_mrt_sunlit_apx
ds['integ_cossza'] = csza_integ
ds['sunlit_cossza'] = csza_sunlit

ds = ds.sel(time='2017-07-01')

ds = ds.resample(time='1D').min(dim='time')

ds['sunlit-era'] = ds['sunlit'] - ds['era']
ds['integ-integ_apx'] = ds['integ'] - ds['integ_apx']
ds['integ-era'] = ds['integ'] - ds['era']
ds['integ-sunlit'] = ds['integ'] - ds['sunlit']
ds['integ_apx-era'] = ds['integ_apx'] - ds['era']

ds['sunlit-sunlit_apx'] = ds['sunlit'] - ds['sunlit_apx']
ds['sunlit-integ'] = ds['sunlit'] - ds['integ']
ds['sunlit_apx-era'] = ds['sunlit_apx'] - ds['era']

strs = ['sunlit-sunlit_apx', 'sunlit-integ']
str = 'sunlit_apx-era'

strs = ['sunlit-sunlit_apx', 'sunlit_apx-era']
str = 'sunlit-integ'

ds = ds.mean(dim='time')

strs = ['era', 'integ', 'sunlit', 'integ_apx', 'sunlit_apx']

strs = ['sunlit-era', 'integ-integ_apx', 'integ-era', 'integ-sunlit', 'integ_apx-era']

for str in strs:
    min_avg = ds[str].min().values # .sel(time='2017-07-15-T12')
    mean_avg = ds[str].mean().values # .sel(time='2017-07-15-T12')
    max_avg = ds[str].max().values
    print(f'For {str} min mean max:{min_avg} {mean_avg} {max_avg}')

str = '2017-01-01-T08'
strs = ['2017-01-01-T00', '2017-01-01-T02', '2017-01-01-T03', '2017-01-01-T04', '2017-01-01-T05', '2017-01-01-T06', '2017-01-01-T07', '2017-01-01-T08', '2017-01-01-T09',
        '2017-01-01-T10', '2017-01-01-T11', '2017-01-01-T12', '2017-01-01-T14', '2017-01-01-T16', '2017-01-01-T18', '2017-01-01-T20', '2017-01-01-T22', '2017-01-01-T23']

strs = ['2017-07-01-T00', '2017-07-01-T02', '2017-07-01-T03', '2017-07-01-T04', '2017-07-01-T05', '2017-07-01-T06', '2017-07-01-T07', '2017-07-01-T08', '2017-07-01-T09',
        '2017-07-01-T10', '2017-07-01-T11', '2017-07-01-T12', '2017-07-01-T14', '2017-07-01-T16', '2017-07-01-T18', '2017-07-01-T20', '2017-07-01-T22', '2017-07-01-T23']

cerra_proj = ccrs.LambertConformal(
    central_longitude=8,
    central_latitude=50,
    standard_parallels=(50, 50),
    globe=ccrs.Globe(ellipse='sphere', semimajor_axis=6371229)
)
import cmocean
for str in strs:
    fig = plt.figure(figsize=(3, 3.1)) #figsize=(3, 3.1) figsize=(3, 2.6) figsize=(4.3, 3) figsize=(6, 7)
    ax = plt.axes(projection=cerra_proj)
    # ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', linewidth=0.5) #10m, 50m, 110m
    # ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.BORDERS, linestyle='--', edgecolor='black', linewidth=0.5)
    # plt.contour(lsm.longitude, lsm.latitude, lsm, levels=[0.3, 1], colors='black', transform=ccrs.PlateCarree())
    # norm = colors.BoundaryNorm(boundaries=[-40, -27, -13, 0, 9, 26, 32, 38, 46], ncolors=256, extend='both') #UTCIcats
    # norm = colors.BoundaryNorm(boundaries=[0, 0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8], ncolors=256, extend='both')#COSSZA
    # norm = colors.BoundaryNorm(boundaries=[-50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10], ncolors=256, extend='both')
    # norm = colors.BoundaryNorm(boundaries=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75], ncolors=256, extend='both')
    # norm = colors.BoundaryNorm(boundaries=[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], ncolors=256, extend='both')
    # norm = colors.BoundaryNorm(boundaries=[-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1], ncolors=256, extend='both')
    norm = colors.BoundaryNorm(boundaries=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], ncolors=256, extend='both')
    filled_c = plt.pcolormesh(ds.longitude.values, ds.latitude.values, ds[str].values, transform=ccrs.PlateCarree(), #ds[str].resample(time='1D').max(dim='time').mean(dim='time')
                              cmap=cmocean.cm.balance, norm=norm) #'RdYlBu_r' cmocean.cm.balance thermal
    # JANMIN, vmin=-54, vmax=13 JANMAX, vmin=-43, vmax=65
    # JULtest, vmin=3, vmax=88, JULMIN, vmin=-25, vmax=27 JULMAX, vmin=24, vmax=78
    #norm=norm, vmin=5, vmax=75, vmin=-25, vmax=55
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    # ax.set_extent([-19, 35, 25, 74], crs=ccrs.PlateCarree()) #[-19, 35.1, 24.2, 74]
    ax.margins(0, x=None, y=None, tight=True)
    # cbar = fig.colorbar(filled_c, orientation="vertical", pad=0.12)
    cbar = fig.colorbar(filled_c, orientation="horizontal", pad=0.01)
    # cbar.set_label('Mean Radiant Temperature (°C)')
    # cbar.set_label('Universal Thermal Climate Index (°C)')
    plt.tight_layout()
    plt.show()
    # fig.savefig('paper/mrt_maps/!2017-07-1Dmax-bar.png', dpi=1000)
    fig.savefig('paper/mrt_maps/diff/2017-01-1Dmin-diff-' + str + '.png', dpi=1000)
    # fig.savefig('paper/csza_maping_bias/bound/' + time_str + '_csza_sunlit', dpi=1000)
    plt.close()




fdir_ratio = _fdir_ratio(dates, csza_i, csza_s, rsds=rad.ssrd)
rsds_direct = fdir_ratio * rad.ssrd
rad['fdir_apx'] = rsds_direct
rad['diffuse'] = rad.ssrd - rad.tidirswrf
rad['diffuse_apx'] = rad.ssrd - rad.fdir_apx

rad['mrt'] = cerra_mrt
rad['mrt_apx'] = cerra_mrt_apx

sel = rad.sel(time='2017-07')

sel = rad.sel(time=rad.time.dt.hour == 12)

era_rad_interp

plt.figure()
plt.hexbin(era_rad_interp.strd.values.flatten(), era_rad_interp.mrt.values.flatten(), gridsize=100, cmap='viridis', bins='log', mincnt=1)
plt.xlabel('Surface short-wave radiation downwards (W/m\u00b2)') # Original Sdn,directsurf (W/m2)
plt.ylabel('Mean Radiant Temperature (°C)') # Derived Sdn,directsurf (W/m2)
plt.colorbar(label='Density (log scale)')
plt.tight_layout()
plt.savefig('poster_output/fin/strd_vs_mrt_era5.png', dpi=1000)

variables_to_plot = ['ssrd', 'tidirswrf', 'diffuse',
                     'ssru', 'strd', 'stru']
variables_lables = ['Short-wave downwards (W/m\u00b2)', 'Short-wave direct (total sky) downwards (W/m\u00b2)', 'Short-wave diffuse downwards (W/m\u00b2)',
                    'Short-wave upwards (W/m\u00b2)', 'Long-wave downwards (W/m\u00b2)', 'Long-wave upwards (W/m\u00b2)']
for variable, label in zip(variables_to_plot, variables_lables):
    plt.figure()
    plt.hexbin(rad[variable].values.flatten(), rad['mrt'].values.flatten(), gridsize=100, cmap='viridis', bins='log', mincnt=1)
    plt.xlabel(label)
    plt.ylabel('Mean Radiant Temperature (°C)')
    plt.colorbar(label='Density (log scale)')
    plt.tight_layout()
    plt.savefig(f'poster_output/fin/sensitivity/cerra_mrt_{variable}.png', dpi=1000)
    plt.close()
    print('Saved: {}'.format(variable))
print("Plots saved successfully.")

plt.close()