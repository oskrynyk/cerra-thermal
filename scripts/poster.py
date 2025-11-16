"""
Poster Figure Generation

This script creates figures for conference posters and presentations.
It processes multi-lead-time forecast data and generates visualizations
of thermal comfort indices across different seasonal periods.

Configuration:
    Modify src/config.py to change data paths and output locations.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import xclim
import thermofeel
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import dask.array as da
import seaborn as sns
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime
import dask

from src.config import DATA_DIR, OUTPUT_DIR, PLOTS_DIR

print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Load radiation data from lead times
# Universal naming: radiation_lead1.nc, radiation_lead2.nc, radiation_lead3.nc
# For seasonal comparison, load appropriate seasonal data
rad1 = xr.open_dataset(str(DATA_DIR / "radiation_lead1.nc"))
rad2 = xr.open_dataset(str(DATA_DIR / "radiation_lead2.nc"))
rad3 = xr.open_dataset(str(DATA_DIR / "radiation_lead3.nc"))

rad3 = rad3 - rad2
rad2 = rad2 - rad1

rad1 = rad1.drop_vars(['step', 'valid_time'], errors='ignore')
rad2.coords["time"] = rad2.coords["time"] + np.timedelta64(1, 'h')
rad3.coords["time"] = rad3.coords["time"] + np.timedelta64(2, 'h')
rad = xr.concat(objs=[rad1, rad2, rad3], dim='time', data_vars='all', coords='different', compat='broadcast_equals')
rad = rad.assign_coords(longitude=(((rad.longitude + 180) % 360) - 180))
del rad1, rad2, rad3
rad = rad.sortby('time')
rad = rad / 3600
rad['ssru'] = rad.ssrd - rad.ssr  # Upward shortwave radiation flux
rad['stru'] = rad.strd - rad.str  # Upward longwave radiation flux

rad.assign_coords(longitude=(((rad.longitude + 180) % 360) - 180), inplace=True)
variables = ["ssr", "str", "ssrd", "strd", "tidirswrf", 'ssru', 'stru']
for variable in variables:
    rad[variable].attrs.update({"units": "W m-2"})
rad.longitude.attrs["units"] = "degrees"

# Save processed radiation data
output_radiation = PLOTS_DIR / "radiation_processed.nc"
rad.to_netcdf(str(output_radiation))
print(f"Processed radiation data saved to: {output_radiation}")

rad = xr.open_dataset('poster_input/rad_jan.nc')
rad = rad.sel(time='2017-01-01')
rad['diffuse'] = rad.ssrd - rad.tidirswrf
rad.diffuse.attrs["units"] = "W m-2"

rad = xr.open_dataset('poster_input/rad_jul.nc')

rad = rad.sel(time='2017-07-01')
# rad['time'] = rad['time'] + np.timedelta64(30, 'm')
rad = rad.drop_vars(['step', 'surface', 'valid_time'])

rad.longitude.attrs["units"] = "degrees"
dates = rad.tidirswrf.time
lat = _gather_lat(rad.tidirswrf)
lon = _gather_lon(rad.tidirswrf)
dec = solar_declination(dates)
csza_i = cosine_of_solar_zenith_angle(dates, dec, lat, lon=lon, stat="average", sunlit=False)
csza_s = cosine_of_solar_zenith_angle(dates, dec, lat, lon=lon, stat="average", sunlit=True)
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

sel = era_rad_interp.sel(time='2017-07')


cerra_mrt = mean_radiant_temperature_from_DIR_DIF(rsds=rad.ssrd,
                                                  rsus=rad.ssru,
                                                  rlds=rad.strd,
                                                  rlus=rad.stru,
                                                  rsds_direct=rad.tidirswrf,
                                                  rsds_diffuse=rad.diffuse,
                                                  stat='sunlit')



cerra_mrt = xclim.indicators.atmos.mean_radiant_temperature(rsds=rad.ssrd, rsus=rad.ssru, rlds=rad.strd, rlus=rad.stru, stat='sunlit', ds=rad)

cerra_mrt.to_netcdf("poster_input/mrt_jan_from_fdir.nc")
cerra_mrt.to_netcdf("poster_input/mrt_jan.nc")
cerra_mrt.to_netcdf("poster_input/mrt_jul_from_fdir.nc")
cerra_mrt.to_netcdf("poster_input/mrt_jul.nc")

i_star.to_netcdf("poster_input/istar_jul_from_fdir.nc")

cerra_mrt_apx = xr.open_dataarray("poster_input/mrt_jan.nc")
cerra_mrt = xr.open_dataarray("poster_input/mrt_jan_from_fdir.nc")
cerra_mrt = cerra_mrt.sel(time='2017-01-01') - 273.15

cerra_mrt_apx = xr.open_dataarray("poster_input/mrt_jul.nc")#-273.15
cerra_mrt_apx = cerra_mrt_apx.drop_vars(['step', 'surface', 'valid_time'])

cerra_mrt = xr.open_dataarray("poster_input/mrt_jul_from_fdir.nc")-273.15
cerra_mrt = cerra_mrt.drop_vars(['step', 'surface', 'valid_time'])

cerra_mrt_apx = cerra_mrt_apx - 273.15
cerra_mrt_fdir = cerra_mrt_fdir - 273.15
era_mrt_interp = era_mrt_interp - 273.15

sel = cerra_mrt_fdir.sel(time='2017-07') - 273.15
daily_min = sel.resample(time='1D').min(dim='time')
daily_mean = sel.resample(time='1D').mean(dim='time')
daily_max = sel.resample(time='1D').max(dim='time')

# calculate averages for each grid point
min_avg = daily_min.mean(dim='time')
mean_avg = daily_mean.mean(dim='time')
max_avg = daily_max.mean(dim='time')

# Define the Lambert Conformal projection with parameters according to CERRA dataset
cerra_proj = ccrs.LambertConformal(
    central_longitude=8,
    central_latitude=50,
    standard_parallels=(50, 50),
    globe=ccrs.Globe(ellipse='sphere', semimajor_axis=6371229)
)
cerra_utci.to_netcdf("poster_output/cerra_utci_2017-07-15T12.nc")
cerra_utci_apx.to_netcdf("poster_output/cerra_utci_apx_2017-07-15T12.nc")

sel = cerra_utci.sel(time='2017-07-15T12')
sel = cerra_utci_apx.sel(time='2017-07-01T12')
sel = era_utci_interp.sel(time='2017-07-01T12')



sel = cerra_mrt.sel(time='2017-07-15T12')
sel = cerra_mrt_apx.sel(time='2017-07-15T12')
sel = era_mrt.sel(time='2017-07-15T12') #6, 76

fig = plt.figure()
ax = plt.axes(projection=cerra_proj)
# ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution='50m')
# ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.BORDERS, linestyle='--', edgecolor='black')
# plt.contour(lsm.longitude, lsm.latitude, lsm, levels=[0.3, 1], colors='black', transform=ccrs.PlateCarree())
# norm = colors.BoundaryNorm(boundaries=[-40, -27, -13, 0, 9, 26, 32, 38, 46], ncolors=256, extend='both')
norm = colors.BoundaryNorm(boundaries=[20, 30, 40, 50, 60, 70, 80], ncolors=256, extend='both')
filled_c = plt.pcolormesh(sel.longitude, sel.latitude, sel, transform=ccrs.PlateCarree(), cmap="viridis", vmin=-25, vmax=55) #, vmin=3, vmax=88, vmin=5, vmax=75, norm=norm
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.set_extent([-19, 35, 25, 74], crs=ccrs.PlateCarree())
ax.margins(0, x=None, y=None, tight=True)
cbar = fig.colorbar(filled_c, orientation="vertical", pad=0.08)
cbar.set_label('Mean Radiant Temperature (°C)')
cbar.set_label('Universal Thermal Climate Index (°C)')
plt.tight_layout()
fig.savefig('poster_output/fin/15_utci_cerra_apx', dpi=1000)
plt.close()
plt.show()




cerra_mrt_apx.sel(time='2017-07-15T12').plot()
plt.figure()
cerra_mrt_fdir.sel(time='2017-07-15T12').plot()
plt.show()
plt.figure()
era_mrt_interp.sel(time='2017-07-21T12').plot()
plt.show()
# wind0 = xr.open_dataset("poster_input/wind_jul0.nc")['si10']
# wind1 = xr.open_dataset("poster_input/wind_jul1.nc")['si10']
# wind2 = xr.open_dataset("poster_input/wind_jul2.nc")['si10']
# temp0 = xr.open_dataset("poster_input/temp_jul0.nc")
# temp1 = xr.open_dataset("poster_input/temp_jul1.nc")
# temp2 = xr.open_dataset("poster_input/temp_jul2.nc")

wind0 = xr.open_dataset("poster_input/wind_jul0.nc")['si10']
wind1 = xr.open_dataset("poster_input/wind_jul1.nc")['si10']
wind2 = xr.open_dataset("poster_input/wind_jul2.nc")['si10']
temp0 = xr.open_dataset("poster_input/temp_jul0.nc")
temp1 = xr.open_dataset("poster_input/temp_jul1.nc")
temp2 = xr.open_dataset("poster_input/temp_jul2.nc")

wind = xr.concat(objs=[wind0, wind1, wind2], dim='time', data_vars='all', coords='different', compat='broadcast_equals')
wind = wind.assign_coords(longitude=(((wind.longitude + 180) % 360) - 180), time=wind.valid_time)
wind = wind.drop_vars(['step', 'heightAboveGround', 'valid_time'])
wind = wind.sortby('time')
wind = wind.clip(min=0.5, max=17)
# del wind0, wind1, wind2

temp = xr.concat(objs=[temp0, temp1, temp2], dim='time', data_vars='all', coords='different', compat='broadcast_equals')
temp = temp.assign_coords(longitude=(((temp.longitude + 180) % 360) - 180), time=temp.valid_time, step=np.timedelta64(0, 'ns'))
temp = temp.drop_vars(['step', 'heightAboveGround', 'valid_time'])
temp = temp.sortby('time')
# del temp0, temp1, temp2
temp['t2m'] = temp.t2m - 273.15

temp['si10'] = wind
temp['mrt'] = cerra_mrt
temp['utci'] = cerra_utci

temp['mrt'] = cerra_mrt_apx
temp['utci_apx'] = cerra_utci_apx

# sel = temp.sel(time='2017-07-01')
sel = temp.sel(time=slice('2017-07-01', '2017-07-05'))

sel = era_comp_interp.sel(time=slice('2017-07-01', '2017-07-05'))

# utci_flat = temp['utci'].values.flatten()
variables_to_plot = ['r2', 't2m', 'si10', 'mrt']
variables_lables = ['relative humidity (%)', '2m temperature (°C)', '10m wind speed (m/s)', 'Mean Radiant Temperature (°C)']

for variable, label in zip(variables_to_plot, variables_lables):
    plt.figure()
    plt.hexbin(sel[variable].values.flatten(), sel['utci'].values.flatten(), gridsize=100, cmap='viridis', bins='log', mincnt=1)
    plt.xlabel(label)
    plt.ylabel('Universal Thermal Climate Index (°C)')
    plt.colorbar(label='Density (log scale)')
    plt.tight_layout()
    plt.savefig(f'poster_output/fin/era_utci_vs_{variable}.png', dpi=1000)
    plt.close()
    print('Saved: {}'.format(variable))
print("Plots saved successfully.")


################################   MRT upload   #####################################
cerra_mrt = xr.open_dataarray("poster_input/mrt_jul_from_fdir.nc")-273.15
cerra_mrt = cerra_mrt.drop_vars(['step', 'surface', 'valid_time'])
cerra_mrt_apx = xr.open_dataarray("poster_input/mrt_jul.nc")-273.15
cerra_mrt_apx = cerra_mrt_apx.drop_vars(['step', 'surface', 'valid_time'])

e_mrt = cerra_mrt - temp.t2m
# count_less_than = (temp.t2m.values < -50).sum()
# count_more_than = (temp.t2m.values > 50).sum()

# count_less_than = (e_mrt.values < -30).sum()
# count_more_than = (e_mrt.values > 70).sum() #apx - 45171, dir - 486233

# count_less_than = (wind.values < 0.5).sum()
# count_more_than = (wind.values > 17).sum()

temp['t2m'] = temp.t2m-273.15
temp.t2m.attrs["units"] = "degC"
temp.r2.attrs["units"] = "%"
wind.attrs["units"] = "m/s"

cerra_mrt.attrs["units"] = "degC"
cerra_mrt_apx.attrs["units"] = "degC"

era_mrt_interp.attrs["units"] = "K"

cerra_utci = universal_thermal_climate_index(tas=temp.t2m.sel(time='2017-07-15T12'),
                                             hurs=temp.r2.sel(time='2017-07-15T12'),
                                             sfcWind=wind.sel(time='2017-07-15T12'),
                                             mrt=cerra_mrt.sel(time='2017-07-15T12'),
                                             mask_invalid=False)

cerra_utci_apx = universal_thermal_climate_index(tas=temp.t2m.sel(time='2017-07-15T12'),
                                                 hurs=temp.r2.sel(time='2017-07-15T12'),
                                                 sfcWind=wind.sel(time='2017-07-15T12'),
                                                 mrt=cerra_mrt_apx.sel(time='2017-07-15T12'),
                                                 mask_invalid=False)

cerra_utci = universal_thermal_climate_index(tas=temp.t2m,
                                             hurs=temp.r2,
                                             sfcWind=wind,
                                             mrt=cerra_mrt,
                                             mask_invalid=True)

np.isnan(cerra_utci).sum()
cerra_utci = xclim.indicators.atmos.universal_thermal_climate_index() #78490589 = 0.092
cerra_utci.to_netcdf("poster_input/utci_jul_from_fdir.nc") #16968431/850214184=0.0199
cerra_utci.to_netcdf("poster_input/utci_jul.nc") #          16531112/850214184=0.0194


cerra_utci = xr.open_dataarray("poster_input/utci_jul_from_fdir.nc")
cerra_utci_apx = xr.open_dataarray("poster_input/utci_jul.nc")



# MRT interp
path = 'era5/'
file_pattern = 'ECMWF_mrt_*.nc'
era_mrt = xr.open_mfdataset(f'{path}/{file_pattern}', combine='by_coords')['mrt'] - 273.15
era_mrt = era_mrt.rename({'lon': 'longitude', 'lat': 'latitude'})
era_mrt = era_mrt.sel(time='2017-07')

f1 = xr.open_dataset("meteo_test.nc")
f1 = f1.assign_coords(longitude=(((f1.longitude + 180) % 360) - 180))
coords = {"latitude": f1.latitude, "longitude": f1.longitude}
era_mrt_interp = era_mrt.interp(coords=coords, method="nearest")

max_value = era_mrt.max().compute()
count_above_80 = era_mrt.where(era_mrt > 80).count().compute()
unique_years = np.unique(era_mrt.time.dt.month)


# UTCI interp
path = 'era5/'
file_pattern = 'ECMWF_utci_*.nc'
era_utci = xr.open_mfdataset(f'{path}/{file_pattern}', combine='by_coords')['utci']-273.15
era_utci = era_utci.rename({'lon': 'longitude', 'lat': 'latitude'})
era_utci = era_utci.sel(time='2017-07-15T12')
coords = {"latitude": cerra_utci.latitude, "longitude": cerra_utci.longitude}
era_utci_interp = era_utci.interp(coords=coords, method="nearest")


# ERA5 components
era_comp = xr.open_dataset('poster_input/comp_era5.nc')
era_comp = era_comp.sel(time='2017-07')
era_comp['si10'] = np.sqrt(era_comp.u10 ** 2 + era_comp.v10 ** 2)
from xclim.indices import relative_humidity
era_comp['r2'] = relative_humidity(tas=era_comp.t2m, tdps=era_comp.d2m)
era_comp['t2m'] = era_comp['t2m']-273.15
era_comp['mrt'] = era_mrt
era_comp['utci'] = era_utci

f1 = xr.open_dataset("meteo_test.nc")
f1 = f1.assign_coords(longitude=(((f1.longitude + 180) % 360) - 180))
coords = {"latitude": f1.latitude, "longitude": f1.longitude}
era_comp_interp = era_comp.interp(coords=coords, method="nearest")





# cerra_mrt['time'] = cerra_mrt['time'] + np.timedelta64(1, 'h')
daily_min = cerra_utci.resample(time='1D').min(dim='time')
daily_mean = cerra_utci.resample(time='1D').mean(dim='time')
daily_max = cerra_utci.resample(time='1D').max(dim='time')

# calculate averages for each grid point
min_avg = daily_min.mean(dim='time')
mean_avg = daily_mean.mean(dim='time')
max_avg = daily_max.mean(dim='time')

daily_max = cerra_utci.resample(time='1D').max(dim='time')
max_avg = daily_max.mean(dim='time')

daily_max = era_utci_interp.resample(time='1D').max(dim='time')
max_avg_era = daily_max.mean(dim='time')

diff = max_avg - max_avg_era
diff = diff.clip(min=-10, max=10)
diff.plot()

plt.suptitle("Spatial Pattern at")
plt.show()


# lon_mg, lat_mg = xr.broadcast(rad.longitude, rad.latitude)

# Create new datasets to store the results
# cerra_mrt = xr.DataArray(np.zeros_like(rad.ssr), dims=rad.ssr.dims, coords=rad.ssr.coords)
cerra_utci = xr.DataArray(np.zeros_like(f3.t2m), dims=f3.t2m.dims, coords=f3.t2m.coords)

for timestamp in np.unique(rad.time.values):
    # cossza_integrated = thermofeel.calculate_cos_solar_zenith_angle(lon=rad.longitude.values, lat=rad.latitude.values,
    #                                                             y=timestamp.astype('datetime64[Y]').astype(int) + 1970,
    #                                                             m=timestamp.astype('datetime64[M]').astype(int) % 12 + 1,
    #                                                             d=timestamp.astype('datetime64[D]').astype(int) % 31 + 1,
    #                                                             h=timestamp.astype('datetime64[h]').astype(int) % 24)
    cossza_integrated = thermofeel.calculate_cos_solar_zenith_angle_integrated(lon=rad.longitude.values, lat=rad.latitude.values,
                                                                               y=timestamp.astype('datetime64[Y]').astype(int) + 1970,
                                                                               m=timestamp.astype('datetime64[M]').astype(int) % 12 + 1,
                                                                               d=timestamp.astype('datetime64[D]').astype(int) % 31 + 1,
                                                                               h=timestamp.astype('datetime64[h]').astype(int) % 24,
                                                                               tbegin=0,
                                                                               tend=1)
    fdir = rad.tidirswrf.sel(time=timestamp)
    dsrp_approx = thermofeel.approximate_dsrp(fdir=fdir.values, cossza=cossza_integrated)

    ssr = rad.ssr.sel(time=timestamp)  # Surface net solar radiation [W m-2]
    ssrd = rad.ssrd.sel(time=timestamp) # Surface solar radiation downwards [W m-2]
    strr = rad.str.sel(time=timestamp) # Surface net thermal radiation [W m-2]
    strd = rad.strd.sel(time=timestamp) # Surface thermal radiation downwards [W m-2]
    mrt = thermofeel.calculate_mean_radiant_temperature(cossza=cossza_integrated,
                                                        fdir=fdir.values,
                                                        dsrp=dsrp_approx,
                                                        ssr=ssr.values,
                                                        ssrd=ssrd.values,
                                                        strr=strr.values,
                                                        strd=strd.values)
    mrt_c = mrt - 273.15
    cerra_mrt.loc[{'time': timestamp}] = mrt_c
    print('Timestamp: {}'.format(timestamp))
    mrt_c = mrt_c.clip(min=-30, max=70)
    windspeed = f2.si10.sel(time=timestamp)
    windspeed = windspeed.clip(min=0.5, max=17)

    rh = f3.r2.sel(time=timestamp)
    t2k = f3.t2m.sel(time=timestamp)
    t2m = f3.t2m.sel(time=timestamp) - 273.15
    t2m = t2m.clip(min=-50, max=50)
    t2m_phase = (t2m <= 0).astype(int)

    # t2m = np.random.uniform(-20, 20, size=(4, 4))
    # t2m[0, 0] = 0.00
    # t2m[3, 0] = 0.00
    # t2k = t2m + 273.15
    # t2m_phase = (t2m <= 0).astype(int)
    # rh_kPa = cal_sat_vp_multi_corrected(tk=t2k, phase=t2m_phase)


    rh_kPa = thermofeel.calculate_saturation_vapour_pressure(tk=t2k.values) * rh.values / 1000.0 # Calculate rh in kPa
    # rh_kPa = cal_sat_vp_multi_corrected(tk=t2k.values, phase=t2m_phase) * rh.values / 1000.0 # Calculate rh in kPa considering 0 - water and 1 - ice

    utci = thermofeel.calculate_utci_polynomial(t2m=t2m.values,
                                                mrt=mrt_c,
                                                va=windspeed.values,
                                                rh=rh_kPa) # Calculate UTCI with a 6th order polynomial approximation
    cerra_utci.loc[{'time': timestamp}] = utci
    print('Timestamp: {}'.format(timestamp))



# utci_xr.to_netcdf('output/utci_2017_wi.nc')
# utci_xr = xr.open_dataset('output/utci_1985.nc')
# utci_xr = xr.open_dataset('output/utci_2017.nc')

cerra_thermal = xr.Dataset({
    'mrt': cerra_mrt,
    'utci': cerra_utci
})







mrt = rad['mrt'] # Extract the 'mrt' variable
variables_to_plot = ['rsds', 'rlds', 'rsus', 'rlus'] # Variables to plot against 'mrt'

mrt = rad['mrt'] # Extract the 'mrt' variable
variables_to_plot = ['ssr', 'str', 'ssrd', 'strd', 'tidirswrf'] # Variables to plot against 'mrt'

for i, variable in enumerate(variables_to_plot):
    # Print progress for current variable
    print(f"Processing variable {i + 1}/{len(variables_to_plot)}: {variable}")
    # Extract the current variable
    current_var = rad[variable]
    # Flatten the data for scatter plot
    mrt_flat = mrt.values.flatten()
    current_var_flat = current_var.values.flatten()
    # Create a scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(current_var_flat, mrt_flat, s=0.1, marker='.', alpha=0.5)
    plt.xlabel(f'{variable}')
    plt.ylabel('mrt')
    plt.title(f'{variable} vs mrt')
    # Save the plot to a file
    output_file = f'poster_output/{variable}_integ_rev001_vs_mrt_scatter.png'
    plt.savefig(output_file)
    # Close the plot to release memory
    plt.close()
print("Plots saved successfully.")


path = 'era5/'
file_pattern = 'ECMWF_mrt_*.nc'
era_mrt = xr.open_mfdataset(f'{path}/{file_pattern}', combine='by_coords')['mrt']
era_mrt = era_mrt.sel(time='2017-07') - 273.15
era_mrt = era_mrt.rename({'lon': 'longitude', 'lat': 'latitude'})
coords = {"latitude": cerra_mrt.latitude, "longitude": cerra_mrt.longitude}
era_mrt_interp = era_mrt.interp(coords=coords, method="nearest")

# file_pattern = 'ECMWF_utci_*.nc'
# era_utci = xr.open_mfdataset(f'{path}/{file_pattern}', combine='by_coords')['utci'] - 273.15
# extent = [-60, 75, 20, 75]
# mrt_xr = mrt_xr.sel(lon=slice(extent[0], extent[1]), lat=slice(extent[3], extent[2]))
# utci_xr = utci_xr.sel(lon=slice(extent[0], extent[1]), lat=slice(extent[3], extent[2]))

era_rad = xr.open_dataset("poster_input/rad_era5.nc")
era_rad = era_rad.sel(time='2017-07')
era_rad = era_rad / 3600
era_rad['ssru'] = era_rad.ssrd - era_rad.ssr # rsw or uswrf Upward short-wave radiation flux
era_rad['stru'] = era_rad.strd - era_rad.str # lur or ulwrf Upward long-wave radiation flux
era_rad['diffuse'] = era_rad.ssrd - era_rad.fdir
era_rad['mrt'] = era_mrt

coords = {"latitude": rad.latitude, "longitude": rad.longitude}
era_rad_interp = era_rad.interp(coords=coords, method="nearest")
era_rad_interp['mrt'] = era_mrt_interp




mrt_flat = rad['mrt'] # Extract the 'mrt' variable
variables_to_plot = ['ssrd', 'strd', 'ssru', 'stru', 'fdir_apx', 'diffuse_apx'] # Variables to plot against 'mrt': 'ssrd', 'strd', 'ssru', 'stru', 'tidirswrf'

# Create a scatter plot for each variable and save to files
for i, variable in enumerate(variables_to_plot):
    print(f"Processing variable {i + 1}/{len(variables_to_plot)}: {variable}")
    # mrt_flat = mrt.values.flatten()
    # current_var = rad[variable]
    # current_var_flat = current_var.values.flatten()
    # Create a hexbin plot
    plt.figure()
    plt.hexbin(rad[variable].values.flatten(), rad['mrt'].values.flatten(), gridsize=50, cmap='viridis', bins='log', mincnt=5)
    plt.xlabel(f'{variable} (W/m\u00b2)')
    plt.ylabel('mrt')
    plt.title(f'{variable} vs mrt')
    # Add a colorbar
    plt.colorbar(label='Density (log scale)')
    plt.tight_layout()
    # Save the plot to a file
    output_file = f'poster_output/jan/cerra_jan_hexbin_{variable}_vs_mrt.png'
    plt.savefig(output_file, dpi=600)
    # Close the plot to release memory
    plt.close()
print("Plots saved successfully.")



from scipy.stats import gaussian_kde
mrt = rad['mrt']  # Extract the 'mrt' variable
variables_to_plot = ['rsds', 'rlds', 'rsus', 'rlus']  # Variables to plot against 'mrt'
for i, variable in enumerate(variables_to_plot):
    # Print progress for current variable
    print(f"Processing variable {i + 1}/{len(variables_to_plot)}: {variable}")

    # Extract the current variable
    current_var = rad[variable]
    mrt_flat = mrt.values.flatten()
    current_var_flat = current_var.values.flatten()
    # Calculate the 2D kernel density estimate
    kde = gaussian_kde(np.vstack([current_var_flat, mrt_flat]))
    x, y = np.mgrid[current_var_flat.min():current_var_flat.max():100j,
                    mrt_flat.min():mrt_flat.max():100j]
    z = kde(np.vstack([x.flatten(), y.flatten()]))
    plt.figure(figsize=(10, 10))
    plt.imshow(z.reshape(x.shape), origin='lower', cmap='inferno', aspect='auto',
               extent=[current_var_flat.min(), current_var_flat.max(), mrt_flat.min(), mrt_flat.max()])
    plt.colorbar(label='Density')
    plt.xlabel(f'{variable}')
    plt.ylabel('mrt')
    plt.title(f'{variable} vs mrt')
    output_file = f'poster_output/cerra_jul_2Ddensity_{variable}_vs_mrt.png'
    plt.savefig(output_file)
    plt.close()
print("Plots saved successfully.")