"""Thermal comfort analysis pipeline: MRT, UTCI, NET calculation and visualization."""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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
    OUTPUT_DIR,
    PLOTS_DIR,
    DATASETS_DIR,
    RESULTS_DIR,
    RADIATION_FILE,
    WIND_FILE,
    TEMPERATURE_HUMIDITY_FILE,
    ERA5_DATA_DIR,
    T2M_CLIP,
    MRT_CLIP,
    WIND_CLIP,
)

print("\n" + "="*80)
print("THERMAL COMFORT ANALYSIS PIPELINE")
print("="*80)
print(f"\nData directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Open the radiation dataset
f = xr.open_dataset(str(DATA_DIR / RADIATION_FILE), engine="cfgrib")
f = f.assign_coords(longitude=(((f.longitude + 180) % 360) - 180))
lon_mg, lat_mg = xr.broadcast(f.longitude, f.latitude)
print(f"✓ Radiation data loaded: shape {f.dims}")

# Open the wind speed and humidity datasets
f2 = xr.open_dataset(str(DATA_DIR / WIND_FILE), engine="cfgrib")
f2 = f2.assign_coords(longitude=(((f2.longitude + 180) % 360) - 180))
print(f"✓ Wind data loaded: shape {f2.dims}")

f3 = xr.open_dataset(str(DATA_DIR / TEMPERATURE_HUMIDITY_FILE))
f3 = f3.assign_coords(longitude=(((f3.longitude + 180) % 360) - 180))
print(f"✓ Temperature/Humidity data loaded: shape {f3.dims}")

# Create new datasets to store the results
cerra_mrt = xr.DataArray(np.zeros_like(f3.t2m), dims=f3.t2m.dims, coords=f3.t2m.coords)
cerra_utci = xr.DataArray(np.zeros_like(f3.t2m), dims=f3.t2m.dims, coords=f3.t2m.coords)

print("\n" + "="*80)
print("CALCULATING THERMAL INDICES FOR ALL TIMESTAMPS")
print("="*80)

def cal_sat_vp_multi_corrected(tk, phase):
    """Corrected saturation vapor pressure calculation."""
    es = np.zeros_like(tk)
    y = (tk - 273.15) / (tk - 32.18)
    es[phase == 0] = 6.1121 * np.exp(17.502 * y[phase == 0])
    y = (tk - 273.15) / (tk - 0.6)
    es[phase == 1] = 6.1115 * np.exp(22.452 * y[phase == 1])
    es = 1.004 * es
    return es

timestamps = np.unique(f3.time.values)
total_timestamps = len(timestamps)

for idx, timestamp in enumerate(timestamps):
    cossza_integrated = thermofeel.calculate_cos_solar_zenith_angle_integrated(
        lon=lon_mg.values, lat=lat_mg.values,
        y=timestamp.astype('datetime64[Y]').astype(int) + 1970,
        m=timestamp.astype('datetime64[M]').astype(int) % 12 + 1,
        d=timestamp.astype('datetime64[D]').astype(int) % 31 + 1,
        h=timestamp.astype('datetime64[h]').astype(int) % 24,
        tbegin=timestamp.astype('datetime64[h]').astype(int) % 24,
        tend=timestamp.astype('datetime64[h]').astype(int) % 24 + 3
    )

    ssrd = f.ssrd.sel(time=timestamp) / 10800
    ssr = f.ssr.sel(time=timestamp) / 10800
    strd = f.strd.sel(time=timestamp) / 10800
    fdir = f.tidirswrf.sel(time=timestamp) / 10800
    strr = f.str.sel(time=timestamp) / 10800

    approx_dsrp = thermofeel.approximate_dsrp(fdir=fdir.values, cossza=cossza_integrated)
    mrt = thermofeel.calculate_mean_radiant_temperature(
        ssrd=ssrd.values, ssr=ssr.values, dsrp=approx_dsrp,
        strd=strd.values, fdir=fdir.values, strr=strr.values,
        cossza=cossza_integrated
    )
    cerra_mrt.loc[{'time': timestamp}] = mrt
    mrt_c = mrt - 273.15
    mrt_c = mrt_c.clip(min=MRT_CLIP['min'], max=MRT_CLIP['max'])

    windspeed = f2.si10.sel(time=timestamp)
    windspeed = windspeed.clip(min=WIND_CLIP['min'], max=WIND_CLIP['max'])
    
    rh = f3.r2.sel(time=timestamp)
    t2k = f3.t2m.sel(time=timestamp)
    t2m = t2k - 273.15
    t2m = t2m.clip(min=T2M_CLIP['min'], max=T2M_CLIP['max'])

    rh_kPa = thermofeel.calculate_saturation_vapour_pressure(tk=t2k.values) * rh.values / 1000.0
    utci = thermofeel.calculate_utci_polynomial(
        t2m=t2m.values,
        mrt=mrt_c,
        va=windspeed.values,
        rh=rh_kPa
    )
    cerra_utci.loc[{'time': timestamp}] = utci

    if (idx + 1) % max(1, total_timestamps // 10) == 0:
        print(f"  ✓ Progress: {idx + 1:4d}/{total_timestamps} timestamps processed")

print(f"✓ All {total_timestamps} timestamps processed successfully")

cerra_thermal = xr.Dataset({
    'mrt': cerra_mrt,
    'utci': cerra_utci
})

print("\n" + "="*80)
print("COMPUTING STATISTICS")
print("="*80)

# Temporal aggregation
print("Computing daily and hourly statistics...")
daily_mrt = cerra_mrt.resample(time='1D').mean()
daily_utci = cerra_utci.resample(time='1D').mean()
daily_mrt_max = cerra_mrt.resample(time='1D').max()
daily_mrt_min = cerra_mrt.resample(time='1D').min()
daily_utci_max = cerra_utci.resample(time='1D').max()
daily_utci_min = cerra_utci.resample(time='1D').min()
print("✓ Daily statistics computed")

# Period statistics
mrt_c_all = cerra_mrt - 273.15
period_mrt_mean = mrt_c_all.mean()
period_mrt_std = mrt_c_all.std()
period_mrt_min = mrt_c_all.min()
period_mrt_max = mrt_c_all.max()
period_utci_mean = cerra_utci.mean()
period_utci_std = cerra_utci.std()
period_utci_min = cerra_utci.min()
period_utci_max = cerra_utci.max()

print(f"""
Global Period Statistics:
  MRT:  mean={period_mrt_mean.values:6.2f}°C, std={period_mrt_std.values:5.2f}°C
        min={period_mrt_min.values:6.2f}°C, max={period_mrt_max.values:6.2f}°C
  UTCI: mean={period_utci_mean.values:6.2f}°C, std={period_utci_std.values:5.2f}°C
        min={period_utci_min.values:6.2f}°C, max={period_utci_max.values:6.2f}°C
""")

# Thermal stress analysis
print("\n" + "="*80)
print("THERMAL STRESS ANALYSIS")
print("="*80)

heat_extreme = (cerra_utci > 46).sum().values
heat_strong = ((cerra_utci > 32) & (cerra_utci <= 46)).sum().values
heat_moderate = ((cerra_utci > 26) & (cerra_utci <= 32)).sum().values
comfort_warm = ((cerra_utci > 9) & (cerra_utci <= 26)).sum().values
comfort_neutral = ((cerra_utci > 0) & (cerra_utci <= 9)).sum().values
cold_weak = ((cerra_utci > -13) & (cerra_utci <= 0)).sum().values
cold_moderate = ((cerra_utci > -27) & (cerra_utci <= -13)).sum().values
cold_strong = (cerra_utci <= -27).sum().values

total_conditions = (heat_extreme + heat_strong + heat_moderate + 
                   comfort_warm + comfort_neutral + 
                   cold_weak + cold_moderate + cold_strong).values

print(f"""
Thermal Stress Distribution (% of conditions):
  Extreme heat stress (UTCI > 46°C):         {100*heat_extreme/total_conditions:6.2f}%
  Strong heat stress (32 < UTCI ≤ 46°C):    {100*heat_strong/total_conditions:6.2f}%
  Moderate heat stress (26 < UTCI ≤ 32°C):  {100*heat_moderate/total_conditions:6.2f}%
  Warm comfort (9 < UTCI ≤ 26°C):           {100*comfort_warm/total_conditions:6.2f}%
  Near-neutral (0 < UTCI ≤ 9°C):            {100*comfort_neutral/total_conditions:6.2f}%
  Weak cold stress (-13 < UTCI ≤ 0°C):      {100*cold_weak/total_conditions:6.2f}%
  Moderate cold stress (-27 < UTCI ≤ -13°C):{100*cold_moderate/total_conditions:6.2f}%
  Strong cold stress (UTCI ≤ -27°C):        {100*cold_strong/total_conditions:6.2f}%
""")





# Spatial analysis
print("\n" + "="*80)
print("SPATIAL ANALYSIS")
print("="*80)

mrt_c_mean = mrt_c_all.mean(dim='time')
utci_mean = cerra_utci.mean(dim='time')

mrt_grad_x = np.gradient(mrt_c_mean.values, axis=0)
mrt_grad_y = np.gradient(mrt_c_mean.values, axis=1)
mrt_grad_mag = np.sqrt(mrt_grad_x**2 + mrt_grad_y**2)

utci_grad_x = np.gradient(utci_mean.values, axis=0)
utci_grad_y = np.gradient(utci_mean.values, axis=1)
utci_grad_mag = np.sqrt(utci_grad_x**2 + utci_grad_y**2)

print(f"""
Spatial Variability:
  MRT spatial std dev:  {mrt_c_mean.std().values:6.2f}°C
  UTCI spatial std dev: {utci_mean.std().values:6.2f}°C
""")

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

output_file = DATASETS_DIR / "thermal_indices_full.nc"
cerra_thermal.to_netcdf(str(output_file))
print(f"✓ Full dataset saved: {output_file.name}")

daily_dataset = xr.Dataset({
    'mrt_mean': daily_mrt - 273.15,
    'mrt_max': daily_mrt_max - 273.15,
    'mrt_min': daily_mrt_min - 273.15,
    'utci_mean': daily_utci,
    'utci_max': daily_utci_max,
    'utci_min': daily_utci_min,
})
daily_file = DATASETS_DIR / "thermal_indices_daily.nc"
daily_dataset.to_netcdf(str(daily_file))
print(f"✓ Daily statistics saved: {daily_file.name}")

stats_file = DATASETS_DIR / "thermal_statistics.txt"
with open(str(stats_file), 'w') as f:
    f.write("THERMAL COMFORT ANALYSIS STATISTICS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Period Mean Statistics:\n")
    f.write(f"  MRT:  {period_mrt_mean.values:6.2f}°C ± {period_mrt_std.values:5.2f}°C\n")
    f.write(f"  UTCI: {period_utci_mean.values:6.2f}°C ± {period_utci_std.values:5.2f}°C\n\n")
    f.write(f"Thermal Stress Distribution:\n")
    f.write(f"  Extreme heat stress: {100*heat_extreme/total_conditions:6.2f}%\n")
    f.write(f"  Strong heat stress:  {100*heat_strong/total_conditions:6.2f}%\n")
    f.write(f"  Comfort conditions:  {100*(comfort_warm+comfort_neutral)/total_conditions:6.2f}%\n")
    f.write(f"  Cold stress:         {100*(cold_weak+cold_moderate+cold_strong)/total_conditions:6.2f}%\n")
print(f"✓ Statistics saved: {stats_file.name}")

# Visualizations
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

cerra_proj = ccrs.LambertConformal(
    central_longitude=8,
    central_latitude=50,
    standard_parallels=(50, 50),
    globe=ccrs.Globe(ellipse='sphere', semimajor_axis=6371229)
)

# Mean field maps
fig, axes = plt.subplots(2, 1, figsize=(14, 12), 
                         subplot_kw={'projection': cerra_proj}, 
                         constrained_layout=True)

mrt_plot = axes[0]
mrt_plot.coastlines(resolution='50m')
mrt_plot.add_feature(cfeature.BORDERS, linestyle='--', alpha=0.5)
filled_mrt = mrt_plot.pcolormesh(mrt_c_mean.longitude, mrt_c_mean.latitude, mrt_c_mean,
                                transform=ccrs.PlateCarree(), cmap='RdYlBu_r')
mrt_plot.set_title('Mean Radiant Temperature - Period Mean', fontsize=13, fontweight='bold')
gl1 = mrt_plot.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha=0.3)
gl1.top_labels = False
cbar1 = plt.colorbar(filled_mrt, ax=mrt_plot, orientation='horizontal', pad=0.08)
cbar1.set_label('Temperature (°C)', fontsize=11)

utci_plot = axes[1]
utci_plot.coastlines(resolution='50m')
utci_plot.add_feature(cfeature.BORDERS, linestyle='--', alpha=0.5)
filled_utci = utci_plot.pcolormesh(utci_mean.longitude, utci_mean.latitude, utci_mean,
                                  transform=ccrs.PlateCarree(), cmap='RdYlBu_r')
utci_plot.set_title('Universal Thermal Climate Index - Period Mean', fontsize=13, fontweight='bold')
gl2 = utci_plot.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha=0.3)
gl2.top_labels = False
cbar2 = plt.colorbar(filled_utci, ax=utci_plot, orientation='horizontal', pad=0.08)
cbar2.set_label('UTCI (°C)', fontsize=11)

map_file = PLOTS_DIR / "thermal_indices_mean_maps.png"
fig.savefig(str(map_file), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Mean maps saved: {map_file.name}")

# Daily time series
fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
daily_mrt_c = daily_mrt - 273.15
ax.plot(daily_mrt_c.time, daily_mrt_c.mean(dim=['x', 'y']), 'r-', linewidth=2, label='Daily Mean MRT', alpha=0.8)
ax.fill_between(daily_mrt_c.time, 
                daily_mrt_c.min(dim=['x', 'y']),
                daily_mrt_c.max(dim=['x', 'y']),
                alpha=0.2, color='red', label='MRT range')
ax.plot(daily_utci.time, daily_utci.mean(dim=['x', 'y']), 'b-', linewidth=2, label='Daily Mean UTCI', alpha=0.8)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Temperature (°C)', fontsize=11)
ax.set_title('Temporal Evolution of Thermal Indices', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
ts_file = PLOTS_DIR / "thermal_indices_timeseries.png"
fig.savefig(str(ts_file), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Time series saved: {ts_file.name}")

# Additional analysis: Regional extremes
print("\n" + "="*80)
print("REGIONAL THERMAL EXTREMES")
print("="*80)

# Find hottest and coldest regions
mrt_max_idx = np.unravel_index(np.argmax(mrt_c_mean.values), mrt_c_mean.shape)
mrt_max_val = mrt_c_mean.values[mrt_max_idx]
mrt_max_lat = mrt_c_mean.latitude.values[mrt_max_idx[0]]
mrt_max_lon = mrt_c_mean.longitude.values[mrt_max_idx[1]]

mrt_min_idx = np.unravel_index(np.argmin(mrt_c_mean.values), mrt_c_mean.shape)
mrt_min_val = mrt_c_mean.values[mrt_min_idx]
mrt_min_lat = mrt_c_mean.latitude.values[mrt_min_idx[0]]
mrt_min_lon = mrt_c_mean.longitude.values[mrt_min_idx[1]]

print(f"\nHottest location: ({mrt_max_lat:.2f}°N, {mrt_max_lon:.2f}°E) - MRT: {mrt_max_val:.1f}°C")
print(f"Coldest location: ({mrt_min_lat:.2f}°N, {mrt_min_lon:.2f}°E) - MRT: {mrt_min_val:.1f}°C")
print(f"Temperature range: {mrt_max_val - mrt_min_val:.1f}°C")

# Heat stress intensity map
fig, axes = plt.subplots(1, 2, figsize=(14, 6), 
                         subplot_kw={'projection': cerra_proj}, 
                         constrained_layout=True)

# Extreme heat frequency (UTCI > 32°C)
heat_stress = (cerra_utci > 305.15).sum(dim='time') / len(cerra_utci.time) * 100  # UTCI threshold for strong heat
ax1 = axes[0]
ax1.coastlines(resolution='50m')
ax1.add_feature(cfeature.BORDERS, linestyle='--', alpha=0.5)
heat_plot = ax1.pcolormesh(heat_stress.longitude, heat_stress.latitude, heat_stress,
                           transform=ccrs.PlateCarree(), cmap='YlOrRd')
ax1.set_title('Heat Stress Days (UTCI > 32°C)', fontsize=12, fontweight='bold')
gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha=0.3)
gl1.top_labels = False
cbar1 = plt.colorbar(heat_plot, ax=ax1, orientation='horizontal', pad=0.08)
cbar1.set_label('Days (%)', fontsize=10)

# Cold stress frequency (UTCI < 0°C)
cold_stress = (cerra_utci < 273.15).sum(dim='time') / len(cerra_utci.time) * 100
ax2 = axes[1]
ax2.coastlines(resolution='50m')
ax2.add_feature(cfeature.BORDERS, linestyle='--', alpha=0.5)
cold_plot = ax2.pcolormesh(cold_stress.longitude, cold_stress.latitude, cold_stress,
                           transform=ccrs.PlateCarree(), cmap='Blues_r')
ax2.set_title('Cold Stress Days (UTCI < 0°C)', fontsize=12, fontweight='bold')
gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha=0.3)
gl2.top_labels = False
cbar2 = plt.colorbar(cold_plot, ax=ax2, orientation='horizontal', pad=0.08)
cbar2.set_label('Days (%)', fontsize=10)

stress_file = PLOTS_DIR / "thermal_stress_distribution.png"
fig.savefig(str(stress_file), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Stress distribution map saved: {stress_file.name}")

# Histogram of thermal indices
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

mrt_flat = mrt_c.values.flatten()
utci_flat = cerra_utci.values.flatten()
mrt_flat = mrt_flat[~np.isnan(mrt_flat)]
utci_flat = utci_flat[~np.isnan(utci_flat)]

axes[0].hist(mrt_flat, bins=50, color='red', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('MRT (°C)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Distribution of Mean Radiant Temperature', fontsize=11, fontweight='bold')
axes[0].axvline(np.mean(mrt_flat), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mrt_flat):.1f}°C')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].hist(utci_flat, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('UTCI (°C)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Distribution of Universal Thermal Climate Index', fontsize=11, fontweight='bold')
axes[1].axvline(np.mean(utci_flat), color='darkblue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(utci_flat):.1f}°C')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

hist_file = PLOTS_DIR / "thermal_indices_distributions.png"
fig.savefig(str(hist_file), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Distribution histograms saved: {hist_file.name}")

# Enhanced statistics file
stats_file = RESULTS_DIR / "thermal_analysis_detailed.txt"
with open(str(stats_file), 'w') as f:
    f.write("DETAILED THERMAL COMFORT ANALYSIS\n")
    f.write("="*70 + "\n\n")
    f.write("GLOBAL STATISTICS:\n")
    f.write(f"  MRT  - Mean: {period_mrt_mean.values:6.2f}°C, Std: {period_mrt_std.values:5.2f}°C\n")
    f.write(f"       - Min:  {mrt_flat.min():6.2f}°C, Max: {mrt_flat.max():6.2f}°C\n")
    f.write(f"  UTCI - Mean: {period_utci_mean.values:6.2f}°C, Std: {period_utci_std.values:5.2f}°C\n")
    f.write(f"       - Min:  {utci_flat.min():6.2f}°C, Max: {utci_flat.max():6.2f}°C\n\n")
    f.write("REGIONAL EXTREMES:\n")
    f.write(f"  Hottest: {mrt_max_val:.1f}°C at ({mrt_max_lat:.2f}°N, {mrt_max_lon:.2f}°E)\n")
    f.write(f"  Coldest: {mrt_min_val:.1f}°C at ({mrt_min_lat:.2f}°N, {mrt_min_lon:.2f}°E)\n")
    f.write(f"  Range:   {mrt_max_val - mrt_min_val:.1f}°C\n\n")
    f.write("THERMAL STRESS DISTRIBUTION:\n")
    f.write(f"  Extreme heat (UTCI > 46°C):  {100*heat_extreme/total_conditions:6.2f}%\n")
    f.write(f"  Strong heat (32-46°C):       {100*heat_strong/total_conditions:6.2f}%\n")
    f.write(f"  Moderate heat (26-32°C):     {100*comfort_warm/total_conditions:6.2f}%\n")
    f.write(f"  Neutral/Comfort (9-26°C):    {100*comfort_neutral/total_conditions:6.2f}%\n")
    f.write(f"  Moderate cold (0-9°C):       {100*cold_weak/total_conditions:6.2f}%\n")
    f.write(f"  Strong cold (<0°C):          {100*(cold_moderate+cold_strong)/total_conditions:6.2f}%\n")

print(f"✓ Detailed statistics saved: {stats_file.name}")

# Create summary comparison figure
fig = plt.figure(figsize=(16, 10), constrained_layout=True)
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Title
fig.suptitle('Thermal Comfort Analysis Summary', fontsize=16, fontweight='bold', y=0.98)

# Row 1: Mean maps
ax1 = fig.add_subplot(gs[0, :2], projection=cerra_proj)
ax1.coastlines(resolution='50m')
ax1.add_feature(cfeature.BORDERS, linestyle='--', alpha=0.5)
mrt_map = ax1.pcolormesh(mrt_c_mean.longitude, mrt_c_mean.latitude, mrt_c_mean,
                         transform=ccrs.PlateCarree(), cmap='RdYlBu_r')
ax1.set_title('Mean Radiant Temperature', fontsize=11, fontweight='bold')
cbar_mrt = plt.colorbar(mrt_map, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
cbar_mrt.set_label('°C', fontsize=9)

# Statistics box
ax_stats = fig.add_subplot(gs[0, 2])
ax_stats.axis('off')
stats_text = f"""
SUMMARY STATISTICS

MRT:
  Mean: {period_mrt_mean.values:.1f}°C
  Std:  {period_mrt_std.values:.1f}°C
  
UTCI:
  Mean: {period_utci_mean.values:.1f}°C
  Std:  {period_utci_std.values:.1f}°C
  
Timesteps: {total_timestamps:,}
"""
ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, fontsize=9,
              verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Row 2: UTCI map and stress distribution bar
ax2 = fig.add_subplot(gs[1, :2], projection=cerra_proj)
ax2.coastlines(resolution='50m')
ax2.add_feature(cfeature.BORDERS, linestyle='--', alpha=0.5)
utci_map = ax2.pcolormesh(utci_mean.longitude, utci_mean.latitude, utci_mean,
                          transform=ccrs.PlateCarree(), cmap='RdYlBu_r')
ax2.set_title('Universal Thermal Climate Index (UTCI)', fontsize=11, fontweight='bold')
cbar_utci = plt.colorbar(utci_map, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)
cbar_utci.set_label('°C', fontsize=9)

# Stress distribution pie chart
ax_pie = fig.add_subplot(gs[1, 2])
stress_cat = ['Extreme\nHeat', 'Strong\nHeat', 'Moderate\nHeat', 'Neutral', 'Cold']
stress_pct = [
    100*heat_extreme/total_conditions,
    100*heat_strong/total_conditions,
    100*comfort_warm/total_conditions,
    100*comfort_neutral/total_conditions,
    100*(cold_weak+cold_moderate+cold_strong)/total_conditions
]
colors_pie = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4']
wedges, texts, autotexts = ax_pie.pie(stress_pct, labels=stress_cat, autopct='%1.1f%%',
                                       colors=colors_pie, startangle=90, textprops={'fontsize': 8})
ax_pie.set_title('Thermal Stress\nDistribution', fontsize=10, fontweight='bold')

# Row 3: Histograms
ax_mrt_hist = fig.add_subplot(gs[2, 0])
ax_mrt_hist.hist(mrt_flat, bins=40, color='red', alpha=0.7, edgecolor='black')
ax_mrt_hist.axvline(np.mean(mrt_flat), color='darkred', linestyle='--', linewidth=2)
ax_mrt_hist.set_xlabel('MRT (°C)', fontsize=9)
ax_mrt_hist.set_ylabel('Frequency', fontsize=9)
ax_mrt_hist.set_title('MRT Distribution', fontsize=10, fontweight='bold')
ax_mrt_hist.grid(True, alpha=0.3, axis='y')

ax_utci_hist = fig.add_subplot(gs[2, 1])
ax_utci_hist.hist(utci_flat, bins=40, color='blue', alpha=0.7, edgecolor='black')
ax_utci_hist.axvline(np.mean(utci_flat), color='darkblue', linestyle='--', linewidth=2)
ax_utci_hist.set_xlabel('UTCI (°C)', fontsize=9)
ax_utci_hist.set_ylabel('Frequency', fontsize=9)
ax_utci_hist.set_title('UTCI Distribution', fontsize=10, fontweight='bold')
ax_utci_hist.grid(True, alpha=0.3, axis='y')

# Extremes info box
ax_extremes = fig.add_subplot(gs[2, 2])
ax_extremes.axis('off')
extremes_text = f"""
REGIONAL EXTREMES

Hottest:
  {mrt_max_val:.1f}°C
  ({mrt_max_lat:.1f}°N, {mrt_max_lon:.1f}°E)

Coldest:
  {mrt_min_val:.1f}°C
  ({mrt_min_lat:.1f}°N, {mrt_min_lon:.1f}°E)

Range: {mrt_max_val - mrt_min_val:.1f}°C
"""
ax_extremes.text(0.05, 0.95, extremes_text, transform=ax_extremes.transAxes, fontsize=9,
                 verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

summary_file = PLOTS_DIR / "thermal_analysis_summary.png"
fig.savefig(str(summary_file), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Analysis summary saved: {summary_file.name}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - ALL RESULTS SAVED TO outputs/ FOLDER")
print("="*80 + "\n")
