# compuring Normal Effective Temperature from E-OBS dataset

import time
import os
from pathlib import Path
import numba as nb
import numpy as np
import xarray as xr
from numba import vectorize
from xclim.core.units import convert_units_to
from xclim.indices import specific_humidity, relative_humidity
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import rasterio
import rioxarray
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.features import shapes
from dask.diagnostics import ProgressBar

# --- Configuration ---
# Define the path to the external data directory and output directory
DATA_DIR = Path(r'C:\Users\olehs\e-obs')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
PLOTS_DIR = OUTPUT_DIR / 'plots'

# Create output directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

print(f"Using data directory: {DATA_DIR}")
print(f"Using output directory: {OUTPUT_DIR}")

# --- Main Script ---

# Load initial datasets
da_elev = xr.open_dataarray(DATA_DIR / 'elev_ens_0.1deg_reg_v30.0e.nc', chunks={'time': 1000}) #elevation (m)
da_slp = xr.open_dataarray(DATA_DIR / 'pp_ens_mean_0.1deg_reg_v30.0e.nc') #mean sea level preassure (hPa)
da_ta = xr.open_dataarray(DATA_DIR / 'tg_ens_mean_0.1deg_reg_v30.0e.nc', chunks={'time': 1000}) #daily 2m mean air temperature (°C)
da_tn = xr.open_dataarray(DATA_DIR / 'tn_ens_mean_0.1deg_reg_v30.0e.nc', chunks={'time': 1000}) #daily 2m min air temperature (°C)
da_tx = xr.open_dataarray(DATA_DIR / 'tx_ens_mean_0.1deg_reg_v30.0e.nc', chunks={'time': 1000}) #daily 2m max air temperature (°C)
da_rh = xr.open_dataarray(DATA_DIR / 'hu_ens_mean_0.1deg_reg_v30.0e.nc', chunks={'time': 1000}) #relative humidity (%)
da_va = xr.open_dataarray(DATA_DIR / 'fg_ens_mean_0.1deg_reg_v30.0e.nc', chunks={'time': 1000}) #wind speed (m/s)

# Example: Plotting Mean Temperature
plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
da_slp.count(dim='time').plot(ax=ax, transform=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_title('Mean 2m Air Temperature from E-OBS')
plt.show()


# --- Wind Estimation ---
da_elev_v29 = xr.open_dataarray(DATA_DIR / 'elev_ens_0.1deg_reg_v29.0e.nc')
da_va_v28 = xr.open_dataarray(DATA_DIR / 'fg_ens_mean_0.1deg_reg_v28.0e.nc') #.sel(time=slice('1991', '2020')) # wind speed

@nb.vectorize([nb.float32(nb.float32, nb.float32)], target='parallel')
def wind_estimation(va, h):
    v = va * np.log10(h / 0.01) * 0.333333333333 # a = 1 / np.log10(10 / 0.01)
    return v

start_time = time.time()
da_v = wind_estimation(da_va_v28, 1.2)
end_time = time.time()
print(f"Wind estimation elapsed time: {end_time - start_time:.2f}s")

_time = da_v.coords['time']
latitude = da_elev_v29.coords['latitude']
longitude = da_elev_v29.coords['longitude']
wind_da = xr.DataArray(
    da_v,
    dims=['time', 'latitude', 'longitude'],
    coords={'time': _time, 'latitude': latitude, 'longitude': longitude},
    name='fg'
)
wind_da.attrs['units'] = 'm/s'
wind_da.to_netcdf(OUTPUT_DIR / 'fg_ens_mean_0.1deg_reg_v28.1.nc')


# --- Pressure at Elevation Calculation ---
da_slp_v29 = xr.open_dataarray(DATA_DIR / 'pp_ens_mean_0.1deg_reg_v29.0e.nc') #mean sea level preassure (hPa)
da_ta_v29 = xr.open_dataarray(DATA_DIR / 'tg_ens_mean_0.1deg_reg_v29.0e.nc') #daily 2m mean air temperature (°C)

# Constants
M = 0.0289644  # Molar mass of Earth's air (kg/mol)
g = 9.80665    # Acceleration due to gravity (m/s^2)
R = 8.314462618 # Universal gas constant (J/(mol*K))
LAPSE_RATE = 0.0065  # Standard temperature lapse rate (K/m)

temperature_difference = (da_elev_v29 - 0) * LAPSE_RATE
da_temp_at_sea_level = da_ta_v29 + temperature_difference
da_temp_at_sea_level_kelvins = da_temp_at_sea_level + 273.15

da_slp_pa = da_slp_v29 * 100
da_slp_pa.attrs['units'] = 'Pa'
with ProgressBar():
    da_comp = da_slp_pa.compute()

da_pressure_at_elev = da_slp_v29 * np.exp(-M * g * da_elev_v29 / (R * da_temp_at_sea_level_kelvins))
with ProgressBar():
    da_comp = da_pressure_at_elev.compute()
da_pressure_at_elev.to_netcdf(OUTPUT_DIR / 'pressure_at_elevation.nc')


# --- Normal Effective Temperature (NET) Calculation ---
da_v_fg1 = xr.open_dataarray(DATA_DIR / '1991-fg1.nc').sel(time=slice('1991', '1991'))

ts = ['n', 'g']
for s in ts:
    da_t = xr.open_dataarray(DATA_DIR / f't{s}_ens_mean_0.1deg_reg_v29.0e.nc').sel(time=slice('1991', '2020'))

    @nb.vectorize([nb.float32(nb.float32, nb.float32, nb.float32)], target='parallel')
    def calculate_normal_effective_temperature(t2m, rh, v):
        net = (37 -
               ((37 - t2m) /
                (0.68 - 0.0014 * rh +
                 (1 / (1.76 + 1.4 * v ** 0.75)))) -
               (0.29 * t2m * (1 - 0.01 * rh)))
        return net

    start_time = time.time()
    # Note: This assumes da_rh and da_v_fg1 have compatible coordinates with da_t
    net_x = calculate_normal_effective_temperature(da_t.values, da_rh.values, da_v_fg1.values)
    end_time = time.time()
    print(f"NET calculation for '{s}' elapsed time: {end_time - start_time:.2f}s")

    net_da = xr.DataArray(
        net_x,
        dims=['time', 'latitude', 'longitude'],
        coords=da_t.coords,
        name='net_' + s
    )
    net_da.attrs['units'] = 'Celsius'
    net_da.to_netcdf(OUTPUT_DIR / f'1991-net_{s}.nc')


# --- Monthly Mean Analysis and Plotting ---
da_net_n = xr.open_dataarray(OUTPUT_DIR / '1991-net_n.nc')
da_net_x = xr.open_dataarray(OUTPUT_DIR / '1991-net_x.nc') # Assuming this is net_g

cts_27 = xr.open_dataarray(DATA_DIR / '1991-era5_cts_27_float32.nc').astype(np.int32)
cts_11 = xr.open_dataarray(DATA_DIR / '1991-era5_cts_11_float32.nc').astype(np.int32)

# Interpolate classification data to match NET grid
cts_27 = cts_27.interp_like(da_net_x, method='linear')

mean_net_x = da_net_x.mean(dim='time')
mean_jan = da_net_n.sel(time=da_net_x['time.month'] == 1).mean(dim='time')
mean_jul = da_net_x.sel(time=da_net_x['time.month'] == 7).mean(dim='time')

# Plot single month (January)
fig = plt.figure(figsize=(5, 4.5), edgecolor=None, constrained_layout=True)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-10, 40, 35.1, 72], crs=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.add_feature(cfeature.BORDERS, linestyle='--', edgecolor='black')
filled_c = plt.pcolormesh(mean_net_x.longitude, mean_net_x.latitude, mean_jan, transform=ccrs.PlateCarree(), cmap="RdYlBu_r")
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.margins(0, tight=True)
cbar = fig.colorbar(filled_c, ax=ax, orientation="horizontal", pad=0.03)
cbar.set_label('NET (°C)')
plt.show()
fig.savefig(PLOTS_DIR / 'mean_jan.png', dpi=600)
plt.close()

# Plot January vs July comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
extent = [-10, 40, 35.1, 72]
cmap = "RdYlBu_r"
vmin = min(mean_jan.min(), mean_jul.min())
vmax = max(mean_jan.max(), mean_jul.max())

ax1.set_extent(extent)
ax1.coastlines(resolution='10m')
filled_jan = ax1.pcolormesh(mean_jan.longitude, mean_jan.latitude, mean_jan, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax1.set_title('January')
ax1.margins(0, tight=True)

ax2.set_extent(extent)
ax2.coastlines(resolution='10m')
filled_jul = ax2.pcolormesh(mean_jul.longitude, mean_jul.latitude, mean_jul, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax2.set_title('July')
ax2.margins(0, tight=True)

cbar = fig.colorbar(filled_jul, ax=[ax1, ax2], orientation="horizontal", pad=0.05)
cbar.set_label('NET (°C)')
plt.show()
fig.savefig(PLOTS_DIR / 'jan_jul_comparison.png', dpi=600)
plt.close()


# --- TIF to Shapefile Conversion ---
def tif_to_shapefile(tif_path, shapefile_path):
    """Converts a classified TIF to a Shapefile."""
    try:
        with rioxarray.open_rasterio(tif_path, masked=True) as src:
            raster_data = src.squeeze()
            crs = src.rio.crs
            image = raster_data.values.astype('int16')
            shapes_gen = shapes(image, transform=src.rio.transform())
            geometries = []
            values = []
            for (geom, value) in shapes_gen:
                if value != src.rio.nodata:
                    geometries.append(Polygon(geom['coordinates'][0]))
                    values.append(value)
            gdf = gpd.GeoDataFrame({'value': values, 'geometry': geometries}, crs=crs)
            gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.1)
            shapefile_path.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(shapefile_path)
            print(f"Shapefile saved to: {shapefile_path}")
    except rasterio.RasterioIOError as e:
        print(f"Error opening or reading the TIF file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

kg_path = DATA_DIR / r'kg_maps\koppen_geiger_tif\1991_2020\koppen_geiger_0p1.tif'
shapefile_output_path = OUTPUT_DIR / r'kg_maps\koppen_geiger_shp\1991_2020\koppen_geiger.shp'
tif_to_shapefile(kg_path, shapefile_output_path)


# --- Plotting with Shapefile Overlay ---
kg_zones = gpd.read_file(shapefile_output_path)

fig = plt.figure(figsize=(5, 4.5), edgecolor=None, constrained_layout=True)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-10, 40, 35.1, 72], crs=ccrs.PlateCarree())
ax.coastlines(resolution='10m')

filled_c = plt.pcolormesh(mean_net_x.longitude, mean_net_x.latitude, mean_jul,
                           transform=ccrs.PlateCarree(), cmap="RdYlBu_r")
kg_zones.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.margins(0, tight=True)
cbar = fig.colorbar(filled_c, ax=ax, orientation="horizontal", pad=0.03)
cbar.set_label('NET (°C)')
plt.show()
fig.savefig(PLOTS_DIR / 'mean_jul_with_kg_zones.png', dpi=600)
plt.close()