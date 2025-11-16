# creating atmospheric classification dataset

from pathlib import Path
import os

# --- Configuration ---
# Define the path to the external data directory.
# This path is currently set to 'C:\\Users\\olehs\\e-obs'.
# You can change this to the correct location of your data.
DATA_DIR = Path(r'C:\Users\olehs\e-obs')
print(f"Using data directory: {DATA_DIR}")

# Ensure the data directory exists
if not DATA_DIR.exists():
    raise FileNotFoundError(f"The specified data directory does not exist: {DATA_DIR}")

import time
import numpy as np
import xarray as xr
from jcclass import jc
from dask.distributed import Client
from dask.diagnostics import ProgressBar

client = Client(n_workers=8, threads_per_worker=2)
print(client)
print(client.dashboard_link)

# mslp.rename({'valid_time': 'time'})
mslp = xr.open_dataset(DATA_DIR / '1991-era5_mslp_time.nc', chunks={'time': 1000})['msl']
selected_times = mslp.sel(time=mslp['time'].dt.hour.isin([0, 6, 12, 18]))
# daily_means = selected_times.groupby('time.date').mean(skipna=True)
daily_means = selected_times.resample(time='1D').mean(skipna=True)

with ProgressBar():
    daily_means_computed = daily_means.compute()

daily_means_computed.to_netcdf(DATA_DIR / '1991-era5_mslp_daily.nc')
daily_means_computed = xr.open_dataarray(DATA_DIR / '1991-era5_mslp_daily.nc')

da = xr.open_dataarray(DATA_DIR / '1991-era5_mslp_daily.nc').sel(time=slice('2020-01', '2020-01'))
jc_res = jc(da)
cts_27 = jc_res.classification()
cts_11 = jc_res.eleven_cts(cts_27)

start_time = time.time()
filename = DATA_DIR / '1991-era5_mslp_daily.nc'
cts_27 = jc(filename).classification()
print("Elapsed time: ", time.time() - start_time)
# cts_27.to_netcdf(DATA_DIR / '1991-era5_cts_27.nc')
cts_27 = cts_27.astype(np.float32)
cts_27.to_netcdf(DATA_DIR / '1991-era5_cts_27_float32.nc')

cts_11 = jc.eleven_cts(cts_27)
print("Elapsed time: ", time.time() - start_time)
# cts_11.to_netcdf(DATA_DIR / '1991-era5_cts_11.nc')
cts_11 = cts_11.astype(np.float32)
cts_11.to_netcdf(DATA_DIR / '1991-era5_cts_11_float32.nc')

cts_27 = xr.open_dataarray(DATA_DIR / '1991-era5_cts_27_float32.nc').astype(np.int32)
cts_11 = xr.open_dataarray(DATA_DIR / '1991-era5_cts_11_float32.nc').astype(np.int32)
print(np.unique(cts_27.values))
print(np.unique(cts_11.values))



date = cts_27.time[175]
cts = cts_27.sel(time=date)
mslp = xr.open_dataset(filename).sel(time=date)

fig = jc.plot_cts(cts)
fig = jc.plot_cts_mslp(cts, mslp)
fig = jc.plot_cts_globe(cts, mslp)