# some data opening and Jenkinson and Collison grid points plots

import os
os.chdir('D:\\cerra-thermal')
print(" ".join(['\nCurrent path is', os.getcwd()]))

import os
os.chdir('C:\\Users\olehs\cerra-thermal\poster_input\multi')
print(" ".join(['\nCurrent path is', os.getcwd()]))

import cfgrib
grib_data = cfgrib.open_datasets("159e040e873f74df10103b9e33462cf40.grib")



import xarray as xr
era5 = xr.open_dataset("13d464c92ea7e25dd197b3ec7ad8213ffera5.grib", engine="cfgrib")
land1 = xr.open_dataset("159e040e873f74df10103b9e33462cf40.grib", engine="cfgrib")
land2 = xr.open_dataset("1b07f3ce5d8f63480aec8cd49f98c16c7.nc")
land3 = xr.open_dataset("34fd5edd8925afd784df3f4041fddfd5.grib", engine="cfgrib")
land4 = xr.open_dataset("3651ef40ba73db9ed2afea6689f0c27f0.nc")
land31 = xr.open_dataset("31-dd93eea0992e2d59a52981ebd13e1131.grib", engine="cfgrib")


land1 = xr.open_dataarray("159e040e873f74df10103b9e33462cf40.grib", engine="cfgrib")
land1_fixed = land1.stack(valid_time=('time', 'step'))

    .set_index(z='valid_time').rename({'z': 'valid_time'})

land1.dims
land1.coords.values
land1.time.values
land1.step.values
land1.valid_time.values
point = land1.sel(latitude=48.8, longitude=2.3, method='nearest')
point.values

point = jan1.sel(latitude=48.8, longitude=2.3, method='nearest')
point.values

import xarray as xr
import time
start_time = time.time()
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")
# rad = xr.open_dataset("jan1.grib", engine="cfgrib")
# rad = xr.open_dataset("jan2.grib", engine="cfgrib")
# rad = xr.open_dataset("jan3.grib", engine="cfgrib")
rad1 = xr.open_dataset("jul1.grib", engine="cfgrib")
rad2 = xr.open_dataset("jul2.grib", engine="cfgrib")
rad3 = xr.open_dataset("jul3.grib", engine="cfgrib")

rad3 = rad3 - rad2
rad2 = rad2 - rad1

rad1 = rad1.drop_vars(['step', 'valid_time'])
rad2.coords["time"] = rad2.coords["time"] + np.timedelta64(1, 'h')
rad3.coords["time"] = rad3.coords["time"] + np.timedelta64(2, 'h')
rad = xr.concat(objs=[rad1, rad2, rad3], dim='time', data_vars='all', coords='different', compat='broadcast_equals')
rad = rad.assign_coords(longitude=(((rad.longitude + 180) % 360) - 180))
del rad1, rad2, rad3
rad = rad.sortby('time')
rad = rad / 3600
rad['ssru'] = rad.ssrd - rad.ssr # rsw or uswrf Upward short-wave radiation flux
rad['stru'] = rad.strd - rad.str # lur or ulwrf Upward long-wave radiation flux

rad.assign_coords(longitude=(((rad.longitude + 180) % 360) - 180), inplace=True)
variables = ["ssr", "str", "ssrd", "strd", "tidirswrf", 'ssru', 'stru']
for variable in variables:
    rad[variable].attrs.update({"units": "W m-2"})
rad.longitude.attrs["units"] = "degrees"

# rad.to_netcdf("poster_input/rad_jan.nc")
rad.to_netcdf("poster_input/rad_jul.nc")


# Distribution of Jenkinson and Collison grid points
import pandas as pd
data = [                ['x1', 57, 17], ['x2', 57, 27],
        ['x3', 52, 7], ['x4', 52, 17], ['x5', 52, 27], ['x6', 52, 37],
        ['x7', 47, 7], ['x8', 47, 17], ['x9', 47, 27], ['x10', 47, 37],
        ['x11', 42, 7], ['x12', 42, 17], ['x13', 42, 27], ['x14', 42, 37],
                        ['x15', 37, 17], ['x16', 37, 27]]