# MRT and UTCI evaluation against BSRN observations

'''latitude and longitude dimensions insted of y and x
import xarray as xr
# Open the dataset
rad_jul = xr.open_dataset('poster_input/rad_jul.nc')
# Assuming latitude and longitude are defined as coordinates
# Create new data variables with latitude and longitude dimensions
ssr_new = xr.DataArray(rad_jul['ssr'], coords=[rad_jul['time'], rad_jul['latitude'], rad_jul['longitude']], dims=['time', 'latitude', 'longitude'])
str_new = xr.DataArray(rad_jul['str'], coords=[rad_jul['time'], rad_jul['latitude'], rad_jul['longitude']], dims=['time', 'latitude', 'longitude'])
# Create a new dataset with the new data variables
rad_jul_new = xr.Dataset({'ssr': ssr_new, 'str': str_new})
# Drop the old data variables that depend on y and x coordinates
rad_jul_new = rad_jul_new.drop_vars(['ssr', 'str'])
# Print the updated dataset
print(rad_jul_new)'''
import os
os.chdir('C:\\Users\\olehs\\cerra-thermal')
print(" ".join(['\nCurrent path is', os.getcwd()]))
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PLOTS_DIR
import sys

jan_cerra_mrt = xr.open_dataarray("poster_input/mrt_jan_from_fdir.nc")

jan_cerra_mrt_apx = xr.open_dataarray("poster_input/mrt_jan.nc")
jan_cerra_mrt_apx = jan_cerra_mrt_apx.drop_vars(['step', 'surface', 'valid_time'])

jul_cerra_mrt = xr.open_dataarray("poster_input/mrt_jul_from_fdir.nc")
jul_cerra_mrt = jul_cerra_mrt.drop_vars(['step', 'surface', 'valid_time'])
# cerra_mrt_cabauw = cerra_mrt.interp(coords={"x": 51.9680, "y": 4.9280}, method="nearest")

jul_cerra_mrt_apx = xr.open_dataarray("poster_input/mrt_jul.nc")
jul_cerra_mrt_apx = jul_cerra_mrt_apx.drop_vars(['step', 'surface', 'valid_time'])
# cerra_mrt_apx_cabauw = cerra_mrt_apx.interp(coords={"x": 51.9680, "y": 4.9280}, method="nearest")

cerra_mrt = xr.concat(objs=[jan_cerra_mrt, jul_cerra_mrt], dim='time', data_vars='all', coords='different', compat='broadcast_equals')
cerra_mrt_apx = xr.concat(objs=[jan_cerra_mrt_apx, jul_cerra_mrt_apx], dim='time', data_vars='all', coords='different', compat='broadcast_equals')

# Cabauw	CAB	51.968	4.928	0
# Toravere	TOR	58.2641	26.4613	70
# Payerne	PAY	46.8123	6.9422	491
# Izana	IZA	28.3093	-16.4993	2373
# Target BSRN station latitude and longitude
station_code = 'CAB'
target_latitude = 51.968
target_longitude = 4.928

station_code = 'TOR'
target_latitude = 58.2641
target_longitude = 26.4613
# old location 58.254°N, 26.462°E

station_code = 'PAY'
target_latitude = 46.8123
target_longitude = 6.9422

station_code = 'IZA'
target_latitude = 28.29 #from 28.3093    to ------, 28.29, 28.28
target_longitude = -16.58 #from -16.4993 to -16.53, 16.58, 16.63

path = 'era5/'
file_pattern = 'ECMWF_mrt_*.nc'
era_mrt = xr.open_mfdataset(f'{path}/{file_pattern}', combine='by_coords')['mrt'].sel(time='2017')
era_mrt = era_mrt.rename({'lon': 'longitude', 'lat': 'latitude'})
era_mrt = era_mrt.sel(time='2017')
# era_mrt = era_mrt.sel(time='2017-07')
era_mrt_cabauw = era_mrt.sel(latitude=target_latitude, longitude=target_longitude, method="nearest")

# Calculate the squared distances between target and dataset coordinates
lat_diff_sq = (cerra_mrt.latitude - target_latitude)**2
lon_diff_sq = (cerra_mrt.longitude - target_longitude)**2
# Calculate the squared distance from the target point
distance_sq = lat_diff_sq + lon_diff_sq
# Find the index of the minimum squared distance
min_distance_idx = np.unravel_index(distance_sq.argmin(), distance_sq.shape)
# Extract data at the closest grid point
cerra_mrt_cabauw = cerra_mrt.isel(y=min_distance_idx[0], x=min_distance_idx[1])
cerra_mrt_apx_cabauw = cerra_mrt_apx.isel(y=min_distance_idx[0], x=min_distance_idx[1])

f1 = xr.open_dataset("meteo_test.nc")
f1 = f1.assign_coords(longitude=(((f1.longitude + 180) % 360) - 180))
cerra_lsm = f1.lsm.isel(time=28)
cerra_or = f1.orog.isel(time=28)
cerra_or_sel = cerra_or.isel(y=min_distance_idx[0], x=min_distance_idx[1])
#
# era5_gp = xr.open_dataarray("poster_input/era5_geopotential.nc")
# era5_or = era5_gp.isel(time=0)/9.80665 # The (surface) geopotential height (orography) can be calculated by dividing the (surface) geopotential by the Earth's gravitational acceleration, g (=9.80665 m s-2 ).
# era5_or_sel = era5_or.sel(latitude=target_latitude, longitude=target_longitude, method="nearest")
#
# print(era5_or_sel.values, cerra_or_sel.values)
#
# print("Closest Data at Lat:", cerra_mrt.latitude[min_distance_idx].values)
# print("Closest Data at Lon:", cerra_mrt.longitude[min_distance_idx].values)
# print("MRT Values at Closest Point:", cerra_mrt_cabauw.values)

# df = pd.read_csv('poster_input/' + station_code + '_radiation_2017-07.csv', encoding='utf-8')
# # df = pd.read_csv('poster_input/IZA_radiation_2017-01.tab', sep='\t')
# df['latitude'] = target_latitude
# df['longitude'] = target_longitude
# df = df[["time", "latitude", "longitude", "SWD", "DIR", "DIF", "LWD", "SWU", "LWU"]]

df1 = pd.read_csv('poster_input/' + station_code + '_radiation_2017-01.csv', encoding='utf-8')
# df = pd.read_csv('poster_input/IZA_radiation_2017-01.tab', sep='\t')
df1['latitude'] = target_latitude
df1['longitude'] = target_longitude
df1 = df1[["time", "latitude", "longitude", "SWD", "DIR", "DIF", "LWD", "SWU", "LWU"]]

df7 = pd.read_csv('poster_input/' + station_code + '_radiation_2017-07.csv', encoding='utf-8')
# df = pd.read_csv('poster_input/IZA_radiation_2017-01.tab', sep='\t')
df7['latitude'] = target_latitude
df7['longitude'] = target_longitude
df7 = df7[["time", "latitude", "longitude", "SWD", "DIR", "DIF", "LWD", "SWU", "LWU"]]

df = pd.concat([df1, df7], ignore_index=True)
df['time'] = pd.to_datetime(df['time'])
df.set_index(['time', 'latitude', 'longitude'], inplace=True)
bsrn = xr.Dataset.from_dataframe(df)

bsrn["DIFe"] = bsrn.SWD - bsrn.DIR
variables = ["SWD", "DIR", "DIF", "DIFe", "LWD", "SWU", "LWU"]
for variable in variables:
    bsrn[variable].attrs.update({"units": "W m-2"})
bsrn.longitude.attrs["units"] = "degrees"
bsrn.latitude.attrs["units"] = "degrees"


# bsrn1h = bsrn.resample(time='1H').mean(dim='time')

# sel = bsrn.sel(time='2017-07-24T03')

''' calculating average(i) and sunlit(s) cosine of solar zenith angle
dates = cerra_mrt.time
lat = _gather_lat(cerra_mrt)
lon = _gather_lon(cerra_mrt)
dec = solar_declination(dates)
csza_i = cosine_of_solar_zenith_angle(dates, dec, lat, lon=lon, stat="average", sunlit=False)
csza_i.to_netcdf("poster_output/csza_i_jul.nc")
csza_s = cosine_of_solar_zenith_angle(dates, dec, lat, lon=lon, stat="average", sunlit=True)
csza_s.to_netcdf("poster_output/csza_s_jul.nc")'''

bsrn = bsrn.resample(time='1H').mean(dim='time')
bsrn_mrt_cabauw_in_test = mean_radiant_temperature_from_DIR_DIF_INST(rsds=bsrn.SWD,
                                                         rsus=bsrn.SWU,
                                                         rlds=bsrn.LWD,
                                                         rlus=bsrn.LWU,
                                                         rsds_direct=bsrn.DIR,
                                                         rsds_diffuse=bsrn.DIF,
                                                         stat='sunlit')

bsrn_mrt_cabauw_in_test2 = mean_radiant_temperature_from_DIR_DIF_INST(rsds=bsrn.SWD,
                                                         rsus=bsrn.SWU,
                                                         rlds=bsrn.LWD,
                                                         rlus=bsrn.LWU,
                                                         rsds_direct=bsrn.DIR,
                                                         rsds_diffuse=bsrn.DIF,
                                                         stat='sunlit') #sunlit 2 times

bsrn_mrt_cabauw_in_test3 = mean_radiant_temperature_from_DIR_DIF_INST(rsds=bsrn.SWD,
                                                         rsus=bsrn.SWU,
                                                         rlds=bsrn.LWD,
                                                         rlus=bsrn.LWU,
                                                         rsds_direct=bsrn.DIR,
                                                         rsds_diffuse=bsrn.DIF,
                                                         stat='average')

plt.figure()
plt.hexbin(bsrn_mrt_cabauw_in_test2.values.flatten(), bsrn_mrt_cabauw_in_test3.values.flatten(), gridsize=100, cmap='viridis', bins='log', mincnt=1)
plt.xlabel('MRT TEST')
plt.ylabel('MRT XCLIM')
plt.colorbar(label='Density (log scale)')
plt.tight_layout()
plt.show()

plt.savefig(str(PLOTS_DIR / 'hexbin_mrt_jul_old.png'), dpi=1000)
plt.close()

bsrn_mrt_cabauw_in = mean_radiant_temperature_from_DIR_DIF_INST(rsds=bsrn.SWD,
                                                         rsus=bsrn.SWU,
                                                         rlds=bsrn.LWD,
                                                         rlus=bsrn.LWU,
                                                         rsds_direct=bsrn.DIR,
                                                         rsds_diffuse=bsrn.DIF,
                                                         stat='instant')
bsrn_mrt_cabauw_in = bsrn_mrt_cabauw_in.resample(time='1H').mean(dim='time')

bsrn_mrt_cabauw_in = bsrn_mrt_cabauw_in.resample(time='3H').mean(dim='time')
era_mrt_cabauw = era_mrt_cabauw.resample(time='3H').mean(dim='time')
cerra_mrt_cabauw = cerra_mrt_cabauw.resample(time='3H').mean(dim='time')
cerra_mrt_apx_cabauw = cerra_mrt_apx_cabauw.resample(time='3H').mean(dim='time')




import skill_metrics as sm

# # Preprocess data arrays
# bsrn_mrt_preproc = bsrn_mrt_cabauw_in.values.reshape(-1)
# era_mrt_preproc = era_mrt_cabauw.values
# cerra_mrt_preproc = cerra_mrt_cabauw.values
# cerra_mrt_apx_preproc = cerra_mrt_apx_cabauw.values
#
# print("bsrn_mrt_preproc has NaN values:", np.isnan(bsrn_mrt_preproc).any())
# print("era_mrt_preproc has NaN values:", np.isnan(era_mrt_preproc).any())
# print("cerra_mrt_preproc has NaN values:", np.isnan(cerra_mrt_preproc).any())
# print("cerra_mrt_apx_preproc has NaN values:", np.isnan(cerra_mrt_apx_preproc).any())
#
# # Check shapes of the data arrays
# print("bsrn_mrt_preproc shape:", bsrn_mrt_preproc.shape)
# print("era_mrt_preproc shape:", era_mrt_preproc.shape)
# print("cerra_mrt_preproc shape:", cerra_mrt_preproc.shape)
# print("cerra_mrt_apx_preproc shape:", cerra_mrt_apx_preproc.shape)
#
# def handle_non_finite(data_array):
#     finite_mask = np.isfinite(data_array)
#     finite_data = data_array[finite_mask]
#     return finite_data
# bsrn_mrt_preproc = handle_non_finite(bsrn_mrt_preproc)
# era_mrt_preproc = handle_non_finite(era_mrt_preproc)
# cerra_mrt_preproc = handle_non_finite(cerra_mrt_preproc)
# cerra_mrt_apx_preproc = handle_non_finite(cerra_mrt_apx_preproc)
#
# print("bsrn_mrt_preproc has NaN values:", np.isnan(bsrn_mrt_preproc).any())
# print("era_mrt_preproc has NaN values:", np.isnan(era_mrt_preproc).any())
# print("cerra_mrt_preproc has NaN values:", np.isnan(cerra_mrt_preproc).any())
# print("cerra_mrt_apx_preproc has NaN values:", np.isnan(cerra_mrt_apx_preproc).any())
# # Check shapes of the data arrays
# print("bsrn_mrt_preproc shape:", bsrn_mrt_preproc.shape)
# print("era_mrt_preproc shape:", era_mrt_preproc.shape)
# print("cerra_mrt_preproc shape:", cerra_mrt_preproc.shape)
# print("cerra_mrt_apx_preproc shape:", cerra_mrt_apx_preproc.shape)
#
#
# min_shape = min(bsrn_mrt_preproc.shape[0], era_mrt_preproc.shape[0], cerra_mrt_preproc.shape[0], cerra_mrt_apx_preproc.shape[0])
# bsrn_mrt_preproc = bsrn_mrt_preproc[:min_shape]
# era_mrt_preproc = era_mrt_preproc[:min_shape]
# cerra_mrt_preproc = cerra_mrt_preproc[:min_shape]
# cerra_mrt_apx_preproc = cerra_mrt_apx_preproc[:min_shape]
#
# taylor_stats_era = sm.taylor_statistics(era_mrt_preproc, bsrn_mrt_preproc, 'data')
# taylor_stats_cerra = sm.taylor_statistics(cerra_mrt_preproc, bsrn_mrt_preproc, 'data')
# taylor_stats_cerra_apx = sm.taylor_statistics(cerra_mrt_apx_preproc, bsrn_mrt_preproc, 'data')


def handle_non_finite(*data_arrays):
    # Create a mask for each data array to identify non-finite values
    finite_masks = [np.isfinite(data_array) for data_array in data_arrays]

    # Combine the masks using logical AND to identify timestamps with non-finite values in any array
    combined_mask = np.all(np.vstack(finite_masks), axis=0)

    # Apply the combined mask to each data array
    finite_data_arrays = [data_array[combined_mask] for data_array in data_arrays]

    return finite_data_arrays

# Apply handle_non_finite to all data arrays
bsrn_mrt_preproc, era_mrt_preproc, cerra_mrt_preproc, cerra_mrt_apx_preproc = handle_non_finite(
    bsrn_mrt_cabauw_in.values.reshape(-1),
    era_mrt_cabauw.values,
    cerra_mrt_cabauw.values,
    cerra_mrt_apx_cabauw.values
)

# Calculate Taylor statistics using the matched timestamps
taylor_stats_era = sm.taylor_statistics(era_mrt_preproc, bsrn_mrt_preproc, 'data')
taylor_stats_cerra = sm.taylor_statistics(cerra_mrt_preproc, bsrn_mrt_preproc, 'data')
taylor_stats_cerra_apx = sm.taylor_statistics(cerra_mrt_apx_preproc, bsrn_mrt_preproc, 'data')

sdev = np.array([taylor_stats_era['sdev'][0], taylor_stats_era['sdev'][1], taylor_stats_cerra['sdev'][1], taylor_stats_cerra_apx['sdev'][1]])
crmsd = np.array([taylor_stats_era['crmsd'][0], taylor_stats_era['crmsd'][1], taylor_stats_cerra['crmsd'][1], taylor_stats_cerra_apx['crmsd'][1]])
ccoef = np.array([taylor_stats_era['ccoef'][0], taylor_stats_era['ccoef'][1], taylor_stats_cerra['ccoef'][1], taylor_stats_cerra_apx['ccoef'][1]])



label = ['BSRN', 'ERA5-HEAT', 'CERRA original', 'CERRA empirical']
intervalsCOR = np.concatenate((np.arange(0,1.0,0.2),[0.9, 0.95, 0.99, 1]))
plt.figure
# sm.taylor_diagram(sdev,crmsd,ccoef)
# sm.taylor_diagram(sdev,crmsd,ccoef, styleOBS = '-', colOBS = 'r', markerobs = 'o', titleOBS = 'BSRN obsv.')
sm.taylor_diagram(sdev, crmsd, ccoef, markerLabel=label,
                  tickCOR=intervalsCOR, rmsLabelFormat=':.1f',
                  styleOBS='-', colOBS='r', markerobs='o', titleOBS=station_code+' obsv.',
                  markerLegend='on')
# sm.taylor_diagram(sdev, crmsd, ccoef, markerLabel=label,
#                   markerLabelColor='r', markerLegend='on',
#                   tickRMS=range(0, 60, 10), tickRMSangle=110.0,
#                   colRMS='m', styleRMS=':', widthRMS=2.0,
#                   titleRMS='off', tickSTD=range(0, 80, 20),
#                   axismax=60.0, colSTD='b', styleSTD='-.',
#                   widthSTD=1.0, titleSTD='off',
#                   colCOR='k', styleCOR='--', widthCOR=1.0,
#                   titleCOR='off')
plt.title(station_code)
plt.tight_layout()
plt.savefig(str(PLOTS_DIR / f'tailor_{station_code}_1.png'), dpi=1000)
plt.close()

era5_bias = era_mrt_cabauw - bsrn_mrt_cabauw_in
cerra_original_bias = cerra_mrt_cabauw - bsrn_mrt_cabauw_in
cerra_empirical_bias = cerra_mrt_apx_cabauw - bsrn_mrt_cabauw_in

# era5_bias = era5_bias.resample(time='D').apply(lambda x: np.abs(x).max(dim='time'))
# cerra_original_bias = cerra_original_bias.resample(time='D').apply(lambda x: np.abs(x).max(dim='time'))
# cerra_empirical_bias = cerra_empirical_bias.resample(time='D').apply(lambda x: np.abs(x).max(dim='time'))

# era5_bias = era5_bias.resample(time='D').max(dim='time')
# cerra_original_bias = cerra_original_bias.resample(time='D').max(dim='time')
# cerra_empirical_bias = cerra_empirical_bias.resample(time='D').max(dim='time')

era5_bias_m = era5_bias.mean(dim='time')
cerra_original_bias_m = cerra_original_bias.mean(dim='time')
cerra_empirical_bias_m = cerra_empirical_bias.mean(dim='time')

era5_bias_m = era5_bias.sel(time='2017-01').mean(dim='time')
cerra_original_bias_m = cerra_original_bias.sel(time='2017-01').mean(dim='time')
cerra_empirical_bias_m = cerra_empirical_bias.sel(time='2017-01').mean(dim='time')

era5_bias_m = era_mrt_cabauw.sel(time='2017-07').mean(dim='time') - bsrn_mrt_cabauw_in.sel(time='2017-07').mean(dim='time')
cerra_original_bias_m = cerra_mrt_cabauw.sel(time='2017-07').mean(dim='time') - bsrn_mrt_cabauw_in.sel(time='2017-07').mean(dim='time')
cerra_empirical_bias_m = cerra_mrt_apx_cabauw.sel(time='2017-07').mean(dim='time') - bsrn_mrt_cabauw_in.sel(time='2017-07').mean(dim='time')

print(era5_bias_m.values, cerra_original_bias_m.values, cerra_empirical_bias_m.values)

# Convert xarray DataArrays to pandas DataFrames
era5_bias_df = era5_bias.to_dataframe(name="bias")
cerra_original_bias_df = cerra_original_bias.to_dataframe(name="bias")
cerra_empirical_bias_df = cerra_empirical_bias.to_dataframe(name="bias")
era5_bias_df['model'] = "ERA5-HEAT"
cerra_original_bias_df['model'] = "CERRA original"
cerra_empirical_bias_df['model'] = "CERRA empirical"

bias_df = pd.concat([era5_bias_df, cerra_original_bias_df, cerra_empirical_bias_df], ignore_index=False)
bias_df = bias_df.reset_index()
bias_df['month'] = bias_df['time'].dt.strftime('%b')

cab = bias_df.copy()
cab['station'] = 'CAB'

tor = bias_df.copy()
tor['station'] = 'TOR'

pay = bias_df.copy()
pay['station'] = 'PAY'

iza = bias_df.copy()
iza['station'] = 'IZA' #station_code

bias_formated = pd.concat([cab, tor, pay, iza], ignore_index=False)

# csza_i = xr.open_dataarray("poster_output/csza_i_jul.nc")
# csza_s = xr.open_dataarray("poster_output/csza_s_jul.nc")
# csza_i_cabauw = csza_i.isel(y=min_distance_idx[0], x=min_distance_idx[1])
# csza_s_cabauw = csza_s.isel(y=min_distance_idx[0], x=min_distance_idx[1])
#
# bsrn_mrt_cabauw_ave = mean_radiant_temperature_from_DIR_DIF_COS(rsds=bsrn1h.SWD,
#                                                        rsus=bsrn1h.SWU,
#                                                        rlds=bsrn1h.LWD,
#                                                        rlus=bsrn1h.LWU,
#                                                        rsds_direct=bsrn1h.DIR,
#                                                        rsds_diffuse=bsrn1h.DIF,
#                                                        csza_i=csza_i_cabauw,
#                                                        csza_s=csza_s_cabauw)
# bsrn_mrt_cabauw_ave = bsrn_mrt_cabauw_ave.resample(time='3H').mean(dim='time')

time = bsrn_mrt_cabauw_in.sel(time=slice('2017-07-01', '2017-07-04')).time #.sel(time=bsrn_mrt_cabauw.time.dt.hour == 12)
plt.figure(figsize=(8,2.5))
plt.rcParams['font.size'] = plt.rcParamsDefault['font.size']
plt.plot(time, bsrn_mrt_cabauw_in.sel(time=slice('2017-07-01', '2017-07-04')).values.reshape(-1)-273.15, label='BSRN', color='y')
# plt.plot(time, bsrn_mrt_cabauw_ave.values.reshape(-1)-273.15, label='BSRN COSave', color='c')
# plt.plot(time, era5_bias.values.reshape(-1), label='ERA5-HEAT', color='r')
# plt.plot(time, cerra_original_bias.values.reshape(-1), label='CERRA original', color='g')
# plt.plot(time, cerra_empirical_bias.values.reshape(-1), label='CERRA empirical', color='b')
plt.plot(time, era_mrt_cabauw.sel(time=slice('2017-07-01', '2017-07-04'))-273.15, label='ERA5-HEAT', color='r')
plt.plot(time, cerra_mrt_cabauw.sel(time=slice('2017-07-01', '2017-07-04'))-273.15, label='CERRA original', color='g')
plt.plot(time, cerra_mrt_apx_cabauw.sel(time=slice('2017-07-01', '2017-07-04'))-273.15, label='CERRA empirical', color='b')
plt.xlabel('Time')
plt.ylabel('MRT in CAB (°C)')
plt.legend()
plt.tight_layout()
plt.savefig(str(PLOTS_DIR / 'MRT_CAB.png'), dpi=1000)

plt.close()

plt.show()

era_rad = xr.open_dataset("poster_input/rad_era5.nc")
era_rad = era_rad.sel(time='2017-01-01')
era_rad = era_rad / 3600

cerra_original_bias_jan = cerra_original_bias.sel(time='2017-01').mean(dim='time')
cerra_original_bias_jul = cerra_original_bias.sel(time='2017-07').mean(dim='time')
cerra_original = [-2.12243931, -1.5820727, -1.22437111, -1.86401933, -2.51381168, 0.54171359, 0.50884134, -0.90407469]

cerra_empirical_bias_jan = cerra_empirical_bias.sel(time='2017-01').mean(dim='time')
cerra_empirical_bias_jul = cerra_empirical_bias.sel(time='2017-07').mean(dim='time')
cerra_empirical = [-2.57897565, -2.47483429, -1.46063932, -2.63172938, -2.87032617, -0.3649205, 0.09065792, -2.15919951]

categories = ['CAB Jan', 'CAB Jul', 'TOR Jan', 'TOR Jul', 'PAY Jan', 'PAY Jul', 'IZA Jan', 'IZA Jul']
era5_heat = [-0.86890794, -1.2926863, -0.60306062, -0.7469185, -0.98096218, -3.00451045, 8.86100739, 2.9475632]
cerra_original = [-2.12243931, -1.5820727, -1.22437111, -1.86401933, -2.51381168, 0.54171359, 1.2634548, -0.0298884]
cerra_empirical = [-2.57897565, -2.47483429, -1.46063932, -2.63172938, -2.87032617, -0.3649205, 0.85641386, -1.18029722]

# Create a scatter plot
plt.figure()
plt.scatter(categories, cerra_original, label='CERRA original', marker='^', color='g')
plt.scatter(categories, cerra_empirical, label='CERRA empirical', marker='s', color='b')
plt.scatter(categories, era5_heat, label='ERA5-HEAT', marker='o', color='r')
# plt.xlabel('Categories')
plt.ylabel('Mean Radiant Temperature Bias')
# plt.title('Scatter Plot with Different Markers')
plt.legend()
plt.tight_layout()
plt.savefig(str(PLOTS_DIR / 'mean_bias_2078.png'), dpi=1000)
plt.show()
plt.close()

bias_formated.to_csv('poster_output/fin/bias_formated.csv', index=False)
# bias_formated = pd.read_csv('poster_output/fin/bias_formated.csv')
def map_model_to_abr(model):
    if model == "CERRA original":
        return "Co"
    elif model == "CERRA empirical":
        return "Сe"
    elif model == "ERA5-HEAT":
        return "Eh"
    else:
        return None
bias_formated['model_abr'] = bias_formated['model'].apply(map_model_to_abr)
bias_formated["station_model"] = bias_formated["station"] + bias_formated["model_abr"]

# Create a scatter plot
import seaborn as sns
for scale in scales:
    plt.figure()
    sns.boxenplot(data=bias_formated, y='bias', x='station_model', hue='month',
                   order=['CAB(Co)', 'CAB(Ce)', 'CAB(Eh)', 'TOR(Co)', 'TOR(Ce)','TOR(Eh)', 'PAY(Co)', 'PAY(Ce)', 'PAY(Eh)', 'IZA(Co)', 'IZA(Ce)', 'IZA(Eh)'], #['CAB', 'TOR', 'PAY', 'IZA']
                   # hue_order=["CERRA original", "CERRA empirical", "ERA5-HEAT"],
                   hue_order=['Jan', 'Jul'])
    plt.ylabel('Bias')
    plt.xlabel('')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(str(PLOTS_DIR / 'bias_3h_box.png'), dpi=1000)
    plt.close()


'''
target_latitude = 51.9680
target_longitude = 4.9280

# Get latitude and longitude coordinates from the DataArray
latitudes = cerra_mrt.latitude.values
longitudes = cerra_mrt.longitude.values
import numpy as np
# Calculate the distances using Haversine formula
R = 6371  # Earth radius in km
lat_diff = np.radians(latitudes - target_latitude)
lon_diff = np.radians(longitudes - target_longitude)
a = np.sin(lat_diff / 2) ** 2 + np.cos(np.radians(target_latitude)) * np.cos(np.radians(latitudes)) * np.sin(lon_diff / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distances = R * c
# Find the index of the minimum distance
min_distance_index = np.unravel_index(np.argmin(distances), distances.shape)
# Get the nearest grid point's orog value
nearest_orog = cerra_mrt[min_distance_index]

print("Nearest orog value:", nearest_orog)
print("Nearest latitude:", latitudes[min_distance_index])
print("Nearest longitude:", longitudes[min_distance_index])


target_latitude = 51.9680
target_longitude = 4.9280

# Find the nearest grid point using xarray's sel() function
nearest_point = orog.sel(latitude=51.98915252148688, longitude=4.948920323953786, method='nearest')

nearest_latitude = orog.latitude.interp(y=target_latitude)
nearest_longitude = orog.longitude.interp(x=target_longitude)
nearest_point = orog.interp(latitude=target_latitude, longitude=target_longitude)
'''
# Define the Lambert Conformal projection with parameters according to CERRA dataset
cerra_proj = ccrs.LambertConformal(
    central_longitude=8,
    central_latitude=50,
    standard_parallels=(50, 50),
    globe=ccrs.Globe(ellipse='sphere', semimajor_axis=6371229)
)

#upload Land Sea Mask and Orography
# f1 = xr.open_dataset("radiation_from_thermofeel.nc")
f1 = xr.open_dataset("meteo_test.nc")
f1 = f1.assign_coords(longitude=(((f1.longitude + 180) % 360) - 180))
# cerra_lsm = f1.lsm.isel(time=28)
cerra_or = f1.orog.isel(time=28)

station_code = 'IZA'
target_latitude = 28.29 #from 28.3093    to ------, 28.29, 28.28
target_longitude = -16.58 #from -16.4993 to -16.53, 16.58, 16.63

lat_diff_sq = (cerra_or.latitude - target_latitude)**2
lon_diff_sq = (cerra_or.longitude - target_longitude)**2
distance_sq = lat_diff_sq + lon_diff_sq
min_distance_idx = np.unravel_index(distance_sq.argmin(), distance_sq.shape)
cerra_or_sel = cerra_or.isel(y=min_distance_idx[0], x=min_distance_idx[1])

era5_gp = xr.open_dataarray("poster_input/era5_geopotential.nc").sel(time="2017-01-01")
era5_gp = era5_gp.assign_coords(longitude=(((era5_gp.longitude + 180) % 360) - 180))
era5_or = era5_gp/9.80665 # The (surface) geopotential height (orography) can be calculated by dividing the (surface) geopotential by the Earth's gravitational acceleration, g (=9.80665 m s-2 ).
era5_or_sel = era5_or.sel(latitude=28.25, longitude=-16.5, method="nearest")
era5_or_sel = era5_or.sel(latitude=target_latitude, longitude=target_longitude, method="nearest")

era5_gp = xr.open_dataset("poster_input/era5_geopotential2.nc")
era5_gp = era5_gp.assign_coords(longitude=(((era5_gp.longitude + 180) % 360) - 180))
era5_or = era5_gp.z/9.80665

lat_diff_sq = (era5_or.latitude - target_latitude)**2
lon_diff_sq = (era5_or.longitude - target_longitude)**2
distance_sq = lat_diff_sq + lon_diff_sq
min_distance_idx = np.unravel_index(distance_sq.argmin(), distance_sq.shape)
era5_or_sel = era5_or.isel(latitude=min_distance_idx[0], longitude=min_distance_idx[1])

era_comp = xr.open_dataset('poster_input/comp_era5.nc')['t2m'].sel(time="2017-07-01T12")-273.15
era_comp_sel = era_comp.sel(latitude=target_latitude, longitude=target_longitude, method="nearest")
# plot
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
fig = plt.figure()
# ax = plt.axes(projection=cerra_proj)
ax = plt.axes(projection=ccrs.PlateCarree())

ax.coastlines(resolution='50m')
# ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.BORDERS, linestyle='--', edgecolor='black')
# plt.contour(lsm.longitude, lsm.latitude, lsm, levels=[0.3, 1], colors='black', transform=ccrs.PlateCarree())
norm = colors.BoundaryNorm(boundaries=[0, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], ncolors=256, extend='both')
# norm = colors.BoundaryNorm(boundaries=[1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, ], ncolors=256, extend='both')
# filled_c = plt.pcolormesh(cerra_or.longitude, cerra_or.latitude, cerra_or, transform=ccrs.PlateCarree(), cmap="terrain")
filled_c = plt.pcolormesh(era5_or.longitude, era5_or.latitude, era5_or, transform=ccrs.PlateCarree(), cmap="terrain", norm=norm, shading='auto') #, vmin=-50, vmax=50 , norm=norm

# norm = colors.BoundaryNorm(boundaries=[17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 21, 22], ncolors=256, extend='both')
# filled_c = plt.pcolormesh(era_comp.longitude, era_comp.latitude, era_comp, transform=ccrs.PlateCarree(), cmap="terrain", vmin=18, vmax=21, norm=norm) #, vmin=-50, vmax=50 , norm=norm

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
# ax.set_extent([-19, 35, 25, 74], crs=ccrs.PlateCarree())
ax.margins(0, x=None, y=None, tight=True)
# Plot station markers
for station, loc in locations.items():
    ax.plot(loc['longitude'], loc['latitude'], marker='o', color='red', markersize=8, transform=ccrs.PlateCarree())
    ax.text(loc['longitude'] + 1, loc['latitude'], station, color='black', transform=ccrs.PlateCarree())

cbar = fig.colorbar(filled_c, orientation="vertical", pad=0.08)
cbar.set_label('Orography (m)')
plt.tight_layout()
plt.show()

plt.close()

locations = {
    'CAB': {'latitude': 51.968, 'longitude': 4.928},
    'TOR': {'latitude': 58.2641, 'longitude': 26.4613},
    'PAY': {'latitude': 46.8123, 'longitude': 6.9422},
    'IZA': {'latitude': 28.3093, 'longitude': -16.4993},
    'IZA-CERRA': {'latitude': 28.3093, 'longitude': -16.53},
    'IZA-CERRA2': {'latitude': 28.29, 'longitude': -16.58},
    'IZA-CERRA3': {'latitude': 28.28, 'longitude': -16.63},
    'IZA-ERA': {'latitude': 28.25, 'longitude': -16.5},
    '807.49': {'latitude': 34.25, 'longitude': 51.5}
}

# filled_c = plt.pcolormesh(orog.longitude, orog.latitude, orog, transform=ccrs.PlateCarree(), cmap="terrain", norm=norm)
# contour = plt.contour(orog.longitude, orog.latitude, orog, levels=10, colors='black', transform=ccrs.PlateCarree())
# plt.clabel(contour, inline=True, fontsize=8, colors='black', fmt='%1.0f')  # Add contour labels

# fig.savefig('output_fig/integrated/c_mrt_2017_jan_min', dpi=600)
plt.close()

import thermofeel
f = bsrn
f['ssr'] = f.SWD - f.SWU
f['str'] = f.SWD - f.SWU
cerra_mrt_tf = xr.DataArray(np.zeros_like(f.SWD), dims=f.SWD.dims, coords=f.SWD.coords)

for timestamp in np.unique(f.time.values):
    cossza_integrated = thermofeel.calculate_cos_solar_zenith_angle_integrated(lon=f.longitude.values, lat=f.latitude.values,
                                                                               y=timestamp.astype('datetime64[Y]').astype(int) + 1970,
                                                                               m=timestamp.astype('datetime64[M]').astype(int) % 12 + 1,
                                                                               d=timestamp.astype('datetime64[D]').astype(int) % 31 + 1,
                                                                               h=timestamp.astype('datetime64[h]').astype(int) % 24,
                                                                               tbegin=timestamp.astype('datetime64[h]').astype(int) % 24,
                                                                               tend=timestamp.astype('datetime64[h]').astype(int) % 24 + 3)

    ssrd = f.SWD.sel(time=timestamp) # Surface solar radiation downwards [W/m^-2]
    ssr = f.ssr.sel(time=timestamp) # Surface net solar radiation [W/m^-2]
    strd = f.LWD.sel(time=timestamp) # Surface thermal radiation downwards [W/m^-2]
    fdir = f.DIR.sel(time=timestamp) # Time-integrated surface direct short wave radiation flux - counterpart of the Total sky direct solar radiation at surface [W m-2]
    strr = f.str.sel(time=timestamp) # Surface net thermal radiation [W/m^-2]

    # testing fdir
    approx_dsrp = thermofeel.approximate_dsrp(fdir=fdir.values, cossza=cossza_integrated) # approximation of the Direct radiation from the Sun [W/m^-2]
    mrt = thermofeel.calculate_mean_radiant_temperature(ssrd=ssrd.values, ssr=ssr.values, dsrp=approx_dsrp,
                                                        strd=strd.values, fdir=fdir.values, strr=strr.values,
                                                        cossza=cossza_integrated) # calculate MRT in K
    cerra_mrt_tf.loc[{'time': timestamp}] = mrt
    print('Timestamp: {}'.format(timestamp))
    mrt_c = mrt - 273.15 # convert MRT to °C