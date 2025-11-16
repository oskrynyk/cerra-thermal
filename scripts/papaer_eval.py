# MRT and UTCI evaluation agains BSRN observations

import os
os.chdir('C:\\Users\\olehs\\cerra-thermal')
print(" ".join(['\nCurrent path is', os.getcwd()]))
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

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
target_latitude = 28.29 #from 28.3093  to 28.3093, 28.29, 28.28
target_longitude = -16.58 #from -16.4993 to -16.53, 16.58, 16.63


csza_thr = 0.07
from sklearn.metrics import mean_squared_error

csza_threshholds = [0.001, 0.01, 0.05, 0.1]
for csza_thr in csza_threshholds:

station_codes = ['CAB', 'TOR', 'PAY', 'IZA']
target_latitudes = [51.968, 58.2641, 46.8123, 28.29]
target_longitudes = [4.928, 26.4613, 6.9422, -16.58]
for station_code, target_latitude, target_longitude in zip(station_codes, target_latitudes, target_longitudes):
    path = 'era5/'
    file_pattern = 'ECMWF_mrt_*.nc'
    era_mrt = xr.open_mfdataset(f'{path}/{file_pattern}', combine='by_coords')['mrt'].sel(time='2017')
    era_mrt = era_mrt.rename({'lon': 'longitude', 'lat': 'latitude'})
    era_mrt = era_mrt.sel(time='2017-' + month) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    era_mrt_sel = era_mrt.sel(latitude=target_latitude, longitude=target_longitude, method="nearest")

    # Calculate the squared distances between target and dataset coordinates, the squared distance from the target point
    # and find the index of the minimum squared distance from defined xr
    lat_diff_sq = (rad.latitude - target_latitude)**2
    lon_diff_sq = (rad.longitude - target_longitude)**2
    distance_sq = lat_diff_sq + lon_diff_sq
    min_distance_idx = np.unravel_index(distance_sq.argmin(), distance_sq.shape)
    # Extract data at the closest grid point from defined xrs
    rad_sel = rad.isel(y=min_distance_idx[0], x=min_distance_idx[1])
    sunlit_sel = csza_sunlit.isel(y=min_distance_idx[0], x=min_distance_idx[1])
    integ_sel = csza_integ.isel(y=min_distance_idx[0], x=min_distance_idx[1])
    cerra_mrt_sunlit_sel = mean_radiant_temperature(rsds=rad_sel.ssrd, rsus=rad_sel.ssru, rlds=rad_sel.strd, rlus=rad_sel.stru,
                                                    csza=sunlit_sel, csza_thr=csza_thr, rsds_direct=rad_sel.tidirswrf)
    cerra_mrt_integ_sel = mean_radiant_temperature(rsds=rad_sel.ssrd, rsus=rad_sel.ssru, rlds=rad_sel.strd, rlus=rad_sel.stru,
                                                   csza=integ_sel, csza_thr=csza_thr, rsds_direct=rad_sel.tidirswrf)

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
    bsrn = xr.Dataset.from_dataframe(df).sel(time='2017-' + month) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    bsrn["DIFe"] = bsrn.SWD - bsrn.DIR
    variables = ["SWD", "DIR", "DIF", "DIFe", "LWD", "SWU", "LWU"]
    for variable in variables:
        bsrn[variable].attrs.update({"units": "W m-2"})
    bsrn.longitude.attrs["units"] = "degrees"
    bsrn.latitude.attrs["units"] = "degrees"

    bsrn_mrt_inst = mean_radiant_temperature_2(rsds=bsrn.SWD, rsus=bsrn.SWU, rlds=bsrn.LWD, rlus=bsrn.LWU, rsds_direct=bsrn.DIR, rsds_diffuse=bsrn.DIF)
    bsrn_mrt = bsrn_mrt_inst.resample(time='1H').mean(dim='time')

    # bsrn_mrt = bsrn_mrt.resample(time='3H').mean(dim='time')
    # era_mrt_sel = era_mrt_sel.resample(time='3H').mean(dim='time')
    # cerra_mrt_sunlit_sel = cerra_mrt_sunlit_sel.resample(time='3H').mean(dim='time')
    # cerra_mrt_integ_sel = cerra_mrt_integ_sel.resample(time='3H').mean(dim='time')
    # cerra_mrt_sunlit_apx_sel = cerra_mrt_sunlit_apx_sel.resample(time='3H').mean(dim='time')
    # cerra_mrt_integ_apx_sel = cerra_mrt_integ_apx_sel.resample(time='3H').mean(dim='time')

    # bsrn_mrt = bsrn_mrt.sel(time=bsrn_mrt.time.dt.hour == 12)
    # era_mrt_sel = era_mrt_sel.sel(time=era_mrt_sel.time.dt.hour == 12)
    # cerra_mrt_sunlit_sel = cerra_mrt_sunlit_sel.sel(time=cerra_mrt_sunlit_sel.time.dt.hour == 12)
    # cerra_mrt_integ_sel = cerra_mrt_integ_sel.sel(time=cerra_mrt_integ_sel.time.dt.hour == 12)
    # cerra_mrt_sunlit_apx_sel = cerra_mrt_sunlit_apx_sel.sel(time=cerra_mrt_sunlit_apx_sel.time.dt.hour == 12)
    # cerra_mrt_integ_apx_sel = cerra_mrt_integ_apx_sel.sel(time=cerra_mrt_integ_apx_sel.time.dt.hour == 12)

    def handle(*data_arrays):
        # Create a mask for each data array to identify non-finite values
        finite_masks = [np.isfinite(data_array) for data_array in data_arrays]
        # Combine the masks using logical AND to identify timestamps with non-finite values in any array
        combined_mask = np.all(np.vstack(finite_masks), axis=0)
        # Apply the combined mask to each data array
        finite_data_arrays = [data_array[combined_mask] for data_array in data_arrays]
        return finite_data_arrays

    # Apply handle_non_finite to all data arrays
    _bsrn_mrt, _era_mrt_sel, _cerra_mrt_sunlit_sel, _cerra_mrt_integ_sel = handle(
        bsrn_mrt.values.reshape(-1),
        era_mrt_sel.values,
        cerra_mrt_sunlit_sel.values,
        cerra_mrt_integ_sel.values,
    )

    era5_er = mean_squared_error(_bsrn_mrt, _era_mrt_sel, squared=True)
    cerra_er_s = mean_squared_error(_bsrn_mrt, _cerra_mrt_sunlit_sel, squared=True)
    cerra_er_i = mean_squared_error(_bsrn_mrt, _cerra_mrt_integ_sel, squared=True)
    # era5_er = mean_squared_error(_bsrn_mrt, _era_mrt_sel, squared=False)
    # cerra_er_s = mean_squared_error(_bsrn_mrt, _cerra_mrt_sunlit_sel, squared=False)
    # cerra_er_i = mean_squared_error(_bsrn_mrt, _cerra_mrt_integ_sel, squared=False)

    print(f"MSE for {station_code, csza_thr}: ERA5={era5_er}, CERRA sunlit={cerra_er_s}, CERRA integ={cerra_er_i}")

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
target_latitude = 28.29 #from 28.3093  to 28.3093, 28.29, 28.28
target_longitude = -16.58 #from -16.4993 to -16.53, 16.58, 16.63

month = '01'

station_codes = ['CAB', 'TOR', 'PAY', 'IZA']
target_latitudes = [51.968, 58.2641, 46.8123, 28.29]
target_longitudes = [4.928, 26.4613, 6.9422, -16.58]
for station_code, target_latitude, target_longitude in zip(station_codes, target_latitudes, target_longitudes):
    print(f"Processeding station: {station_code}")
    path = 'era5/'
    file_pattern = 'ECMWF_mrt_*.nc'
    era_mrt = xr.open_mfdataset(f'{path}/{file_pattern}', combine='by_coords')['mrt'].sel(time='2017')
    era_mrt = era_mrt.rename({'lon': 'longitude', 'lat': 'latitude'})
    era_mrt = era_mrt.sel(time='2017-' + month) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    era_mrt_sel = era_mrt.sel(latitude=target_latitude, longitude=target_longitude, method="nearest")

    # Calculate the squared distances between target and dataset coordinates, the squared distance from the target point
    # and find the index of the minimum squared distance from defined xr
    lat_diff_sq = (cerra_mrt_sunlit.latitude - target_latitude)**2
    lon_diff_sq = (cerra_mrt_sunlit.longitude - target_longitude)**2
    distance_sq = lat_diff_sq + lon_diff_sq
    min_distance_idx = np.unravel_index(distance_sq.argmin(), distance_sq.shape)
    # Extract data at the closest grid point from defined xrs
    cerra_mrt_sunlit_sel = cerra_mrt_sunlit.isel(y=min_distance_idx[0], x=min_distance_idx[1])+273.15
    cerra_mrt_integ_sel = cerra_mrt_integ.isel(y=min_distance_idx[0], x=min_distance_idx[1])+273.15
    cerra_mrt_sunlit_apx_sel = cerra_mrt_sunlit_apx.isel(y=min_distance_idx[0], x=min_distance_idx[1])+273.15
    cerra_mrt_integ_apx_sel = cerra_mrt_integ_apx.isel(y=min_distance_idx[0], x=min_distance_idx[1])+273.15


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
    bsrn = xr.Dataset.from_dataframe(df).sel(time='2017-' + month) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    bsrn["DIFe"] = bsrn.SWD - bsrn.DIR
    variables = ["SWD", "DIR", "DIF", "DIFe", "LWD", "SWU", "LWU"]
    for variable in variables:
        bsrn[variable].attrs.update({"units": "W m-2"})
    bsrn.longitude.attrs["units"] = "degrees"
    bsrn.latitude.attrs["units"] = "degrees"

    bsrn_mrt_inst = mean_radiant_temperature_2(rsds=bsrn.SWD, rsus=bsrn.SWU, rlds=bsrn.LWD, rlus=bsrn.LWU, rsds_direct=bsrn.DIR, rsds_diffuse=bsrn.DIF)
    bsrn_mrt = bsrn_mrt_inst.resample(time='1H').mean(dim='time')

    # bsrn_mrt_avg = bsrn_mrt.mean(dim='time').values-273.15
    # bsrn_mrt_min = bsrn_mrt.resample(time='1D').min(dim='time').mean(dim='time').values-273.15
    # bsrn_mrt_max = bsrn_mrt.resample(time='1D').max(dim='time').mean(dim='time').values-273.15
    # bsrn_mrt_abs = bsrn_mrt.resample(time='1D').max(dim='time').max(dim='time').values - 273.15
    # print(f"MSE for {station_code}: AVG={bsrn_mrt_avg}, MIN={bsrn_mrt_min}, MAX={bsrn_mrt_max}, ABS={bsrn_mrt_abs}")

    bsrn_mrt = bsrn_mrt.resample(time='3H').mean(dim='time')
    era_mrt_sel = era_mrt_sel.resample(time='3H').mean(dim='time')
    cerra_mrt_integ_sel = cerra_mrt_integ_sel.resample(time='3H').mean(dim='time')
    cerra_mrt_sunlit_sel = cerra_mrt_sunlit_sel.resample(time='3H').mean(dim='time')
    cerra_mrt_integ_apx_sel = cerra_mrt_integ_apx_sel.resample(time='3H').mean(dim='time')
    cerra_mrt_sunlit_apx_sel = cerra_mrt_sunlit_apx_sel.resample(time='3H').mean(dim='time')

    # bsrn_mrt = bsrn_mrt.sel(time=bsrn_mrt.time.dt.hour == 12)
    # era_mrt_sel = era_mrt_sel.sel(time=era_mrt_sel.time.dt.hour == 12)
    # cerra_mrt_sunlit_sel = cerra_mrt_sunlit_sel.sel(time=cerra_mrt_sunlit_sel.time.dt.hour == 12)
    # cerra_mrt_integ_sel = cerra_mrt_integ_sel.sel(time=cerra_mrt_integ_sel.time.dt.hour == 12)
    # cerra_mrt_sunlit_apx_sel = cerra_mrt_sunlit_apx_sel.sel(time=cerra_mrt_sunlit_apx_sel.time.dt.hour == 12)
    # cerra_mrt_integ_apx_sel = cerra_mrt_integ_apx_sel.sel(time=cerra_mrt_integ_apx_sel.time.dt.hour == 12)

    # bsrn_mrt = bsrn_mrt.resample(time='1D').min()
    # era_mrt_sel = era_mrt_sel.resample(time='1D').min()
    # cerra_mrt_sunlit_sel = cerra_mrt_sunlit_sel.resample(time='1D').min()
    # cerra_mrt_integ_sel = cerra_mrt_integ_sel.resample(time='1D').min()
    # cerra_mrt_sunlit_apx_sel = cerra_mrt_sunlit_apx_sel.resample(time='1D').min()
    # cerra_mrt_integ_apx_sel = cerra_mrt_integ_apx_sel.resample(time='1D').min()

    def handle_non_finite(*data_arrays):
        # Create a mask for each data array to identify non-finite values
        finite_masks = [np.isfinite(data_array) for data_array in data_arrays]
        # Combine the masks using logical AND to identify timestamps with non-finite values in any array
        combined_mask = np.all(np.vstack(finite_masks), axis=0)
        # Apply the combined mask to each data array
        finite_data_arrays = [data_array[combined_mask] for data_array in data_arrays]
        return finite_data_arrays

    # Apply handle_non_finite to all data arrays
    _bsrn_mrt, _era_mrt_sel, _cerra_mrt_sunlit_sel, _cerra_mrt_integ_sel, _cerra_mrt_sunlit_apx_sel, _cerra_mrt_integ_apx_sel = handle_non_finite(
        bsrn_mrt.values.reshape(-1),
        era_mrt_sel.values,
        cerra_mrt_sunlit_sel.values,
        cerra_mrt_integ_sel.values,
        cerra_mrt_sunlit_apx_sel.values,
        cerra_mrt_integ_apx_sel.values,
    )

    x = _bsrn_mrt - 273.15
    y_values = [_era_mrt_sel - 273.15, _cerra_mrt_integ_sel - 273.15, _cerra_mrt_sunlit_sel - 273.15,
                _cerra_mrt_integ_apx_sel - 273.15, _cerra_mrt_sunlit_apx_sel - 273.15]
    labels = ['ERA5-HEAT', 'CERRA (quadrature)', 'CERRA (sunlit average)', 'CERRA (quadrature, calc. fdir)',
              'CERRA (sunlit average, calc. fdir)']
    kind = ['+', 'o', 'x', 's', 'd']
    colorm = ['r', 'b', 'g', 'c', 'm']

    plt.figure(figsize=(5, 5), constrained_layout=True)
    for i, (y, marker) in enumerate(zip(y_values, kind)):
        if marker in ['o', 's', 'd']:
            plt.scatter(x, y, label=f"{labels[i]}: {np.mean(y - x):.2f}°C", marker=marker, edgecolor=colorm[i], facecolor='none', alpha=1)
        else:
            plt.scatter(x, y, label=f"{labels[i]}: {np.mean(y - x):.2f}°C", marker=marker, color=colorm[i], alpha=1)
    plt.xlabel(f"{station_code} MRT (°C)")
    plt.ylabel('NWPs MRT (°C)')
    plt.legend(loc='upper left', framealpha=0)
    plt.savefig('paper/scatter/scatter_' + month + '_!3h_' + station_code + '.png', dpi=1000)
    # plt.show()

    # plt.figure(figsize=(5, 5), constrained_layout=True)
    # for i, y in enumerate(y_values):
    #     plt.scatter(x, y, label=f"{labels[i]} Bias: {np.mean(y - x):.2f}°C", marker=kind[i], color=colorm[i], facecolor='none')
    # plt.xlabel('BSRN MRT (°C)')
    # plt.ylabel('NWPs MRT (°C)')
    # plt.legend(loc='upper left', framealpha=0)
    # plt.savefig('paper/scatter/scatter_' + month + '_3h_' + station_code + '.png', dpi=1000)
    # # plt.show()

    import skill_metrics as sm
    # Calculate Taylor statistics using the matched timestamps
    ts_era = sm.taylor_statistics(_era_mrt_sel, _bsrn_mrt, 'data')
    ts_cerra_i = sm.taylor_statistics(_cerra_mrt_integ_sel, _bsrn_mrt, 'data')
    ts_cerra_s = sm.taylor_statistics(_cerra_mrt_sunlit_sel, _bsrn_mrt, 'data')
    ts_cerra_i_apx = sm.taylor_statistics(_cerra_mrt_integ_apx_sel, _bsrn_mrt, 'data')
    ts_cerra_s_apx = sm.taylor_statistics(_cerra_mrt_sunlit_apx_sel, _bsrn_mrt, 'data')

    sdev = np.array([ts_era['sdev'][0], ts_era['sdev'][1],  ts_cerra_i['sdev'][1], ts_cerra_s['sdev'][1], ts_cerra_i_apx['sdev'][1], ts_cerra_s_apx['sdev'][1]])
    crmsd = np.array([ts_era['crmsd'][0], ts_era['crmsd'][1], ts_cerra_i['crmsd'][1], ts_cerra_s['crmsd'][1], ts_cerra_i_apx['crmsd'][1], ts_cerra_s_apx['crmsd'][1]])
    ccoef = np.array([ts_era['ccoef'][0], ts_era['ccoef'][1], ts_cerra_i['ccoef'][1], ts_cerra_s['ccoef'][1], ts_cerra_i_apx['ccoef'][1], ts_cerra_s_apx['ccoef'][1]])

    # FONT_FAMILY = 'Times New Roman'
    # FONT_SIZE = 9
    # plt.rcParams.update({'font.size': FONT_SIZE, 'font.family': FONT_FAMILY})

    # label = ['BSRN', 'ERA5-HEAT', 'CERRA α', 'CERRA β', 'CERRA γ', 'CERRA δ']
    # label = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
    label = ['BSRN', 'ERA5-HEAT',  'CERRA (quadrature)', 'CERRA (sunlit average)', 'CERRA (quadrature, calc. fdir)', 'CERRA (sunlit average, calc. fdir)']
    intervalsCOR = np.concatenate((np.arange(0,1.0,0.2),[0.9, 0.95, 0.99, 1]))
    COLS_COR = {'grid': '#DDDDDD', 'tick_labels': '#000000', 'title': '#000000'} # specify some styles for the correlation component
    COLS_STD = { 'grid': '#DDDDDD', 'tick_labels': '#000000', 'ticks': '#DDDDDD', 'title': '#000000'} # specify some styles for the standard deviation
    # STYLES_RMS = {'color': '#AAAADD','linestyle': '--'} # # specify some styles for the root mean square deviation

    plt.figure(figsize=(3, 3), constrained_layout=True)
    sm.taylor_diagram(sdev, crmsd, ccoef, markerLabel=label, alpha=0.0,
                      tickCOR=intervalsCOR,
                      styleOBS=':', colOBS='k', markerOBS='o', titleOBS=station_code+' Obsv.',
                      markerLegend='on', colframe='#DDDDDD',
                      titleCOR='on',
                      colsCOR=COLS_COR,
                      colsSTD=COLS_STD,
                      styleCOR='-',
                      titleSTD='on',
                      styleSTD='-',
                      # rincstd=[0, 10, 20, 30],
                      # tickSTD=[0, 10, 20, 30],
                      titleRMS='off',
                      showlabelsRMS='on', RMSLabelFormat=':.0f', tickRMS=[5, 10, 15], #tickRMS=[5, 10, 20, 30], #tickRMS=[5, 10, 15, 20],
                      colRMS='tab:grey', styleRMS='--', tickRMSangle=115, #IZA 125 130
                      labelweight='normal',
                      titlecorshape='linear',
                      )
    def format_ticks(x, pos):
        return f"{x:.0f}"
    from matplotlib.ticker import FuncFormatter
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks))
    # plt.gca().set_xticks([0, 10, 20, 30])
    # plt.gca().set_yticks([0, 10, 20, 30])

    plt.gca().get_legend().get_frame().set_alpha(1.0)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles, labels, loc='upper right')

    # plt.tight_layout()
    plt.show()
    plt.savefig('paper/taylor/taylor_' + month + '_3h_' + station_code + '3.png', dpi=1000) #_' + station_code + '2
    plt.close()

month = '01'

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
target_latitude = 28.29  # from 28.3093  to 28.3093, 28.29, 28.28
target_longitude = -16.58  # from -16.4993 to -16.53, 16.58, 16.63

    # Append the statistics for each model to the corresponding lists
    all_sdev.append([ts_era['sdev'][0], ts_era['sdev'][1], ts_cerra_s['sdev'][1], ts_cerra_i['sdev'][1],
                     ts_cerra_s_apx['sdev'][1], ts_cerra_i_apx['sdev'][1]])
    all_crmsd.append([ts_era['crmsd'][0], ts_era['crmsd'][1], ts_cerra_s['crmsd'][1], ts_cerra_i['crmsd'][1],
                      ts_cerra_s_apx['crmsd'][1], ts_cerra_i_apx['crmsd'][1]])
    all_ccoef.append([ts_era['ccoef'][0], ts_era['ccoef'][1], ts_cerra_s['ccoef'][1], ts_cerra_i['ccoef'][1],
                      ts_cerra_s_apx['ccoef'][1], ts_cerra_i_apx['ccoef'][1]])

# Convert lists to numpy arrays for easy computation
all_sdev_np = np.array(all_sdev)
all_crmsd_np = np.array(all_crmsd)
all_ccoef_np = np.array(all_ccoef)

# Compute the mean of the statistics across all stations
avg_sdev = np.mean(all_sdev_np, axis=0)
avg_crmsd = np.mean(all_sdev_np, axis=0)
avg_ccoef = np.mean(all_sdev_np, axis=0)

plt.figure(figsize=(3, 3))
sm.taylor_diagram(avg_sdev, avg_crmsd, avg_ccoef, markerLabel=label, alpha=0.0,
                  tickCOR=intervalsCOR,
                  styleOBS=':', colOBS='k', markerOBS='o', titleOBS='BSRN Obsv.',
                  markerLegend='on', colframe='#DDDDDD',
                  titleCOR='on',
                  colsCOR=COLS_COR,
                  colsSTD=COLS_STD,
                  styleCOR='-',
                  titleSTD='on',
                  styleSTD='-',
                  # rincstd=[0, 10, 20, 30],
                  # tickSTD=[0, 10, 20, 30],
                  titleRMS='off',
                  showlabelsRMS='on', RMSLabelFormat=':.0f', tickRMS=[5, 10, 20],  # tickRMS=[5, 10, 15, 20],
                  colRMS='tab:grey', styleRMS='--', tickRMSangle=125,
                  labelweight='normal',
                  titlecorshape='linear',
                  )


def format_ticks(x, pos):
    return f"{x:.0f}"
from matplotlib.ticker import FuncFormatter
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks))
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks))
# plt.gca().set_xticks([0, 10, 20, 30])
# plt.gca().set_yticks([0, 10, 20, 30])

# plt.gca().get_legend().get_frame().set_alpha(1.0)

handles, labels = plt.gca().get_legend_handles_labels()
plt.gca().legend(handles, labels, loc='upper right')

plt.tight_layout()
plt.savefig('paper/taylor/taylor_' + month + '_3h.png', dpi=1000)
plt.close()


station1_data = [9, 5, 4, 5, 8,]  # Time series data for station 1
station2_data = [3, 5, 8, 5, 2,]  # Time series data for station 2
station3_data = [5, 5, 8, 5, 4,]  # Time series data for station 3
station4_data = [4, 5, 9, 3, 2,]  # Time series data for station 4

station_data = np.array([station1_data, station2_data, station3_data, station4_data])

station_data_t = station_data.T
