"""Thermal comfort index calculations and meteorological functions."""

import numpy as np
from numba import vectorize, float32

@vectorize([float32(float32, float32)], target='parallel')
def wind_estimation(va, h):
    """Estimate wind speed at height h using logarithmic wind profile."""
    v = va * np.log10(h / 0.01) * 0.333333333333
    return v


@vectorize([float32(float32, float32, float32)], target='parallel')
def calculate_normal_effective_temperature(t2m, rh, v):
    """Calculate Normal Effective Temperature (NET) from temperature, humidity, wind."""
    net = (37 -
           ((37 - t2m) /
            (0.68 - 0.0014 * rh +
             (1 / (1.76 + 1.4 * v ** 0.75)))) -
           (0.29 * t2m * (1 - 0.01 * rh)))
    return net


def tif_to_shapefile(tif_path, shapefile_path):
    """Convert classified TIF to Shapefile."""
    import rioxarray
    import geopandas as gpd
    from shapely.geometry import Polygon
    from rasterio.features import shapes
    import rasterio
    import os

    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(shapefile_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

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

            gdf.to_file(shapefile_path)
            print(f"Shapefile saved to: {shapefile_path}")

    except rasterio.RasterioIOError as e:
        print(f"Error opening or reading the TIF file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# ============================================================================
# ADVANCED THERMAL ANALYSIS FUNCTIONS
# ============================================================================

def apparent_temperature(t2m, rh, windspeed, mrt=None, elevation=0):
    """
    Calculate Apparent Temperature (AT) / Feels-like temperature.
    
    Combines air temperature, humidity, wind, and radiation into perceived temperature.
    Based on Steadman approach.
    
    Parameters
    ----------
    t2m : ndarray
        2m air temperature [°C]
    rh : ndarray
        Relative humidity [%]
    windspeed : ndarray
        Wind speed [m/s]
    mrt : ndarray, optional
        Mean Radiant Temperature [°C]
    elevation : float
        Elevation above sea level [m]
    
    Returns
    -------
    ndarray
        Apparent Temperature [°C]
    """
    # Wind chill effect
    v_ms = windspeed
    wind_chill = t2m - 0.55 * (1 - 0.02 * rh / 100) * (10 * np.sqrt(v_ms) - v_ms - 4.8) * (t2m - 10) / 22
    
    # Humidity effect
    es = 6.112 * np.exp((17.67 * t2m) / (t2m + 243.5))
    rh_vp = rh / 100 * es
    humidity_effect = -0.29 * rh_vp / 100 * (t2m + 273.15)
    
    AT = t2m + wind_chill + humidity_effect
    
    # Radiation effect if MRT provided
    if mrt is not None:
        radiation_effect = 0.1 * (mrt - t2m)
        AT = AT + radiation_effect
    
    return AT


def wind_power_density(windspeed, air_density=1.225):
    """
    Calculate wind power density.
    
    Parameters
    ----------
    windspeed : ndarray
        Wind speed [m/s]
    air_density : float
        Air density [kg/m³], default 1.225 at sea level 15°C
    
    Returns
    -------
    ndarray
        Wind power density [W/m²]
    """
    return 0.5 * air_density * windspeed ** 3


def compute_cooling_degree_days(t2m, base_temp=18.3):
    """
    Compute Cooling Degree Days (CDD).
    
    Measure of how much cooling is required.
    
    Parameters
    ----------
    t2m : ndarray or xr.DataArray
        Daily mean temperature [°C]
    base_temp : float
        Base temperature for calculation [°C]
    
    Returns
    -------
    ndarray or float
        Accumulated cooling degree days
    """
    return np.maximum(0, t2m - base_temp)


def compute_heating_degree_days(t2m, base_temp=18.3):
    """
    Compute Heating Degree Days (HDD).
    
    Measure of how much heating is required.
    
    Parameters
    ----------
    t2m : ndarray or xr.DataArray
        Daily mean temperature [°C]
    base_temp : float
        Base temperature for calculation [°C]
    
    Returns
    -------
    ndarray or float
        Accumulated heating degree days
    """
    return np.maximum(0, base_temp - t2m)


def human_comfort_index(t2m, rh, windspeed, radiation=None):
    """
    Calculate Human Comfort Index (HCI).
    
    Combines multiple factors into single comfort metric (0-100).
    100 = perfect comfort, 0 = extreme discomfort
    
    Parameters
    ----------
    t2m : ndarray
        Temperature [°C]
    rh : ndarray
        Relative humidity [%]
    windspeed : ndarray
        Wind speed [m/s]
    radiation : ndarray, optional
        Solar radiation [W/m²]
    
    Returns
    -------
    ndarray
        Comfort index (0-100)
    """
    # Temperature comfort (optimal ~20-24°C)
    t_comfort = np.exp(-((t2m - 22) ** 2) / 50)
    
    # Humidity comfort (optimal 40-60%)
    rh_comfort = np.exp(-((rh - 50) ** 2) / 300)
    
    # Wind comfort (optimal 2-4 m/s)
    wind_comfort = np.exp(-((windspeed - 3) ** 2) / 10)
    
    # Radiation comfort (if provided)
    if radiation is not None:
        rad_comfort = np.exp(-radiation / 500)
        HCI = 100 * (0.4 * t_comfort + 0.3 * rh_comfort + 0.15 * wind_comfort + 0.15 * rad_comfort)
    else:
        HCI = 100 * (0.4 * t_comfort + 0.35 * rh_comfort + 0.25 * wind_comfort)
    
    return np.clip(HCI, 0, 100)


def temperature_humidity_index(t2m, rh):
    """
    Calculate Temperature-Humidity Index (THI) / Heat Index.
    
    Simple measure combining temperature and humidity.
    
    Parameters
    ----------
    t2m : ndarray
        Temperature [°C]
    rh : ndarray
        Relative humidity [%]
    
    Returns
    -------
    ndarray
        THI [°C]
    """
    # Steadman formula for apparent temperature
    THI = t2m + 0.5555 * (rh / 100 * 6.112 * np.exp((17.67 * t2m) / (t2m + 243.5)) - 10)
    return THI


def perceived_temperature_thermal_index(t2m, rh, windspeed, mrt=None):
    """
    Calculate Perceived Temperature / Thermal Index.
    
    Advanced index combining multiple thermal factors.
    
    Parameters
    ----------
    t2m : ndarray
        Temperature [°C]
    rh : ndarray
        Relative humidity [%]
    windspeed : ndarray
        Wind speed [m/s]
    mrt : ndarray, optional
        Mean Radiant Temperature [°C]
    
    Returns
    -------
    ndarray
        Perceived temperature [°C]
    """
    # Base temperature with wind chill
    PT = t2m - 0.55 * (1 - 0.02 * rh / 100) * (10.29 * np.sqrt(windspeed) + 10.45 - 1.3 * windspeed)
    
    # Adjust if above 10°C
    PT = np.where(
        PT > 10,
        t2m + 0.2206 * (rh / 100) * 6.105 * np.exp((17.27 * t2m) / (237.7 + t2m)),
        PT
    )
    
    # Radiation adjustment
    if mrt is not None:
        PT = PT + 0.2 * (mrt - t2m)
    
    return PT


def thermal_strain_index(core_temp, skin_temp, windspeed):
    """
    Calculate Thermal Strain Index (TSI).
    
    Measures human thermal stress from core and skin temperatures.
    
    Parameters
    ----------
    core_temp : float
        Core body temperature [°C]
    skin_temp : ndarray
        Skin temperature [°C]
    windspeed : ndarray
        Wind speed [m/s]
    
    Returns
    -------
    ndarray
        TSI (0=no strain, 1=maximum strain)
    """
    # Normalized thermal strain
    TSI = ((core_temp - 37) / 1.3) + ((skin_temp - 33.7) / 3.8)
    # Wind effect (cooling)
    wind_factor = np.log(windspeed + 1) / 10
    return np.clip((TSI - wind_factor) / 2, 0, 1)
