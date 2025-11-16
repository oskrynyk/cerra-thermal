# helper functions
# some redefined here from thermofeel or xclim libraries for tests

from __future__ import annotations
import numpy as np
import xarray as xr
from numba import float32, float64, vectorize  # noqa
from xclim.core.calendar import date_range
from xclim.core.units import (
    amount2rate,
    convert_units_to,
    declare_units,
    flux2rate,
    rate2flux,
    units,
    units2pint,
)
from xclim.core.utils import Quantified
from xclim.indices.helpers import (
    _gather_lat,
    _gather_lon,
    cosine_of_solar_zenith_angle,
    day_lengths,
    distance_from_sun,
    extraterrestrial_solar_radiation,
    solar_declination,
    time_correction_for_solar_angle,
    wind_speed_height_conversion,
)

def _fdir_ratio(
    dates: xr.DataArray,
    csza: xr.DataArray,
    rsds: xr.DataArray,
) -> xr.DataArray:
    r"""Return ratio of direct solar radiation.

    The ratio of direct solar radiation is the fraction of the total horizontal solar irradiance
    due to the direct beam of the sun.

    Parameters
    ----------
    dates : xr.DataArray
        Series of dates and time of day
    csza : xr.DataArray
        Cosine of the solar zenith angle during the sunlit period of each interval or at an instant
    rsds : xr.DataArray
        Surface Downwelling Shortwave Radiation

    Returns
    -------
    xarray.DataArray, [dimensionless]
        Ratio of direct solar radiation

    Notes
    -----
    This code was inspired by the `PyWBGT` package.

    References
    ----------
    :cite:cts:`liljegren_modeling_2008,kong_explicit_2022`
    """
    d = distance_from_sun(dates)
    s_star = rsds * ((1367 * csza * (d ** (-2))) ** (-1))
    s_star = xr.where(s_star > 0.85, 0.85, s_star)
    fdir_ratio = np.exp(3 - 1.34 * s_star - 1.65 * (s_star ** (-1)))
    fdir_ratio = xr.where(fdir_ratio > 0.9, 0.9, fdir_ratio)
    return xr.where(
        (fdir_ratio <= 0) | (csza <= np.cos(89.5 / 180 * np.pi)) | (rsds <= 0),
        0,
        fdir_ratio,
    )

from typing import Optional

@declare_units(
    rsds="[radiation]", rsus="[radiation]", rlds="[radiation]", rlus="[radiation]"
)
def mean_radiant_temperature(
    rsds: xr.DataArray,
    rsus: xr.DataArray,
    rlds: xr.DataArray,
    rlus: xr.DataArray,
    csza: xr.DataArray,
    csza_thr = 0.001,
    rsds_direct: Optional[xr.DataArray] = None,
    # stat: str = "sunlit",
) -> xr.DataArray:
    r"""Mean radiant temperature.

    The mean radiant temperature is the incidence of radiation on the body from all directions.

    Parameters
    ----------
    rsds : xr.DataArray
       Surface Downwelling Shortwave Radiation
    rsus : xr.DataArray
        Surface Upwelling Shortwave Radiation
    rlds : xr.DataArray
        Surface Downwelling Longwave Radiation
    rlus : xr.DataArray
        Surface Upwelling Longwave Radiation
    stat : {'instant', 'sunlit'}
        Which statistic to apply. If "instant", the instantaneous cosine
        of the solar zenith angle is calculated. If "sunlit", the cosine of the
        solar zenith angle is calculated during the sunlit period of each interval.

    Returns
    -------
    xarray.DataArray, [K]
        Mean radiant temperature

    Warnings
    --------
    There are some issues in the calculation of mrt in polar regions.

    Notes
    -----
    This code was inspired by the `thermofeel` package :cite:p:`brimicombe_thermofeel_2021`.

    References
    ----------
    :cite:cts:`di_napoli_mean_2020`
    """
    rsds = convert_units_to(rsds, "W m-2")
    rsus = convert_units_to(rsus, "W m-2")
    rlds = convert_units_to(rlds, "W m-2")
    rlus = convert_units_to(rlus, "W m-2")

    # lat = _gather_lat(rsds)
    # lon = _gather_lon(rsds)
    # dec = solar_declination(dates)

    # if stat == "sunlit":
    #     csza = cosine_of_solar_zenith_angle(
    #         dates,
    #         dec,
    #         lat,
    #         lon=lon,
    #         stat="average",
    #         sunlit=True,
    #         chunks=rsds.chunksizes,
    #     )
    # elif stat == "instant":
    #     tc = time_correction_for_solar_angle(dates)
    #     csza = cosine_of_solar_zenith_angle(
    #         dates,
    #         dec,
    #         lat,
    #         lon=lon,
    #         time_correction=tc,
    #         stat="instant",
    #         chunks=rsds.chunksizes,
    #     )
    # else:
    #     raise NotImplementedError(
    #         "Argument 'stat' must be one of 'instant' or 'sunlit'."
    #     )

    if rsds_direct is None:
        dates = rsds.time
        fdir_ratio = _fdir_ratio(dates, csza, rsds)
        rsds_direct = fdir_ratio * rsds

    rsds_diffuse = rsds - rsds_direct

    gamma = np.arcsin(csza)
    fp = 0.308 * np.cos(gamma * 0.988 - (gamma**2 / 50000))
    i_star = xr.where(csza > csza_thr, rsds_direct / csza, 0) #csza > 0.001

    mrt = np.power(
        (
            (1 / 5.67e-8)  # Stefan-Boltzmann constant
            * (
                0.5 * rlds
                + 0.5 * rlus
                + (0.7 / 0.97) * (0.5 * rsds_diffuse + 0.5 * rsus + fp * i_star)
            )
        ),
        0.25,
    )
    return mrt.assign_attrs({"units": "K"})


@declare_units(
    rsds="[radiation]", rsus="[radiation]", rlds="[radiation]", rlus="[radiation]"
)
def mean_radiant_temperature_2(
    rsds: xr.DataArray,
    rsus: xr.DataArray,
    rlds: xr.DataArray,
    rlus: xr.DataArray,
    rsds_direct: xr.DataArray,
    rsds_diffuse: xr.DataArray
) -> xr.DataArray:
    r"""Mean radiant temperature.
    """
    rsds = convert_units_to(rsds, "W m-2")
    rsus = convert_units_to(rsus, "W m-2")
    rlds = convert_units_to(rlds, "W m-2")
    rlus = convert_units_to(rlus, "W m-2")

    dates = rsds.time
    lat = _gather_lat(rsds)
    lon = _gather_lon(rsds)
    dec = solar_declination(dates)
    tc = time_correction_for_solar_angle(dates)
    csza = cosine_of_solar_zenith_angle(
        dates,
        dec,
        lat,
        lon=lon,
        time_correction=tc,
        stat="instant",
    )

    # fdir_ratio = _fdir_ratio(dates, csza, rsds)
    #
    # rsds_direct = fdir_ratio * rsds
    # rsds_diffuse = rsds - rsds_direct

    gamma = np.arcsin(csza)
    fp = 0.308 * np.cos(gamma * 0.988 - (gamma**2 / 50000))
    # i_star = xr.where(csza > 0.001, rsds_direct / csza, 0)
    i_star = rsds_direct

    mrt = np.power(
        (
            (1 / 5.67e-8)  # Stefan-Boltzmann constant
            * (
                0.5 * rlds
                + 0.5 * rlus
                + (0.7 / 0.97) * (0.5 * rsds_diffuse + 0.5 * rsus + fp * i_star)
            )
        ),
        0.25,
    )
    return mrt.assign_attrs({"units": "K"})












# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import numpy as np

DAYS_PER_YEAR = 365.25

"""
WARNING: This code is incomplete and not tested. DO NOT USE.
"""


def julian_day(date):
    delta = date - datetime.datetime(date.year, 1, 1)
    return delta.days + delta.seconds / 86400.0


def solar_declination_angle(date):
    angle = julian_day(date) / DAYS_PER_YEAR * np.pi * 2

    # declination in [degrees]
    declination = (
        0.396372
        - 22.91327 * np.cos(angle)
        + 4.025430 * np.sin(angle)
        - 0.387205 * np.cos(2 * angle)
        + 0.051967 * np.sin(2 * angle)
        - 0.154527 * np.cos(3 * angle)
        + 0.084798 * np.sin(3 * angle)
    )
    # time correction in [ h.degrees ]
    time_correction = (
        0.004297
        + 0.107029 * np.cos(angle)
        - 1.837877 * np.sin(angle)
        - 0.837378 * np.cos(2 * angle)
        - 2.340475 * np.sin(2 * angle)
    )
    return declination, time_correction


def cos_solar_zenith_angle(date, latitudes, longitudes):
    """Cosine of solar zenith angle.

    Parameters
    ----------
    date: datetime.datetime
        Date
    lat: float array
        Latitude [degrees]
    lon: float array
        Longitude [degrees]

    Returns
    -------
    float array
        Cosine of the solar zenith angle (all values, including negatives)
        [Hogan_and_Hirahara2015]_. See also:
        http://answers.google.com/answers/threadview/id/782886.html

    """
    # declination angle + time correction for solar angle
    declination, time_correction = solar_declination_angle(date)

    # solar_declination_angle returns degrees
    declination = np.deg2rad(declination)

    latitudes = np.deg2rad(latitudes)

    sindec_sinlat = np.sin(declination) * np.sin(latitudes)
    cosdec_coslat = np.cos(declination) * np.cos(latitudes)

    # solar hour angle [h.deg]
    solar_angle = np.deg2rad((date.hour - 12) * 15 + longitudes + time_correction)
    zenith_angle = sindec_sinlat + cosdec_coslat * np.cos(solar_angle)

    # Clip negative values
    return np.clip(zenith_angle, 0, None)


def _integrate(
    func,
    begin_date,
    end_date,
    latitudes,
    longitudes,
    *,
    intervals_per_hour=1,
    integration_order=3,
):
    # Ensure that begin_date and end_date are datetime objects
    assert isinstance(begin_date, datetime.datetime)
    assert isinstance(end_date, datetime.datetime)

    # Gauss-Integration coefficients
    if integration_order == 3:  # default, good speed and accuracy (3 points)
        E = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        W = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    elif integration_order == 1:  # fastest, worse accuracy (1 point)
        E = np.array([0.0])
        W = np.array([2.0])
    elif integration_order == 2:  # faster, less accurate (2 points)
        E = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        W = np.array([1.0, 1.0])
    elif integration_order == 4:  # slower, more accurate (4 points)
        E = np.array(
            [
                -np.sqrt(3.0 / 7.0 + 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
                -np.sqrt(3.0 / 7.0 - 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
                np.sqrt(3.0 / 7.0 - 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
                np.sqrt(3.0 / 7.0 + 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
            ]
        )
        W = np.array(
            [
                (18 - np.sqrt(30)) / 36,
                (18 + np.sqrt(30)) / 36,
                (18 + np.sqrt(30)) / 36,
                (18 - np.sqrt(30)) / 36,
            ]
        )
    else:
        raise ValueError("Invalid integration order %d", integration_order)

    assert intervals_per_hour > 0
    assert end_date > begin_date

    date = begin_date
    interval_size_hours = (end_date - begin_date).total_seconds() / 3600.0

    nsplits = int(interval_size_hours * intervals_per_hour + 0.5)

    assert nsplits > 0

    time_steps = np.linspace(0, interval_size_hours, num=nsplits + 1)

    integral = np.zeros_like(latitudes)
    for s in range(len(time_steps) - 1):
        ti = time_steps[s]
        tf = time_steps[s + 1]

        deltat = tf - ti
        jacob = deltat / 2.0

        w = jacob * W
        w /= interval_size_hours  # average of integral
        t = jacob * E
        t += (tf + ti) / 2.0

        for n in range(len(w)):
            integral += w[n] * func(
                begin_date + datetime.timedelta(hours=t[n]),  # Correct usage of timedelta
                latitudes,
                longitudes,
            )

    return integral


def cos_solar_zenith_angle_integrated(
    begin_date,
    end_date,
    latitudes,
    longitudes,
    *,
    intervals_per_hour=1,
    integration_order=3,
):
    """Average of solar zenith angle based on numerical integration.

    Parameters
    ----------
    begin_date: datetime.datetime
    end_date: datetime.datetime
    lat: int darray
        Latitude [degrees].
    lon: int darray
        Longitude [degrees].
    tbegin: int
        Offset in hours from forecast time to begin of time interval for integration.
    tend: int
        Offset in hours from forecast time to end of time interval for integration.
    intervals_per_hour: int
        Number of time integrations per hour.
    integration order: int
        Order of gauss integration, valid = (1, 2, 3, 4)

    Returns
    -------
    float array
        Average of cosine of the solar zenith angle during interval [degrees]. Based on
        numerical integration using the 3 point
        `Gauss integration <https://en.wikipedia.org/wiki/Gaussian_quadrature>`_ rule.
        [Hogan_and_Hirahara2015]_, [Biricombe2022]_

    """
    return _integrate(
        cos_solar_zenith_angle,
        begin_date,
        end_date,
        latitudes,
        longitudes,
        intervals_per_hour=intervals_per_hour,
        integration_order=integration_order,
    )


def incoming_solar_radiation(date):
    # To be replaced with improved formula
    (a, b) = (165120.0, 4892416.0)
    angle = julian_day(date) / DAYS_PER_YEAR * np.pi * 2
    return np.cos(angle) * a + b


def toa_incident_solar_radiation(
    begin_date,
    end_date,
    latitudes,
    longitudes,
    *,
    intervals_per_hour=1,
    integration_order=3,
):
    def func(date, latitudes, longitudes):
        isr = incoming_solar_radiation(date)
        csza = cos_solar_zenith_angle(date, latitudes, longitudes)
        return isr * csza

    return _integrate(
        func,
        begin_date,
        end_date,
        latitudes,
        longitudes,
        intervals_per_hour=intervals_per_hour,
        integration_order=integration_order,
    )