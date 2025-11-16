# poster data preparation

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr
from numba import float32, float64, vectorize  # noqa

from xclim.core.calendar import date_range, datetime_to_decimal_year
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
from xclim.indices import saturation_vapor_pressure

def _fdir_ratio(
    dates: xr.DataArray,
    csza_i: xr.DataArray,
    csza_s: xr.DataArray,
    rsds: xr.DataArray,
) -> xr.DataArray:
    d = distance_from_sun(dates)
    s_star = rsds * ((1367 * csza_s * (d ** (-2))) ** (-1))
    s_star = xr.where(s_star > 0.85, 0.85, s_star)
    fdir_ratio = np.exp(3 - 1.34 * s_star - 1.65 * (s_star ** (-1)))
    fdir_ratio = xr.where(fdir_ratio > 0.9, 0.9, fdir_ratio)
    return xr.where(
        (fdir_ratio <= 0) | (csza_i <= np.cos(89.5 / 180 * np.pi)) | (rsds <= 0),
        0,
        fdir_ratio,
    )

@declare_units(
    rsds="[radiation]", rsus="[radiation]", rlds="[radiation]", rlus="[radiation]", rsds_direct="[radiation]", rsds_diffuse="[radiation]"
)
def mean_radiant_temperature_from_DIR_DIF_INST(
    rsds: xr.DataArray,
    rsus: xr.DataArray,
    rlds: xr.DataArray,
    rlus: xr.DataArray,
    rsds_direct: xr.DataArray,
    rsds_diffuse: xr.DataArray,
    stat: str = "sunlit",
) -> xr.DataArray:
    rsds = convert_units_to(rsds, "W m-2")
    rsus = convert_units_to(rsus, "W m-2")
    rlds = convert_units_to(rlds, "W m-2")
    rlus = convert_units_to(rlus, "W m-2")

    dates = rsds.time
    lat = _gather_lat(rsds)
    lon = _gather_lon(rsds)
    dec = solar_declination(dates)

    if stat == "sunlit":
        csza_i = cosine_of_solar_zenith_angle(
            dates, dec, lat, lon=lon, stat="average", sunlit=True
        )
        csza_s = cosine_of_solar_zenith_angle(
            dates, dec, lat, lon=lon, stat="average", sunlit=True
        )
    elif stat == "instant":
        tc = time_correction_for_solar_angle(dates)
        csza = cosine_of_solar_zenith_angle(
            dates, dec, lat, lon=lon, time_correction=tc, stat="instant"
        )
        csza_i = csza.copy()
        csza_s = csza.copy()
    elif stat == "average":
        csza = cosine_of_solar_zenith_angle(
            dates, dec, lat, stat="average", sunlit=False
        )
        csza_i = csza.copy()
        csza_s = csza.copy()
    else:
        raise NotImplementedError(
            "Argument 'stat' must be one of 'average', 'instant' or 'sunlit'."
        )

    # fdir_ratio = _fdir_ratio(dates, csza_i, csza_s, rsds)

    # rsds_direct = fdir_ratio * rsds
    # rsds_diffuse = rsds - rsds_direct

    gamma = np.arcsin(csza_i)
    fp = 0.308 * np.cos(gamma * 0.988 - (gamma**2 / 50000))
    # i_star = xr.where(csza_s > 0.001, rsds_direct / csza_s, 0)
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

@declare_units(
    rsds="[radiation]", rsus="[radiation]", rlds="[radiation]", rlus="[radiation]", rsds_direct="[radiation]", rsds_diffuse="[radiation]"
)
def mean_radiant_temperature_from_DIR_DIF_COS(
    rsds: xr.DataArray,
    rsus: xr.DataArray,
    rlds: xr.DataArray,
    rlus: xr.DataArray,
    rsds_direct: xr.DataArray,
    rsds_diffuse: xr.DataArray,
    csza_i: xr.DataArray,
    csza_s: xr.DataArray,
) -> xr.DataArray:
    rsds = convert_units_to(rsds, "W m-2")
    rsus = convert_units_to(rsus, "W m-2")
    rlds = convert_units_to(rlds, "W m-2")
    rlus = convert_units_to(rlus, "W m-2")

    dates = rsds.time
    lat = _gather_lat(rsds)
    lon = _gather_lon(rsds)
    dec = solar_declination(dates)

    gamma = np.arcsin(csza_i)
    fp = 0.308 * np.cos(gamma * 0.988 - (gamma**2 / 50000))
    i_star = xr.where(csza_s > 0.001, rsds_direct / csza_s, 0)
    # i_star = rsds_direct

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


@vectorize(
    # [
    #     float64(float64, float64, float64, float64),
    #     float32(float32, float32, float32, float32),
    # ],
)
def _utci(tas, sfcWind, dt, wvp):
    """Return the empirical polynomial function for UTCI. See :py:func:`universal_thermal_climate_index`."""
    # Taken directly from the original Fortran code by Peter Bröde.
    # http://www.utci.org/public/UTCI%20Program%20Code/UTCI_a002.f90
    # tas -> Ta (surface temperature, °C)
    # sfcWind -> va (surface wind speed, m/s)
    # dt -> D_Tmrt (tas - t_mrt, K)
    # wvp -> Pa (water vapour partial pressure, kPa)
    return (
        tas
        + 6.07562052e-1
        + -2.27712343e-2 * tas
        + 8.06470249e-4 * tas * tas
        + -1.54271372e-4 * tas * tas * tas
        + -3.24651735e-6 * tas * tas * tas * tas
        + 7.32602852e-8 * tas * tas * tas * tas * tas
        + 1.35959073e-9 * tas * tas * tas * tas * tas * tas
        + -2.25836520e0 * sfcWind
        + 8.80326035e-2 * tas * sfcWind
        + 2.16844454e-3 * tas * tas * sfcWind
        + -1.53347087e-5 * tas * tas * tas * sfcWind
        + -5.72983704e-7 * tas * tas * tas * tas * sfcWind
        + -2.55090145e-9 * tas * tas * tas * tas * tas * sfcWind
        + -7.51269505e-1 * sfcWind * sfcWind
        + -4.08350271e-3 * tas * sfcWind * sfcWind
        + -5.21670675e-5 * tas * tas * sfcWind * sfcWind
        + 1.94544667e-6 * tas * tas * tas * sfcWind * sfcWind
        + 1.14099531e-8 * tas * tas * tas * tas * sfcWind * sfcWind
        + 1.58137256e-1 * sfcWind * sfcWind * sfcWind
        + -6.57263143e-5 * tas * sfcWind * sfcWind * sfcWind
        + 2.22697524e-7 * tas * tas * sfcWind * sfcWind * sfcWind
        + -4.16117031e-8 * tas * tas * tas * sfcWind * sfcWind * sfcWind
        + -1.27762753e-2 * sfcWind * sfcWind * sfcWind * sfcWind
        + 9.66891875e-6 * tas * sfcWind * sfcWind * sfcWind * sfcWind
        + 2.52785852e-9 * tas * tas * sfcWind * sfcWind * sfcWind * sfcWind
        + 4.56306672e-4 * sfcWind * sfcWind * sfcWind * sfcWind * sfcWind
        + -1.74202546e-7 * tas * sfcWind * sfcWind * sfcWind * sfcWind * sfcWind
        + -5.91491269e-6 * sfcWind * sfcWind * sfcWind * sfcWind * sfcWind * sfcWind
        + 3.98374029e-1 * dt
        + 1.83945314e-4 * tas * dt
        + -1.73754510e-4 * tas * tas * dt
        + -7.60781159e-7 * tas * tas * tas * dt
        + 3.77830287e-8 * tas * tas * tas * tas * dt
        + 5.43079673e-10 * tas * tas * tas * tas * tas * dt
        + -2.00518269e-2 * sfcWind * dt
        + 8.92859837e-4 * tas * sfcWind * dt
        + 3.45433048e-6 * tas * tas * sfcWind * dt
        + -3.77925774e-7 * tas * tas * tas * sfcWind * dt
        + -1.69699377e-9 * tas * tas * tas * tas * sfcWind * dt
        + 1.69992415e-4 * sfcWind * sfcWind * dt
        + -4.99204314e-5 * tas * sfcWind * sfcWind * dt
        + 2.47417178e-7 * tas * tas * sfcWind * sfcWind * dt
        + 1.07596466e-8 * tas * tas * tas * sfcWind * sfcWind * dt
        + 8.49242932e-5 * sfcWind * sfcWind * sfcWind * dt
        + 1.35191328e-6 * tas * sfcWind * sfcWind * sfcWind * dt
        + -6.21531254e-9 * tas * tas * sfcWind * sfcWind * sfcWind * dt
        + -4.99410301e-6 * sfcWind * sfcWind * sfcWind * sfcWind * dt
        + -1.89489258e-8 * tas * sfcWind * sfcWind * sfcWind * sfcWind * dt
        + 8.15300114e-8 * sfcWind * sfcWind * sfcWind * sfcWind * sfcWind * dt
        + 7.55043090e-4 * dt * dt
        + -5.65095215e-5 * tas * dt * dt
        + -4.52166564e-7 * tas * tas * dt * dt
        + 2.46688878e-8 * tas * tas * tas * dt * dt
        + 2.42674348e-10 * tas * tas * tas * tas * dt * dt
        + 1.54547250e-4 * sfcWind * dt * dt
        + 5.24110970e-6 * tas * sfcWind * dt * dt
        + -8.75874982e-8 * tas * tas * sfcWind * dt * dt
        + -1.50743064e-9 * tas * tas * tas * sfcWind * dt * dt
        + -1.56236307e-5 * sfcWind * sfcWind * dt * dt
        + -1.33895614e-7 * tas * sfcWind * sfcWind * dt * dt
        + 2.49709824e-9 * tas * tas * sfcWind * sfcWind * dt * dt
        + 6.51711721e-7 * sfcWind * sfcWind * sfcWind * dt * dt
        + 1.94960053e-9 * tas * sfcWind * sfcWind * sfcWind * dt * dt
        + -1.00361113e-8 * sfcWind * sfcWind * sfcWind * sfcWind * dt * dt
        + -1.21206673e-5 * dt * dt * dt
        + -2.18203660e-7 * tas * dt * dt * dt
        + 7.51269482e-9 * tas * tas * dt * dt * dt
        + 9.79063848e-11 * tas * tas * tas * dt * dt * dt
        + 1.25006734e-6 * sfcWind * dt * dt * dt
        + -1.81584736e-9 * tas * sfcWind * dt * dt * dt
        + -3.52197671e-10 * tas * tas * sfcWind * dt * dt * dt
        + -3.36514630e-8 * sfcWind * sfcWind * dt * dt * dt
        + 1.35908359e-10 * tas * sfcWind * sfcWind * dt * dt * dt
        + 4.17032620e-10 * sfcWind * sfcWind * sfcWind * dt * dt * dt
        + -1.30369025e-9 * dt * dt * dt * dt
        + 4.13908461e-10 * tas * dt * dt * dt * dt
        + 9.22652254e-12 * tas * tas * dt * dt * dt * dt
        + -5.08220384e-9 * sfcWind * dt * dt * dt * dt
        + -2.24730961e-11 * tas * sfcWind * dt * dt * dt * dt
        + 1.17139133e-10 * sfcWind * sfcWind * dt * dt * dt * dt
        + 6.62154879e-10 * dt * dt * dt * dt * dt
        + 4.03863260e-13 * tas * dt * dt * dt * dt * dt
        + 1.95087203e-12 * sfcWind * dt * dt * dt * dt * dt
        + -4.73602469e-12 * dt * dt * dt * dt * dt * dt
        + 5.12733497e0 * wvp
        + -3.12788561e-1 * tas * wvp
        + -1.96701861e-2 * tas * tas * wvp
        + 9.99690870e-4 * tas * tas * tas * wvp
        + 9.51738512e-6 * tas * tas * tas * tas * wvp
        + -4.66426341e-7 * tas * tas * tas * tas * tas * wvp
        + 5.48050612e-1 * sfcWind * wvp
        + -3.30552823e-3 * tas * sfcWind * wvp
        + -1.64119440e-3 * tas * tas * sfcWind * wvp
        + -5.16670694e-6 * tas * tas * tas * sfcWind * wvp
        + 9.52692432e-7 * tas * tas * tas * tas * sfcWind * wvp
        + -4.29223622e-2 * sfcWind * sfcWind * wvp
        + 5.00845667e-3 * tas * sfcWind * sfcWind * wvp
        + 1.00601257e-6 * tas * tas * sfcWind * sfcWind * wvp
        + -1.81748644e-6 * tas * tas * tas * sfcWind * sfcWind * wvp
        + -1.25813502e-3 * sfcWind * sfcWind * sfcWind * wvp
        + -1.79330391e-4 * tas * sfcWind * sfcWind * sfcWind * wvp
        + 2.34994441e-6 * tas * tas * sfcWind * sfcWind * sfcWind * wvp
        + 1.29735808e-4 * sfcWind * sfcWind * sfcWind * sfcWind * wvp
        + 1.29064870e-6 * tas * sfcWind * sfcWind * sfcWind * sfcWind * wvp
        + -2.28558686e-6 * sfcWind * sfcWind * sfcWind * sfcWind * sfcWind * wvp
        + -3.69476348e-2 * dt * wvp
        + 1.62325322e-3 * tas * dt * wvp
        + -3.14279680e-5 * tas * tas * dt * wvp
        + 2.59835559e-6 * tas * tas * tas * dt * wvp
        + -4.77136523e-8 * tas * tas * tas * tas * dt * wvp
        + 8.64203390e-3 * sfcWind * dt * wvp
        + -6.87405181e-4 * tas * sfcWind * dt * wvp
        + -9.13863872e-6 * tas * tas * sfcWind * dt * wvp
        + 5.15916806e-7 * tas * tas * tas * sfcWind * dt * wvp
        + -3.59217476e-5 * sfcWind * sfcWind * dt * wvp
        + 3.28696511e-5 * tas * sfcWind * sfcWind * dt * wvp
        + -7.10542454e-7 * tas * tas * sfcWind * sfcWind * dt * wvp
        + -1.24382300e-5 * sfcWind * sfcWind * sfcWind * dt * wvp
        + -7.38584400e-9 * tas * sfcWind * sfcWind * sfcWind * dt * wvp
        + 2.20609296e-7 * sfcWind * sfcWind * sfcWind * sfcWind * dt * wvp
        + -7.32469180e-4 * dt * dt * wvp
        + -1.87381964e-5 * tas * dt * dt * wvp
        + 4.80925239e-6 * tas * tas * dt * dt * wvp
        + -8.75492040e-8 * tas * tas * tas * dt * dt * wvp
        + 2.77862930e-5 * sfcWind * dt * dt * wvp
        + -5.06004592e-6 * tas * sfcWind * dt * dt * wvp
        + 1.14325367e-7 * tas * tas * sfcWind * dt * dt * wvp
        + 2.53016723e-6 * sfcWind * sfcWind * dt * dt * wvp
        + -1.72857035e-8 * tas * sfcWind * sfcWind * dt * dt * wvp
        + -3.95079398e-8 * sfcWind * sfcWind * sfcWind * dt * dt * wvp
        + -3.59413173e-7 * dt * dt * dt * wvp
        + 7.04388046e-7 * tas * dt * dt * dt * wvp
        + -1.89309167e-8 * tas * tas * dt * dt * dt * wvp
        + -4.79768731e-7 * sfcWind * dt * dt * dt * wvp
        + 7.96079978e-9 * tas * sfcWind * dt * dt * dt * wvp
        + 1.62897058e-9 * sfcWind * sfcWind * dt * dt * dt * wvp
        + 3.94367674e-8 * dt * dt * dt * dt * wvp
        + -1.18566247e-9 * tas * dt * dt * dt * dt * wvp
        + 3.34678041e-10 * sfcWind * dt * dt * dt * dt * wvp
        + -1.15606447e-10 * dt * dt * dt * dt * dt * wvp
        + -2.80626406e0 * wvp * wvp
        + 5.48712484e-1 * tas * wvp * wvp
        + -3.99428410e-3 * tas * tas * wvp * wvp
        + -9.54009191e-4 * tas * tas * tas * wvp * wvp
        + 1.93090978e-5 * tas * tas * tas * tas * wvp * wvp
        + -3.08806365e-1 * sfcWind * wvp * wvp
        + 1.16952364e-2 * tas * sfcWind * wvp * wvp
        + 4.95271903e-4 * tas * tas * sfcWind * wvp * wvp
        + -1.90710882e-5 * tas * tas * tas * sfcWind * wvp * wvp
        + 2.10787756e-3 * sfcWind * sfcWind * wvp * wvp
        + -6.98445738e-4 * tas * sfcWind * sfcWind * wvp * wvp
        + 2.30109073e-5 * tas * tas * sfcWind * sfcWind * wvp * wvp
        + 4.17856590e-4 * sfcWind * sfcWind * sfcWind * wvp * wvp
        + -1.27043871e-5 * tas * sfcWind * sfcWind * sfcWind * wvp * wvp
        + -3.04620472e-6 * sfcWind * sfcWind * sfcWind * sfcWind * wvp * wvp
        + 5.14507424e-2 * dt * wvp * wvp
        + -4.32510997e-3 * tas * dt * wvp * wvp
        + 8.99281156e-5 * tas * tas * dt * wvp * wvp
        + -7.14663943e-7 * tas * tas * tas * dt * wvp * wvp
        + -2.66016305e-4 * sfcWind * dt * wvp * wvp
        + 2.63789586e-4 * tas * sfcWind * dt * wvp * wvp
        + -7.01199003e-6 * tas * tas * sfcWind * dt * wvp * wvp
        + -1.06823306e-4 * sfcWind * sfcWind * dt * wvp * wvp
        + 3.61341136e-6 * tas * sfcWind * sfcWind * dt * wvp * wvp
        + 2.29748967e-7 * sfcWind * sfcWind * sfcWind * dt * wvp * wvp
        + 3.04788893e-4 * dt * dt * wvp * wvp
        + -6.42070836e-5 * tas * dt * dt * wvp * wvp
        + 1.16257971e-6 * tas * tas * dt * dt * wvp * wvp
        + 7.68023384e-6 * sfcWind * dt * dt * wvp * wvp
        + -5.47446896e-7 * tas * sfcWind * dt * dt * wvp * wvp
        + -3.59937910e-8 * sfcWind * sfcWind * dt * dt * wvp * wvp
        + -4.36497725e-6 * dt * dt * dt * wvp * wvp
        + 1.68737969e-7 * tas * dt * dt * dt * wvp * wvp
        + 2.67489271e-8 * sfcWind * dt * dt * dt * wvp * wvp
        + 3.23926897e-9 * dt * dt * dt * dt * wvp * wvp
        + -3.53874123e-2 * wvp * wvp * wvp
        + -2.21201190e-1 * tas * wvp * wvp * wvp
        + 1.55126038e-2 * tas * tas * wvp * wvp * wvp
        + -2.63917279e-4 * tas * tas * tas * wvp * wvp * wvp
        + 4.53433455e-2 * sfcWind * wvp * wvp * wvp
        + -4.32943862e-3 * tas * sfcWind * wvp * wvp * wvp
        + 1.45389826e-4 * tas * tas * sfcWind * wvp * wvp * wvp
        + 2.17508610e-4 * sfcWind * sfcWind * wvp * wvp * wvp
        + -6.66724702e-5 * tas * sfcWind * sfcWind * wvp * wvp * wvp
        + 3.33217140e-5 * sfcWind * sfcWind * sfcWind * wvp * wvp * wvp
        + -2.26921615e-3 * dt * wvp * wvp * wvp
        + 3.80261982e-4 * tas * dt * wvp * wvp * wvp
        + -5.45314314e-9 * tas * tas * dt * wvp * wvp * wvp
        + -7.96355448e-4 * sfcWind * dt * wvp * wvp * wvp
        + 2.53458034e-5 * tas * sfcWind * dt * wvp * wvp * wvp
        + -6.31223658e-6 * sfcWind * sfcWind * dt * wvp * wvp * wvp
        + 3.02122035e-4 * dt * dt * wvp * wvp * wvp
        + -4.77403547e-6 * tas * dt * dt * wvp * wvp * wvp
        + 1.73825715e-6 * sfcWind * dt * dt * wvp * wvp * wvp
        + -4.09087898e-7 * dt * dt * dt * wvp * wvp * wvp
        + 6.14155345e-1 * wvp * wvp * wvp * wvp
        + -6.16755931e-2 * tas * wvp * wvp * wvp * wvp
        + 1.33374846e-3 * tas * tas * wvp * wvp * wvp * wvp
        + 3.55375387e-3 * sfcWind * wvp * wvp * wvp * wvp
        + -5.13027851e-4 * tas * sfcWind * wvp * wvp * wvp * wvp
        + 1.02449757e-4 * sfcWind * sfcWind * wvp * wvp * wvp * wvp
        + -1.48526421e-3 * dt * wvp * wvp * wvp * wvp
        + -4.11469183e-5 * tas * dt * wvp * wvp * wvp * wvp
        + -6.80434415e-6 * sfcWind * dt * wvp * wvp * wvp * wvp
        + -9.77675906e-6 * dt * dt * wvp * wvp * wvp * wvp
        + 8.82773108e-2 * wvp * wvp * wvp * wvp * wvp
        + -3.01859306e-3 * tas * wvp * wvp * wvp * wvp * wvp
        + 1.04452989e-3 * sfcWind * wvp * wvp * wvp * wvp * wvp
        + 2.47090539e-4 * dt * wvp * wvp * wvp * wvp * wvp
        + 1.48348065e-3 * wvp * wvp * wvp * wvp * wvp * wvp
    )


@declare_units(
    tas="[temperature]",
    hurs="[]",
    sfcWind="[speed]",
    mrt="[temperature]",
    rsds="[radiation]",
    rsus="[radiation]",
    rlds="[radiation]",
    rlus="[radiation]",
)
def universal_thermal_climate_index(
    tas: xr.DataArray,
    hurs: xr.DataArray,
    sfcWind: xr.DataArray,
    mrt: xr.DataArray = None,
    rsds: xr.DataArray = None,
    rsus: xr.DataArray = None,
    rlds: xr.DataArray = None,
    rlus: xr.DataArray = None,
    stat: str = "average",
    mask_invalid: bool = True,
) -> xr.DataArray:
    r"""Universal thermal climate index (UTCI).

    The UTCI is the equivalent temperature for the environment derived from a
    reference environment and is used to evaluate heat stress in outdoor spaces.

    Parameters
    ----------
    tas : xarray.DataArray
        Mean temperature
    hurs : xarray.DataArray
        Relative Humidity
    sfcWind : xarray.DataArray
        Wind velocity
    mrt: xarray.DataArray, optional
        Mean radiant temperature
    rsds : xr.DataArray, optional
        Surface Downwelling Shortwave Radiation
        This is necessary if mrt is not None.
    rsus : xr.DataArray, optional
        Surface Upwelling Shortwave Radiation
        This is necessary if mrt is not None.
    rlds : xr.DataArray, optional
        Surface Downwelling Longwave Radiation
        This is necessary if mrt is not None.
    rlus : xr.DataArray, optional
        Surface Upwelling Longwave Radiation
        This is necessary if mrt is not None.
    stat : {'average', 'instant', 'sunlit'}
        Which statistic to apply. If "average", the average of the cosine of the
        solar zenith angle is calculated. If "instant", the instantaneous cosine
        of the solar zenith angle is calculated. If "sunlit", the cosine of the
        solar zenith angle is calculated during the sunlit period of each interval.
        If "instant", the instantaneous cosine of the solar zenith angle is calculated.
        This is necessary if mrt is not None.
    mask_invalid: bool
        If True (default), UTCI values are NaN where any of the inputs are outside
        their validity ranges : -50°C < tas < 50°C,  -30°C < tas - mrt < 30°C
        and  0.5 m/s < sfcWind < 17.0 m/s.

    Returns
    -------
    xarray.DataArray
        Universal Thermal Climate Index.

    Notes
    -----
    The calculation uses water vapour partial pressure, which is derived from relative
    humidity and saturation vapour pressure computed according to the ITS-90 equation.

    This code was inspired by the `pythermalcomfort` and `thermofeel` packages.

    Notes
    -----
    See: http://www.utci.org/utcineu/utcineu.php

    References
    ----------
    :cite:cts:`brode_utci_2009,blazejczyk_introduction_2013`
    """
    e_sat = saturation_vapor_pressure(tas=tas, method="its90")
    tas = convert_units_to(tas, "degC")
    sfcWind = convert_units_to(sfcWind, "m/s")
    if mrt is None:
        mrt = mean_radiant_temperature(
            rsds=rsds, rsus=rsus, rlds=rlds, rlus=rlus, stat=stat
        )
    mrt = convert_units_to(mrt, "degC")
    delta = mrt - tas
    pa = convert_units_to(e_sat, "kPa") * convert_units_to(hurs, "1")

    utci = xr.apply_ufunc(
        _utci,
        tas,
        sfcWind,
        delta,
        pa,
        input_core_dims=[[], [], [], []],
        dask="parallelized",
        output_dtypes=[tas.dtype],
    )

    utci = utci.assign_attrs({"units": "degC"})
    if mask_invalid:
        utci = utci.where(
            (-50.0 < tas)
            & (tas < 50.0)
            & (-30 < delta)
            & (delta < 70)
            & (0.5 < sfcWind)
            & (sfcWind < 17.0)
        )
    return utci