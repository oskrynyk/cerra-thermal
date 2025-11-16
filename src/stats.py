"""Thermal comfort statistics and extreme value analysis."""

import numpy as np
import xarray as xr
from scipy import stats
from scipy.ndimage import uniform_filter
import warnings

warnings.filterwarnings('ignore')


class ThermalStatistics:
    """Compute thermal statistics and extremes."""

    def __init__(self, mrt_data, utci_data, net_data=None):
        """Initialize with thermal datasets."""
        self.mrt_c = mrt_data - 273.15
        self.utci = utci_data
        self.net = net_data

    def percentile_analysis(self, percentiles=[5, 25, 50, 75, 95]):
        """Compute percentiles across spatial domain."""
        results = {}
        
        for perc in percentiles:
            results[f'mrt_p{perc}'] = np.percentile(self.mrt_c.values, perc)
            results[f'utci_p{perc}'] = np.percentile(self.utci.values, perc)
            if self.net is not None:
                results[f'net_p{perc}'] = np.percentile(self.net.values, perc)
        
        return results

    def extreme_value_analysis(self):
        """Analyze extreme thermal events."""
        extremes = {}
        
        # MRT extremes
        mrt_max_loc = np.unravel_index(np.argmax(self.mrt_c.values), self.mrt_c.shape)
        mrt_min_loc = np.unravel_index(np.argmin(self.mrt_c.values), self.mrt_c.shape)
        
        extremes['mrt_max_value'] = float(self.mrt_c.values[mrt_max_loc])
        extremes['mrt_max_location'] = (
            float(self.mrt_c.longitude.values[mrt_max_loc[1]]),
            float(self.mrt_c.latitude.values[mrt_max_loc[0]])
        )
        extremes['mrt_min_value'] = float(self.mrt_c.values[mrt_min_loc])
        extremes['mrt_min_location'] = (
            float(self.mrt_c.longitude.values[mrt_min_loc[1]]),
            float(self.mrt_c.latitude.values[mrt_min_loc[0]])
        )
        
        # UTCI extremes
        utci_max_loc = np.unravel_index(np.argmax(self.utci.values), self.utci.shape)
        utci_min_loc = np.unravel_index(np.argmin(self.utci.values), self.utci.shape)
        
        extremes['utci_max_value'] = float(self.utci.values[utci_max_loc])
        extremes['utci_max_location'] = (
            float(self.utci.longitude.values[utci_max_loc[1]]),
            float(self.utci.latitude.values[utci_max_loc[0]])
        )
        extremes['utci_min_value'] = float(self.utci.values[utci_min_loc])
        extremes['utci_min_location'] = (
            float(self.utci.longitude.values[utci_min_loc[1]]),
            float(self.utci.latitude.values[utci_min_loc[0]])
        )
        
        return extremes

    def heat_stress_classification(self):
        """Classify heat stress categories."""
        classifications = {
            'extreme_heat': (self.utci > 46).sum().values,
            'strong_heat': ((self.utci > 32) & (self.utci <= 46)).sum().values,
            'moderate_heat': ((self.utci > 26) & (self.utci <= 32)).sum().values,
            'warm': ((self.utci > 9) & (self.utci <= 26)).sum().values,
            'neutral': ((self.utci > 0) & (self.utci <= 9)).sum().values,
            'weak_cold': ((self.utci > -13) & (self.utci <= 0)).sum().values,
            'moderate_cold': ((self.utci > -27) & (self.utci <= -13)).sum().values,
            'strong_cold': (self.utci <= -27).sum().values,
        }
        
        total = sum(classifications.values())
        return {k: 100 * v / total for k, v in classifications.items()}

    def spatial_correlation(self):
        """Analyze spatial correlation between MRT and UTCI."""
        mrt_flat = self.mrt_c.values.flatten()
        utci_flat = self.utci.values.flatten()
        
        # Remove NaNs
        valid_mask = ~(np.isnan(mrt_flat) | np.isnan(utci_flat))
        mrt_valid = mrt_flat[valid_mask]
        utci_valid = utci_flat[valid_mask]
        
        if len(mrt_valid) > 2:
            corr, pval = stats.pearsonr(mrt_valid, utci_valid)
            return {'correlation': corr, 'p_value': pval}
        else:
            return {'correlation': np.nan, 'p_value': np.nan}

    def temporal_trend(self, time_data):
        """Analyze temporal trend in mean values."""
        mrt_mean_ts = self.mrt_c.mean(dim=['x', 'y']).values
        utci_mean_ts = self.utci.mean(dim=['x', 'y']).values
        
        time_index = np.arange(len(time_data))
        
        # Linear regression
        mrt_slope, mrt_intercept, mrt_r, mrt_p, mrt_se = stats.linregress(time_index, mrt_mean_ts)
        utci_slope, utci_intercept, utci_r, utci_p, utci_se = stats.linregress(time_index, utci_mean_ts)
        
        return {
            'mrt_trend': mrt_slope,
            'mrt_r_squared': mrt_r ** 2,
            'mrt_p_value': mrt_p,
            'utci_trend': utci_slope,
            'utci_r_squared': utci_r ** 2,
            'utci_p_value': utci_p,
        }

    def spatial_anomalies(self):
        """Compute spatial anomalies from mean."""
        mrt_mean = self.mrt_c.mean().values
        utci_mean = self.utci.mean().values
        
        mrt_anom = self.mrt_c - mrt_mean
        utci_anom = self.utci - utci_mean
        
        return mrt_anom, utci_anom

    def regional_statistics(self, regions_dict):
        """
        Compute statistics by region.
        
        Parameters:
            regions_dict: {region_name: (lat_slice, lon_slice), ...}
        """
        regional_stats = {}
        
        for region, (lat_slice, lon_slice) in regions_dict.items():
            mrt_reg = self.mrt_c.sel(latitude=lat_slice, longitude=lon_slice)
            utci_reg = self.utci.sel(latitude=lat_slice, longitude=lon_slice)
            
            regional_stats[region] = {
                'mrt_mean': float(mrt_reg.mean().values),
                'mrt_std': float(mrt_reg.std().values),
                'utci_mean': float(utci_reg.mean().values),
                'utci_std': float(utci_reg.std().values),
            }
        
        return regional_stats

    def distribution_analysis(self):
        """Analyze probability distributions."""
        mrt_flat = self.mrt_c.values.flatten()
        utci_flat = self.utci.values.flatten()
        
        # Remove NaNs
        mrt_clean = mrt_flat[~np.isnan(mrt_flat)]
        utci_clean = utci_flat[~np.isnan(utci_flat)]
        
        # Normality test
        mrt_shapiro = stats.shapiro(mrt_clean[:5000])  # Sample if too large
        utci_shapiro = stats.shapiro(utci_clean[:5000])
        
        return {
            'mrt_skewness': stats.skew(mrt_clean),
            'mrt_kurtosis': stats.kurtosis(mrt_clean),
            'mrt_shapiro_stat': mrt_shapiro.statistic,
            'mrt_shapiro_pval': mrt_shapiro.pvalue,
            'utci_skewness': stats.skew(utci_clean),
            'utci_kurtosis': stats.kurtosis(utci_clean),
            'utci_shapiro_stat': utci_shapiro.statistic,
            'utci_shapiro_pval': utci_shapiro.pvalue,
        }


def compute_heat_accumulation(utci_data, threshold=26):
    """
    Compute heat accumulation index above threshold.
    
    Parameters:
        utci_data (xr.DataArray): UTCI data
        threshold (float): Temperature threshold
        
    Returns:
        xr.DataArray: Accumulated heat above threshold
    """
    heat_excess = (utci_data - threshold).clip(min=0)
    return heat_excess.sum(dim='time')


def compute_cold_accumulation(utci_data, threshold=0):
    """
    Compute cold accumulation index below threshold.
    
    Parameters:
        utci_data (xr.DataArray): UTCI data
        threshold (float): Temperature threshold
        
    Returns:
        xr.DataArray: Accumulated cold below threshold
    """
    cold_excess = (threshold - utci_data).clip(min=0)
    return cold_excess.sum(dim='time')


def compute_thermal_stress_days(utci_data):
    """
    Count number of days with thermal stress.
    
    Parameters:
        utci_data (xr.DataArray): UTCI data with time dimension
        
    Returns:
        dict: Count of different stress categories
    """
    daily_max = utci_data.resample(time='1D').max()
    
    stress_days = {
        'extreme_heat_days': (daily_max > 46).sum().values,
        'strong_heat_days': ((daily_max > 32) & (daily_max <= 46)).sum().values,
        'moderate_heat_days': ((daily_max > 26) & (daily_max <= 32)).sum().values,
        'cold_stress_days': (daily_max < 0).sum().values,
    }
    
    return stress_days


def compute_bioclimatic_indices(mrt_c, utci, net=None):
    """
    Compute various bioclimatic indices.
    
    Parameters:
        mrt_c (xr.DataArray): MRT in Celsius
        utci (xr.DataArray): UTCI
        net (xr.DataArray, optional): NET
        
    Returns:
        dict: Various indices
    """
    indices = {
        'thermal_comfort_index': np.abs(utci - 20).mean().values,  # Distance from comfort zone
        'heat_stress_intensity': (utci[utci > 26]).mean().values if (utci > 26).any() else 0,
        'cold_stress_intensity': (np.abs(utci[utci < 0])).mean().values if (utci < 0).any() else 0,
        'mrt_range': (mrt_c.max() - mrt_c.min()).values,
        'utci_range': (utci.max() - utci.min()).values,
    }
    
    return indices
