"""Dataset comparison and validation metrics."""

import numpy as np
import xarray as xr
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class DatasetComparison:
    """Compare two thermal datasets."""

    def __init__(self, dataset1, dataset2, dataset1_name="Dataset1", dataset2_name="Dataset2"):
        """Initialize with two datasets."""
        self.ds1 = dataset1
        self.ds2 = dataset2
        self.name1 = dataset1_name
        self.name2 = dataset2_name

    def interpolate_to_common_grid(self, method='nearest'):
        """Interpolate ds2 to ds1's grid."""
        coords = {"latitude": self.ds1.latitude, "longitude": self.ds1.longitude}
        self.ds2_interp = self.ds2.interp(coords=coords, method=method)
        return self.ds2_interp

    def compute_bias(self):
        """Compute mean bias (systematic error)."""
        diff = self.ds1 - self.ds2_interp
        bias = diff.mean().values
        return bias

    def compute_rmse(self):
        """Compute Root Mean Square Error."""
        diff = self.ds1 - self.ds2_interp
        rmse = np.sqrt((diff ** 2).mean()).values
        return rmse

    def compute_mae(self):
        """Compute Mean Absolute Error."""
        diff = np.abs(self.ds1 - self.ds2_interp)
        mae = diff.mean().values
        return mae

    def compute_correlation(self):
        """Compute spatial correlation."""
        ds1_flat = self.ds1.values.flatten()
        ds2_flat = self.ds2_interp.values.flatten()
        
        # Remove NaNs
        valid_mask = ~(np.isnan(ds1_flat) | np.isnan(ds2_flat))
        ds1_valid = ds1_flat[valid_mask]
        ds2_valid = ds2_flat[valid_mask]
        
        if len(ds1_valid) > 2:
            corr, pval = stats.pearsonr(ds1_valid, ds2_valid)
            return corr, pval
        else:
            return np.nan, np.nan

    def compute_nse(self):
        """
        Compute Nash-Sutcliffe Efficiency (model performance metric).
        Range: (-∞, 1], where 1 = perfect fit, 0 = as good as mean, <0 = worse than mean
        """
        diff = (self.ds1 - self.ds2_interp) ** 2
        var_obs = ((self.ds1 - self.ds1.mean()) ** 2)
        
        nse = 1 - (diff.sum() / var_obs.sum())
        return float(nse.values)

    def compute_pbias(self):
        """
        Compute Percent Bias.
        Measures over/underestimation. Range: (-100, 100), where 0 = perfect fit
        """
        numerator = (self.ds2_interp - self.ds1).sum()
        denominator = self.ds1.sum()
        pbias = 100 * (numerator / denominator)
        return float(pbias.values)

    def spatial_difference(self):
        """Compute spatial difference field."""
        diff = self.ds1 - self.ds2_interp
        return diff

    def temporal_difference(self):
        """Compute temporal mean of differences."""
        diff = self.ds1 - self.ds2_interp
        temporal_mean_diff = diff.mean(dim='time')
        return temporal_mean_diff

    def quantile_analysis(self, quantiles=[5, 25, 50, 75, 95]):
        """Compare quantiles."""
        results = {}
        
        ds1_flat = self.ds1.values.flatten()
        ds2_flat = self.ds2_interp.values.flatten()
        
        ds1_clean = ds1_flat[~np.isnan(ds1_flat)]
        ds2_clean = ds2_flat[~np.isnan(ds2_flat)]
        
        for q in quantiles:
            results[f'q{q}_ds1'] = np.percentile(ds1_clean, q)
            results[f'q{q}_ds2'] = np.percentile(ds2_clean, q)
            results[f'q{q}_diff'] = results[f'q{q}_ds2'] - results[f'q{q}_ds1']
        
        return results

    def extreme_agreement(self, threshold_low, threshold_high):
        """
        Assess agreement in extreme values.
        
        Parameters
        ----------
        threshold_low : float
            Lower extreme threshold
        threshold_high : float
            Upper extreme threshold
            
        Returns
        -------
        dict : Agreement metrics
        """
        ds1_extreme = (self.ds1 < threshold_low) | (self.ds1 > threshold_high)
        ds2_extreme = (self.ds2_interp < threshold_low) | (self.ds2_interp > threshold_high)
        
        agreement = (ds1_extreme == ds2_extreme).sum()
        total = ds1_extreme.size
        
        return {
            'agreement_ratio': float(agreement / total),
            'ds1_extremes': int(ds1_extreme.sum()),
            'ds2_extremes': int(ds2_extreme.sum()),
        }

    def summary_metrics(self):
        """Compute all comparison metrics."""
        metrics = {
            'bias': self.compute_bias(),
            'rmse': self.compute_rmse(),
            'mae': self.compute_mae(),
            'correlation': self.compute_correlation()[0],
            'p_value': self.compute_correlation()[1],
            'nse': self.compute_nse(),
            'pbias': self.compute_pbias(),
        }
        return metrics


class ValidationMetrics:
    """Compute validation metrics for observational data comparison."""

    def __init__(self, simulated, observed):
        """
        Initialize validation.
        
        Parameters
        ----------
        simulated : ndarray or xr.DataArray
            Simulated/modeled values
        observed : ndarray or xr.DataArray
            Observed/reference values
        """
        self.sim = np.asarray(simulated).flatten()
        self.obs = np.asarray(observed).flatten()
        
        # Remove NaNs
        valid_mask = ~(np.isnan(self.sim) | np.isnan(self.obs))
        self.sim = self.sim[valid_mask]
        self.obs = self.obs[valid_mask]

    def mse(self):
        """Mean Squared Error."""
        return np.mean((self.sim - self.obs) ** 2)

    def rmse(self):
        """Root Mean Squared Error."""
        return np.sqrt(self.mse())

    def mae(self):
        """Mean Absolute Error."""
        return np.mean(np.abs(self.sim - self.obs))

    def bias(self):
        """Mean Bias Error."""
        return np.mean(self.sim - self.obs)

    def r_squared(self):
        """Coefficient of Determination (R²)."""
        ss_res = np.sum((self.obs - self.sim) ** 2)
        ss_tot = np.sum((self.obs - np.mean(self.obs)) ** 2)
        return 1 - (ss_res / ss_tot)

    def nash_sutcliffe(self):
        """Nash-Sutcliffe Efficiency."""
        ss_res = np.sum((self.obs - self.sim) ** 2)
        ss_tot = np.sum((self.obs - np.mean(self.obs)) ** 2)
        if ss_tot == 0:
            return np.nan
        return 1 - (ss_res / ss_tot)

    def pbias(self):
        """Percent Bias."""
        return 100 * np.sum(self.sim - self.obs) / np.sum(self.obs)

    def nrmse(self):
        """Normalized RMSE (by mean of observations)."""
        mean_obs = np.mean(self.obs)
        if mean_obs == 0:
            return np.nan
        return 100 * self.rmse() / mean_obs

    def correlation(self):
        """Pearson correlation coefficient."""
        if len(self.sim) < 3:
            return np.nan, np.nan
        corr, pval = stats.pearsonr(self.sim, self.obs)
        return corr, pval

    def all_metrics(self):
        """Compute all validation metrics."""
        corr, pval = self.correlation()
        return {
            'rmse': self.rmse(),
            'mae': self.mae(),
            'bias': self.bias(),
            'r_squared': self.r_squared(),
            'nash_sutcliffe': self.nash_sutcliffe(),
            'pbias': self.pbias(),
            'nrmse': self.nrmse(),
            'correlation': corr,
            'p_value': pval,
        }


def compare_mrt_methods(mrt_method1, mrt_method2, method_names=None):
    """
    Compare different MRT calculation methods.
    
    Parameters
    ----------
    mrt_method1 : xr.DataArray
        MRT from method 1 [K]
    mrt_method2 : xr.DataArray
        MRT from method 2 [K]
    method_names : tuple
        Names of methods for output
        
    Returns
    -------
    dict : Comparison metrics
    """
    if method_names is None:
        method_names = ("Method 1", "Method 2")
    
    # Convert to Celsius
    mrt1_c = mrt_method1 - 273.15
    mrt2_c = mrt_method2 - 273.15
    
    comparison = DatasetComparison(mrt1_c, mrt2_c, method_names[0], method_names[1])
    comparison.interpolate_to_common_grid()
    
    return comparison.summary_metrics()


def validate_against_observations(modeled_data, observed_data, variable_name="Temperature"):
    """
    Validate modeled data against observations.
    
    Parameters
    ----------
    modeled_data : ndarray
        Modeled values
    observed_data : ndarray
        Observed values
    variable_name : str
        Name of variable for reporting
        
    Returns
    -------
    dict : Validation metrics and report
    """
    validator = ValidationMetrics(modeled_data, observed_data)
    metrics = validator.all_metrics()
    
    report = f"""
VALIDATION REPORT: {variable_name}
{'='*50}
RMSE:               {metrics['rmse']:.3f}
MAE:                {metrics['mae']:.3f}
Bias:               {metrics['bias']:.3f}
R²:                 {metrics['r_squared']:.3f}
Nash-Sutcliffe:     {metrics['nash_sutcliffe']:.3f}
Percent Bias:       {metrics['pbias']:.2f}%
NRMSE:              {metrics['nrmse']:.2f}%
Correlation:        {metrics['correlation']:.3f} (p={metrics['p_value']:.2e})
{'='*50}
"""
    
    return {'metrics': metrics, 'report': report}
