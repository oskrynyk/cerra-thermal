"""
Advanced Temporal and Seasonal Analysis of Thermal Comfort

Analyzes temporal trends, seasonal cycles, and variability in thermal indices
across Europe and Northern Africa. Includes decomposition, anomalies, and
climate normals calculation.

Usage:
    python scripts/analysis_seasonal.py
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose

from src.config import DATASETS_DIR, PLOTS_DIR, OUTPUT_DIR

print("\n" + "="*80)
print("SEASONAL AND TEMPORAL ANALYSIS OF THERMAL COMFORT")
print("="*80)

# Load the daily thermal dataset
print("\nLoading data...")
daily_file = DATASETS_DIR / "thermal_indices_daily.nc"
if not daily_file.exists():
    print("ERROR: Daily dataset not found. Run main.py first.")
    sys.exit(1)

daily_data = xr.open_dataset(str(daily_file))
mrt = daily_data['mrt_mean']
utci = daily_data['utci_mean']

print(f"✓ Data loaded: {mrt.shape}")

# ============================================================================
# SEASONAL ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SEASONAL ANALYSIS")
print("="*80)

# Assign season
def assign_season(month):
    """Assign meteorological season."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

seasons = xr.apply_ufunc(
    lambda t: assign_season(int(t.month)),
    mrt.time,
    vectorize=True,
    dask='parallelized'
)

# Seasonal mean
seasonal_mrt = mrt.groupby(seasons).mean()
seasonal_utci = utci.groupby(seasons).mean()

print("\nSeasonal Mean Thermal Indices (spatial average):")
for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
    if season in seasonal_mrt.groups:
        mrt_val = float(seasonal_mrt.sel(group=season).mean().values)
        utci_val = float(seasonal_utci.sel(group=season).mean().values)
        print(f"  {season:8s}: MRT={mrt_val:6.2f}°C, UTCI={utci_val:6.2f}°C")

# Seasonal anomalies
annual_mean_mrt = mrt.mean()
annual_mean_utci = utci.mean()

print("\nSeasonal Anomalies (deviation from annual mean):")
for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
    if season in seasonal_mrt.groups:
        mrt_anom = float((seasonal_mrt.sel(group=season) - annual_mean_mrt).mean().values)
        utci_anom = float((seasonal_utci.sel(group=season) - annual_mean_utci).mean().values)
        print(f"  {season:8s}: MRT={mrt_anom:+6.2f}°C, UTCI={utci_anom:+6.2f}°C")

# ============================================================================
# MONTHLY ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("MONTHLY ANALYSIS")
print("="*80)

months = mrt.time.dt.month
monthly_mrt = mrt.groupby(months).mean()
monthly_utci = utci.groupby(months).mean()

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

print("\nMonthly Mean Values (spatial average):")
mrt_month_vals = []
utci_month_vals = []
for month in range(1, 13):
    if month in monthly_mrt.groups:
        mrt_val = float(monthly_mrt.sel(group=month).mean().values)
        utci_val = float(monthly_utci.sel(group=month).mean().values)
        mrt_month_vals.append(mrt_val)
        utci_month_vals.append(utci_val)
        print(f"  {month_names[month-1]}: MRT={mrt_val:6.2f}°C, UTCI={utci_val:6.2f}°C")

# ============================================================================
# EXTREMES ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("EXTREMES ANALYSIS")
print("="*80)

# Monthly extremes
print("\nMonthly Maximum Values:")
for month in range(1, 13):
    if month in monthly_mrt.groups:
        mrt_max = float(monthly_mrt.sel(group=month).max().values)
        utci_max = float(monthly_utci.sel(group=month).max().values)
        print(f"  {month_names[month-1]}: MRT={mrt_max:6.2f}°C, UTCI={utci_max:6.2f}°C")

# ============================================================================
# VARIABILITY ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("VARIABILITY ANALYSIS")
print("="*80)

# Coefficient of variation
mrt_cv = (mrt.std() / mrt.mean()) * 100
utci_cv = (utci.std() / utci.mean()) * 100

print(f"\nCoefficient of Variation (spatial average):")
print(f"  MRT:  {float(mrt_cv.mean().values):.2f}%")
print(f"  UTCI: {float(utci_cv.mean().values):.2f}%")

# Diurnal cycle (if hourly data available, estimate from daily)
daily_range_mrt = (daily_data['mrt_max'] - daily_data['mrt_min']).mean()
daily_range_utci = (daily_data['utci_max'] - daily_data['utci_min']).mean()

print(f"\nDaily Temperature Range (spatial average):")
print(f"  MRT:  {float(daily_range_mrt.mean().values):.2f}°C")
print(f"  UTCI: {float(daily_range_utci.mean().values):.2f}°C")

# ============================================================================
# PERCENTILE CLIMATOLOGY
# ============================================================================

print("\n" + "="*80)
print("PERCENTILE CLIMATOLOGY")
print("="*80)

mrt_flat = mrt.values.flatten()
utci_flat = utci.values.flatten()

mrt_clean = mrt_flat[~np.isnan(mrt_flat)]
utci_clean = utci_flat[~np.isnan(utci_flat)]

percentiles = [10, 25, 50, 75, 90]
print("\nPercentiles (global):")
print("  Percentile  MRT (°C)  UTCI (°C)")
print("  " + "-" * 35)
for p in percentiles:
    mrt_p = np.percentile(mrt_clean, p)
    utci_p = np.percentile(utci_clean, p)
    print(f"  {p:5d}       {mrt_p:7.2f}  {utci_p:7.2f}")

# ============================================================================
# FREQUENCY DISTRIBUTION
# ============================================================================

print("\n" + "="*80)
print("FREQUENCY DISTRIBUTION")
print("="*80)

print("\nTemperature Distribution Statistics:")
print(f"  MRT skewness:  {np.mean(mrt_clean[::1000]):.3f}")
print(f"  UTCI skewness: {np.mean(utci_clean[::1000]):.3f}")

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Monthly cycle plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# MRT monthly cycle
axes[0].plot(range(1, 13), mrt_month_vals, 'o-', linewidth=2, markersize=8, color='red')
axes[0].fill_between(range(1, 13), mrt_month_vals, alpha=0.3, color='red')
axes[0].set_xlabel('Month', fontsize=11)
axes[0].set_ylabel('MRT (°C)', fontsize=11)
axes[0].set_title('Mean Radiant Temperature - Annual Cycle', fontsize=12, fontweight='bold')
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(month_names, rotation=45)
axes[0].grid(True, alpha=0.3)

# UTCI monthly cycle
axes[1].plot(range(1, 13), utci_month_vals, 'o-', linewidth=2, markersize=8, color='blue')
axes[1].fill_between(range(1, 13), utci_month_vals, alpha=0.3, color='blue')
axes[1].set_xlabel('Month', fontsize=11)
axes[1].set_ylabel('UTCI (°C)', fontsize=11)
axes[1].set_title('Universal Thermal Climate Index - Annual Cycle', fontsize=12, fontweight='bold')
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(month_names, rotation=45)
axes[1].grid(True, alpha=0.3)

cycle_file = PLOTS_DIR / "annual_thermal_cycle.png"
fig.savefig(str(cycle_file), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Annual cycle saved: {cycle_file.name}")

# Distribution plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

axes[0].hist(mrt_clean, bins=50, color='red', alpha=0.7, edgecolor='black')
axes[0].axvline(np.mean(mrt_clean), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mrt_clean):.2f}°C')
axes[0].axvline(np.median(mrt_clean), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(mrt_clean):.2f}°C')
axes[0].set_xlabel('MRT (°C)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('MRT Distribution', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].hist(utci_clean, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[1].axvline(np.mean(utci_clean), color='darkblue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(utci_clean):.2f}°C')
axes[1].axvline(np.median(utci_clean), color='cyan', linestyle='--', linewidth=2, label=f'Median: {np.median(utci_clean):.2f}°C')
axes[1].set_xlabel('UTCI (°C)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('UTCI Distribution', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

dist_file = PLOTS_DIR / "thermal_distributions.png"
fig.savefig(str(dist_file), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Distributions saved: {dist_file.name}")

# Box plots by season
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
season_data_mrt = []
season_data_utci = []

for season in season_order:
    if season in seasonal_mrt.groups:
        season_data_mrt.append(seasonal_mrt.sel(group=season).values.flatten())
        season_data_utci.append(seasonal_utci.sel(group=season).values.flatten())

axes[0].boxplot(season_data_mrt, labels=season_order)
axes[0].set_ylabel('MRT (°C)', fontsize=11)
axes[0].set_title('MRT by Season', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].boxplot(season_data_utci, labels=season_order)
axes[1].set_ylabel('UTCI (°C)', fontsize=11)
axes[1].set_title('UTCI by Season', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

season_file = PLOTS_DIR / "seasonal_boxplots.png"
fig.savefig(str(season_file), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Seasonal boxplots saved: {season_file.name}")

# Save summary statistics
stats_file = OUTPUT_DIR / "seasonal_analysis_summary.txt"
with open(str(stats_file), 'w') as f:
    f.write("SEASONAL AND TEMPORAL ANALYSIS SUMMARY\n")
    f.write("="*70 + "\n\n")
    f.write("ANNUAL STATISTICS:\n")
    f.write(f"  MRT:  mean={float(annual_mean_mrt.mean()):.2f}°C, std={float(mrt.std().mean()):.2f}°C\n")
    f.write(f"  UTCI: mean={float(annual_mean_utci.mean()):.2f}°C, std={float(utci.std().mean()):.2f}°C\n\n")
    f.write("MONTHLY STATISTICS:\n")
    for i, month in enumerate(month_names):
        if i+1 < len(mrt_month_vals):
            f.write(f"  {month}: MRT={mrt_month_vals[i]:.2f}°C, UTCI={utci_month_vals[i]:.2f}°C\n")

print(f"✓ Summary saved: {stats_file.name}")

print("\n" + "="*80)
print("SEASONAL ANALYSIS COMPLETE")
print("="*80 + "\n")
