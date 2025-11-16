"""
Regional Analysis of Thermal Comfort Indices

Analyzes thermal comfort variations across different geographic regions,
including Mediterranean, Central Europe, and North Africa. Computes regional
statistics and extreme event frequencies.

Usage:
    python scripts/analysis_regional.py
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
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

from src.config import DATASETS_DIR, PLOTS_DIR, OUTPUT_DIR

print("\n" + "="*80)
print("REGIONAL THERMAL COMFORT ANALYSIS")
print("="*80)

# Load data
print("\nLoading data...")
daily_file = DATASETS_DIR / "thermal_indices_daily.nc"
if not daily_file.exists():
    print("ERROR: Daily dataset not found. Run main.py first.")
    sys.exit(1)

daily_data = xr.open_dataset(str(daily_file))
mrt = daily_data['mrt_mean']
utci = daily_data['utci_mean']
lat = daily_data.lat.values
lon = daily_data.lon.values

print(f"✓ Data loaded: {mrt.shape}")

# Define regions (lon, lat bounds)
regions = {
    'Mediterranean': {'lon': (-6, 40), 'lat': (30, 45)},
    'Central Europe': {'lon': (-5, 20), 'lat': (45, 56)},
    'North Africa': {'lon': (-10, 30), 'lat': (15, 30)},
    'Western Europe': {'lon': (-10, 5), 'lat': (43, 56)},
    'Eastern Europe': {'lon': (20, 40), 'lat': (45, 60)},
}

print("\n" + "="*80)
print("REGIONAL STATISTICS")
print("="*80)

regional_stats = {}

for region_name, bounds in regions.items():
    lon_mask = (lon >= bounds['lon'][0]) & (lon <= bounds['lon'][1])
    lat_mask = (lat >= bounds['lat'][0]) & (lat <= bounds['lat'][1])
    
    # Subset data
    mrt_region = mrt[:, lat_mask, :][:, :, lon_mask]
    utci_region = utci[:, lat_mask, :][:, :, lon_mask]
    
    if mrt_region.size == 0:
        continue
    
    # Calculate statistics
    mrt_mean = float(mrt_region.mean().values)
    mrt_std = float(mrt_region.std().values)
    mrt_max = float(mrt_region.max().values)
    mrt_min = float(mrt_region.min().values)
    
    utci_mean = float(utci_region.mean().values)
    utci_std = float(utci_region.std().values)
    utci_max = float(utci_region.max().values)
    utci_min = float(utci_region.min().values)
    
    regional_stats[region_name] = {
        'mrt_mean': mrt_mean, 'mrt_std': mrt_std, 'mrt_max': mrt_max, 'mrt_min': mrt_min,
        'utci_mean': utci_mean, 'utci_std': utci_std, 'utci_max': utci_max, 'utci_min': utci_min,
    }
    
    print(f"\n{region_name}:")
    print(f"  MRT:  {mrt_mean:6.2f} ± {mrt_std:5.2f}°C (range: {mrt_min:6.2f} to {mrt_max:6.2f}°C)")
    print(f"  UTCI: {utci_mean:6.2f} ± {utci_std:5.2f}°C (range: {utci_min:6.2f} to {utci_max:6.2f}°C)")

# ============================================================================
# HEAT STRESS EVENTS BY REGION
# ============================================================================

print("\n" + "="*80)
print("HEAT STRESS DAYS BY REGION")
print("="*80)

for region_name, bounds in regions.items():
    lon_mask = (lon >= bounds['lon'][0]) & (lon <= bounds['lon'][1])
    lat_mask = (lat >= bounds['lat'][0]) & (lat <= bounds['lat'][1])
    
    utci_region = utci[:, lat_mask, :][:, :, lon_mask]
    
    if utci_region.size == 0:
        continue
    
    # UTCI categories
    extreme_heat = (utci_region > 46).sum().values
    strong_heat = ((utci_region > 32) & (utci_region <= 46)).sum().values
    moderate_heat = ((utci_region > 26) & (utci_region <= 32)).sum().values
    neutral = ((utci_region > 9) & (utci_region <= 26)).sum().values
    moderate_cold = ((utci_region > 0) & (utci_region <= 9)).sum().values
    strong_cold = ((utci_region > -13) & (utci_region <= 0)).sum().values
    
    total = extreme_heat + strong_heat + moderate_heat + neutral + moderate_cold + strong_cold
    
    if total > 0:
        print(f"\n{region_name}:")
        print(f"  Extreme Heat (>46°C):       {100*extreme_heat/total:5.2f}%")
        print(f"  Strong Heat (32-46°C):     {100*strong_heat/total:5.2f}%")
        print(f"  Moderate Heat (26-32°C):   {100*moderate_heat/total:5.2f}%")
        print(f"  Neutral (9-26°C):          {100*neutral/total:5.2f}%")
        print(f"  Moderate Cold (0-9°C):     {100*moderate_cold/total:5.2f}%")
        print(f"  Strong Cold (<0°C):        {100*strong_cold/total:5.2f}%")

# ============================================================================
# GENERATE REGIONAL VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Regional comparison bar chart
fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

region_names = list(regional_stats.keys())
mrt_means = [regional_stats[r]['mrt_mean'] for r in region_names]
utci_means = [regional_stats[r]['utci_mean'] for r in region_names]

x_pos = np.arange(len(region_names))
width = 0.35

# MRT comparison
axes[0].bar(x_pos - width/2, mrt_means, width, label='MRT', color='red', alpha=0.7)
axes[0].set_ylabel('Temperature (°C)', fontsize=11)
axes[0].set_title('Mean Radiant Temperature by Region', fontsize=12, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(region_names, rotation=30, ha='right')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].legend(fontsize=10)

# Add value labels on bars
for i, v in enumerate(mrt_means):
    axes[0].text(i - width/2, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

# UTCI comparison
axes[1].bar(x_pos - width/2, utci_means, width, label='UTCI', color='blue', alpha=0.7)
axes[1].set_ylabel('Temperature (°C)', fontsize=11)
axes[1].set_title('Universal Thermal Climate Index by Region', fontsize=12, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(region_names, rotation=30, ha='right')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].legend(fontsize=10)

# Add value labels on bars
for i, v in enumerate(utci_means):
    axes[1].text(i - width/2, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

regional_file = PLOTS_DIR / "regional_thermal_comparison.png"
fig.savefig(str(regional_file), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Regional comparison saved: {regional_file.name}")

# Regional maps with bounds
if mrt.size > 0 and len(lon) > 1 and len(lat) > 1:
    try:
        import matplotlib.patches as patches
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        
        # Global MRT
        im0 = axes[0].contourf(lon, lat, mrt.mean(dim='time').values, 
                               cmap='RdYlBu_r', levels=20)
        axes[0].set_xlabel('Longitude', fontsize=10)
        axes[0].set_ylabel('Latitude', fontsize=10)
        axes[0].set_title('Mean Radiant Temperature (Annual Mean)', fontsize=11, fontweight='bold')
        cbar0 = plt.colorbar(im0, ax=axes[0])
        cbar0.set_label('°C', fontsize=10)
        
        # Add region rectangles
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for (region_name, bounds), color in zip(regions.items(), colors):
            lon_min, lon_max = bounds['lon']
            lat_min, lat_max = bounds['lat']
            rect = patches.Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
                                   linewidth=2, edgecolor=color, facecolor='none', 
                                   label=region_name)
            axes[0].add_patch(rect)
        
        axes[0].legend(fontsize=9, loc='upper left')
        
        # Global UTCI
        im1 = axes[1].contourf(lon, lat, utci.mean(dim='time').values, 
                               cmap='RdYlBu_r', levels=20)
        axes[1].set_xlabel('Longitude', fontsize=10)
        axes[1].set_ylabel('Latitude', fontsize=10)
        axes[1].set_title('Universal Thermal Climate Index (Annual Mean)', fontsize=11, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=axes[1])
        cbar1.set_label('°C', fontsize=10)
        
        # Add region rectangles
        for (region_name, bounds), color in zip(regions.items(), colors):
            lon_min, lon_max = bounds['lon']
            lat_min, lat_max = bounds['lat']
            rect = patches.Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
                                   linewidth=2, edgecolor=color, facecolor='none')
            axes[1].add_patch(rect)
        
        maps_file = PLOTS_DIR / "regional_thermal_maps.png"
        fig.savefig(str(maps_file), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Regional maps saved: {maps_file.name}")
    except Exception as e:
        print(f"Note: Regional maps skipped ({str(e)[:50]})")

# Save summary
stats_file = OUTPUT_DIR / "regional_analysis_summary.txt"
with open(str(stats_file), 'w') as f:
    f.write("REGIONAL THERMAL COMFORT ANALYSIS SUMMARY\n")
    f.write("="*70 + "\n\n")
    for region_name, stats in regional_stats.items():
        f.write(f"{region_name}:\n")
        f.write(f"  MRT:  mean={stats['mrt_mean']:.2f}°C, std={stats['mrt_std']:.2f}°C\n")
        f.write(f"  UTCI: mean={stats['utci_mean']:.2f}°C, std={stats['utci_std']:.2f}°C\n")
        f.write(f"  Range (MRT): {stats['mrt_min']:.2f} to {stats['mrt_max']:.2f}°C\n")
        f.write(f"  Range (UTCI): {stats['utci_min']:.2f} to {stats['utci_max']:.2f}°C\n\n")

print(f"✓ Summary saved: {stats_file.name}")

print("\n" + "="*80)
print("REGIONAL ANALYSIS COMPLETE")
print("="*80 + "\n")
