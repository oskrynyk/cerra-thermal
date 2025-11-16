"""
Extreme Events and Anomalies Analysis

Identifies and characterizes extreme thermal events including heat waves,
cold snaps, and anomalous conditions. Includes event detection, duration analysis,
and spatial extent characterization.

Usage:
    python scripts/analysis_extremes.py
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

from src.config import DATASETS_DIR, PLOTS_DIR, OUTPUT_DIR

print("\n" + "="*80)
print("EXTREME THERMAL EVENTS ANALYSIS")
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

print(f"✓ Data loaded: {mrt.shape}")

# ============================================================================
# HEAT WAVES DETECTION
# ============================================================================

print("\n" + "="*80)
print("HEAT WAVE DETECTION")
print("="*80)

# Define heat wave: 3+ consecutive days with UTCI > 32°C (Strong Heat)
heat_stress = (utci > 32).values

# Spatial coverage for heat waves
heat_days = heat_stress.sum(axis=0)
utci_mean_daily = utci.mean(dim=['lat', 'lon']).values

heat_wave_events = []
in_event = False
start_day = None
event_intensity = 0

for day in range(len(utci_mean_daily)):
    if utci_mean_daily[day] > 32:
        if not in_event:
            start_day = day
            in_event = True
            event_intensity = utci_mean_daily[day]
        else:
            event_intensity = max(event_intensity, utci_mean_daily[day])
    else:
        if in_event and day - start_day >= 3:
            duration = day - start_day
            heat_wave_events.append({
                'start': start_day,
                'end': day,
                'duration': duration,
                'max_intensity': event_intensity
            })
        in_event = False

print(f"\nHeat Wave Events (UTCI > 32°C, duration ≥ 3 days):")
print(f"  Total events: {len(heat_wave_events)}")

if heat_wave_events:
    print(f"\n  Event Summary:")
    avg_duration = np.mean([e['duration'] for e in heat_wave_events])
    max_intensity = np.max([e['max_intensity'] for e in heat_wave_events])
    print(f"    Average duration: {avg_duration:.1f} days")
    print(f"    Maximum intensity: {max_intensity:.2f}°C")
    print(f"    Total heat wave days: {sum([e['duration'] for e in heat_wave_events])}")
    
    print(f"\n  Event Details:")
    for i, event in enumerate(heat_wave_events[:5]):
        print(f"    Event {i+1}: Days {event['start']}-{event['end']}, "
              f"Duration: {event['duration']} days, Max: {event['max_intensity']:.2f}°C")

# ============================================================================
# COLD SNAPS DETECTION
# ============================================================================

print("\n" + "="*80)
print("COLD SNAP DETECTION")
print("="*80)

# Define cold snap: 3+ consecutive days with UTCI < 0°C (Below freezing)
cold_stress = (utci < 0).values
cold_wave_events = []
in_event = False
start_day = None
event_intensity = 0

for day in range(len(utci_mean_daily)):
    if utci_mean_daily[day] < 0:
        if not in_event:
            start_day = day
            in_event = True
            event_intensity = utci_mean_daily[day]
        else:
            event_intensity = min(event_intensity, utci_mean_daily[day])
    else:
        if in_event and day - start_day >= 3:
            duration = day - start_day
            cold_wave_events.append({
                'start': start_day,
                'end': day,
                'duration': duration,
                'min_intensity': event_intensity
            })
        in_event = False

print(f"\nCold Snap Events (UTCI < 0°C, duration ≥ 3 days):")
print(f"  Total events: {len(cold_wave_events)}")

if cold_wave_events:
    print(f"\n  Event Summary:")
    avg_duration = np.mean([e['duration'] for e in cold_wave_events])
    min_intensity = np.min([e['min_intensity'] for e in cold_wave_events])
    print(f"    Average duration: {avg_duration:.1f} days")
    print(f"    Minimum intensity: {min_intensity:.2f}°C")
    print(f"    Total cold snap days: {sum([e['duration'] for e in cold_wave_events])}")
    
    print(f"\n  Event Details:")
    for i, event in enumerate(cold_wave_events[:5]):
        print(f"    Event {i+1}: Days {event['start']}-{event['end']}, "
              f"Duration: {event['duration']} days, Min: {event['min_intensity']:.2f}°C")

# ============================================================================
# SPATIAL ANOMALIES
# ============================================================================

print("\n" + "="*80)
print("SPATIAL ANOMALY ANALYSIS")
print("="*80)

# Compute anomalies
annual_mean = utci.mean()
utci_anomaly = utci - annual_mean

# Maximum positive anomaly
max_anom_idx = np.nanargmax(utci_anomaly.values)
max_anom_idx_3d = np.unravel_index(max_anom_idx, utci_anomaly.shape)
max_anom = utci_anomaly.values[max_anom_idx_3d]
max_utci_value = utci.values[max_anom_idx_3d]

# Maximum negative anomaly
min_anom_idx = np.nanargmin(utci_anomaly.values)
min_anom_idx_3d = np.unravel_index(min_anom_idx, utci_anomaly.shape)
min_anom = utci_anomaly.values[min_anom_idx_3d]
min_utci_value = utci.values[min_anom_idx_3d]

print(f"\nAnomalies (relative to {annual_mean.values:.2f}°C annual mean):")
print(f"  Maximum positive anomaly: +{max_anom:.2f}°C (UTCI={max_utci_value:.2f}°C)")
print(f"  Maximum negative anomaly: {min_anom:.2f}°C (UTCI={min_utci_value:.2f}°C)")

# Spatial extent of extreme anomalies
extreme_hot_anom = (utci_anomaly > np.percentile(utci_anomaly.values, 95)).sum().values
extreme_cold_anom = (utci_anomaly < np.percentile(utci_anomaly.values, 5)).sum().values
total_cells = utci_anomaly.size - np.isnan(utci_anomaly.values).sum()

print(f"\n  Extreme positive anomalies (>95th %ile): {100*extreme_hot_anom/total_cells:.2f}% of grid")
print(f"  Extreme negative anomalies (<5th %ile): {100*extreme_cold_anom/total_cells:.2f}% of grid")

# ============================================================================
# EXTREME VALUE STATISTICS
# ============================================================================

print("\n" + "="*80)
print("EXTREME VALUE STATISTICS")
print("="*80)

utci_flat = utci.values.flatten()
utci_clean = utci_flat[~np.isnan(utci_flat)]

percentiles = [1, 5, 10, 90, 95, 99]
print(f"\nUTCI Percentiles:")
for p in percentiles:
    val = np.percentile(utci_clean, p)
    print(f"  {p:2d}th percentile: {val:6.2f}°C")

# Return period analysis (simplified)
sorted_utci = np.sort(utci_clean)[::-1]
print(f"\nReturn Period (days per occurrence):")
for i, percentile in enumerate([99, 95, 90, 50, 10, 5, 1]):
    idx = int(len(sorted_utci) * (100 - percentile) / 100)
    if idx < len(sorted_utci):
        val = sorted_utci[idx]
        return_period = len(sorted_utci) / (len(sorted_utci) - idx) if idx > 0 else 1
        print(f"  UTCI > {val:6.2f}°C: ~{return_period:.1f} days per occurrence")

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Time series with events marked
fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)

time_days = np.arange(len(utci_mean_daily))
ax.plot(time_days, utci_mean_daily, 'b-', linewidth=1.5, label='Daily UTCI', alpha=0.7)

# Mark heat waves
for event in heat_wave_events[:10]:  # Show first 10
    ax.axvspan(event['start'], event['end'], alpha=0.2, color='red')

# Mark cold snaps
for event in cold_wave_events[:10]:  # Show first 10
    ax.axvspan(event['start'], event['end'], alpha=0.2, color='blue')

# Add thresholds
ax.axhline(32, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Heat stress threshold (32°C)')
ax.axhline(0, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Cold stress threshold (0°C)')
ax.axhline(annual_mean.values, color='green', linestyle=':', linewidth=1, alpha=0.7, label='Annual mean')

ax.set_xlabel('Day of Period', fontsize=11)
ax.set_ylabel('UTCI (°C)', fontsize=11)
ax.set_title('Daily UTCI Time Series with Extreme Events', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

ts_file = PLOTS_DIR / "extreme_events_timeline.png"
fig.savefig(str(ts_file), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Timeline saved: {ts_file.name}")

# Event duration distribution
if heat_wave_events or cold_wave_events:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    
    if heat_wave_events:
        hw_durations = [e['duration'] for e in heat_wave_events]
        axes[0].hist(hw_durations, bins=10, color='red', alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(hw_durations), color='darkred', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(hw_durations):.1f} days')
        axes[0].set_xlabel('Duration (days)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title(f'Heat Wave Duration Distribution ({len(heat_wave_events)} events)', 
                         fontsize=11, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')
    
    if cold_wave_events:
        cw_durations = [e['duration'] for e in cold_wave_events]
        axes[1].hist(cw_durations, bins=10, color='blue', alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(cw_durations), color='darkblue', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(cw_durations):.1f} days')
        axes[1].set_xlabel('Duration (days)', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title(f'Cold Snap Duration Distribution ({len(cold_wave_events)} events)', 
                         fontsize=11, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
    
    event_file = PLOTS_DIR / "extreme_event_durations.png"
    fig.savefig(str(event_file), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Event durations saved: {event_file.name}")

# Anomaly distribution
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

utci_anom_flat = utci_anomaly.values.flatten()
utci_anom_clean = utci_anom_flat[~np.isnan(utci_anom_flat)]

ax.hist(utci_anom_clean, bins=50, color='purple', alpha=0.7, edgecolor='black')
ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='No anomaly')
ax.axvline(np.mean(utci_anom_clean), color='darkviolet', linestyle='--', linewidth=1.5, 
          label=f'Mean: {np.mean(utci_anom_clean):.2f}°C')
ax.set_xlabel('UTCI Anomaly (°C)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of UTCI Anomalies', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

anom_file = PLOTS_DIR / "utci_anomaly_distribution.png"
fig.savefig(str(anom_file), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Anomaly distribution saved: {anom_file.name}")

# Save summary
stats_file = OUTPUT_DIR / "extremes_analysis_summary.txt"
with open(str(stats_file), 'w') as f:
    f.write("EXTREME THERMAL EVENTS ANALYSIS SUMMARY\n")
    f.write("="*70 + "\n\n")
    f.write("HEAT WAVES:\n")
    f.write(f"  Total events: {len(heat_wave_events)}\n")
    if heat_wave_events:
        f.write(f"  Average duration: {np.mean([e['duration'] for e in heat_wave_events]):.1f} days\n")
        f.write(f"  Maximum intensity: {np.max([e['max_intensity'] for e in heat_wave_events]):.2f}°C\n")
    f.write(f"\nCOLD SNAPS:\n")
    f.write(f"  Total events: {len(cold_wave_events)}\n")
    if cold_wave_events:
        f.write(f"  Average duration: {np.mean([e['duration'] for e in cold_wave_events]):.1f} days\n")
        f.write(f"  Minimum intensity: {np.min([e['min_intensity'] for e in cold_wave_events]):.2f}°C\n")
    f.write(f"\nANOMALIES:\n")
    f.write(f"  Maximum positive: +{max_anom:.2f}°C\n")
    f.write(f"  Maximum negative: {min_anom:.2f}°C\n")

print(f"✓ Summary saved: {stats_file.name}")

print("\n" + "="*80)
print("EXTREME EVENTS ANALYSIS COMPLETE")
print("="*80 + "\n")
