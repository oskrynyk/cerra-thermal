"""Configuration: paths, domains, and processing parameters."""

from pathlib import Path

# =============================================================================
# PROJECT ROOT AND BASE DIRECTORIES
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# DATA DIRECTORIES
# =============================================================================

DATA_DIR = Path(r'C:\Users\olehs\cerra-thermal')

RAW_DATA_DIR = DATA_DIR / 'raw'

PROCESSED_DATA_DIR = DATA_DIR / 'processed'

ERA5_DATA_DIR = DATA_DIR / 'input_era5'

EOBS_DATA_DIR = Path(r'C:\Users\olehs\e-obs')

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

OUTPUT_DIR = PROJECT_ROOT / 'outputs'

PLOTS_DIR = OUTPUT_DIR / 'plots'

DATASETS_DIR = OUTPUT_DIR / 'datasets'

RESULTS_DIR = OUTPUT_DIR / 'results'

# Create output directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# SPATIAL DOMAIN CONFIGURATION
# =============================================================================

# Define geographic domain (Europe and North Africa)
DOMAIN = {
    'north': 72.0,
    'south': 25.0,
    'west': -10.0,
    'east': 40.0,
}

# =============================================================================
# DATA FILE NAMING CONVENTIONS
# =============================================================================

# Universal file names for input datasets
RADIATION_FILE = 'radiation.nc'

WIND_FILE = 'wind.nc'

TEMPERATURE_HUMIDITY_FILE = 't2m_rh.nc'
"""Universal filename for temperature and humidity (replaces t2m_rh_2017.nc, etc.)"""

# File patterns for multi-file datasets
RADIATION_GRIB = 'radiation.grib'
"""GRIB format radiation data."""

WIND_GRIB = 'wind.grib'
"""GRIB format wind data."""

# =============================================================================
# PROCESSING PARAMETERS
# =============================================================================

# Clipping ranges for quality control
T2M_CLIP = {'min': -50, 'max': 50}
"""Temperature clipping range (°C)."""

MRT_CLIP = {'min': -30, 'max': 70}
"""Mean Radiant Temperature clipping range (°C)."""

WIND_CLIP = {'min': 0.5, 'max': 17}
"""Wind speed clipping range (m/s)."""

# Standard heights and parameters
STANDARD_HEIGHT = 1.2  # meters
"""Standard height for meteorological measurements (m)."""

REFERENCE_HEIGHT = 10.0  # meters
"""Reference height for wind measurements (m)."""

# MRT calculation parameters
SOLAR_ABSORPTION_COEFFICIENT = 0.68
"""Solar absorption coefficient for MRT calculation."""

LONGWAVE_ABSORPTION_COEFFICIENT = 0.97
"""Longwave absorption coefficient for MRT calculation."""

# =============================================================================
# RADIATION VARIABLES MAPPING
# =============================================================================

# Standard variable names in radiation datasets
RADIATION_VARS = {
    'ssrd': 'Surface Solar Radiation Downward (W m-2)',
    'ssr': 'Surface Net Solar Radiation (W m-2)',
    'strd': 'Surface Thermal (Longwave) Radiation Downward (W m-2)',
    'strr': 'Surface Net Thermal (Longwave) Radiation (W m-2)',
    'fdir': 'Direct Short Wave Radiation Flux (W m-2)',
    'tidirswrf': 'Time-Integrated Direct Short Wave Radiation Flux (W m-2)',
}

# Wind variables
WIND_VARS = {
    'si10': '10m Wind Speed (m/s)',  # CERRA
    'u10': '10m U-wind component (m/s)',  # ERA5
    'v10': '10m V-wind component (m/s)',  # ERA5
}

# Temperature and humidity variables
TEMP_HUMIDITY_VARS = {
    't2m': '2m Temperature (K)',
    'r2': '2m Relative Humidity (%)',  # CERRA
    'd2m': '2m Dew Point Temperature (K)',  # ERA5
}

# =============================================================================
# CONSTANTS FOR CALCULATIONS
# =============================================================================

# Physical constants
STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m2·K4)
"""Stefan-Boltzmann constant."""

# Time constants
HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60
SECONDS_PER_HOUR = 3600

# Angular constants
DEGREES_TO_RADIANS = 3.141592653589793 / 180
RADIANS_TO_DEGREES = 180 / 3.141592653589793

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = 'INFO'
"""Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL."""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_project_root():
    """
    Get the project root directory.
    
    Returns:
        Path: Project root directory
    """
    return PROJECT_ROOT

def ensure_output_dirs():
    """
    Ensure all output directories exist.
    Creates directories if they don't already exist.
    """
    for directory in [OUTPUT_DIR, PLOTS_DIR, DATASETS_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def print_config():
    """
    Print current configuration for debugging.
    """
    print("=" * 70)
    print("THERMAL COMFORT PROJECT CONFIGURATION")
    print("=" * 70)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"\nData Directories:")
    print(f"  - Main: {DATA_DIR}")
    print(f"  - Raw: {RAW_DATA_DIR}")
    print(f"  - Processed: {PROCESSED_DATA_DIR}")
    print(f"  - ERA5: {ERA5_DATA_DIR}")
    print(f"  - E-OBS: {EOBS_DATA_DIR}")
    print(f"\nOutput Directories:")
    print(f"  - Main: {OUTPUT_DIR}")
    print(f"  - Plots: {PLOTS_DIR}")
    print(f"  - Datasets: {DATASETS_DIR}")
    print(f"  - Results: {RESULTS_DIR}")
    print(f"\nSpatial Domain:")
    print(f"  - N: {DOMAIN['north']}°, S: {DOMAIN['south']}°")
    print(f"  - E: {DOMAIN['east']}°, W: {DOMAIN['west']}°")
    print("=" * 70)

# Ensure output directories exist on import
ensure_output_dirs()
