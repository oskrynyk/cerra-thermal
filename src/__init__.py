"""
Thermal Comfort Analysis Package

A comprehensive toolkit for analyzing thermal comfort in complex terrains using
high-resolution climate data. Combines CERRA, ERA5, and E-OBS datasets to compute
thermal stress indices (MRT, UTCI, NET) and classify atmospheric patterns.

Key Modules:
    - config: Configuration and paths
    - thermal_utils: Core thermal comfort calculations
    - func: Utility functions
    - e-obs-thermal: E-OBS data processing
    - era5: ERA5 data processing
    - atmo-class: Atmospheric classification
    - post_data_prep: Post-processing utilities

Example:
    >>> from src.config import DATA_DIR, OUTPUT_DIR
    >>> from src.thermal_utils import calculate_net
    >>> import xarray as xr
    >>>
    >>> # Load data
    >>> ds = xr.open_dataset(DATA_DIR / 'radiation.nc')
    >>> # Process...
"""

__version__ = "1.0.0"
__author__ = "Oleh Skrynyk"
__license__ = "MIT"

from . import config

__all__ = ['config']
