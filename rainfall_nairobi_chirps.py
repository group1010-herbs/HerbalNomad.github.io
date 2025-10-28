#!/usr/bin/env python3
"""
rainfall_nairobi_chirps.py

Purpose:
 - Extract CHIRPS timeseries for Nairobi (configurable years)
 - Compute monthly/annual totals and wet-day frequencies
 - Save CSVs and PNGs to an output directory (default: ../docs/data)

Notes:
 - This script can read a local NetCDF/GeoTIFF dataset or read CHIRPS from S3 with s3fs
 - Replace the data-loading block marked "USER: data source" with your CHIRPS source pattern.
"""

import argparse
import os
from pathlib import Path
import xarray as xr
import rioxarray     # needed for some GeoTIFF/CRS builds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Extract Nairobi rainfall (CHIRPS) and save CSV/PNG outputs.")
    p.add_argument("--lat", type=float, default=-1.286389, help="Latitude (default: Nairobi centre)")
    p.add_argument("--lon", type=float, default=36.817223, help="Longitude (default: Nairobi centre)")
    p.add_argument("--start-year", type=int, default=2015)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--wet-threshold", type=float, default=1.0, help="wet day threshold in mm (default 1.0)")
    p.add_argument("--out-dir", type=str, default="../docs/data", help="Output directory for CSVs and PNGs")
    p.add_argument("--source", type=str, default="placeholder",
                   help="Data source: 'placeholder' (no data), 'local_nc', 's3_chirps' or 'local_tif'.")
    p.add_argument("--input-path", type=str, default=None,
                   help="If using 'local_nc' or 'local_tif' provide path or glob pattern (or S3 prefix for s3_chirps).")
    return p.parse_args()

# ----------------------
def ensure_outdir(outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)

def extract_point_timeseries(ds, lat, lon, var='precip'):
    """
    Extract nearest-gridpoint timeseries for variable var from xarray dataset ds.
    ds must have dims ('time','lat','lon') or ('time','y','x').
    """
    if 'lat' in ds.coords and 'lon' in ds.coords:
        ts = ds[var].sel(lat=lat, lon=lon, method='nearest')
    elif 'y' in ds.coords and 'x' in ds.coords:
        ts = ds[var].sel(y=lat, x=lon, method='nearest')
    else:
        raise ValueError("Dataset coordinate names not recognized. Inspect ds.coords")
    return ts

# ----------------------
def load_timeseries(args):
    """
    Load or create a daily timeseries (xarray.DataArray) named 'precip' with daily frequency.
    Replace or expand the s3/local loading logic with your accessible data source.
    """
    if args.source == "placeholder":
        # small synthetic example (for testing the pipeline)
        dates = pd.date_range(start=f'{args.start_year}-01-01', end=f'{args.end_year}-12-31', freq='D')
        np.random.seed(0)
        fake_daily = 2.0 + 4.0 * (np.sin(2*np.pi*(dates.dayofyear/365.0 - 0.25))) + np.random.gamma(0.5, 1.0, len(dates))
        ts_daily = xr.DataArray(fake_daily, coords=[dates], dims=['time'])
        ts_daily.name = 'precip'
        return ts_daily

    elif args.source == "local_nc":
        if not args.input_path:
            raise ValueError("input_path required for local_nc source")
        # Example: open NetCDF containing variable 'precip' with dims time, lat, lon
        ds = xr.open_dataset(args.input_path)
        ts = extract_point_timeseries(ds, args.lat, args.lon, var='precip')
        return ts

    elif args.source == "local_tif":
        # If you have a Cloud Optimized GeoTIFF per day or a stack, open with rioxarray or rasterio
        # You would use xr.open_mfdataset with combine='by_coords' on a list of files.
        if not args.input_path:
            raise ValueError("input_path required for local_tif source")
        ds = xr.open_mfdataset(args.input_path, combine='by_coords')
        # ensure the variable naming matches; many CHIRPS/in-house sets use 'precip' or 'band1'
        varname = 'precip' if 'precip' in ds.data_vars else list(ds.data_vars)[0]
        ts = extract_point_timeseries(ds, args.lat, args.lon, var=varname)
        return ts

    elif args.source == "s3_chirps":
        # Access S3 via s3fs. input_path should be an S3 glob/prefix (e.g. "s3://bucket/path/*.tif" or list pattern)
        # Example usage requires installing s3fs and configuring xarray to use the s3fs filesystem
        if not args.input_path:
            raise ValueError("input_path required for s3_chirps source")
        # Example pattern: "s3://your-bucket/chirps/daily/*.tif"
        # You may need to provide storage_options for xarray, e.g., {"anon": True} for public buckets
        storage_options = {"anon": True}
        ds = xr.open_mfdataset(args.input_path, engine="rasterio", combine='by_coords', backend_kwargs={"storage_options": storage_options})
        varname = 'precip' if 'precip' in ds.data_vars else list(ds.data_vars)[0]
        ts = extract_point_timeseries(ds, args.lat, args.lon, var=varname)
        return ts

    else:
        raise NotImplementedError("Unknown source: " + args.source)

# ----------------------
def run_analysis(args):
    ensure_outdir(args.out_dir)
    print(f"Loading data (source={args.source}) ...")
    ts_daily = load_timeseries(args)

    # ensure pandas friendly index
    if isinstance(ts_daily, xr.DataArray):
        daily = ts_daily.to_series()
    else:
        # if the load returned a pandas Series already
        daily = pd.Series(ts_daily, index=pd.date_range(start=f'{args.start_year}-01-01', end=f'{args.end_year}-12-31', freq='D'))

    daily.index = pd.DatetimeIndex(daily.index)

    # aggregates
    monthly = daily.resample('M').sum()
    annual = daily.resample('Y').sum()

    wet_days_daily = (daily >= args.wet_threshold).astype(int)
    wet_days_monthly = wet_days_daily.resample('M').sum()
    wet_days_annual = wet_days_daily.resample('Y').sum()

    # output file paths (YYYY format in filenames)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%MZ")
    out_prefix = Path(args.out_dir) / f"nairobi_{args.start_year}_{args.end_year}_{timestamp}"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    monthly.to_csv(f"{out_prefix}_monthly.csv", header=['monthly_mm'])
    annual.to_csv(f"{out_prefix}_annual.csv", header=['annual_mm'])
    wet_days_monthly.to_csv(f"{out_prefix}_wetdays_monthly.csv", header=['wet_days'])
    wet_days_annual.to_csv(f"{out_prefix}_wetdays_annual.csv", header=['wet_days'])

    # quick plots
    try:
        plt.figure(figsize=(10,4))
        plt.plot(annual.index.year, annual.values, marker='o')
        plt.title('Nairobi annual rainfall')
        plt.xlabel('Year')
        plt.ylabel('Total annual rainfall (mm)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_annual.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10,4))
        plt.bar(wet_days_annual.index.year, wet_days_annual.values)
        plt.title(f'Nairobi wet days per year (threshold = {args.wet_threshold} mm)')
        plt.xlabel('Year')
        plt.ylabel('Wet days (count)')
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_wetdays.png", dpi=200)
        plt.close()
    except Exception as e:
        print("Plotting failed:", e)

    print("Saved outputs to:", args.out_dir)
    return True

# ----------------------
if __name__ == "__main__":
    args = parse_args()
    run_analysis(args)
