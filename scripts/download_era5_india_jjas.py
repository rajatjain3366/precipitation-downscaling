#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Provenance: copied unchanged from data/ERA5/code.py (only this comment block
# was added). This script, with the default JJAS months (6-9), produced the
# era5_{year}_coarse.nc files on two machines: 1971-1989 and 2009-2016 on
# Kartik's laptop, 1990-2008 on the teammate's. data/India/ERA5_full is the
# teammate's complete 1971-2016 copy; the 27 years present on both machines
# were byte-identical.
# Each output file holds daily means (resample('1D').mean() over the 24 hourly
# ERA5 fields of each UTC day), 4x4 block-mean coarsened from 38.5 N / 66.5 E.
# Not the repository's own scripts/download_era5_india.py (CDS daily statistics).
# ---------------------------------------------------------------------------
"""
download_era5_standalone.py
===========================
Self-contained, automated ERA5 downloader and 1°x1° coarsener for the India domain.
Designed to be shared with collaborators with zero manual configuration.

By default, downloads the 4 Southwest Monsoon months:
  **June, July, August, September (JJAS)**
as requested for precipitation downscaling research (reducing download time 
and storage by 3x compared to full-year downloads).

Features:
- Self-checks dependencies (`cdsapi`, `xarray`, `netCDF4`, `numpy`) with auto-install prompt.
- Interactive CDS API setup helper if `~/.cdsapirc` is missing.
- Supports single years (`--year 1991`), year ranges (`--years 1991-1995`), or multiple years (`--years 1991 1992`).
- Downloads 850 hPa predictors: q, t, u, v, w over India (38.5°N - 6.5°N, 66.5°E - 100.0°E).
- Automatically converts hourly data to daily mean and 1.0°x1.0° coarse grid.
- Auto-cleans large raw hourly files (saving ~1.5 GB/year per download).
- Resume capability: skips years and months already downloaded.

Quick Start:
  # Download June to September (JJAS) for 1991:
  python download_era5_standalone.py --year 1991

  # Download multiple years (June to September):
  python download_era5_standalone.py --years 1991-1995

  # Download all 12 months if ever needed:
  python download_era5_standalone.py --year 1991 --all-months
"""

import os
import sys
import time
import glob
import calendar
import argparse
import subprocess

# -----------------------------------------------------------------------------
# 1. DEPENDENCY CHECK & AUTO-INSTALL
# -----------------------------------------------------------------------------
REQUIRED_PACKAGES = {
    "cdsapi": "cdsapi",
    "xarray": "xarray",
    "netCDF4": "netCDF4",
    "numpy": "numpy",
}

def check_dependencies():
    missing = []
    for import_name, pkg_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)
    
    if missing:
        print("=" * 70)
        print("MISSING PYTHON PACKAGES DETECTED:")
        print(f"  {', '.join(missing)}")
        print("=" * 70)
        user_input = input("Would you like to install them automatically using pip? (y/n): ").strip().lower()
        if user_input in ['y', 'yes']:
            print(f"\nInstalling {missing} via pip...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("[OK] Dependencies installed successfully!\n")
        else:
            print("\nPlease install the missing packages manually by running:")
            print(f"  pip install {' '.join(missing)}\n")
            sys.exit(1)

check_dependencies()

import cdsapi
import xarray as xr
import numpy as np


# -----------------------------------------------------------------------------
# 2. INTERACTIVE CDS API KEY CONFIGURATION HELPER
# -----------------------------------------------------------------------------
def setup_cdsapirc_if_needed():
    rc_path = os.path.join(os.path.expanduser("~"), ".cdsapirc")
    if os.path.exists(rc_path):
        return True

    print("\n" + "=" * 70)
    print(" ECMWF COPERNICUS CDS API KEY SETUP REQUIRED")
    print("=" * 70)
    print("A CDS API key was not found at ~/.cdsapirc.")
    print("To download ERA5 reanalysis data from Copernicus:")
    print("  1. Create a free account at: https://cds.climate.copernicus.eu")
    print("  2. Open your profile page:    https://cds.climate.copernicus.eu/profile")
    print("  3. Copy your 'Personal Access Token' (API key).")
    print("  4. NOTE: Make sure to accept the Terms of Use for ERA5 at:")
    print("     https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels")
    print("     (Scroll down to the 'Terms of use' tab and click 'Accept').")
    print("=" * 70)

    try:
        token = input("\nEnter your CDS Personal Access Token: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
        sys.exit(1)

    if not token:
        print("[ERROR] Token cannot be empty. Setup aborted.")
        sys.exit(1)

    content = f"url: https://cds.climate.copernicus.eu/api\nkey: {token}\n"
    try:
        with open(rc_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[OK] Successfully saved API key to {rc_path}!\n")
        return True
    except Exception as e:
        print(f"[ERROR] Could not write {rc_path}: {e}")
        sys.exit(1)


# -----------------------------------------------------------------------------
# 3. ARGUMENT PARSER
# -----------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Download and coarsen ERA5 850 hPa reanalysis predictors over India (June-Sept default)."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Single year to download (e.g. --year 1991)"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=None,
        help="List of years or year range (e.g. --years 1991 1992 OR --years 1991-1995)"
    )
    parser.add_argument(
        "--months",
        nargs="+",
        type=int,
        default=[6, 7, 8, 9],
        help="Months to download (default: 6 7 8 9 for June to September Southwest Monsoon)"
    )
    parser.add_argument(
        "--all-months",
        action="store_true",
        default=False,
        help="Download all 12 months (Jan-Dec) instead of only June-September"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/India/ERA5",
        help="Directory to save final processed files (default: ./data/India/ERA5)"
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="./data/India/raw_temp",
        help="Directory for temporary raw monthly files (cleaned up automatically)"
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        default=False,
        help="Keep large raw hourly monthly files instead of deleting them"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-download and re-process even if output file already exists"
    )
    return parser.parse_args()


def parse_year_list(year_arg, years_arg):
    """Parses year arguments into a sorted list of unique integers."""
    years = []
    if year_arg is not None:
        years.append(int(year_arg))

    if years_arg is not None:
        for item in years_arg:
            if "-" in item:
                parts = item.split("-")
                start, end = int(parts[0]), int(parts[1])
                years.extend(list(range(start, end + 1)))
            else:
                years.append(int(item))

    if not years:
        years = [1991]  # default fallback

    return sorted(list(set(years)))


# -----------------------------------------------------------------------------
# 4. PROCESSING LOGIC
# -----------------------------------------------------------------------------
def process_month_to_daily_and_coarse(raw_nc_path, out_daily_path, out_coarse_path):
    """Converts a single monthly hourly NetCDF into daily mean and 1°x1° coarse."""
    print(f"    [Process] Resampling {os.path.basename(raw_nc_path)} to daily and 1x1 deg coarse...")
    ds_raw = xr.open_dataset(raw_nc_path)
    time_dim = 'valid_time' if 'valid_time' in ds_raw.coords else 'time'

    # Resample hourly to daily mean
    ds_daily = ds_raw.resample({time_dim: '1D'}).mean()
    if 'pressure_level' in ds_daily.sizes and ds_daily.sizes['pressure_level'] == 1:
        ds_daily = ds_daily.squeeze('pressure_level')

    ds_daily.to_netcdf(out_daily_path)

    # 4x spatial coarsening (0.25° -> 1.0° spatial pooling)
    ds_coarse = ds_daily.coarsen(latitude=4, longitude=4, boundary='pad').mean()
    ds_coarse.to_netcdf(out_coarse_path)

    ds_raw.close()
    ds_daily.close()
    ds_coarse.close()


def download_single_month(client, year, month, raw_nc_path):
    """Downloads one month of hourly 850 hPa ERA5 data over India from CDS."""
    _, num_days = calendar.monthrange(year, month)
    days_list = [f"{d:02d}" for d in range(1, num_days + 1)]
    times_list = [f"{h:02d}:00" for h in range(24)]

    request = {
        'product_type': ['reanalysis'],
        'variable': [
            'specific_humidity',
            'temperature',
            'u_component_of_wind',
            'v_component_of_wind',
            'vertical_velocity',
        ],
        'pressure_level': ['850'],
        'year': [str(year)],
        'month': [f"{month:02d}"],
        'day': days_list,
        'time': times_list,
        'data_format': 'netcdf',
        'download_format': 'unarchived',
        'area': [38.5, 66.5, 6.5, 100.0],  # India domain [North, West, South, East]
    }

    month_name = calendar.month_abbr[month]
    print(f"\n  >> Fetching from CDS: {year}-{month:02d} ({month_name}, {num_days} days)...")
    t0 = time.time()
    try:
        client.retrieve('reanalysis-era5-pressure-levels', request, raw_nc_path)
    except Exception as e:
        err_msg = str(e)
        if "terms" in err_msg.lower() or "license" in err_msg.lower():
            print("\n[ERROR] You have not accepted the Copernicus Terms of Use for ERA5!")
            print("Please visit this link, log in, scroll down, and accept the terms:")
            print("  https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels\n")
        raise e

    size_mb = os.path.getsize(raw_nc_path) / (1024 * 1024)
    print(f"  [OK] Downloaded {year}-{month:02d} ({size_mb:.1f} MB in {time.time()-t0:.1f}s)")


def download_and_process_year(client, year, months, era5_dir, temp_dir, keep_raw=False, force=False):
    os.makedirs(era5_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    annual_coarse_file = os.path.join(era5_dir, f"era5_{year}_coarse.nc")
    annual_daily_file = os.path.join(era5_dir, f"era5_{year}_daily.nc")

    # Resume check: if already completed, skip
    if not force and os.path.exists(annual_coarse_file) and os.path.exists(annual_daily_file):
        print(f"\n[SKIP] Year {year} already exists in {era5_dir}!")
        print(f"  -> Coarse: {annual_coarse_file}")
        print(f"  -> Daily:  {annual_daily_file}")
        print("  (Use --force to overwrite if you wish to re-download)")
        return

    month_names = [calendar.month_abbr[m] for m in months]
    season_desc = "Southwest Monsoon (JJAS)" if months == [6, 7, 8, 9] else f"{len(months)} Selected Months"

    print("\n" + "=" * 70)
    print(f" ERA5 INDIA PIPELINE: YEAR {year} ({season_desc})")
    print("=" * 70)
    print(f"Output Directory:    {os.path.abspath(era5_dir)}")
    print(f"Domain Extent:       India [38.5N - 6.5N, 66.5E - 100.0E]")
    print(f"Target Months:       {month_names} (Months: {months})")
    print(f"Coarse Resolution:   1.0 deg x 1.0 deg (~100 km, 4x pooled)")
    print(f"Variables (850 hPa): Specific Humidity (q), Temp (t), Wind (u, v), Vert Vel (w)")
    print("=" * 70)

    monthly_coarse_files = []
    monthly_daily_files = []
    monthly_raw_files = []

    monthly_proc_dir = os.path.join(temp_dir, f"proc_{year}")
    os.makedirs(monthly_proc_dir, exist_ok=True)

    year_t0 = time.time()

    for month in months:
        raw_nc = os.path.join(temp_dir, f"india_era5_{year}_{month:02d}_raw.nc")
        daily_nc = os.path.join(monthly_proc_dir, f"india_era5_{year}_{month:02d}_daily.nc")
        coarse_nc = os.path.join(monthly_proc_dir, f"india_era5_{year}_{month:02d}_coarse.nc")

        # Process if not already done
        if force or not (os.path.exists(daily_nc) and os.path.exists(coarse_nc)):
            if force or not os.path.exists(raw_nc) or os.path.getsize(raw_nc) < 10 * 1024 * 1024:
                download_single_month(client, year, month, raw_nc)
            
            process_month_to_daily_and_coarse(raw_nc, daily_nc, coarse_nc)

        monthly_coarse_files.append(coarse_nc)
        monthly_daily_files.append(daily_nc)
        if os.path.exists(raw_nc):
            monthly_raw_files.append(raw_nc)

    # Concatenate downloaded months into full datasets
    print("\n-------------------------------------------------------")
    print(f" Concatenating {len(months)} months into final dataset for {year}...")
    print("-------------------------------------------------------")

    # 1. Merge Coarse 1x1 deg
    print(f"  --> Merging into {annual_coarse_file}...")
    coarse_ds_list = [xr.open_dataset(f) for f in monthly_coarse_files]
    time_dim = 'valid_time' if 'valid_time' in coarse_ds_list[0].coords else 'time'
    ds_coarse_annual = xr.concat(coarse_ds_list, dim=time_dim).sortby('latitude')
    ds_coarse_annual.attrs['months_included'] = str(months)
    ds_coarse_annual.attrs['season'] = season_desc
    ds_coarse_annual.to_netcdf(annual_coarse_file)
    coarse_size = os.path.getsize(annual_coarse_file) / (1024 * 1024)
    num_days = len(ds_coarse_annual[time_dim])
    print(f"  [OK] Saved Coarse File: {annual_coarse_file} ({coarse_size:.2f} MB, {num_days} days, shape: {len(ds_coarse_annual.latitude)}x{len(ds_coarse_annual.longitude)})")
    
    # Close datasets before potential file cleanup
    ds_coarse_annual.close()
    for ds in coarse_ds_list:
        ds.close()

    # 2. Merge Daily Fine 0.25 deg
    print(f"  --> Merging into {annual_daily_file}...")
    daily_ds_list = [xr.open_dataset(f) for f in monthly_daily_files]
    ds_daily_annual = xr.concat(daily_ds_list, dim=time_dim).sortby('latitude')
    ds_daily_annual.attrs['months_included'] = str(months)
    ds_daily_annual.attrs['season'] = season_desc
    ds_daily_annual.to_netcdf(annual_daily_file)
    daily_size = os.path.getsize(annual_daily_file) / (1024 * 1024)
    print(f"  [OK] Saved Daily File:  {annual_daily_file} ({daily_size:.2f} MB, {num_days} days, shape: {len(ds_daily_annual.latitude)}x{len(ds_daily_annual.longitude)})")
    
    # Close datasets before potential file cleanup
    ds_daily_annual.close()
    for ds in daily_ds_list:
        ds.close()

    # Cleanup temporary monthly files to save disk space
    if not keep_raw:
        print("\n  Cleaning up intermediate monthly files to save disk space...")
        for f in monthly_raw_files:
            try:
                os.remove(f)
            except Exception:
                pass
        for f in monthly_coarse_files + monthly_daily_files:
            try:
                os.remove(f)
            except Exception:
                pass
        try:
            os.rmdir(monthly_proc_dir)
        except Exception:
            pass
        print("  [OK] Intermediate monthly files cleaned.")

    print("\n" + "=" * 70)
    print(f" YEAR {year} ({season_desc}) COMPLETE in {(time.time() - year_t0)/60:.1f} minutes!")
    print(f" Coarse File: {annual_coarse_file}")
    print(f" Daily File:  {annual_daily_file}")
    print("=" * 70)


# -----------------------------------------------------------------------------
# 5. MAIN ENTRY POINT
# -----------------------------------------------------------------------------
def main():
    args = parse_arguments()
    years = parse_year_list(args.year, args.years)

    if args.all_months:
        months = list(range(1, 13))
    else:
        months = sorted(list(set(args.months)))

    print("=" * 70)
    print(" ERA5 INDIA PREDICTOR DOWNLOADER (STANDALONE)")
    print("=" * 70)
    print(f" Target Years:   {years}")
    print(f" Target Months:  {months} ({[calendar.month_abbr[m] for m in months]})")
    print(f" Destination:    {os.path.abspath(args.output_dir)}")
    print("=" * 70)

    # Ensure CDS API credentials are ready
    setup_cdsapirc_if_needed()
    client = cdsapi.Client()

    overall_t0 = time.time()
    for y in years:
        download_and_process_year(
            client=client,
            year=y,
            months=months,
            era5_dir=args.output_dir,
            temp_dir=args.temp_dir,
            keep_raw=args.keep_raw,
            force=args.force
        )

    print("\n" + "#" * 70)
    print(f" ALL REQUESTED YEARS COMPLETED in {(time.time() - overall_t0)/60:.1f} minutes!")
    print(f" Files saved in: {os.path.abspath(args.output_dir)}")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()