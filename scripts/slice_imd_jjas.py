# Provenance: the teammate's script, copied unchanged (only this comment block
# was added). It made data/India/IMD_jjas/ (it writes "IMD_JJAS"; the same
# folder on Windows) by cutting IMD's yearly 0.25 deg files
# RF25_ind<year>_rfp25.nc, 1971-2016, read from "data/India/IMD data/" (not in
# the repository), to 1 June - 30 September (122 days per year).
# Its ERA5 date cross-check reads data/India/ERA5, which is now
# data/India/ERA5_full (made by scripts/download_era5_india_jjas.py).
"""
slice_imd_jjas.py
=================
Slices all 46 years (1971-2016) of IMD daily rainfall datasets down to 
the 4 Southwest Monsoon months: June 1 to September 30 (JJAS, 122 days).

Saves processed files into `data/India/IMD_JJAS/RF25_ind<year>_jjas.nc`.
Every file will have exactly 122 days, aligning day-for-day with `era5_<year>_coarse.nc`.
"""

import os
import glob
import time
import xarray as xr


def slice_all_imd(
    input_dir="data/India/IMD data",
    output_dir="data/India/IMD_JJAS",
    era5_dir="data/India/ERA5"
):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(" SLICING IMD DATASET TO JUNE-SEPTEMBER MONSOON (JJAS)")
    print("=" * 70)
    print(f" Source Directory:      {os.path.abspath(input_dir)}")
    print(f" Destination Directory: {os.path.abspath(output_dir)}")
    print(f" Target Season:         June 1 to September 30 (122 days/year)")
    print("=" * 70)

    t0 = time.time()
    total_processed = 0
    total_orig_bytes = 0
    total_new_bytes = 0

    for year in range(1971, 2017):
        in_file = os.path.join(input_dir, f"RF25_ind{year}_rfp25.nc")
        out_file = os.path.join(output_dir, f"RF25_ind{year}_jjas.nc")

        if not os.path.exists(in_file):
            print(f"  [ERROR] Missing input file for year {year}: {in_file}")
            continue

        orig_size = os.path.getsize(in_file)
        total_orig_bytes += orig_size

        # Check if already processed
        if os.path.exists(out_file) and os.path.getsize(out_file) > 1024 * 1024:
            new_size = os.path.getsize(out_file)
            total_new_bytes += new_size
            total_processed += 1
            continue

        # Load and slice
        with xr.open_dataset(in_file) as ds:
            ds_loaded = ds.load()

        time_dim = 'TIME' if 'TIME' in ds_loaded.coords else 'time'
        ds_jjas = ds_loaded.sel({time_dim: slice(f"{year}-06-01", f"{year}-09-30")})
        num_days = len(ds_jjas[time_dim])

        if num_days != 122:
            print(f"  [WARNING] Year {year} has {num_days} days instead of 122!")

        ds_jjas.attrs['season'] = "Southwest Monsoon (JJAS)"
        ds_jjas.attrs['months_included'] = "June, July, August, September"
        ds_jjas.attrs['dates'] = f"{year}-06-01 to {year}-09-30"

        ds_jjas.to_netcdf(out_file)
        new_size = os.path.getsize(out_file)
        total_new_bytes += new_size
        total_processed += 1

        print(f"  [OK] {year}: {num_days} days saved to {os.path.basename(out_file)} ({new_size / (1024*1024):.2f} MB)")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f" PROCESSED {total_processed} YEARS IN {elapsed:.1f} SECONDS!")
    print(f" Original Size: {total_orig_bytes / (1024**3):.2f} GB")
    print(f" Sliced Size:   {total_new_bytes / (1024**2):.2f} MB ({total_new_bytes / (1024**3):.2f} GB)")
    print(f" Space Saved:   {(total_orig_bytes - total_new_bytes) / (1024**2):.2f} MB")
    print("=" * 70)

    # Cross-validation with ERA5
    print("\n--- Cross-Validating with ERA5 Coarse Datasets ---")
    mismatches = []
    for year in range(1971, 2017):
        era_path = os.path.join(era5_dir, f"era5_{year}_coarse.nc")
        imd_path = os.path.join(output_dir, f"RF25_ind{year}_jjas.nc")

        if os.path.exists(era_path) and os.path.exists(imd_path):
            with xr.open_dataset(era_path) as ds_era, xr.open_dataset(imd_path) as ds_imd:
                t_era = [str(t)[:10] for t in ds_era.valid_time.values]
                t_imd = [str(t)[:10] for t in ds_imd.TIME.values]
                if t_era != t_imd:
                    mismatches.append(year)

    if not mismatches:
        print("[SUCCESS] All 46 years match ERA5 day-for-day with 100% exact date alignment!")
    else:
        print(f"[!] Mismatches found in years: {mismatches}")


if __name__ == "__main__":
    # Handle paths relative to repo root or scripts folder
    repo_root = "." if os.path.exists("data") else ".."
    slice_all_imd(
        input_dir=os.path.join(repo_root, "data/India/IMD data"),
        output_dir=os.path.join(repo_root, "data/India/IMD_JJAS"),
        era5_dir=os.path.join(repo_root, "data/India/ERA5")
    )
