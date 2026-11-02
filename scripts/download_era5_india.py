"""
download_era5_india.py
======================
Downloads ERA5 q, t, u, v, w at 850 hPa over India from the Copernicus Climate
Data Store and coarsens them by 4 x 4 block means of the 0.25 deg grid, the
same coarsening as data/India_sample/india_era5_1990_01_coarse_4x.nc (blocks
start at the north-west corner 38.5 N, 66.5 E; the partial last row/column is
kept here and dropped by the loader). The coarse coordinates this produces are
checked against the sample file in tests/test_india.py.

NOT run in this repository. Needs `pip install cdsapi` and a CDS account with
~/.cdsapirc (https://cds.climate.copernicus.eu/how-to-api), and the dataset's
licence accepted on its CDS page.

Products (--product):
  daily_mean  derived-era5-pressure-levels-daily-statistics, daily mean of hourly
              data, UTC days (preferred: a daily mean, not one snapshot)
  hourly      reanalysis-era5-pressure-levels at --hours (default 00:00, as in the
              sample); several hours are averaged per day by the loader
              (data.aggregation_shift_hours = 3 groups 03 UTC .. 02 UTC, the IMD day)

The request keys follow the CDS API as documented for these datasets; before a
long download, compare them with "Show API request" on the dataset's download
form and adjust if the CDS has changed them.

Usage:
  python scripts/download_era5_india.py --years 1990 2020 --product daily_mean --out data/India
  python scripts/download_era5_india.py --coarsen-only --out data/India     # raw files already downloaded
Raw files go to <out>/era5_raw/, coarse files to <out>/era5/ (data.era5_pattern = "era5/*.nc").
"""
import argparse
import calendar
import glob
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.india_data import coarsen_blocks  # noqa: E402

AREA = [38.5, 66.5, 6.5, 100.0]  # N, W, S, E: the IMD 0.25 deg grid (129 x 135 points)
CDS_VARIABLES = ["specific_humidity", "temperature", "u_component_of_wind", "v_component_of_wind",
                 "vertical_velocity"]
SHORT = ["q", "t", "u", "v", "w"]
FACTOR = 4


def request(product, year, month, hours):
    days = [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)]
    common = {"product_type": "reanalysis", "variable": CDS_VARIABLES, "year": str(year), "month": f"{month:02d}",
              "day": days, "pressure_level": ["850"], "area": AREA}
    if product == "daily_mean":
        return "derived-era5-pressure-levels-daily-statistics", dict(
            common, daily_statistic="daily_mean", time_zone="utc+00:00", frequency="1_hourly")
    return "reanalysis-era5-pressure-levels", dict(
        common, product_type=["reanalysis"], time=hours, data_format="netcdf", download_format="unarchived")


def coarsen_file(src, dst):
    with xr.open_dataset(src) as ds:
        ds = ds.load()
    lat = "latitude" if "latitude" in ds.dims else "lat"
    lon = "longitude" if "longitude" in ds.dims else "lon"
    tname = "valid_time" if "valid_time" in ds.dims else "time"
    la, lo = ds[lat].values, ds[lon].values
    if not (len(la) == 129 and len(lo) == 135 and np.isclose(la[0], 38.5) and np.isclose(la[-1], 6.5)
            and np.isclose(lo[0], 66.5) and np.isclose(lo[-1], 100.0)):
        raise ValueError(f"{src}: expected the 0.25 deg grid 38.5..6.5 N x 66.5..100 E (129 x 135), got "
                         f"{len(la)} x {len(lo)} from {la[0]}..{la[-1]}, {lo[0]}..{lo[-1]}")
    if "pressure_level" in ds.dims:
        ds = ds.sel(pressure_level=850.0)
    lat_c = coarsen_blocks(np.repeat(la[:, None], len(lo), 1), FACTOR, (0, 1))[:, 0]
    lon_c = coarsen_blocks(np.repeat(lo[None, :], len(la), 0), FACTOR, (0, 1))[0]
    out = xr.Dataset(coords={tname: ds[tname].values, "latitude": lat_c, "longitude": lon_c})
    for v in SHORT:
        da = ds[v].transpose(tname, lat, lon)
        out[v] = ((tname, "latitude", "longitude"), coarsen_blocks(da.values, FACTOR, (1, 2)).astype(np.float32),
                  dict(da.attrs))
    out["pressure_level"] = 850.0
    out.attrs = {"history": f"4x4 block mean of {os.path.basename(src)} (scripts/download_era5_india.py)"}
    out.to_netcdf(dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, nargs=2, metavar=("FIRST", "LAST"))
    p.add_argument("--product", choices=["daily_mean", "hourly"], default="daily_mean")
    p.add_argument("--hours", nargs="+", default=["00:00"])
    p.add_argument("--out", default="data/India")
    p.add_argument("--coarsen-only", action="store_true")
    a = p.parse_args()
    raw_dir, coarse_dir = os.path.join(a.out, "era5_raw"), os.path.join(a.out, "era5")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(coarse_dir, exist_ok=True)
    if not a.coarsen_only:
        import cdsapi
        client = cdsapi.Client()
        for year in range(a.years[0], a.years[1] + 1):
            for month in range(1, 13):
                dst = os.path.join(raw_dir, f"era5_{a.product}_{year}_{month:02d}.nc")
                if os.path.exists(dst):
                    continue  # resume: finished months are skipped
                name, req = request(a.product, year, month, a.hours)
                print("requesting", name, year, month, flush=True)
                client.retrieve(name, req).download(dst + ".part")
                os.replace(dst + ".part", dst)
    for src in sorted(glob.glob(os.path.join(raw_dir, "*.nc"))):
        dst = os.path.join(coarse_dir, os.path.basename(src).replace(".nc", "_coarse_4x.nc"))
        if not os.path.exists(dst):
            coarsen_file(src, dst)
            print("coarsened", dst, flush=True)


if __name__ == "__main__":
    main()
