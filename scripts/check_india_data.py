"""
check_india_data.py
===================
File-by-file checks of the India ERA5 and IMD data before any training,
independent of the loader. Read-only; writes a JSON report.

ERA5 (each file): 122 JJAS days of its year, unique and complete; q, t, u, v, w
with the expected units; no NaN; latitude order detected; identical grid and
coordinates in every file; the grid is a 4 x 4 block mean of the IMD grid, and
dropping the partial row/column gives 32 x 33.
IMD (each file): 122 JJAS days; the same 0.25 deg grid as the sample file; a
valid-cell mask that is constant over time and identical in every year (and
compared with the sample's); no NaN inside the mask; no negative values.
Rainfall statistics inside the mask: fraction of exact zeros, 99th / 99.9th
percentiles (all years and training years).
Date matching ERA5 <-> IMD for day offsets 0 and 1, with the season edges.

Usage:
  python scripts/check_india_data.py --config configs/india_v1.json
"""
import argparse
import json
import os
import sys

import numpy as np
import xarray as xr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.india_data import (EXPECTED_UNITS, _rename_dims, block_alignment, list_files,  # noqa: E402
                            match_days, month_of)

SAMPLE_IMD = os.path.join(ROOT, "data", "India_sample", "RF25_ind1990_rfp25.nc")


def season_days(year, months):
    d = np.arange(np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01"), dtype="datetime64[D]")
    return d[np.isin(month_of(d), months)]


def year_of(path):
    digits = "".join(ch if ch.isdigit() else " " for ch in os.path.basename(path)).split()
    return int([t for t in digits if len(t) == 4][0])


def check_era5(paths, variables, months):
    out, ref, problems = {"files": len(paths)}, None, []
    hours, orders = set(), set()
    for p in paths:
        y = year_of(p)
        with xr.open_dataset(p) as ds:
            ds = _rename_dims(ds)
            t = ds["time"].values
            days = t.astype("datetime64[D]")
            hours |= set(((t - days) // np.timedelta64(1, "h")).astype(int).tolist())
            if not np.array_equal(np.unique(days), season_days(y, months)) or len(days) != len(np.unique(days)):
                problems.append(f"{os.path.basename(p)}: days are not exactly the {len(season_days(y, months))} season days of {y}")
            missing = [v for v in variables if v not in ds]
            if missing:
                problems.append(f"{os.path.basename(p)}: missing variables {missing}")
            for v in variables:
                if v in ds:
                    if ds[v].attrs.get("units") != EXPECTED_UNITS[v]:
                        problems.append(f"{os.path.basename(p)}: {v} units {ds[v].attrs.get('units')!r}")
                    if np.isnan(ds[v].values).any():
                        problems.append(f"{os.path.basename(p)}: NaN in {v}")
            lat, lon = ds["lat"].values, ds["lon"].values
            orders.add("descending" if lat[0] > lat[-1] else "ascending")
            if ref is None:
                ref = (lat, lon, dict(ds.sizes))
            elif not (np.array_equal(lat, ref[0]) and np.array_equal(lon, ref[1])):
                problems.append(f"{os.path.basename(p)}: grid differs from {os.path.basename(paths[0])}")
    out.update({"years": [year_of(paths[0]), year_of(paths[-1])], "days_per_file": len(season_days(1990, months)),
                "sizes": ref[2], "latitude_order": sorted(orders), "hours_utc": sorted(hours),
                "lat_range": [float(ref[0].min()), float(ref[0].max())],
                "lon_range": [float(ref[1].min()), float(ref[1].max())], "problems": problems})
    return out, np.sort(ref[0]), ref[1]


def check_imd(paths, variable, months, train_years):
    out, problems = {"files": len(paths)}, []
    with xr.open_dataset(SAMPLE_IMD) as s:
        s = _rename_dims(s[variable])
        s_lat, s_lon, s_mask = s["lat"].values, s["lon"].values, ~np.isnan(s.isel(time=0).values)
    mask0, vals, vals_train = None, [], []
    for p in paths:
        y = year_of(p)
        with xr.open_dataset(p) as ds:
            da = _rename_dims(ds[variable]).transpose("time", "lat", "lon")
            if da["lat"].values[0] > da["lat"].values[-1]:
                da = da.isel(lat=slice(None, None, -1))
            days = da["time"].values.astype("datetime64[D]")
            if not np.array_equal(days, season_days(y, months)):
                problems.append(f"{os.path.basename(p)}: days are not exactly the season days of {y}")
            if not (np.array_equal(da["lat"].values, s_lat) and np.array_equal(da["lon"].values, s_lon)):
                problems.append(f"{os.path.basename(p)}: grid differs from the sample IMD grid")
            r = da.values
            valid = ~np.isnan(r)
            mask = valid[0]
            if np.any(valid != mask):
                problems.append(f"{os.path.basename(p)}: mask changes over time (NaN inside the mask on some days)")
            if mask0 is None:
                mask0 = mask
            elif not np.array_equal(mask, mask0):
                problems.append(f"{os.path.basename(p)}: mask differs from {os.path.basename(paths[0])} "
                                f"({int(mask.sum())} vs {int(mask0.sum())} cells)")
            inside = r[:, mask0]
            if np.isnan(inside).any():
                problems.append(f"{os.path.basename(p)}: NaN inside the mask")
            if (inside < 0).any():
                problems.append(f"{os.path.basename(p)}: negative rainfall")
            vals.append(inside.ravel())
            if train_years[0] <= y <= train_years[1]:
                vals_train.append(inside.ravel())
    allv, trv = np.concatenate(vals), np.concatenate(vals_train)

    def stats(v):
        return {"values": int(v.size), "fraction_exact_zero": float(np.mean(v == 0)),
                "fraction_in_0_0.1": float(np.mean((v > 0) & (v < 0.1))), "mean_mm_per_day": float(v.mean()),
                "p99_mm_per_day": float(np.percentile(v, 99)), "p99_9_mm_per_day": float(np.percentile(v, 99.9)),
                "max_mm_per_day": float(v.max())}

    out.update({"years": [year_of(paths[0]), year_of(paths[-1])], "grid": [len(s_lat), len(s_lon)],
                "valid_cells": int(mask0.sum()), "sample_valid_cells": int(s_mask.sum()),
                "mask_equals_sample": bool(np.array_equal(mask0, s_mask)),
                "rain_all_years": stats(allv), "rain_train_years": stats(trv), "problems": problems})
    return out, s_lat, s_lon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/india_v1.json")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(ROOT, a.config)))
    d = cfg["data"]
    data_dir = os.path.join(ROOT, d["data_dir"])
    months = d["months"]
    train_years = [int(cfg["split"]["train"][0][:4]), int(cfg["split"]["train"][1][:4])]
    era5_paths = list_files(data_dir, d["era5_pattern"])
    imd_paths = list_files(data_dir, d["imd_pattern"])
    rep = {"config": a.config, "months": months}
    rep["era5"], e_lat, e_lon = check_era5(era5_paths, d["era5_variables"], months)
    rep["imd"], i_lat, i_lon = check_imd(imd_paths, d["imd_variable"], months, train_years)
    al = block_alignment(e_lat, e_lon, i_lat, i_lon, d["coarsen_factor"])
    rep["alignment"] = {"input_grid": [len(al["lat_keep"]), len(al["lon_keep"])],
                        "dropped_lat": al["dropped_lat"], "dropped_lon": al["dropped_lon"]}
    imd_days = np.concatenate([season_days(year_of(p), months) for p in imd_paths])
    era_days = np.concatenate([season_days(year_of(p), months) for p in era5_paths])
    rep["date_matching"] = {}
    for off in (0, 1):
        _, _, r = match_days(imd_days, era_days, off, d["period"], months)
        rep["date_matching"][str(off)] = {
            "matched_days": r["matched_days"], "missing_imd_days": len(r["missing_imd_days"]),
            "missing_era5_days": len(r["missing_era5_days"]),
            "season_edge_dropped": len(r["season_edge_dropped"]),
            "season_edge_examples": r["season_edge_dropped"][:3]}
    rep["ok"] = not rep["era5"]["problems"] and not rep["imd"]["problems"] and rep["alignment"]["input_grid"] == [32, 33]
    out = os.path.join(ROOT, cfg["save_dir"], "data_check.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=2)
    print(json.dumps(rep, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
