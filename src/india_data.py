"""
India data pipeline: ERA5 predictors (q, t, u, v, w at 850 hPa) -> IMD 0.25 deg
gridded rainfall.

ERA5 comes on a coarse grid made by 4x4 block means of the same 0.25 deg grid
as IMD (see coarsen_blocks and scripts/download_era5_india.py). Partial edge
blocks (fewer than 4x4 fine cells) are dropped, and the remaining blocks are
checked to tile the IMD grid exactly (block_alignment).

Days: predictors for IMD day D come from ERA5 day D + day_offset (0 or 1).
IMD day D is a 24 h accumulation ending 08:30 IST (03 UTC), so an ERA5 00 UTC
snapshot of day D or D + 1 falls inside the accumulation window depending on
the convention; experiments/india_day_offset.py measures which offset fits
better. Sub-daily ERA5 (hourly / 6-hourly) is averaged to daily means first,
optionally shifted (aggregation_shift_hours = 3 groups 03 UTC .. 02 UTC, the
IMD window); each window is labelled by its start date, and day_offset is
applied to these labels.

IMD cells that are NaN are outside India; the mask is computed once, checked
to be constant over time, and NaN is never replaced by zero: targets are
returned only for valid cells, Y (T, n_valid).
"""
import glob
import os

import numpy as np
import xarray as xr

EARTH_RADIUS_KM = 6371.0
ERA5_VARIABLES = ["q", "t", "u", "v", "w"]
EXPECTED_UNITS = {"q": "kg kg**-1", "t": "K", "u": "m s**-1", "v": "m s**-1", "w": "Pa s**-1"}

_DIM_ALIASES = {"time": ("time", "valid_time", "TIME", "date"),
                "lat": ("lat", "latitude", "LATITUDE", "LAT"),
                "lon": ("lon", "longitude", "LONGITUDE", "LON")}


def _rename_dims(obj):
    """Renames time/latitude/longitude dims and coords of an xarray object to time/lat/lon."""
    mapping = {}
    for std, aliases in _DIM_ALIASES.items():
        for a in aliases:
            if a in obj.dims or a in obj.coords:
                if a != std:
                    mapping[a] = std
                break
    return obj.rename(mapping)


def list_files(data_dir, pattern):
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"no files match {os.path.join(data_dir, pattern)}")
    return paths


def _concat_time(objs):
    out = xr.concat(objs, dim="time") if len(objs) > 1 else objs[0]
    out = out.sortby("time")
    t = out["time"].values
    if len(np.unique(t)) != len(t):
        raise ValueError("duplicate time stamps across input files")
    return out


# -----------------------------------------------------------------------------
# IMD
# -----------------------------------------------------------------------------
def load_imd(paths, variable="RAINFALL"):
    """
    IMD rainfall from one or more NetCDF files -> dict with
    rain (T, n_lat, n_lon) float32 with NaN outside India, dates (T,) datetime64[D],
    lat ascending, lon ascending, mask (n_lat, n_lon) bool (constant over time), units.
    """
    parts, units = [], None
    for p in paths:
        with xr.open_dataset(p) as ds:
            if variable not in ds:
                raise KeyError(f"{p}: variable {variable!r} not found (has {list(ds.data_vars)})")
            da = _rename_dims(ds[variable]).load()
        units = units or da.attrs.get("units")
        parts.append(da.transpose("time", "lat", "lon"))
    da = _concat_time(parts)
    if da["lat"].values[0] > da["lat"].values[-1]:
        da = da.isel(lat=slice(None, None, -1))
    rain = da.values.astype(np.float32)
    valid = ~np.isnan(rain)
    mask = valid[0]
    changed = np.flatnonzero(np.any(valid != mask, axis=(1, 2)))
    if len(changed):
        raise ValueError(f"IMD valid-cell mask changes over time on {len(changed)} days "
                         f"(first: {da['time'].values[changed[0]]}); expected a constant mask")
    if np.any(rain[:, mask] < 0):
        raise ValueError("negative IMD rainfall values")
    return {"rain": rain, "dates": da["time"].values.astype("datetime64[D]"),
            "lat": da["lat"].values.astype(np.float64), "lon": da["lon"].values.astype(np.float64),
            "mask": mask, "units": units}


# -----------------------------------------------------------------------------
# ERA5
# -----------------------------------------------------------------------------
def load_era5(paths, variables=ERA5_VARIABLES, level=850.0, aggregation_shift_hours=0):
    """
    ERA5 pressure-level fields from one or more NetCDF files -> dict with
    x (T, n_lat, n_lon, C) float32 in the order of `variables`, dates (T,) datetime64[D],
    lat ascending, lon, units per variable, and a temporal report (snapshot hour(s)
    or number of sub-daily steps averaged per day).
    """
    parts, units = [], {}
    for p in paths:
        with xr.open_dataset(p) as ds:
            ds = _rename_dims(ds)
            missing = [v for v in variables if v not in ds]
            if missing:
                raise KeyError(f"{p}: ERA5 variables {missing} not found (has {list(ds.data_vars)})")
            ds = ds[variables]
            if "pressure_level" in ds.dims:
                ds = ds.sel(pressure_level=level)
            elif "pressure_level" in ds.coords and float(ds["pressure_level"]) != level:
                raise ValueError(f"{p}: pressure level {float(ds['pressure_level'])}, expected {level}")
            ds = ds.load()
        for v in variables:
            units.setdefault(v, ds[v].attrs.get("units"))
        parts.append(ds.transpose("time", "lat", "lon"))
    ds = _concat_time(parts)
    if ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.isel(lat=slice(None, None, -1))  # descending in ERA5 files; IMD is ascending
    if any(np.isnan(ds[v].values).any() for v in variables):
        raise ValueError("NaN in ERA5 predictors")

    times = ds["time"].values.astype("datetime64[ns]")
    shifted = times - np.timedelta64(int(aggregation_shift_hours), "h")
    days = shifted.astype("datetime64[D]")
    uniq, counts = np.unique(days, return_counts=True)
    if counts.max() == 1:
        hours = sorted(set(((times - times.astype("datetime64[D]")) // np.timedelta64(1, "h")).tolist()))
        temporal = {"kind": "one_field_per_day", "hours_utc": hours}
        x = np.stack([ds[v].values for v in variables], axis=-1)
    else:
        x = np.stack([ds[v].values for v in variables], axis=-1)
        sums = np.zeros((len(uniq),) + x.shape[1:], dtype=np.float64)
        np.add.at(sums, np.searchsorted(uniq, days), x)
        x = sums / counts[:, None, None, None]
        full = counts.max()
        temporal = {"kind": "daily_mean_of_sub_daily", "steps_per_full_day": int(full),
                    "aggregation_shift_hours": int(aggregation_shift_hours),
                    "incomplete_days": [str(d) for d, c in zip(uniq, counts) if c < full]}
    return {"x": x.astype(np.float32), "dates": uniq, "lat": ds["lat"].values.astype(np.float64),
            "lon": ds["lon"].values.astype(np.float64), "units": units, "temporal": temporal}


def check_units(units):
    """Variables whose units differ from the ERA5 CF units (empty dict = all as expected)."""
    return {v: u for v, u in units.items() if v in EXPECTED_UNITS and u != EXPECTED_UNITS[v]}


# -----------------------------------------------------------------------------
# Grid alignment and coarsening
# -----------------------------------------------------------------------------
def coarsen_blocks(values, factor, axes):
    """
    factor x factor block means along two axes, starting at index 0 of each axis;
    edge blocks with fewer cells are means of the cells present (NaN-padded).
    Used for both data and coordinates, so the coarse coordinate is the block mean.
    """
    v = np.asarray(values, dtype=np.float64)
    a0, a1 = axes
    pad = [(0, 0)] * v.ndim
    pad[a0] = (0, (-v.shape[a0]) % factor)
    pad[a1] = (0, (-v.shape[a1]) % factor)
    v = np.pad(v, pad, constant_values=np.nan)
    shape = list(v.shape)
    shape[a1:a1 + 1] = [shape[a1] // factor, factor]
    shape[a0:a0 + 1] = [shape[a0] // factor, factor]
    v = v.reshape(shape)
    with np.errstate(invalid="ignore"):
        return np.nanmean(v, axis=(a0 + 1, a1 + 2))


def full_blocks(coarse, fine, factor, fine_res):
    """
    For each coarse coordinate, the fine indices of its block. A block is kept if it
    covers exactly `factor` fine cells whose mean is the coarse coordinate; kept
    blocks must be contiguous and non-overlapping. Returns (kept coarse indices,
    list of fine index arrays, dropped coarse coordinates).
    """
    kept, blocks, dropped = [], [], []
    for j, c in enumerate(coarse):
        idx = np.flatnonzero(np.abs(fine - c) < factor * fine_res / 2 - 1e-9)
        if len(idx) == factor and abs(fine[idx].mean() - c) < 1e-6:
            kept.append(j)
            blocks.append(idx)
        else:
            dropped.append(float(c))
    if not kept:
        raise ValueError("no ERA5 coordinate is the mean of a full block of IMD cells: grids are not aligned")
    for a, b in zip(blocks, blocks[1:]):
        if b[0] != a[-1] + 1:
            raise ValueError("ERA5 blocks do not tile the IMD grid contiguously")
    return np.array(kept), blocks, dropped


def block_alignment(era5_lat, era5_lon, imd_lat, imd_lon, factor=4):
    """Verifies that ERA5 is a factor x factor block mean of the IMD grid; returns the kept blocks."""
    res_lat = float(np.median(np.diff(imd_lat)))
    res_lon = float(np.median(np.diff(imd_lon)))
    klat, blat, dlat = full_blocks(era5_lat, imd_lat, factor, res_lat)
    klon, blon, dlon = full_blocks(era5_lon, imd_lon, factor, res_lon)
    # fine cell -> coarse block (-1 outside the kept blocks), for local predictor-rainfall checks
    lat_block = np.full(len(imd_lat), -1)
    lon_block = np.full(len(imd_lon), -1)
    for i, idx in enumerate(blat):
        lat_block[idx] = i
    for i, idx in enumerate(blon):
        lon_block[idx] = i
    return {"lat_keep": klat, "lon_keep": klon, "dropped_lat": dlat, "dropped_lon": dlon,
            "lat_block_of_fine": lat_block, "lon_block_of_fine": lon_block,
            "fine_res": [res_lat, res_lon], "factor": factor}


# -----------------------------------------------------------------------------
# Cell areas
# -----------------------------------------------------------------------------
def cell_areas_km2(lat, lon, res_lat=None, res_lon=None):
    """
    Area of each (lat, lon) grid cell on a sphere of radius 6371 km:
    A = R^2 * dlon * (sin(phi_north) - sin(phi_south)), angles in radians.
    Returns (n_lat, n_lon).
    """
    lat = np.asarray(lat, dtype=np.float64)
    res_lat = res_lat or float(np.median(np.diff(lat)))
    res_lon = res_lon or float(np.median(np.diff(lon)))
    north = np.deg2rad(lat + res_lat / 2)
    south = np.deg2rad(lat - res_lat / 2)
    band = EARTH_RADIUS_KM ** 2 * np.deg2rad(res_lon) * (np.sin(north) - np.sin(south))
    return np.repeat(band[:, None], len(lon), axis=1)


# -----------------------------------------------------------------------------
# Day matching
# -----------------------------------------------------------------------------
def match_days(imd_dates, era5_dates, day_offset, period=None):
    """
    Pairs IMD day D with ERA5 day D + day_offset. Returns (imd_index, era5_index, report);
    the report lists every IMD day in `period` that is missing, and every IMD day whose
    ERA5 day is missing.
    """
    imd_dates = np.asarray(imd_dates).astype("datetime64[D]")
    era5_dates = np.asarray(era5_dates).astype("datetime64[D]")
    sel = np.ones(len(imd_dates), bool)
    report = {"day_offset": int(day_offset)}
    if period is not None:
        start, end = np.datetime64(period[0], "D"), np.datetime64(period[1], "D")
        sel = (imd_dates >= start) & (imd_dates <= end)
        wanted = np.arange(start, end + np.timedelta64(1, "D"), dtype="datetime64[D]")
        report["missing_imd_days"] = [str(d) for d in np.setdiff1d(wanted, imd_dates)]
    imd_idx = np.flatnonzero(sel)
    need = imd_dates[imd_idx] + np.timedelta64(int(day_offset), "D")
    pos = np.searchsorted(era5_dates, need)
    found = (pos < len(era5_dates)) & (era5_dates[np.minimum(pos, len(era5_dates) - 1)] == need)
    report["missing_era5_days"] = [{"imd_day": str(d), "era5_day_needed": str(n)}
                                   for d, n in zip(imd_dates[imd_idx][~found], need[~found])]
    report["matched_days"] = int(found.sum())
    return imd_idx[found], pos[found], report


def load_india(dcfg, verbose=True):
    """
    Full pipeline from the data section of an India config. Returns
    x (T, H, W, C) float32, y (T, n_valid) float32, dates, mask, lat/lon of IMD and of
    the valid cells, valid-cell areas (km^2), alignment info and a report dict.
    """
    imd = load_imd(list_files(dcfg["data_dir"], dcfg["imd_pattern"]), dcfg["imd_variable"])
    era = load_era5(list_files(dcfg["data_dir"], dcfg["era5_pattern"]), dcfg["era5_variables"],
                    dcfg["pressure_level"], dcfg["aggregation_shift_hours"])
    align = block_alignment(era["lat"], era["lon"], imd["lat"], imd["lon"], dcfg["coarsen_factor"])
    imd_idx, era_idx, days = match_days(imd["dates"], era["dates"], dcfg["day_offset"], dcfg.get("period"))
    n_missing = len(days["missing_era5_days"]) + len(days.get("missing_imd_days", []))
    if n_missing and not dcfg["allow_missing_days"]:
        raise ValueError(f"{n_missing} missing days (IMD: {days.get('missing_imd_days', [])[:10]}, "
                         f"ERA5: {days['missing_era5_days'][:10]}); download them or set "
                         f"data.allow_missing_days=true to drop them (they are listed in the report)")
    x = era["x"][era_idx][:, align["lat_keep"]][:, :, align["lon_keep"]]
    mask = imd["mask"]
    y = imd["rain"][imd_idx][:, mask]
    areas = cell_areas_km2(imd["lat"], imd["lon"])
    lat2d, lon2d = np.meshgrid(imd["lat"], imd["lon"], indexing="ij")
    report = {
        "imd_files": int(len(list_files(dcfg["data_dir"], dcfg["imd_pattern"]))),
        "era5_files": int(len(list_files(dcfg["data_dir"], dcfg["era5_pattern"]))),
        "imd_units": imd["units"], "era5_units": era["units"], "era5_unit_mismatches": check_units(era["units"]),
        "era5_temporal": era["temporal"], "imd_grid": [len(imd["lat"]), len(imd["lon"])],
        "imd_valid_cells": int(mask.sum()),
        "era5_grid_in_file": [len(era["lat"]), len(era["lon"])], "input_grid": list(x.shape[1:3]),
        "dropped_partial_blocks": {"lat": align["dropped_lat"], "lon": align["dropped_lon"]},
        "days": days, "first_day": str(imd["dates"][imd_idx[0]]), "last_day": str(imd["dates"][imd_idx[-1]]),
        "n_days": int(len(imd_idx)),
    }
    if verbose:
        print(f"India data: X {x.shape}, Y {y.shape}, days {report['first_day']} .. {report['last_day']}, "
              f"input grid {report['input_grid']} (dropped lat {align['dropped_lat']}, lon {align['dropped_lon']})",
              flush=True)
    return {"x": x, "y": y.astype(np.float32), "dates": imd["dates"][imd_idx], "mask": mask,
            "imd_lat": imd["lat"], "imd_lon": imd["lon"], "lat": lat2d[mask], "lon": lon2d[mask],
            "areas_km2": areas[mask], "alignment": align, "era5_lat": era["lat"][align["lat_keep"]],
            "era5_lon": era["lon"][align["lon_keep"]], "report": report}
