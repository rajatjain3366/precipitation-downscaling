"""
india_day_offset.py
===================
Which ERA5 -> IMD day offset (0 or 1) fits better? Uses TRAINING and
VALIDATION days only (never test) and raw predictors.

IMD day D is the 24 h accumulation ending 08:30 IST (03 UTC). Whether an ERA5
field of day D (offset 0) or D + 1 (offset 1) falls inside that window depends
on the convention, so it is measured: for each offset and predictor,
  - domain: Pearson correlation of the domain-mean predictor with the
            area-weighted all-India mean rainfall
  - local:  mean over IMD cells of |corr| between the cell's rainfall and the
            predictor of the ERA5 block containing it
on the days both offsets can use. The offset with the higher mean local |corr|
over the predictors is reported as recommended.

Usage:
  python experiments/india_day_offset.py --config configs/india_v1.json
Writes <save_dir>/day_offset/day_offset.json.
"""
import copy
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
sys.path.insert(0, SCRIPT_DIR)

import run_india as I  # noqa: E402
import run_cordex_v2 as R  # noqa: E402
from src.india_data import load_india  # noqa: E402
from src.preprocessing import split_masks_by_dates  # noqa: E402


def corr_columns(a, b):
    """Pearson correlation of matching columns of a and b, (T, K) -> (K,); NaN for constant columns."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(axis=0) / den, np.nan)


def main():
    args = I.parse_args()
    cfg = I.load_config(args)
    loaded = {}
    for offset in (0, 1):
        dcfg = copy.deepcopy(cfg["data"])
        dcfg.update(day_offset=offset, allow_missing_days=True)
        full = load_india(dcfg, verbose=False)
        m = split_masks_by_dates(full["dates"], cfg["split"])
        keep = m["train"] | m["val"]  # never the test split
        loaded[offset] = {"full": full, "keep": keep}
    common = np.intersect1d(*[loaded[o]["full"]["dates"][loaded[o]["keep"]] for o in (0, 1)])

    out = {"days_used": int(len(common)), "first_day": str(common[0]), "last_day": str(common[-1]),
           "splits_used": ["train", "val"], "offsets": {}}
    for offset in (0, 1):
        full = loaded[offset]["full"]
        sel = np.isin(full["dates"], common)
        x, y = full["x"][sel].astype(np.float64), full["y"][sel].astype(np.float64)
        w = full["areas_km2"] / full["areas_km2"].sum()
        india = y @ w
        al = full["alignment"]
        # valid IMD cell -> ERA5 block (cells outside the kept blocks are skipped)
        lat_idx = np.searchsorted(full["imd_lat"], full["lat"])
        lon_idx = np.searchsorted(full["imd_lon"], full["lon"])
        bi, bj = al["lat_block_of_fine"][lat_idx], al["lon_block_of_fine"][lon_idx]
        inside = (bi >= 0) & (bj >= 0)
        res = {}
        for v, name in enumerate(cfg["data"]["era5_variables"]):
            dom = corr_columns(x[..., v].mean(axis=(1, 2))[:, None], india[:, None])[0]
            local = corr_columns(x[:, bi[inside], bj[inside], v], y[:, inside])
            res[name] = {"domain_corr": float(dom), "local_mean_abs_corr": float(np.nanmean(np.abs(local))),
                         "local_mean_corr": float(np.nanmean(local)), "cells_used": int(np.isfinite(local).sum())}
        out["offsets"][str(offset)] = {
            "per_variable": res,
            "score_mean_local_abs_corr": float(np.mean([r["local_mean_abs_corr"] for r in res.values()])),
            "missing_era5_days": full["report"]["days"]["missing_era5_days"][:20]}
    s = {o: out["offsets"][o]["score_mean_local_abs_corr"] for o in out["offsets"]}
    out["recommended_day_offset"] = int(max(s, key=s.get))
    out["score_difference"] = float(abs(s["0"] - s["1"]))
    path = os.path.join(cfg["save_dir"], "day_offset", "day_offset.json")
    R.save_json(out, path)
    for o in ("0", "1"):
        r = out["offsets"][o]["per_variable"]
        print(f"offset {o}: score {s[o]:.4f}  " + "  ".join(
            f"{k}: local|r|={v['local_mean_abs_corr']:.3f} domain r={v['domain_corr']:+.3f}" for k, v in r.items()))
    print(f"recommended data.day_offset = {out['recommended_day_offset']} ({out['days_used']} train+val days) -> {path}")


if __name__ == "__main__":
    main()
