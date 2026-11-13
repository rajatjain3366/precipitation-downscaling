"""
NZ mass-conservation study: cell areas, land mask, and a smoke run on the
synthetic fixture (lambda = 0 and lambda > 0; lambda = 0 must reproduce the
NZ v2 QRE exactly).
Run with:  python -m pytest tests/test_nz_mass.py -q
Needs data/CORDEX_NZ_sample (written by tests/generate_sample_cordex.py); the
real-data mask test needs data/CORDEX_NZ and is skipped without it.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.grid_areas import EARTH_RADIUS_KM, cell_areas_from_centres, cell_edges  # noqa: E402
from src.india_data import cell_areas_km2  # noqa: E402

SAMPLE = os.path.join(WORKSPACE_ROOT, "data", "CORDEX_NZ_sample")
REAL_STATIC = os.path.join(WORKSPACE_ROOT, "data", "CORDEX_NZ", "train", "ESD_pseudo_reality", "predictors",
                           "Static_fields.nc")
needs_sample = pytest.mark.skipif(not os.path.isdir(SAMPLE), reason="run tests/generate_sample_cordex.py first")


# cell areas ---------------------------------------------------------------------------------
def test_areas_match_uniform_grid_formula():
    lat, lon = np.arange(6.5, 38.51, 0.25), np.arange(66.5, 100.01, 0.25)
    np.testing.assert_allclose(cell_areas_from_centres(lat, lon), cell_areas_km2(lat, lon), rtol=1e-9)


def test_areas_nz_like_grid_positive_and_sum_to_box():
    # the CORDEX NZ target grid: 128 x 128, ~0.1073 deg, southern hemisphere
    lat = -47.5641 + 0.10732918 * np.arange(128)
    lon = 166.0406 + 0.10732924 * np.arange(128)
    a = cell_areas_from_centres(lat, lon)
    assert a.shape == (128, 128) and np.all(a > 0)
    le, lo = cell_edges(lat), cell_edges(lon)
    box = EARTH_RADIUS_KM ** 2 * np.deg2rad(lo[-1] - lo[0]) * (np.sin(np.deg2rad(le[-1])) - np.sin(np.deg2rad(le[0])))
    assert np.isclose(a.sum(), box, rtol=1e-10)
    assert a[-1, 0] > a[0, 0]                      # cells grow towards the equator
    assert 100 < a.mean() < 120                    # ~12 km x ~9 km cells
    np.testing.assert_allclose(cell_areas_from_centres(lat[::-1], lon), a[::-1], rtol=1e-12)  # descending latitude


def test_areas_irregular_spacing_and_bad_grids():
    a = cell_areas_from_centres(np.array([0.0, 1.0, 3.0]), np.array([10.0, 11.0]))
    assert a.shape == (3, 2) and a[2, 0] > a[1, 0] > 0       # the wider last band is larger
    with pytest.raises(ValueError):
        cell_areas_from_centres(np.zeros((4, 4)), np.arange(4.0))  # 2-D (rotated / projected) lat-lon
    with pytest.raises(ValueError):
        cell_areas_from_centres(np.array([0.0, 1.0, 0.5]), np.arange(4.0))


# land mask ------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(REAL_STATIC), reason="data/CORDEX_NZ not present")
def test_real_land_mask_is_orog_above_zero_with_2804_cells():
    import xarray as xr
    from src.data_adapter_cordex import extract_land_mask
    with xr.open_dataset(REAL_STATIC) as st:
        assert "sftlf" not in st and "orog" in st      # no land-fraction field: the mask comes from orography
        mask, n = extract_land_mask(st)
        assert n == 2804 and np.array_equal(mask, st["orog"].values > 0)


@needs_sample
def test_one_mask_for_targets_areas_and_mass_loss():
    """Target columns, per-cell coordinates and mass-loss areas all come from the same mask, in the same order."""
    from src.data_adapter_cordex import load_cordex_full_period
    from src.mass_conservation import MassConservation
    d = load_cordex_full_period(SAMPLE)
    mask, lat, lon = d["mask"], np.asarray(d["coords"]["lat"]), np.asarray(d["coords"]["lon"])
    assert d["y"].shape[1] == mask.sum() == len(d["coords"]["lat_land"])
    areas = cell_areas_from_centres(lat, lon)
    lat2d = np.repeat(lat[:, None], len(lon), 1)
    np.testing.assert_array_equal(lat2d[mask], d["coords"]["lat_land"])          # row-major order of y's columns
    band = areas[:, 0]
    np.testing.assert_allclose(areas[mask], band[np.searchsorted(lat, d["coords"]["lat_land"])], rtol=1e-12)
    mass = MassConservation(areas[mask])
    assert mass.weights.shape[0] == d["y"].shape[1] and np.isclose(float(np.sum(mass.weights)), 1.0, atol=1e-5)
    assert np.isclose(mass.total_area_km2, areas[mask].sum())


# smoke run: lambda = 0 and lambda > 0 ----------------------------------------------------------
@needs_sample
def test_nz_mass_smoke_and_lambda0_reproduces_v2_qre(tmp_path):
    out, ck = tmp_path.as_posix(), (tmp_path / "ck").as_posix()
    flags = ["--sanity", "--set", 'data.data_dir="data/CORDEX_NZ_sample"',
             "--set", "architecture.channels=[8,16,32]", "--set", "architecture.dense=[16]",
             "--set", "segmentation.n_runs=6", "--set", "segmentation.weight_concentration_prior=10.0",
             "--set", "n_intervals=3", "--set", "data.train_years=[1961,1962]", "--set", "data.val_years=[1963,1963]",
             "--set", "data.test_years=[1981,1981]", "--set", "sanity.members_max_epochs=1",
             "--set", "sanity.weight_network_max_epochs=1", "--set", "mass.lambdas=[0,0.5]",
             "--set", "ensemble.max_epochs=2", "--set", "ensemble.max_epochs_hard=2",
             "--set", f'checkpoints.dir="{ck}"']
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2")

    def run(script, save):
        p = subprocess.run([sys.executable, os.path.join(WORKSPACE_ROOT, "experiments", script)] + flags
                           + ["--save-dir", f"{out}/{save}"], capture_output=True, text=True, env=env, cwd=WORKSPACE_ROOT)
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]

    run("run_cordex_v2.py", "v2")      # the current NZ QRE, in its own process
    run("nz_mass.py", "mass")          # loads those models, adds the mass study
    v2 = json.load(open(f"{out}/v2/sanity/repetition_0.json"))
    comp = json.load(open(f"{out}/mass/sanity/comparison.json"))
    sweep = json.load(open(f"{out}/mass/sanity/sweep/sweep.json"))

    # lambda = 0 is the NZ v2 QRE, to the last digit
    assert comp["lambda0_reproduces_nz_v2_qre_exactly"] is True
    assert comp["test"]["qre_lambda0"]["mse"] == {k: v["mse"] for k, v in v2["metrics"]["test"]["qre"].items()}
    assert sweep["0"]["val"]["qre"]["regime_mse"] == v2["metrics"]["val"]["qre"]
    assert sweep["0"]["training"]["ensemble_stage"] is None            # no extra training at lambda = 0
    # lambda > 0 trained the ensemble stage with a positive effective lambda, both loss terms logged
    stage = sweep["0.5"]["training"]["ensemble_stage"]
    assert stage["lambda_effective"] > 0 and {"qre_loss", "mass_loss"} <= set(stage["at_best_epoch"])
    assert comp["selected_lambda"] == 0.5 and comp["mass_mode"] == "absolute"
    for key in ("bgnet", "qre_lambda0", "qre_mass"):
        r = comp["test"][key]
        assert all(np.isfinite(v) for v in r["mse"].values()) and np.isfinite(r["daily_total_mae_mm_per_day"])
    assert "Mid-term" in open(f"{out}/mass/sanity/comparison.md", encoding="utf-8").read()
    for png in ("daily_totals_timeseries.png", "daily_totals_scatter.png", "example_maps.png", "error_maps.png"):
        assert os.path.exists(f"{out}/mass/sanity/plots/{png}")
