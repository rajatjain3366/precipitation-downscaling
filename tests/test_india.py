"""
India pipeline and mass-conservation tests (unit + sample-file + runner resume).
Run with:  python -m pytest tests/test_india.py -q
The sample-file tests need data/India_sample/ (skipped if absent).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import tensorflow as tf
import xarray as xr

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from gpu_tol import assert_arrays_close, assert_close  # noqa: E402  (exact on CPU, rtol 1e-5 on GPU)
from src.builders import build_bgnet, build_omega  # noqa: E402
from src.GammaLoss import GammaLoss  # noqa: E402
from src.india_data import (EARTH_RADIUS_KM, block_alignment, cell_areas_km2, check_units, coarsen_blocks,  # noqa: E402
                            load_era5, load_imd, load_india, match_days)
from src.mass_conservation import (JointEnsemble, MassConservation, MemberLossWithMass, bg_expected,  # noqa: E402
                                   member_metrics, weight_network_loss)

SAMPLE = os.path.join(WORKSPACE_ROOT, "data", "India_sample")
IMD_FILE = os.path.join(SAMPLE, "RF25_ind1990_rfp25.nc")
ERA5_FILE = os.path.join(SAMPLE, "india_era5_1990_01_coarse_4x.nc")
needs_sample = pytest.mark.skipif(not (os.path.exists(IMD_FILE) and os.path.exists(ERA5_FILE)),
                                  reason="data/India_sample not present")
SMOKE_CFG = os.path.join(WORKSPACE_ROOT, "configs", "india_smoke.json")


@pytest.fixture(scope="module")
def cfg():
    with open(SMOKE_CFG) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def sample(cfg):
    return load_india(cfg["data"] | {"data_dir": SAMPLE}, verbose=False)


# 1. both files load; variables and units -------------------------------------------------
@needs_sample
def test_sample_files_load_with_units():
    imd = load_imd([IMD_FILE])
    assert imd["rain"].shape == (365, 129, 135) and imd["units"] == "mm"
    era = load_era5([ERA5_FILE])
    assert era["x"].shape == (31, 33, 34, 5)
    assert check_units(era["units"]) == {}  # q kg/kg, t K, u/v m/s, w Pa/s
    assert era["temporal"] == {"kind": "one_field_per_day", "hours_utc": [0]}


# 2. alignment: latitude flip, dropped partial blocks, day offset ---------------------------
@needs_sample
def test_latitude_flip_and_partial_blocks():
    imd, era = load_imd([IMD_FILE]), load_era5([ERA5_FILE])
    assert np.all(np.diff(era["lat"]) > 0) and np.all(np.diff(imd["lat"]) > 0)  # both ascending
    al = block_alignment(era["lat"], era["lon"], imd["lat"], imd["lon"], 4)
    assert al["dropped_lat"] == [6.5] and al["dropped_lon"] == [99.75]
    assert len(al["lat_keep"]) == 32 and len(al["lon_keep"]) == 33
    for j, k in enumerate(al["lat_keep"]):  # every kept coarse coordinate is the mean of its 4 fine cells
        assert np.isclose(imd["lat"][al["lat_block_of_fine"] == j].mean(), era["lat"][k])
    for j, k in enumerate(al["lon_keep"]):
        assert np.isclose(imd["lon"][al["lon_block_of_fine"] == j].mean(), era["lon"][k])


@needs_sample
def test_coarsening_reproduces_sample_grid():
    """The download script's coarsening of the 0.25 deg grid gives exactly the sample's ERA5 coordinates."""
    imd = load_imd([IMD_FILE])
    lat, lon = imd["lat"][::-1], imd["lon"]  # ERA5 order: north to south
    lat_c = coarsen_blocks(np.repeat(lat[:, None], len(lon), 1), 4, (0, 1))[:, 0]
    lon_c = coarsen_blocks(np.repeat(lon[None, :], len(lat), 0), 4, (0, 1))[0]
    with xr.open_dataset(ERA5_FILE) as e:
        assert np.allclose(lat_c, e["latitude"].values) and np.allclose(lon_c, e["longitude"].values)


def test_coarsen_and_load_values(tmp_path):
    """Synthetic fine field -> coarsen_file -> load_era5: block means in the right (flipped) place."""
    sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "scripts"))
    from download_era5_india import coarsen_file
    lat = np.arange(38.5, 6.49, -0.25)
    lon = np.arange(66.5, 100.01, 0.25)
    t = np.array(["1990-01-01", "1990-01-02"], dtype="datetime64[ns]")
    field = (lat[:, None] * 100 + lon[None, :])[None] + np.arange(2)[:, None, None]
    ds = xr.Dataset({v: (("valid_time", "latitude", "longitude"), field.astype(np.float32), {"units": "x"})
                     for v in ["q", "t", "u", "v", "w"]}, coords={"valid_time": t, "latitude": lat, "longitude": lon})
    src, dst = tmp_path / "fine.nc", tmp_path / "coarse.nc"
    ds.to_netcdf(src)
    coarsen_file(str(src), str(dst))
    era = load_era5([str(dst)])
    # block at the north-west corner: fine lats 38.5..37.75, lons 66.5..67.25; after the flip it is the last
    # row (the partial 6.5 row comes first)
    expect = field[1, :4, :4].mean()
    assert np.isclose(era["lat"][-1], 38.125) and np.isclose(era["lat"][0], 6.5)
    assert np.isclose(era["x"][1, -1, 0, 0], expect, rtol=1e-6)


def test_day_offset_pairs_and_reports_missing():
    d = np.arange(np.datetime64("1990-01-01"), np.datetime64("1990-01-06"))
    imd_idx, era_idx, rep = match_days(d, d, day_offset=1)
    assert np.array_equal(d[era_idx], d[imd_idx] + np.timedelta64(1, "D"))
    assert rep["missing_era5_days"] == [{"imd_day": "1990-01-05", "era5_day_needed": "1990-01-06"}]
    _, _, rep0 = match_days(d[[0, 1, 3, 4]], d, 0, period=["1990-01-01", "1990-01-05"])
    assert rep0["missing_imd_days"] == ["1990-01-03"]


def test_hourly_era5_is_averaged_to_days(tmp_path):
    t = np.arange(np.datetime64("1990-01-01T00"), np.datetime64("1990-01-03T00"), np.timedelta64(1, "h"))
    vals = np.arange(len(t), dtype=np.float32)[:, None, None] * np.ones((1, 2, 2), np.float32)
    ds = xr.Dataset({v: (("valid_time", "latitude", "longitude"), vals) for v in ["q", "t", "u", "v", "w"]},
                    coords={"valid_time": t, "latitude": [10.0, 9.0], "longitude": [70.0, 71.0]})
    ds.to_netcdf(tmp_path / "h.nc")
    utc = load_era5([str(tmp_path / "h.nc")])
    assert utc["x"].shape[0] == 2 and np.isclose(utc["x"][0, 0, 0, 0], np.mean(np.arange(24)))
    imd_window = load_era5([str(tmp_path / "h.nc")], aggregation_shift_hours=3)  # 03 UTC .. 02 UTC
    # windows are labelled by their start date: 1989-12-31 (00-02 UTC), 1990-01-01 (03 UTC .. 02 UTC), ...
    assert str(imd_window["dates"][1]) == "1990-01-01"
    assert np.isclose(imd_window["x"][1, 0, 0, 0], np.mean(np.arange(3, 27)))
    assert imd_window["temporal"]["incomplete_days"]  # first and last windows are partial and reported


# 3. shapes and 4. NaN cells masked, never zero-filled ----------------------------------------
@needs_sample
def test_shapes(sample):
    assert sample["x"].shape == (31, 32, 33, 5) and sample["y"].shape == (31, 4964)


@needs_sample
def test_nan_cells_masked_not_zero_filled(sample):
    imd = load_imd([IMD_FILE])
    assert sample["mask"].sum() == 4964 and not np.isnan(sample["y"]).any()
    jan = imd["rain"][:31]
    assert np.isnan(jan[:, ~sample["mask"]]).all()               # outside India stays NaN in the source
    assert (sample["y"] == 0).sum() == (jan[:, sample["mask"]] == 0).sum()  # no extra zeros created


# 5. cell areas ----------------------------------------------------------------------------
def test_cell_areas_positive_and_sum_to_box():
    lat, lon = np.arange(6.5, 38.51, 0.25), np.arange(66.5, 100.01, 0.25)
    a = cell_areas_km2(lat, lon)
    assert a.shape == (129, 135) and np.all(a > 0)
    box = EARTH_RADIUS_KM ** 2 * np.deg2rad(lon[-1] - lon[0] + 0.25) * \
        (np.sin(np.deg2rad(38.625)) - np.sin(np.deg2rad(6.375)))
    assert np.isclose(a.sum(), box, rtol=1e-10)
    assert a[0, 0] > a[-1, 0]  # cells shrink towards the pole


# 6-9. mass loss -----------------------------------------------------------------------------
def _mass(D=6, mode="relative", mask=None):
    rng = np.random.default_rng(0)
    return MassConservation(rng.uniform(500, 800, D), mask=mask, mode=mode, eps=0.1)


@pytest.mark.parametrize("mode", ["relative", "absolute"])
def test_mass_loss_zero_iff_totals_match(mode):
    mass = _mass(mode=mode)
    y = tf.constant(np.random.default_rng(1).gamma(1.0, 2.0, (4, 6)).astype(np.float32))
    w = mass.weights.numpy()
    d = np.zeros(6, np.float32)
    d[0], d[1] = w[1], -w[0]  # area-weighted sum of d is 0: same total, different pattern
    assert float(mass(y, y)) == 0.0
    assert float(mass(y, y + 0.1 * d)) < 1e-6
    assert float(mass(y, 2.0 * y)) > 0.0


def test_masked_cells_do_not_affect_loss():
    mask = np.array([1, 1, 1, 0, 1, 0], bool)
    mass = _mass(mask=mask)
    y = np.random.default_rng(2).gamma(1.0, 2.0, (3, 6)).astype(np.float32)
    p = y * 1.3
    base = float(mass(y, p))
    y2, p2 = y.copy(), p.copy()
    y2[:, ~mask], p2[:, ~mask] = np.nan, 1e6
    assert float(mass(y2, p2)) == base
    pt = tf.Variable(p2)
    with tf.GradientTape() as tape:
        loss = mass(tf.constant(y2), pt)
    g = tape.gradient(loss, pt).numpy()
    assert np.all(np.isfinite(g)) and np.all(g[:, ~mask] == 0)


def test_loss_is_per_day():
    mass = _mass()
    rng = np.random.default_rng(3)
    y = rng.gamma(1.0, 2.0, (5, 6)).astype(np.float32)
    p = (y * rng.uniform(0.5, 1.5, (5, 1))).astype(np.float32)
    per_day = [float(mass(y[t:t + 1], p[t:t + 1])) for t in range(5)]
    assert np.isclose(float(mass(y, p)), np.mean(per_day), rtol=1e-6)  # mean of days, not one pooled total
    p2 = p.copy()
    p2[3] *= 5
    assert np.isclose(mass.per_day(y, p2).numpy()[0], mass.per_day(y, p).numpy()[0])
    assert float(mass(y, p2)) > float(mass(y, p))


def test_gradients_flow_through_mass_loss():
    mass = _mass()
    y = tf.constant(np.random.default_rng(4).gamma(1.0, 2.0, (3, 6)).astype(np.float32))
    p = tf.Variable(y * 1.5)
    with tf.GradientTape() as tape:
        loss = mass(y, p)
    g = tape.gradient(loss, p).numpy()
    assert np.all(np.isfinite(g)) and np.abs(g).sum() > 0


# 10-11. training steps and lambda = 0 ---------------------------------------------------------
D_SYN, N_SYN, B_SYN = 40, 3, 8


@pytest.fixture(scope="module")
def syn(cfg):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(B_SYN, 32, 33, 5)).astype(np.float32)
    y = np.where(rng.random((B_SYN, D_SYN)) < 0.4, 0.0, rng.gamma(1.0, 3.0, (B_SYN, D_SYN))).astype(np.float32)
    lab = np.eye(N_SYN, dtype=np.float32)[np.arange(B_SYN) % N_SYN]
    mass = MassConservation(rng.uniform(600, 760, D_SYN), eps=0.1)
    nll = GammaLoss(y_thrs=0.0)
    return {"x": x, "y": y, "lab": lab, "mass": mass, "nll": nll, "cfg": cfg}


def _member(s):
    m = build_bgnet(s["cfg"], D_SYN)
    m(s["x"][:1])
    return m


def _omega(s):
    o = build_omega(s["cfg"], N_SYN)
    o(s["x"][:1])
    return o


def test_training_step_members_placement(syn):
    m = _member(syn)
    m.compile(optimizer="adam", loss=MemberLossWithMass(syn["nll"], syn["mass"], 0.5),
              metrics=member_metrics(syn["nll"], syn["mass"]))
    h = m.fit(syn["x"], syn["y"], batch_size=B_SYN, epochs=1, verbose=0).history
    assert np.isfinite(h["loss"][0]) and {"bg_nll", "mass_loss"} <= set(h)
    assert np.isclose(h["loss"][0], h["bg_nll"][0] + 0.5 * h["mass_loss"][0], rtol=1e-4)


@pytest.mark.parametrize("train_members", [False, True])
def test_training_step_ensemble_placement(syn, train_members):
    members = [_member(syn) for _ in range(N_SYN)]
    om = _omega(syn)
    if train_members:  # "both" / joint fine-tuning of members and Weight Network
        joint = JointEnsemble(om, weight_network_loss(syn["cfg"]["weight_network"], N_SYN), syn["mass"], 0.5,
                              members=members, train_members=True, nll=syn["nll"])
        x_in = [syn["x"]]
    else:  # members frozen, predictions precomputed
        F = np.stack([bg_expected(m(syn["x"])).numpy() for m in members], axis=1)
        joint = JointEnsemble(om, weight_network_loss(syn["cfg"]["weight_network"], N_SYN), syn["mass"], 0.5)
        x_in = [syn["x"], F]
    before = [v.numpy().copy() for v in members[0].trainable_weights] if train_members else None
    joint.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    h = joint.fit(x_in, [syn["lab"], syn["y"]], batch_size=B_SYN, epochs=1, verbose=0).history
    assert {"loss", "qre_loss", "mass_loss"} <= set(h) and np.isfinite(h["loss"][0])
    if train_members:
        assert any(not np.array_equal(b, v.numpy()) for b, v in zip(before, members[0].trainable_weights))


def test_lambda_zero_reproduces_original_objective(syn):
    # members: NLL + 0 * mass is exactly the NLL (on a GPU: up to float noise, rtol 1e-5)
    m = _member(syn)
    p = m(syn["x"])
    assert_close(float(MemberLossWithMass(syn["nll"], syn["mass"], 0.0)(syn["y"], p)), float(syn["nll"](syn["y"], p)),
                 "member loss")
    # ensemble stage: loss and gradients are exactly those of the Weight Network loss alone
    om = _omega(syn)
    wn = weight_network_loss(syn["cfg"]["weight_network"], N_SYN)
    F = np.random.default_rng(6).gamma(1.0, 2.0, (B_SYN, N_SYN, D_SYN)).astype(np.float32)
    joint = JointEnsemble(om, wn, syn["mass"], 0.0)
    lab, y = tf.constant(syn["lab"]), tf.constant(syn["y"])
    with tf.GradientTape() as t1:
        total, qre, mass = joint.loss_terms([syn["x"], F], lab, y, training=False)
    g1 = t1.gradient(total, om.trainable_variables)
    with tf.GradientTape() as t2:
        ref = wn(lab, om(syn["x"], training=False))
    g2 = t2.gradient(ref, om.trainable_variables)
    assert_close(float(total), float(ref), "ensemble loss")
    assert float(mass) > 0
    for i, (a, b) in enumerate(zip(g1, g2)):
        assert_arrays_close(a.numpy(), b.numpy(), f"gradient {i}")
    # and lambda > 0 does change the gradient
    joint.lam = 1.0
    with tf.GradientTape() as t3:
        total3, _, _ = joint.loss_terms([syn["x"], F], lab, y, training=False)
    g3 = t3.gradient(total3, om.trainable_variables)
    assert any(not np.allclose(a.numpy(), b.numpy()) for a, b in zip(g1, g3))


# 12. checkpoint resume of the India runner -------------------------------------------------
@needs_sample
def test_india_runner_resumes_from_checkpoints(tmp_path):
    out, ck = (tmp_path / "out").as_posix(), (tmp_path / "ck").as_posix()
    cmd = [sys.executable, os.path.join(WORKSPACE_ROOT, "experiments", "run_india.py"), "--config", SMOKE_CFG,
           "--stage", "A", "--save-dir", out, "--set", f'checkpoints.dir="{ck}"']
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2", TF_FORCE_GPU_ALLOW_GROWTH="true")
    first = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=WORKSPACE_ROOT)
    assert first.returncode == 0, first.stdout[-2000:] + first.stderr[-2000:]
    m1 = json.load(open(os.path.join(out, "A", "metrics.json")))
    files = {p: os.path.getmtime(os.path.join(dp, p)) for dp, _, fs in os.walk(ck) for p in fs}
    second = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=WORKSPACE_ROOT)
    assert second.returncode == 0, second.stdout[-2000:] + second.stderr[-2000:]
    assert "loaded bgnet from checkpoint" in second.stdout and "loaded ensemble_lam0/omega" in second.stdout
    after = {p: os.path.getmtime(os.path.join(dp, p)) for dp, _, fs in os.walk(ck) for p in fs}
    assert after == files  # nothing retrained or rewritten
    assert_close(json.load(open(os.path.join(out, "A", "metrics.json"))), m1, "A/metrics.json")


# real JJAS data (two seasons), only if data/India/ERA5_full and IMD_jjas are present -----------
REAL_INDIA = os.path.join(WORKSPACE_ROOT, "data", "India")
needs_real = pytest.mark.skipif(not (os.path.isdir(os.path.join(REAL_INDIA, "ERA5_full"))
                                     and os.path.isdir(os.path.join(REAL_INDIA, "IMD_jjas"))),
                                reason="real India JJAS data not present")


@needs_real
@pytest.mark.parametrize("offset", [0, 1])
def test_real_two_seasons_load(offset):
    with open(os.path.join(WORKSPACE_ROOT, "configs", "india_v1.json")) as f:
        d = json.load(f)["data"]
    d = dict(d, data_dir=REAL_INDIA, era5_pattern="ERA5_full/era5_199[01]_coarse.nc",
             imd_pattern="IMD_jjas/RF25_ind199[01]_jjas.nc", period=["1990-06-01", "1991-09-30"], day_offset=offset)
    out = load_india(d, verbose=False)
    n = 2 * 122 - (2 if offset == 1 else 0)          # offset 1: 30 September has no ERA5 1 October
    assert out["x"].shape == (n, 32, 33, 5) and out["y"].shape == (n, 4964)
    months = out["dates"].astype("datetime64[M]").astype(int) % 12 + 1
    assert set(months) == {6, 7, 8, 9} and str(out["dates"][0]) == "1990-06-01"
    assert out["report"]["days"]["missing_imd_days"] == [] and out["report"]["days"]["missing_era5_days"] == []
    assert len(out["report"]["days"]["season_edge_dropped"]) == (2 if offset == 1 else 0)
    assert out["mask"].sum() == 4964 and not np.isnan(out["y"]).any() and not np.isnan(out["x"]).any()
    assert np.all(np.diff(out["era5_lat"]) > 0)       # ascending after load whatever the file order


def test_months_restrict_completeness_check():
    jjas = np.concatenate([np.arange(np.datetime64(f"{y}-06-01"), np.datetime64(f"{y}-10-01")) for y in (1990, 1991)])
    _, _, rep = match_days(jjas, jjas, 0, ["1990-06-01", "1991-09-30"], [6, 7, 8, 9])
    assert rep["missing_imd_days"] == [] and rep["matched_days"] == 244   # Oct-May are not "missing"
    _, _, rep = match_days(jjas, jjas, 0, ["1990-06-01", "1991-09-30"], None)
    assert len(rep["missing_imd_days"]) == 365 - 122                       # without months they would be
    _, _, rep = match_days(np.delete(jjas, 10), jjas, 0, ["1990-06-01", "1991-09-30"], [6, 7, 8, 9])
    assert rep["missing_imd_days"] == ["1990-06-11"]                        # a gap inside the season is
