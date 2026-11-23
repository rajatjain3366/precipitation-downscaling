"""
experiments/mass_study.py and the new stages: India member study, placement study and
repetitions (run_india.py --stage members / placements / repeats), and the NZ placement
study and repetitions on the member-study models (nz_mass.py --stage placements / repeats).

Unit tests for the Wilcoxon tests, the summary and the placement selection; small
integration runs on the India sample and the NZ fixture (2 repetitions, so the Wilcoxon
"needs >= 6" path; the 6-repetition path is covered by the unit tests).
The two integration runs are marked slow (about 10 min on a laptop CPU); the smoke
suite runs only the unit tests.
Run with:  python -m pytest tests/test_mass_study.py -q               (everything)
           python -m pytest -m "not slow" tests/test_mass_study.py -q  (as in the smoke suite)
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (WORKSPACE_ROOT, os.path.join(WORKSPACE_ROOT, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mass_study as S  # noqa: E402

INDIA_SAMPLE = os.path.join(WORKSPACE_ROOT, "data", "India_sample")
NZ_SAMPLE = os.path.join(WORKSPACE_ROOT, "data", "CORDEX_NZ_sample")
ENV = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2", TF_FORCE_GPU_ALLOW_GROWTH="true")


# --- statistics ------------------------------------------------------------------------------
def test_wilcoxon_needs_six_pairs():
    w = S.wilcoxon([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    assert w["p_value"] is None and w["significant"] is None and "at least 6" in w["note"]


def test_wilcoxon_six_pairs_all_lower_is_significant():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    w = S.wilcoxon(a, a + np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    assert w["p_value"] == pytest.approx(2 / 64) and w["significant"] and w["lower"] == "first"
    assert w["median_difference"] < 0
    w = S.wilcoxon(a, a + np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]))
    assert w["p_value"] > 0.05 and not w["significant"] and w["lower"] is None


def test_wilcoxon_all_equal():
    w = S.wilcoxon([1.0] * 6, [1.0] * 6)
    assert w["p_value"] == 1.0 and w["significant"] is False


def rep(k, bg, q, c, m):
    mk = lambda v: {"mse_0_1": v, "mse_0_0.2": v / 10, "mse_0.9_1": v * 3, "daily_total_mae": v / 100,
                    "abs_total_rel_discrepancy": 0.04, "total_rel_discrepancy": -0.04}
    return {"repetition": k, "test": {"bgnet": mk(bg), "qre": mk(q), "control": mk(c), "qre_mass": mk(m)}}


def test_summarize_and_table():
    reps = [rep(k, 50 + k, 52 + k, 52 + k, 51.5 + k) for k in range(6)]
    st = S.summarize(reps)
    assert st["n_repetitions"] == 6
    assert st["summary"]["qre"]["mse_0_1"]["mean"] == pytest.approx(54.5)
    assert st["summary"]["qre"]["mse_0_1"]["std"] == pytest.approx(np.std(np.arange(6), ddof=1))
    t = st["wilcoxon"]
    assert set(t) == {"qre_vs_bgnet", "qre_mass_vs_qre", "qre_mass_vs_control"}
    assert t["qre_vs_bgnet"]["mse_0_1"]["significant"] and t["qre_vs_bgnet"]["mse_0_1"]["lower"] == "second"
    assert t["qre_mass_vs_qre"]["mse_0_1"]["lower"] == "first"
    assert t["qre_mass_vs_qre"]["abs_total_rel_discrepancy"]["p_value"] == 1.0   # identical
    md = S.repetitions_md({"placement": "ensemble", "selected_lambda": 1.0, "stats": st})
    assert "QRE + mass (ensemble, λ = 1)" in md and "*" in md and "± " in md


# --- placement selection ---------------------------------------------------------------------
def prow(lam, mse, mae):
    return {"lambda_mass": lam, "val": {"qre": {"regime_mse": {"[0.0,1.0]": {"mse": mse}},
                                                "all_india_totals": {"daily_mae_mm_per_day": mae}}}}


SCFG = {"mse_regime": "[0.0,1.0]", "mse_tolerance": 0.02, "total_metric": "daily_mae_mm_per_day"}


def test_select_placement_uses_each_placements_own_control():
    res = {
        # within 2 % of its control, lowest totals error overall -> selected
        "members": {"rows": {"0": prow(0, 100.0, 5.0), "1": prow(1, 101.5, 3.0), "10": prow(10, 110.0, 1.0)}},
        # MSE better in absolute terms, but its totals error is higher
        "ensemble": {"rows": {"0": prow(0, 90.0, 5.0), "1": prow(1, 90.5, 4.0)}},
    }
    sel = S.select_placement(res, SCFG)
    assert (sel["placement"], sel["selected_lambda"], sel["tolerance_met"]) == ("members", 1, True)
    assert sel["val_totals_improved_vs_control"] is True
    lam10 = [c for c in sel["candidates"] if c["lambda_mass"] == 10][0]
    assert lam10["within_tolerance"] is False          # 10 % worse than its control: excluded


def test_select_placement_fallback_flags_it():
    res = {"members": {"rows": {"0": prow(0, 100.0, 5.0), "1": prow(1, 120.0, 1.0)}},
           "both": {"rows": {"0": prow(0, 100.0, 5.0), "1": prow(1, 105.0, 4.0)}}}
    sel = S.select_placement(res, SCFG)
    assert (sel["placement"], sel["tolerance_met"]) == ("both", False)


# --- integration: India sample ---------------------------------------------------------------
def run(script, *args):
    p = subprocess.run([sys.executable, os.path.join(WORKSPACE_ROOT, "experiments", script), *args],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV, cwd=WORKSPACE_ROOT)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return p.stdout


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(INDIA_SAMPLE), reason="needs data/India_sample")
def test_india_member_placement_repetition_stages(tmp_path):
    out, ck = (tmp_path / "out").as_posix(), (tmp_path / "ck").as_posix()
    common = ["--config", "configs/india_smoke.json", "--save-dir", out, "--set", f'checkpoints.dir="{ck}"',
              "--set", 'study.placements=["ensemble","members"]', "--set", "study.repetitions=2"]
    run("run_india.py", "--stage", "members", *common)
    ms = json.load(open(f"{out}/member_study/results.json"))
    sel = json.load(open(f"{out}/member_study/selection.json"))
    assert set(sel["perfect_member_val_mse_0_1"]) == {"scratch", "warm_start"}
    assert sel["selected_init"] == min(sel["perfect_member_val_mse_0_1"], key=sel["perfect_member_val_mse_0_1"].get)
    assert all(o["best_epoch"] is not None for o in ms["warm_start"]["own_interval_val"])   # epochs are reported

    run("run_india.py", "--stage", "placements", *common)
    psel = json.load(open(f"{out}/placements/selection.json"))
    assert psel["placement"] in ("ensemble", "members") and psel["selected_lambda"] > 0
    assert {c["placement"] for c in psel["candidates"]} == {"ensemble", "members"}
    comp = json.load(open(f"{out}/placements/comparison.json"))
    assert set(comp["test"]) == {"bgnet", "qre", "control", "qre_mass", "perfect_member_qre"}
    ens = json.load(open(f"{out}/placements/ensemble/sweep.json"))
    assert ens["0"]["is_control"] and ens["0"]["training"]["ensemble_stage"]["lambda_effective"] == 0.0
    mem = json.load(open(f"{out}/placements/members/sweep.json"))
    assert mem["0"]["training"]["ensemble_stage"] is None                       # control = the QRE itself
    assert mem["1"]["training"]["members"][0]["lambda_effective"] > 0          # members trained with the mass term

    stdout = run("run_india.py", "--stage", "repeats", *common)
    summ = json.load(open(f"{out}/repetitions/summary.json"))
    assert summ["stats"]["n_repetitions"] == 2 and summ["placement"] == psel["placement"]
    assert summ["stats"]["wilcoxon"]["qre_vs_bgnet"]["mse_0_1"]["p_value"] is None   # needs >= 6
    r1 = json.load(open(f"{out}/repetitions/rep_1.json"))
    assert r1["seed"] == 42 + 1000 and os.path.isdir(f"{ck}/repetitions/rep1")
    # rerun: finished repetitions are skipped
    assert "repetition 1: already done" in run("run_india.py", "--stage", "repeats", *common)
    assert "repetition 0" in stdout


# --- integration: NZ fixture on the member-study models --------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(NZ_SAMPLE), reason="run tests/generate_sample_cordex.py first")
def test_nz_placements_and_repetitions_use_the_member_study_models(tmp_path):
    out, ck = (tmp_path / "out").as_posix(), (tmp_path / "ck").as_posix()
    flags = ["--sanity", "--set", 'data.data_dir="data/CORDEX_NZ_sample"',
             "--set", "architecture.channels=[8,16,32]", "--set", "architecture.dense=[16]",
             "--set", "segmentation.n_runs=6", "--set", "segmentation.weight_concentration_prior=10.0",
             "--set", "n_intervals=3", "--set", "data.train_years=[1961,1962]", "--set", "data.val_years=[1963,1963]",
             "--set", "data.test_years=[1981,1981]", "--set", "sanity.members_max_epochs=1",
             "--set", "sanity.weight_network_max_epochs=1", "--set", "ensemble.max_epochs=2",
             "--set", "ensemble.max_epochs_hard=2"]
    for part in ("scratch", "warm_start", "bgnet_converged"):
        flags += ["--set", f"member_study.{part}.max_epochs=2", "--set", f"member_study.{part}.max_epochs_hard=2"]
    flags += ["--set", "study.lambdas=[0,1]", "--set", 'study.placements=["ensemble","both"]',
              "--set", "study.repetitions=2",
              "--set", f'study.member_study_results="{out}/sanity/member_study/results.json"',
              "--set", f'checkpoints.dir="{ck}"', "--save-dir", out]
    run("run_cordex_v2.py", *flags)                                            # step 0
    run("member_study.py", *flags, "--set", "member_study.retrain_bgnet=true")  # steps 1-3 in one go
    ms = json.load(open(f"{out}/sanity/member_study/results.json"))
    variant = ms["selection"]["best_by_perfect_member_val_mse_0_1"]

    stdout = run("nz_mass.py", "--stage", "placements", *flags)
    assert f"loaded rep0/members_{variant}/member_0 from checkpoint" in stdout   # the member-study models
    assert "loaded rep0/bgnet_converged from checkpoint" in stdout
    comp = json.load(open(f"{out}/sanity/mass_study/placements/comparison.json"))
    chk = comp["member_study_check"]
    assert chk["qre"]["max_relative_difference"] <= 1e-3 and chk["bgnet"]["max_relative_difference"] <= 1e-3

    run("nz_mass.py", "--stage", "repeats", *flags)
    summ = json.load(open(f"{out}/sanity/mass_study/repetitions/summary.json"))
    assert summ["stats"]["n_repetitions"] == 2
    assert summ["stats"]["summary"]["bgnet"]["mse_0_1"]["values"][0] == pytest.approx(comp["test"]["bgnet"]["mse"]["[0.0,1.0]"])
    assert os.path.exists(f"{ck}/rep1/members_{variant}/member_0.weights.h5")   # rep 1 trained the same recipe
