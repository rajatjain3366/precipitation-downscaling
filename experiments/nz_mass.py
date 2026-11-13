"""
nz_mass.py
==========
NZ (CORDEX-ML-Bench): QRE vs QRE with global mass-conservation regularization,
on the v2 split (train 1961-1988, val 1989-1993, test 1994-2000).

The members, Weight Network and BGNet are the NZ v2 models: they are trained or
loaded by run_cordex_v2.run_repetition with the same checkpoint names,
signatures and seeds as `python experiments/run_cordex_v2.py`. With
mass.lambda0 = "original", lambda = 0 IS that QRE (checked: its test MSEs must
equal run_repetition's exactly). A fine-tuned lambda = 0 control (the same
Weight Network fine-tune as lambda > 0, mass term off) is reported next to it,
so QRE + mass can be compared with a model that had the same extra training. For lambda > 0 the mass term of
src/mass_conservation.py (the India module: same formula, absolute mode by
default, placements members / ensemble / both, lambda list) is added; lambda is
selected on validation only, then the test split is evaluated once.

Cell areas come from the real latitude/longitude of the 128 x 128 target grid
(a regular 1-D lat/lon grid; src/grid_areas.py), restricted to the 2,804 land
cells (orography > 0 m) in the same order as the target columns.

Outputs in <save_dir>: repetition_0.json (the NZ v2 run), sweep/sweep.json,
sweep/selection.json, comparison.json, comparison.md (one table, with the
mid-term numbers of results/final_cordex_qre as a reference on a different
split), loss_scales.json, plots/.

Usage (same flags as run_cordex_v2.py):
  python experiments/nz_mass.py --set 'checkpoints.dir="checkpoints/nz_mass"' --save-dir results/fixes_v2/nz_mass
"""
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for p in [WORKSPACE_ROOT, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import run_cordex_v2 as R  # noqa: E402
import run_india as I  # noqa: E402
from src import india_metrics as M  # noqa: E402
from src.ConvolutionalNetworks import BGCallWrapper  # noqa: E402
from src.GammaLoss import GammaLoss  # noqa: E402
from src.grid_areas import cell_areas_from_centres  # noqa: E402
from src.mass_conservation import MassConservation  # noqa: E402
from src.quantiles import bin_y_var, ynetwork  # noqa: E402
from src.segmentation import daily_totals, partition_indices  # noqa: E402

MIDTERM_DIR = os.path.join(WORKSPACE_ROOT, "results", "final_cordex_qre")


class NZContext:
    """The attributes run_india's model and evaluation functions expect, built from the NZ v2 data."""

    def __init__(self, cfg, data, seg):
        self.cfg, self.seed, self.N = cfg, cfg["base_seed"], cfg["n_intervals"]
        s = data["split"]
        self.X = {n: s[n]["x"] for n in s}
        self.Y = {n: s[n]["y"] for n in s}
        self.dates = {n: s[n]["time"] for n in s}
        self.D = data["D"]
        lat, lon, mask = np.asarray(data["coords"]["lat"]), np.asarray(data["coords"]["lon"]), data["mask"]
        areas = cell_areas_from_centres(lat, lon)[mask]  # row-major, the order of the target columns
        if len(areas) != self.D:
            raise ValueError(f"{len(areas)} masked cells but the target has {self.D} columns")
        self.mass = MassConservation(areas, mode=cfg["mass"]["mode"], eps=cfg["mass"]["eps_mm_per_day"])
        self.data = {"aux": data["aux"], "areas": areas, "mask": mask, "grid_lat": lat, "grid_lon": lon,
                     "obs_label": "observed (CORDEX)", "region": "NZ land"}
        self.thr, b = seg["thresholds_daily_total_mm"], seg["bounds"]
        self.q_ranges = [(b[i], b[i + 1]) for i in range(self.N)]
        self.train_totals = daily_totals(self.Y["train"])
        self.parts = partition_indices(self.train_totals, self.thr, self.N)
        self.L = {n: np.array(bin_y_var(self.Y[n], self.q_ranges, thresholds=self.thr), dtype=np.float32) for n in s}
        self.labels = {n: np.argmax(self.L[n], axis=1) for n in s}
        self.val_parts = [np.flatnonzero(self.labels["val"] == i) for i in range(self.N)]
        self.nll = GammaLoss(epsilon=cfg["loss"]["epsilon"], y_thrs=cfg["loss"]["rain_threshold"])
        self.mode = cfg["prediction"]["mode"]
        self.out = cfg["save_dir"]


def build(c, lam, base):
    """Members and Weight Network for one lambda; lambda = 0 with lambda0 = "original" returns the v2 models."""
    placement = c.cfg["mass"]["placement"]
    lam_m = float(lam) if placement in ("members", "both") else 0.0
    members, infos = base["members"], {"members": None, "ensemble_stage": None}
    if lam_m:
        members, infos["members"] = I.get_members(c, lam_m, base["bgnet"])
    omega = base["omega"]
    if placement in ("ensemble", "both") and not I.skip_lambda0_stage(c.cfg, lam):
        omega, members, infos["ensemble_stage"] = I.get_ensemble_stage(c, float(lam), lam_m, members, omega)
    return members, omega, infos


def build_control(c, base):
    """
    Fine-tuned lambda = 0 control: the same ensemble stage as lambda > 0 (same epochs, LR, early stopping),
    with the mass term off, so "QRE + mass" vs this row isolates the mass term. With placement "members"
    there is no ensemble stage, and the control is the v2 QRE itself.
    """
    if c.cfg["mass"]["placement"] not in ("ensemble", "both"):
        return base["members"], base["omega"], None
    omega, members, info = I.get_ensemble_stage(c, 0.0, 0.0, base["members"], base["omega"])
    return members, omega, info


def qre(c, members, omega):
    """Same ensemble function as run_cordex_v2 (ynetwork), so lambda = 0 gives its predictions exactly."""
    return ynetwork({f"CNN_{i}": [BGCallWrapper(m, mean_mode=c.mode)] for i, m in enumerate(members)}, omega)


def evaluate(c, members, omega, split):
    X, Y = c.X[split], c.Y[split]
    g = R.predict(qre(c, members, omega), X)
    w = R.predict(omega, X)
    F = np.stack([R.predict(BGCallWrapper(m, mean_mode=c.mode), X) for m in members], axis=1)
    ev = lambda pred: M.evaluate(Y, pred, c.train_totals, c.labels[split], c.N, c.data["areas"], c.mass.eps)
    res = {"qre": ev(g), "perfect_member": ev(np.einsum("tn,tnd->td", c.L[split], F)),
           "weight_network": R.weight_network_diagnostics(w, c.labels[split], c.N,
                                                          c.cfg["weight_network"]["collapse_fraction"])}
    return res, g


def midterm_reference():
    """Mid-term (pre-fix) numbers: different split (train 1961-75, val 76-80, test 1981-2000), 3 repetitions."""
    path = os.path.join(MIDTERM_DIR, "aggregate_metrics.json")
    with open(path) as f:
        agg = json.load(f)["metrics"]
    out = {}
    for name, key in (("qre", "qre"), ("bgnet", "single_bgnet")):
        m = agg[key]
        out[name] = {"mse": {r: m[f"Q_{r.replace(',', '_')}"]["mean_mse"] for r in ("[0.0,1.0]", "[0.0,0.2]", "[0.9,1.0]")},
                     "mse_std": {r: m[f"Q_{r.replace(',', '_')}"]["std_mse"] for r in ("[0.0,1.0]", "[0.0,0.2]", "[0.9,1.0]")},
                     "mae": m["Q_[0.0_1.0]"]["mean_mae"]}
    out["note"] = ("results/final_cordex_qre: pipeline before the fixes, split train 1961-1975 / val 1976-1980 / "
                   "test 1981-2000 (7,300 test days), mean of 3 repetitions. Not the same test years as the rows below.")
    return out


def row(m, wn=None):
    t = m["all_india_totals"]  # area-weighted domain totals (for NZ: the 2,804 land cells)
    r = {"mse": {k: v["mse"] for k, v in m["regime_mse"].items()}, "mae": m["spatial"]["mae"],
         "rmse": m["spatial"]["rmse"], "spatial_corr": m["spatial"]["spatial_corr_daily_mean"],
         "daily_total_mae_mm_per_day": t["daily_mae_mm_per_day"],
         "total_relative_discrepancy": t["total_relative_discrepancy"]}
    if wn is not None:
        r["weight_network"] = {k: wn[k] for k in ("accuracy", "within_one_interval", "argmax_histogram",
                                                  "mean_max_weight", "collapsed")}
    return r


def fmt(v, d=2):
    return "n/a" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def table_md(comp):
    head = ("| Model | Test years | MSE [0,1] | MSE [0,0.2] | MSE [0.9,1] | MAE | RMSE | Spatial corr | "
            "Daily-total MAE (mm/day) | Total rel. discrepancy | WN accuracy | WN collapsed |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    lines = []
    for name, r in comp["midterm_reference"].items():
        if name == "note":
            continue
        mse = [f"{r['mse'][k]:.2f} ± {r['mse_std'][k]:.2f}" for k in ("[0.0,1.0]", "[0.0,0.2]", "[0.9,1.0]")]
        lines.append(f"| Mid-term {name.upper() if name == 'qre' else 'BGNet'} (reference) | 1981–2000 | "
                     + " | ".join(mse) + f" | {r['mae']:.2f} | n/a | n/a | n/a | n/a | n/a | n/a |")
    years = "–".join(str(y) for y in comp["split"]["test"])
    for label, key in (("BGNet (converged)", "bgnet"), ("QRE (λ = 0, no fine-tune)", "qre_lambda0"),
                       ("QRE (λ = 0, fine-tuned)", "qre_lambda0_finetuned"),
                       (f"QRE + mass (λ = {comp['selected_lambda']:g})", "qre_mass")):
        r = comp["test"][key]
        wn = r.get("weight_network", {})
        lines.append(f"| {label} | {years} | " + " | ".join(fmt(r["mse"][k]) for k in ("[0.0,1.0]", "[0.0,0.2]", "[0.9,1.0]"))
                     + f" | {fmt(r['mae'])} | {fmt(r['rmse'])} | {fmt(r['spatial_corr'], 3)} | "
                     f"{fmt(r['daily_total_mae_mm_per_day'], 3)} | {fmt(r['total_relative_discrepancy'], 3)} | "
                     f"{fmt(wn.get('accuracy'), 3)} | {fmt(wn.get('collapsed'))} |")
    note = ("\n\n\"QRE (λ = 0, no fine-tune)\" is the current NZ v2 QRE. \"QRE (λ = 0, fine-tuned)\" gets the same "
            "extra Weight Network fine-tune as QRE + mass with the mass term off: compare QRE + mass with it to see "
            "the effect of the mass term alone."
            f"\n\nMid-term rows: {comp['midterm_reference']['note']}\n\n"
            f"λ selected on validation only ({comp['selection']['reason']}). "
            "Daily-total errors are area-weighted over the land cells.\n")
    return head + "\n".join(lines) + note


def main():
    args = R.parse_args()
    cfg = R.load_config(args)
    if cfg["checkpoints"]["dir"] is None:
        raise ValueError("checkpoints.dir is required")
    mcfg = cfg["mass"]
    if mcfg["placement"] not in I.PLACEMENTS or 0.0 not in [float(v) for v in mcfg["lambdas"]]:
        raise ValueError("mass.placement must be members/ensemble/both and mass.lambdas must contain 0")
    out = cfg["save_dir"]
    t0 = time.time()
    data = R.prepare_data(cfg)
    seg = R.derive_segmentation(cfg, data)
    if seg["method"] == "legacy":
        raise ValueError("needs segmentation.method = bayesian_gmm")

    # 1. the NZ v2 models, exactly as run_cordex_v2.py trains / loads them
    log, base = R.run_repetition(cfg, data, seg, 0, return_models=True)
    R.save_json(log, os.path.join(out, "repetition_0.json"))
    c = NZContext(cfg, data, seg)

    # 2. lambda sweep, validation only
    rows = {}
    for lam in [float(v) for v in mcfg["lambdas"]]:
        members, omega, infos = build(c, lam, base)
        r, _ = evaluate(c, members, omega, "val")
        rows[f"{lam:g}"] = {"lambda_mass": lam, "val": r,
                            "training": {"members": [I.summary(i) for i in infos["members"]] if infos["members"] else None,
                                         "ensemble_stage": I.summary(infos["ensemble_stage"])}}
        q = r["qre"]
        print(f"  lambda {lam:g}: val MSE {I.r3(v['mse'] for v in q['regime_mse'].values())} "
              f"daily-total MAE {q['all_india_totals']['daily_mae_mm_per_day']:.4f} mm/day, "
              f"WN acc {r['weight_network']['accuracy']:.3f} collapsed={r['weight_network']['collapsed']}", flush=True)
        R.save_json(rows, os.path.join(out, "sweep", "sweep.json"))
    sel = I.select_lambda(rows, cfg["selection"])
    R.save_json(sel, os.path.join(out, "sweep", "selection.json"))
    lam_b = float(sel["selected_lambda"])
    print(f"  selected lambda_mass = {lam_b:g} ({sel['reason']})", flush=True)

    # fine-tuned lambda = 0 control (not part of the selection; reported next to QRE + mass)
    memc, omc, info_c = build_control(c, base)
    rc_val, _ = evaluate(c, memc, omc, "val")
    R.save_json({"val": rc_val, "training": I.summary(info_c)}, os.path.join(out, "sweep", "control_lambda0_finetuned.json"))

    # 3. test, once, after the selection
    mem0, om0, _ = build(c, 0.0, base)
    memb, omb, _ = build(c, lam_b, base)
    r0, g0 = evaluate(c, mem0, om0, "test")
    rc, gc = evaluate(c, memc, omc, "test")
    rb, gb = evaluate(c, memb, omb, "test")
    bg = R.predict(BGCallWrapper(base["bgnet"], mean_mode=c.mode), c.X["test"])
    bg_m = M.evaluate(c.Y["test"], bg, c.train_totals, c.labels["test"], c.N, c.data["areas"], c.mass.eps)
    lambda0_exact = None
    if I.skip_lambda0_stage(cfg, 0.0):
        lambda0_exact = R.regime_mse(c.Y["test"], g0, c.train_totals) == log["metrics"]["test"]["qre"]
        if not lambda0_exact:
            raise AssertionError("lambda = 0 does not reproduce the NZ v2 QRE test MSEs")
    comp = {"split": {"train": cfg["data"]["train_years"], "val": cfg["data"]["val_years"],
                      "test": cfg["data"]["test_years"]},
            "selected_lambda": lam_b, "selection": sel, "placement": mcfg["placement"], "mass_mode": mcfg["mode"],
            "lambda0_reproduces_nz_v2_qre_exactly": lambda0_exact,
            "test": {"bgnet": row(bg_m), "qre_lambda0": row(r0["qre"], r0["weight_network"]),
                     "qre_lambda0_finetuned": row(rc["qre"], rc["weight_network"]),
                     "qre_mass": row(rb["qre"], rb["weight_network"]),
                     "perfect_member_lambda0": row(r0["perfect_member"])},
            "control_lambda0_finetuned_training": I.summary(info_c),
            "test_full": {"qre_lambda0": r0, "qre_lambda0_finetuned": rc, "qre_mass": rb, "bgnet": bg_m},
            "midterm_reference": midterm_reference(), "seconds": time.time() - t0}
    R.save_json(comp, os.path.join(out, "comparison.json"))
    with open(os.path.join(out, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(table_md(comp))
    I.plots(c, os.path.join(out, "plots"), "test",
            {"QRE (λ=0)": g0, "QRE (λ=0, fine-tuned)": gc, f"QRE+mass (λ={lam_b:g})": gb, "BGNet": bg})
    pt = {"qre_lambda0": {"train": R.predict(qre(c, mem0, om0), c.X["train"]), "val": evaluate(c, mem0, om0, "val")[1]},
          "bgnet": {s: R.predict(BGCallWrapper(base["bgnet"], mean_mode=c.mode), c.X[s]) for s in ("train", "val")}}
    R.save_json(I.loss_scales(c, pt), os.path.join(out, "loss_scales.json"))
    print(table_md(comp), flush=True)


if __name__ == "__main__":
    main()
