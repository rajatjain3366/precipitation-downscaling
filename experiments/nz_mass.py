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
selected on validation only, against the fine-tuned lambda = 0 control below
(the same rule as India), then the test split is evaluated once.

Cell areas come from the real latitude/longitude of the 128 x 128 target grid
(a regular 1-D lat/lon grid; src/grid_areas.py), restricted to the 2,804 land
cells (orography > 0 m) in the same order as the target columns.

Outputs in <save_dir>: repetition_0.json (the NZ v2 run), sweep/sweep.json,
sweep/selection.json, comparison.json, comparison.md (one table, with the
mid-term numbers of results/final_cordex_qre as a reference on a different
split), loss_scales.json, plots/.

Usage (same flags as run_cordex_v2.py):
  python experiments/nz_mass.py --set 'checkpoints.dir="checkpoints/nz_mass"' --save-dir results/fixes_v2/nz_mass

--stage v2 (default) is the study above, on the NZ v2 models (run_repetition:
members from scratch, BGNet with the v2 settings, all in checkpoints/nz_mass).
Those are NOT the models of the NZ member study (RUNBOOK steps 1-3), which is why
its lambda = 0 QRE and BGNet differ from the member-study numbers.

--stage placements / repeats (experiments/mass_study.py, config section "study")
use the member-study models instead: the member set selected in step 2
(study.member_set = "auto" reads it from study.member_study_results), the
converged BGNet of step 3 as the baseline, the 10-epoch BGNet as the warm start
and the squared-EMD Weight Network. They must run with the member study's flags
(--sanity, its intervals file, epochs and checkpoints.dir), so that the
checkpoint signatures match; repetition 0 then loads those exact models, and the
QRE / BGNet test MSEs are checked against the member study's (relative 1e-3).
Repetitions k >= 1 train the same recipe with seed base_seed + 1000 k, in
rep<k>/ of the same checkpoints.dir; the mass models go to rep<k>/mass/<placement>/.
  python experiments/nz_mass.py --stage placements --sanity <member-study flags>
  python experiments/nz_mass.py --stage repeats --sanity <member-study flags>
Outputs: <save_dir>/mass_study/placements/, <save_dir>/mass_study/repetitions/.
"""
import argparse
import copy
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
import mass_study as S  # noqa: E402
from src import checkpoints  # noqa: E402
from src import india_metrics as M  # noqa: E402
from src.builders import build_bgnet, build_omega  # noqa: E402
from src.ConvolutionalNetworks import BGCallWrapper  # noqa: E402
from src.GammaLoss import GammaLoss  # noqa: E402
from src.grid_areas import cell_areas_from_centres  # noqa: E402
from src.mass_conservation import MassConservation  # noqa: E402
from src.quantiles import bin_y_var, ynetwork  # noqa: E402
from src.segmentation import daily_totals, partition_indices  # noqa: E402

MIDTERM_DIR = os.path.join(WORKSPACE_ROOT, "results", "final_cordex_qre")
# lambda = 0 vs the NZ v2 QRE (and the member study's) test MSEs: exact on a CPU; on a GPU the two evaluations
# can pick other cuDNN kernels, so allow float noise (raised from 1e-5 after the first lab run)
LAMBDA0_REL_TOL = 1e-3


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
            f"λ selected on validation only, against the fine-tuned λ = 0 control ({comp['selection']['reason']}). "
            "Daily-total errors are area-weighted over the land cells.\n")
    return head + "\n".join(lines) + note


def max_rel_diff(mine, theirs):
    """Largest relative difference of the regime MSEs (inf if the day counts differ)."""
    return max(abs(mine[k]["mse"] - theirs[k]["mse"]) / max(abs(theirs[k]["mse"]), 1e-12)
               if mine[k]["days"] == theirs[k]["days"] else float("inf") for k in theirs)


# -----------------------------------------------------------------------------
# Member-study models (RUNBOOK steps 1-3) for the placement study and the repetitions
# -----------------------------------------------------------------------------
def member_set(cfg):
    """(variant, member-study results or None): study.member_set, or the set step 2 selected."""
    study = cfg["study"]
    path = study["member_study_results"]
    path = path if os.path.isabs(path) else os.path.join(WORKSPACE_ROOT, path)
    results = None
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
    if study.get("member_set", "auto") != "auto":
        return study["member_set"], results
    best = (results or {}).get("selection", {}).get("best_by_perfect_member_val_mse_0_1")
    if best not in ("scratch", "warm_start"):
        raise FileNotFoundError(f"no member-study selection in {path}; run RUNBOOK step 2 or set study.member_set")
    return best, results


def base_models(c, k, variant):
    """
    Repetition k of the member-study recipe (same names and signatures as run_cordex_v2 --sanity and
    member_study.py, so k = 0 loads their checkpoints): 10-epoch BGNet (the warm start), squared-EMD Weight
    Network, converged BGNet (the baseline) and the members of `variant`.
    """
    cfg, ms = c.cfg, c.cfg["member_study"]
    seed = cfg["base_seed"] + 1000 * k
    Xtr, Ytr, Xva, Yva = c.X["train"], c.Y["train"], c.X["val"], c.Y["val"]
    D, aux = c.D, c.data["aux"]

    def tl(tcfg, name, sig, build, train):
        R.set_seed(seed)
        return R.train_or_load(tcfg, f"rep{k}/{name}", sig, build, Xtr, train)[0]

    bg_sig = checkpoints.signature(cfg, "bgnet", seed)
    bgnet10 = tl(cfg, "bgnet", bg_sig, lambda: build_bgnet(cfg, D, aux),
                 lambda m: R.fit(m, c.nll, Xtr, Ytr, Xva, Yva, cfg["members"]))
    omega = tl(cfg, f"weight_network_{cfg['weight_network']['loss']}",
               checkpoints.signature(cfg, "weight_network", seed, {"thresholds": c.thr}), lambda: build_omega(cfg, c.N),
               lambda m: R.fit(m, R.weight_network_loss(cfg), Xtr, c.L["train"], Xva, c.L["val"], cfg["weight_network"]))
    bcfg = copy.deepcopy(cfg)
    bcfg["members"].update(ms["bgnet_converged"])
    bgnet = tl(bcfg, "bgnet_converged", checkpoints.signature(bcfg, "bgnet", seed), lambda: build_bgnet(bcfg, D, aux),
               lambda m: R.fit(m, c.nll, Xtr, Ytr, Xva, Yva, bcfg["members"]))
    vcfg = copy.deepcopy(cfg)
    vcfg["members"].update(ms[variant])
    init = {"scratch": "random", "warm_start": {"from": f"rep{k}/bgnet", "signature": bg_sig}}[variant]
    members = []
    R.set_seed(seed)
    for i in range(c.N):
        Xs, Ys = Xtr[c.parts[i]], Ytr[c.parts[i]]
        Xv, Yv = Xva[c.val_parts[i]], Yva[c.val_parts[i]]

        def build():
            m = build_bgnet(vcfg, D, aux)
            if variant == "warm_start":
                m(Xtr[:1], training=False)
                m.set_weights(bgnet10.get_weights())
            return m

        sig = checkpoints.signature(vcfg, f"member_{i}", seed,
                                    {"interval": list(c.q_ranges[i]), "thresholds": c.thr, "init": init})
        members.append(R.train_or_load(vcfg, f"rep{k}/members_{variant}/member_{i}", sig, build, Xs,
                                       lambda m: R.fit(m, c.nll, Xs, Ys, Xv, Yv, vcfg["members"]))[0])
    # members trained with the mass term (placements "members" / "both") use the same recipe
    members_cfg = dict(vcfg["members"], init=variant, warm_start={})
    return {"members": members, "omega": omega, "bgnet": bgnet, "warm_bgnet": bgnet10,
            "members_cfg": members_cfg, "seed": seed, "warm_start_signature": bg_sig}


def rep_base(c, variant):
    ck = c.cfg["checkpoints"]["dir"]

    def models(k):
        b = base_models(c, k, variant)
        ctx = S.view(c, seed=b["seed"], members_cfg=b["members_cfg"])
        ctx.warm_start_signature = b["warm_start_signature"]  # mass-trained members start from the 10-epoch BGNet
        return {"ctx": ctx, "base": b,
                "mass_dir": lambda p: os.path.join(ck, f"rep{k}", "mass", p)}
    return models


def stage_placements(cfg, data, seg):
    c = NZContext(cfg, data, seg)
    variant, ms = member_set(cfg)
    print(f"== NZ placement study on the member-study models (members: {variant}) ==", flush=True)
    out = os.path.join(cfg["save_dir"], "mass_study", "placements")
    comp = S.placement_study(c, rep_base(c, variant)(0), cfg["study"], out)
    # repetition 0 must be the member study's models: compare with its test MSEs
    check, bad = {"tolerance": LAMBDA0_REL_TOL}, []
    ref = (ms or {}).get("selection", {}).get("test", {})
    for mine, theirs in (("qre", f"qre_{variant}_squared_emd"), ("bgnet", "bgnet_converged")):
        if theirs in ref:
            mine_mse = comp["test"][mine]["mse"]
            rel = max(abs(mine_mse[r] - ref[theirs][r]["mse"]) / max(abs(ref[theirs][r]["mse"]), 1e-12) for r in mine_mse)
            check[mine] = {"member_study_key": theirs, "max_relative_difference": rel}
            if rel > LAMBDA0_REL_TOL:
                bad.append(f"{mine} vs member study {theirs}: {rel:.3g}")
        else:
            check[mine] = {"member_study_key": theirs, "max_relative_difference": None,
                           "note": "not in the member-study results (run RUNBOOK steps 2-3)"}
    comp["member_study_check"] = check
    R.save_json(comp, os.path.join(out, "comparison.json"))
    print(f"  member-study check: {check}", flush=True)
    if bad:
        raise AssertionError("repetition 0 does not reproduce the member study's test MSEs "
                             f"(relative tolerance {LAMBDA0_REL_TOL:g}): " + "; ".join(bad))
    return comp


def stage_repeats(cfg, data, seg):
    c = NZContext(cfg, data, seg)
    variant, _ = member_set(cfg)
    placement, lam = S.read_selection(os.path.join(cfg["save_dir"], "mass_study", "placements"))
    print(f"== NZ repetitions: {placement}, lambda {lam:g}, members {variant} ==", flush=True)
    return S.repetitions(c, rep_base(c, variant), cfg["study"], os.path.join(cfg["save_dir"], "mass_study", "repetitions"),
                         placement, lam)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="NZ: QRE vs QRE + mass conservation")
    p.add_argument("--stage", choices=["v2", "placements", "repeats"], default="v2")
    p.add_argument("--config", default="configs/cordex_v2.json")
    p.add_argument("--sanity", action="store_true", help="the member study's --sanity settings")
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY.PATH=JSON")
    return p.parse_args(argv)


def main(argv=None):
    # the table printed at the end contains "λ"; on Windows a piped stdout (tee, tests) is cp1252 and would fail
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    cfg = R.load_config(args)
    if cfg["checkpoints"]["dir"] is None:
        raise ValueError("checkpoints.dir is required")
    data = R.prepare_data(cfg)
    seg = R.derive_segmentation(cfg, data)
    if seg["method"] == "legacy":
        raise ValueError("needs segmentation.method = bayesian_gmm")
    if args.stage == "placements":
        return stage_placements(cfg, data, seg)
    if args.stage == "repeats":
        return stage_repeats(cfg, data, seg)
    return stage_v2(cfg, data, seg)


def stage_v2(cfg, data, seg):
    mcfg = cfg["mass"]
    if mcfg["placement"] not in I.PLACEMENTS or 0.0 not in [float(v) for v in mcfg["lambdas"]]:
        raise ValueError("mass.placement must be members/ensemble/both and mass.lambdas must contain 0")
    out = cfg["save_dir"]
    t0 = time.time()

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
    # fine-tuned lambda = 0 control: the same-training baseline, used for the selection (as India's lambda = 0)
    memc, omc, info_c = build_control(c, base)
    rc_val, _ = evaluate(c, memc, omc, "val")
    R.save_json({"val": rc_val, "training": I.summary(info_c)}, os.path.join(out, "sweep", "control_lambda0_finetuned.json"))
    sel_rows = dict(rows, **{"0": {"lambda_mass": 0.0, "val": rc_val}})
    sel = I.select_lambda(sel_rows, cfg["selection"])
    sel["baseline"] = "lambda = 0 fine-tuned control (same extra Weight Network training as lambda > 0, mass term off)"
    R.save_json(sel, os.path.join(out, "sweep", "selection.json"))
    lam_b = float(sel["selected_lambda"])
    print(f"  selected lambda_mass = {lam_b:g} ({sel['reason']}; baseline: fine-tuned lambda = 0)", flush=True)

    # 3. test, once, after the selection
    mem0, om0, _ = build(c, 0.0, base)
    memb, omb, _ = build(c, lam_b, base)
    r0, g0 = evaluate(c, mem0, om0, "test")
    rc, gc = evaluate(c, memc, omc, "test")
    rb, gb = evaluate(c, memb, omb, "test")
    bg = R.predict(BGCallWrapper(base["bgnet"], mean_mode=c.mode), c.X["test"])
    bg_m = M.evaluate(c.Y["test"], bg, c.train_totals, c.labels["test"], c.N, c.data["areas"], c.mass.eps)
    lambda0_exact, lambda0_rel = None, None
    if I.skip_lambda0_stage(cfg, 0.0):
        mine, theirs = R.regime_mse(c.Y["test"], g0, c.train_totals), log["metrics"]["test"]["qre"]
        lambda0_exact = mine == theirs
        # exact on a CPU; on a GPU cuDNN may pick other kernels between the two evaluations, so allow float noise
        lambda0_rel = max_rel_diff(mine, theirs)
        if lambda0_rel > LAMBDA0_REL_TOL:
            raise AssertionError(f"lambda = 0 does not reproduce the NZ v2 QRE test MSEs "
                                 f"(max relative difference {lambda0_rel:.3g} > {LAMBDA0_REL_TOL:g})")
    comp = {"split": {"train": cfg["data"]["train_years"], "val": cfg["data"]["val_years"],
                      "test": cfg["data"]["test_years"]},
            "selected_lambda": lam_b, "selection": sel, "placement": mcfg["placement"], "mass_mode": mcfg["mode"],
            "lambda0_reproduces_nz_v2_qre_exactly": lambda0_exact,
            "lambda0_max_relative_difference": lambda0_rel,
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
