"""
mass_study.py
=============
Shared by run_india.py (India) and nz_mass.py (NZ): the studies that follow the
first lab run.

  member study      (India; NZ has experiments/member_study.py) members from
                    scratch and warm-started from BGNet; the set with the lower
                    Perfect-Member validation MSE [0,1] is selected.
  placement study   the mass term in each placement ("members", "ensemble",
                    "both") with a wider lambda list (study.lambdas, multipliers
                    of the initial L_QRE / L_mass ratio as before). Each placement
                    has its own lambda = 0 control: the fine-tuned control (same
                    ensemble stage, mass term off) for "ensemble" / "both", the
                    QRE itself for "members" (no extra stage). Selection is on
                    validation only, by the existing rule, across all
                    (placement, lambda > 0) pairs: among those whose val MSE [0,1]
                    is within selection.mse_tolerance of their own control, the
                    lowest val daily-total MAE; if none qualifies, the lowest
                    MSE relative to its control (flagged). Test is evaluated
                    once, for the selected pair.
  repetitions       study.repetitions independent repetitions (seed
                    base_seed + 1000 k) of BGNet, QRE, the control and QRE + mass
                    at the selected placement and lambda; mean +- std and paired
                    two-sided Wilcoxon signed-rank tests at 5 % (QRE vs BGNet,
                    QRE + mass vs QRE, QRE + mass vs the control). The Wilcoxon
                    test needs at least 6 pairs: with 5 or fewer, even all
                    differences of one sign give p >= 0.0625.

Every model goes through checkpoints (resume-safe); each repetition's test
metrics are saved as soon as they exist, and a rerun skips finished repetitions.
"""
import copy
import json
import os
import time

import numpy as np
from scipy import stats

import run_cordex_v2 as R
import run_india as I

MIN_REPS_FOR_WILCOXON = 6
ALPHA = 0.05
# metrics compared across repetitions; all "lower is better"
REP_METRICS = {"mse_0_1": "MSE [0,1]", "mse_0_0.2": "MSE [0,0.2]", "mse_0.9_1": "MSE [0.9,1]",
               "daily_total_mae": "Daily-total MAE (mm/day)", "abs_total_rel_discrepancy": "Abs. total rel. discrepancy"}
MODELS = {"bgnet": "BGNet", "qre": "QRE", "control": "QRE (λ = 0 control)", "qre_mass": "QRE + mass"}


# -----------------------------------------------------------------------------
# Context views: the same data, another checkpoint folder / seed / member settings
# -----------------------------------------------------------------------------
def view(c, ckdir=None, seed=None, init=None, members_cfg=None, placement=None):
    v = copy.copy(c)
    v.cfg = copy.deepcopy(c.cfg)
    if ckdir is not None:
        v.cfg["checkpoints"]["dir"] = ckdir
    if seed is not None:
        v.seed = seed
    if members_cfg is not None:
        v.cfg["members"] = copy.deepcopy(members_cfg)
    if init is not None:
        v.cfg["members"]["init"] = init
    if placement is not None:
        v.cfg["mass"]["placement"] = placement
    return v


def lambdas(study):
    lams = [float(v) for v in study["lambdas"]]
    if 0.0 not in lams or not any(v > 0 for v in lams):
        raise ValueError("study.lambdas must contain 0 and at least one lambda > 0")
    return lams


# -----------------------------------------------------------------------------
# Models for one placement and lambda (members / omega from `base`)
# -----------------------------------------------------------------------------
def control_model(cp, base):
    """lambda = 0 control of placement cp: the fine-tuned ensemble stage, or the QRE itself for "members"."""
    if cp.cfg["mass"]["placement"] in ("ensemble", "both"):
        omega, members, info = I.get_ensemble_stage(cp, 0.0, 0.0, base["members"], base["omega"])
        return members, omega, {"members": None, "ensemble_stage": info}
    return base["members"], base["omega"], {"members": None, "ensemble_stage": None}


def mass_model(cp, lam, base):
    """QRE + mass for placement cp and lambda > 0."""
    p = cp.cfg["mass"]["placement"]
    members, omega, infos = base["members"], base["omega"], {"members": None, "ensemble_stage": None}
    lam_m = float(lam) if p in ("members", "both") else 0.0
    if lam_m:
        members, infos["members"] = I.get_members(cp, lam_m, base["warm_bgnet"])
    if p in ("ensemble", "both"):
        omega, members, infos["ensemble_stage"] = I.get_ensemble_stage(cp, float(lam), lam_m, members, omega)
    return members, omega, infos


def training_summary(infos):
    return {"members": [I.summary(i) for i in infos["members"]] if infos["members"] else None,
            "ensemble_stage": I.summary(infos["ensemble_stage"])}


# -----------------------------------------------------------------------------
# Placement study (validation only, then test once)
# -----------------------------------------------------------------------------
def select_placement(results, scfg):
    """results[placement]["rows"][lambda_key] = {"lambda_mass", "val"}; "0" is that placement's control."""
    key, tm, tol = scfg["mse_regime"], scfg["total_metric"], scfg["mse_tolerance"]
    cands = []
    for p, res in results.items():
        rows = res["rows"]
        mse0 = rows["0"]["val"]["qre"]["regime_mse"][key]["mse"]
        tot0 = rows["0"]["val"]["qre"]["all_india_totals"][tm]
        for k, r in rows.items():
            if r["lambda_mass"] <= 0:
                continue
            mse = r["val"]["qre"]["regime_mse"][key]["mse"]
            tot = r["val"]["qre"]["all_india_totals"][tm]
            cands.append({"placement": p, "lambda_mass": r["lambda_mass"], "val_mse": mse, "val_mse_control": mse0,
                          "val_mse_ratio": mse / mse0, f"val_{tm}": tot, f"val_{tm}_control": tot0,
                          "within_tolerance": bool(mse <= (1 + tol) * mse0)})
    if not cands:
        raise ValueError("no placement has a lambda > 0")
    ok = [x for x in cands if x["within_tolerance"]]
    if ok:
        best = min(ok, key=lambda x: x[f"val_{tm}"])
        reason = f"lowest val {tm} among (placement, lambda > 0) with val MSE{key} within {tol:.0%} of their control"
    else:
        best = min(cands, key=lambda x: x["val_mse_ratio"])
        reason = f"no (placement, lambda > 0) within {tol:.0%} of its control's MSE; lowest MSE relative to its control"
    return {"placement": best["placement"], "selected_lambda": best["lambda_mass"], "tolerance_met": bool(ok),
            "reason": reason, "rule": scfg, "selected": best, "candidates": cands,
            "val_totals_improved_vs_control": bool(best[f"val_{tm}"] < best[f"val_{tm}_control"])}


def placement_study(c, rep0, study, out):
    """
    rep0: {"ctx", "base": {"members", "omega", "bgnet", "warm_bgnet"}, "mass_dir": placement -> checkpoint folder}.
    Writes <out>/<placement>/sweep.json, <out>/selection.json, <out>/comparison.json|.md, <out>/plots/.
    """
    t0 = time.time()
    base, results = rep0["base"], {}
    for p in study["placements"]:
        if p not in I.PLACEMENTS:
            raise ValueError(f"study.placements: unknown placement {p!r}")
        cp = view(rep0["ctx"], ckdir=rep0["mass_dir"](p), placement=p)
        rows, path = {}, os.path.join(out, p, "sweep.json")
        print(f"== placement {p} ==", flush=True)
        for lam in lambdas(study):
            t1 = time.time()
            m, o, infos = control_model(cp, base) if lam == 0 else mass_model(cp, lam, base)
            r, _ = I.evaluate_qre(cp, m, o, "val")
            rows[f"{lam:g}"] = {"lambda_mass": lam, "val": r, "seconds": time.time() - t1,
                                "is_control": lam == 0, "training": training_summary(infos)}
            q = r["qre"]
            print(f"  {p} lambda {lam:g}{' (control)' if lam == 0 else ''}: val MSE "
                  f"{I.r3(v['mse'] for v in q['regime_mse'].values())} daily-total MAE "
                  f"{q['all_india_totals']['daily_mae_mm_per_day']:.4f} mm/day, WN collapsed="
                  f"{r['weight_network']['collapsed']}", flush=True)
            R.save_json(rows, path)
        results[p] = {"rows": rows, "selection_within_placement": I.select_lambda(rows, c.cfg["selection"])}
    sel = select_placement(results, c.cfg["selection"])
    sel["controls"] = {"ensemble": "fine-tuned lambda = 0 control (same ensemble stage, mass term off)",
                       "both": "fine-tuned lambda = 0 control (same ensemble stage, mass term off)",
                       "members": "the QRE itself (no extra stage at lambda = 0)"}
    sel["within_placement"] = {p: r["selection_within_placement"] for p, r in results.items()}
    R.save_json(sel, os.path.join(out, "selection.json"))
    print(f"  selected placement {sel['placement']}, lambda {sel['selected_lambda']:g} ({sel['reason']})", flush=True)

    # test, once, for the selected pair
    p, lam = sel["placement"], float(sel["selected_lambda"])
    cp = view(rep0["ctx"], ckdir=rep0["mass_dir"](p), placement=p)
    test, preds = test_models(cp, base, lam)
    comp = {"note": "test split, evaluated once after the validation-only selection", "placement": p,
            "selected_lambda": lam, "selection": sel,
            "test": {k: row(v) for k, v in test.items()},
            "test_weight_network": {k: v["weight_network"] for k, v in test.items() if "weight_network" in v},
            "seconds": time.time() - t0}
    R.save_json(comp, os.path.join(out, "comparison.json"))
    with open(os.path.join(out, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(comparison_md(comp))
    I.plots(cp, os.path.join(out, "plots"), "test",
            {"BGNet": preds["bgnet"], "QRE": preds["qre"], "QRE (λ=0 control)": preds["control"],
             f"QRE+mass ({p}, λ={lam:g})": preds["qre_mass"]})
    return comp


def test_models(cp, base, lam, split="test"):
    """Metrics (and predictions) of BGNet, QRE, the placement's control and QRE + mass on one split."""
    bg, bg_pred = I.bgnet_eval(cp, base["bgnet"], split)
    q, q_pred = I.evaluate_qre(cp, base["members"], base["omega"], split)
    cm, co, _ = control_model(cp, base)
    ctl, ctl_pred = I.evaluate_qre(cp, cm, co, split)
    mm, mo, _ = mass_model(cp, lam, base)
    qm, qm_pred = I.evaluate_qre(cp, mm, mo, split)
    res = {"bgnet": bg, "qre": dict(q["qre"], weight_network=q["weight_network"]),
           "control": dict(ctl["qre"], weight_network=ctl["weight_network"]),
           "qre_mass": dict(qm["qre"], weight_network=qm["weight_network"])}
    res["perfect_member_qre"] = q["perfect_member"]
    return res, {"bgnet": bg_pred, "qre": q_pred, "control": ctl_pred, "qre_mass": qm_pred}


def row(m):
    t = m["all_india_totals"]
    return {"mse": {k: v["mse"] for k, v in m["regime_mse"].items()}, "mae": m["spatial"]["mae"],
            "daily_total_mae": t["daily_mae_mm_per_day"],
            "total_relative_discrepancy": t["total_relative_discrepancy"]}


def comparison_md(comp):
    fmt = lambda v, d=2: "n/a" if v is None else f"{v:.{d}f}"
    lines = ["| Model | MSE [0,1] | MSE [0,0.2] | MSE [0.9,1] | MAE | Daily-total MAE (mm/day) | Total rel. discrepancy |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    names = dict(MODELS, qre_mass=f"QRE + mass ({comp['placement']}, λ = {comp['selected_lambda']:g})",
                 perfect_member_qre="Perfect-Member (QRE members)")
    for k, r in comp["test"].items():
        lines.append(f"| {names[k]} | " + " | ".join(fmt(r["mse"][x]) for x in ("[0.0,1.0]", "[0.0,0.2]", "[0.9,1.0]"))
                     + f" | {fmt(r['mae'])} | {fmt(r['daily_total_mae'], 3)} | {fmt(r['total_relative_discrepancy'], 3)} |")
    sel = comp["selection"]
    return ("\n".join(lines) + f"\n\nPlacement and λ selected on validation only: {sel['reason']} "
            f"(tolerance met: {sel['tolerance_met']}). The control is {sel['controls'][comp['placement']]}.\n")


# -----------------------------------------------------------------------------
# Repetitions
# -----------------------------------------------------------------------------
def rep_metrics(m):
    t = m["all_india_totals"]
    r = m["regime_mse"]
    disc = t["total_relative_discrepancy"]
    return {"mse_0_1": r["[0.0,1.0]"]["mse"], "mse_0_0.2": r["[0.0,0.2]"]["mse"], "mse_0.9_1": r["[0.9,1.0]"]["mse"],
            "daily_total_mae": t["daily_mae_mm_per_day"], "abs_total_rel_discrepancy": None if disc is None else abs(disc),
            "total_rel_discrepancy": disc}


def wilcoxon(a, b):
    """Paired two-sided Wilcoxon signed-rank test of a vs b (lower is better)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    out = {"n": int(len(d)), "mean_difference": float(d.mean()) if len(d) else None,
           "median_difference": float(np.median(d)) if len(d) else None, "alpha": ALPHA}
    if len(d) < MIN_REPS_FOR_WILCOXON:
        return dict(out, statistic=None, p_value=None, significant=None,
                    note=f"needs at least {MIN_REPS_FOR_WILCOXON} pairs (with fewer, p >= 0.0625 even if all differ)")
    if np.all(d == 0):
        return dict(out, statistic=None, p_value=1.0, significant=False, note="all differences are zero")
    res = stats.wilcoxon(a, b, alternative="two-sided")
    p = float(res.pvalue)
    return dict(out, statistic=float(res.statistic), p_value=p, significant=bool(p < ALPHA),
                lower=("first" if np.median(d) < 0 else "second") if p < ALPHA else None)


def summarize(reps):
    """Mean +- std per model and metric, and the paired tests."""
    models = [m for m in MODELS if all(m in r["test"] for r in reps)]
    summary = {}
    for m in models:
        summary[m] = {}
        for k in REP_METRICS:
            vals = [r["test"][m][k] for r in reps]
            if any(v is None for v in vals):
                summary[m][k] = None
                continue
            v = np.array(vals, dtype=float)
            summary[m][k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else None,
                             "n": int(len(v)), "values": [float(x) for x in v]}
    pairs = [("qre", "bgnet"), ("qre_mass", "qre"), ("qre_mass", "control")]
    tests = {}
    for a, b in pairs:
        if a in models and b in models:
            tests[f"{a}_vs_{b}"] = {k: wilcoxon([r["test"][a][k] for r in reps], [r["test"][b][k] for r in reps])
                                   for k in REP_METRICS if summary[a][k] is not None and summary[b][k] is not None}
    return {"n_repetitions": len(reps), "summary": summary, "wilcoxon": tests,
            "note": (f"paired two-sided Wilcoxon signed-rank tests at {ALPHA:.0%}; "
                     f"needs at least {MIN_REPS_FOR_WILCOXON} repetitions")}


def repetitions_md(res):
    fmt = lambda s, d=2: "n/a" if s is None else (f"{s['mean']:.{d}f} ± {s['std']:.{d}f}" if s["std"] is not None
                                                 else f"{s['mean']:.{d}f}")
    names = dict(MODELS, qre_mass=f"QRE + mass ({res['placement']}, λ = {res['selected_lambda']:g})")
    head = "| Model | " + " | ".join(REP_METRICS.values()) + " |\n| --- |" + " --- |" * len(REP_METRICS) + "\n"
    lines = [f"| {names[m]} | " + " | ".join(fmt(s[k], 3 if "total" in k else 2) for k in REP_METRICS) + " |"
             for m, s in res["stats"]["summary"].items()]
    t_lines = ["\n\n| Test | " + " | ".join(REP_METRICS.values()) + " |", "| --- |" + " --- |" * len(REP_METRICS)]
    for name, t in res["stats"]["wilcoxon"].items():
        cells = []
        for k in REP_METRICS:
            w = t.get(k)
            if w is None or w["p_value"] is None:
                cells.append("n/a")
            else:
                cells.append(f"p = {w['p_value']:.3g}" + (" *" if w["significant"] else "")
                             + f" (Δmed {w['median_difference']:+.3g})")
        a, b = name.split("_vs_")
        t_lines.append(f"| {names[a]} vs {names[b]} | " + " | ".join(cells) + " |")
    return (head + "\n".join(lines) + "\n".join(t_lines)
            + f"\n\nTest split, {res['stats']['n_repetitions']} repetitions (seed base_seed + 1000 k), mean ± std "
            f"(ddof 1). * = significant at {ALPHA:.0%} (paired two-sided Wilcoxon signed-rank; lower is better; "
            f"Δmed = median of first − second). {res['stats']['note']}.\n")


def repetitions(c, rep_base, study, out, placement, lam):
    """rep_base(k) -> {"ctx", "base", "mass_dir"} for repetition k (k = 0: the placement study's models)."""
    n = int(study["repetitions"])
    if n < MIN_REPS_FOR_WILCOXON:
        print(f"! study.repetitions = {n}: the Wilcoxon tests need at least {MIN_REPS_FOR_WILCOXON}; "
              "they are reported as n/a", flush=True)
    reps = []
    for k in range(n):
        path = os.path.join(out, f"rep_{k}.json")
        if os.path.exists(path):
            with open(path) as f:
                done = json.load(f)
            if done.get("placement") == placement and done.get("selected_lambda") == lam:
                reps.append(done)
                print(f"  repetition {k}: already done", flush=True)
                continue
        t0 = time.time()
        rb = rep_base(k)
        cp = view(rb["ctx"], ckdir=rb["mass_dir"](placement), placement=placement)
        test, _ = test_models(cp, rb["base"], lam)
        rec = {"repetition": k, "seed": cp.seed, "placement": placement, "selected_lambda": lam,
               "test": {m: rep_metrics(test[m]) for m in MODELS}, "seconds": time.time() - t0}
        R.save_json(rec, path)
        reps.append(rec)
        print(f"  repetition {k} (seed {cp.seed}): MSE[0,1] " + ", ".join(
            f"{MODELS[m]} {rec['test'][m]['mse_0_1']:.3f}" for m in MODELS) + f" ({rec['seconds']:.0f}s)", flush=True)
    res = {"placement": placement, "selected_lambda": lam, "repetitions": n, "stats": summarize(reps)}
    R.save_json(res, os.path.join(out, "summary.json"))
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(repetitions_md(res))
    print(repetitions_md(res), flush=True)
    return res


def read_selection(out):
    path = os.path.join(out, "selection.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; run the placement study first")
    with open(path) as f:
        sel = json.load(f)
    return sel["placement"], float(sel["selected_lambda"])
