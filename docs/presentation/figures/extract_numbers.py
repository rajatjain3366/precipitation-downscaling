"""
Collect every number used in the end-term presentation from committed files.

Writes docs/presentation/figures/numbers.json: {key: {"value", "display", "source", "field"}}.
  value    the raw value as stored in the file
  display  the text shown on the slides (rounded only to the precision the committed
           markdown summaries use: MSE 2 decimals, MAE / discrepancy 3, p 3 significant)
  source   the committed file it comes from
  field    where in that file (JSON path, CSV row, or the literal text for .md files)

Numbers that exist only as text in README.md / docs/STATUS.md are taken as literal strings
and checked to be present in that file, so the deck cannot drift from the docs.

Run from the repository root:  python docs/presentation/figures/extract_numbers.py
"""
import csv
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "numbers.json")
N = {}


def load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def dig(obj, path):
    for p in path:
        obj = obj[p]
    return obj


def put(key, value, display, source, field):
    assert key not in N, key
    N[key] = {"value": value, "display": display, "source": source, "field": field}


def js(key, rel, path, fmt=None):
    """A value from a JSON file."""
    v = dig(load(rel), path)
    if fmt is None:
        d = v if isinstance(v, str) else json.dumps(v)
    else:
        d = fmt(v)
    put(key, v, d, rel, "/".join(str(p) for p in path))
    return v


def md(key, rel, text, display=None):
    """A literal from a committed markdown file (must appear verbatim)."""
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        content = f.read()
    assert text in content, f"{key}: {text!r} not in {rel}"
    put(key, text, display if display is not None else text, rel, f'text: "{text}"')


f2 = lambda v: f"{v:.2f}"
f3 = lambda v: f"{v:.3f}"
f4 = lambda v: f"{v:.4f}"
pct1 = lambda v: f"{100 * v:.1f} %"
p3 = lambda v: f"{v:.3g}"
comma = lambda v: f"{v:,}"

# ---------------------------------------------------------------- paper and NZ setup
md("paper.points", "README.md", "11,491 points")
md("paper.reps", "README.md", "| Repetitions | 40 |", "40")
js("paper.bounds", "results/india/qre_mass_v1/segmentation.json", ["paper_bounds"], lambda v: " / ".join(f"{b:g}" for b in v))
js("nz.land_cells", "results/baseline_rerun/data_match_check.json", ["D_land"], comma)
js("nz.x_shape", "results/baseline_rerun/data_match_check.json", ["x_train_shape"], lambda v: f"{v[1]} × {v[2]} × {v[3]}")
js("nz.target_res", "results/final_cordex_qre/metadata.json", ["target_resolution"], lambda v: v.replace("x", " × "))
DR = "results/fixes_v2/dry_run/data_report.json"
for s in ("train", "val", "test"):
    js(f"nz.v2.{s}_days", DR, [f"{s}_days"], comma)
    js(f"nz.v2.{s}_years", DR, [f"{s}_years"], lambda v: f"{v[0]}–{v[1]}")
    js(f"nz.v2.{s}_fraction", DR, [f"{s}_fraction"], lambda v: f"{100 * v:g} %")
for i, name in enumerate(["q850", "t850", "u850", "v850"]):
    js(f"nz.raw_mean.{name}", DR, ["standardizer", "mean", i], lambda v: f"{v:.4g}")
    js(f"nz.raw_std.{name}", DR, ["standardizer", "std", i], lambda v: f"{v:.4g}")
md("nz.kernel", "README.md", "**3, \"same\" padding**", "3 × 3, same")

# ---------------------------------------------------------------- mid-term run (results/final_cordex_qre)
MT = "results/final_cordex_qre/metadata.json"
for s in ("train", "val", "test"):
    js(f"mt.{s}_dates", MT, [f"actual_{s}_dates"], lambda v: v.replace("-01-01", "").replace("-12-31", "").replace(" to ", "–"))
js("mt.train_years", MT, ["config", "train_years"], str)
js("mt.val_years", MT, ["config", "validation_years"], str)
js("mt.test_years", MT, ["config", "test_years"], str)
js("mt.batch", MT, ["config", "batch_size"], str)
js("mt.lr", MT, ["config", "specialist_learning_rate"], lambda v: f"{v:g}")
js("mt.decay", MT, ["config", "decay_rate"], lambda v: "none" if v is None else str(v))
js("mt.max_epochs_member", MT, ["config", "max_epochs_specialist"], str)
js("mt.max_epochs_omega", MT, ["config", "max_epochs_omega"], str)
js("mt.reps", MT, ["repeats"], str)
CSV = "results/final_cordex_qre/metrics_table.csv"
with open(os.path.join(ROOT, CSV), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        m, r = row["model"], row["quantile_regime"].replace("Q_[", "").replace("]", "").replace("_", ",")
        for stat in ("mean_mse", "std_mse"):
            v = float(row[stat])
            put(f"mt.{m}.{r}.{stat}", v, f2(v), CSV, f"model={m}, quantile_regime={row['quantile_regime']}, {stat}")
TL = "results/final_cordex_qre/training_history/training_log.json"
for k in range(3):
    js(f"mt.omega.rep{k}.best_epoch", TL, ["omega", f"rep_{k}", "best_epoch"], str)
    js(f"mt.omega.rep{k}.epochs", TL, ["omega", f"rep_{k}", "epochs_completed"], str)
    sc = js(f"mt.gmm.rep{k}.sample_counts", TL, ["gmm_diagnostics", f"rep_{k}", "sample_counts"], lambda v: " / ".join(map(str, v)))
    put(f"mt.gmm.rep{k}.sample_sum", sum(sc), comma(sum(sc)), TL, f"sum of gmm_diagnostics/rep_{k}/sample_counts")
    js(f"mt.gmm.rep{k}.intervals", TL, ["gmm_diagnostics", f"rep_{k}", "intervals"],
       lambda v: " / ".join(f"{b[1]:.3f}" for b in v[:-1]))
    js(f"mt.gmm.rep{k}.weights", TL, ["gmm_diagnostics", f"rep_{k}", "weights"], lambda v: " / ".join(f"{w:.3f}" for w in v))
js("mt.train_days", "results/baseline_rerun/data_match_check.json", ["train_days"], comma)
md("mt.problems.normalize", "README.md", "the predictors were not normalized")
md("mt.problems.bgnet", "README.md", "the BGNet baseline had a different architecture from the members")
md("mt.problems.threshold", "README.md", "predictions used a $p > 0.5$ threshold instead of $p\\cdot\\alpha\\beta$",
   "p > 0.5 threshold instead of p·α·β")
md("mt.problems.rain_threshold_old", "README.md", "| `loss.rain_threshold` | `0.1` | `0.5` |", "0.5 (old) → 0.1")

# ---------------------------------------------------------------- v2 sanity run (after fixes, EMD loss)
S = "results/fixes_v2/nz/sanity/repetition_0.json"
js("s.wn.argmax", S, ["omega_diagnostics_val", "argmax_histogram"], lambda v: "[" + ", ".join(map(str, v)) + "]")
js("s.wn.true", S, ["omega_diagnostics_val", "true_histogram"], lambda v: "[" + ", ".join(map(str, v)) + "]")
js("s.wn.accuracy", S, ["omega_diagnostics_val", "accuracy"], f3)
js("s.wn.mean_max", S, ["omega_diagnostics_val", "mean_max_weight"], f3)
js("s.wn.best_epoch", S, ["omega", "best_epoch"], str)
js("s.wn.epochs", S, ["omega", "epochs"], str)
for m in ("qre", "bgnet", "bgnet_minus", "bagging", "probability"):
    for r in ("[0.0,1.0]", "[0.0,0.2]", "[0.9,1.0]"):
        js(f"s.test.{m}.{r}", S, ["metrics", "test", m, r, "mse"], f2)
for i in range(6):
    js(f"s.own_ratio.{i}", S, ["member_vs_bgnet_own_interval_val", i, "member_mse"], f2)
    js(f"s.own_bg.{i}", S, ["member_vs_bgnet_own_interval_val", i, "bgnet_mse"], f2)
js("s.bgnet.still_improving", S, ["bgnet", "still_improving_at_cap"], str)
js("s.bgnet.epochs", S, ["bgnet", "epochs"], str)

# ---------------------------------------------------------------- Weight Network loss study
W = "results/fixes_v2/nz_wn/sanity/weight_network_losses/results.json"
for v in ("emd_paper", "cross_entropy", "emd_squared_cdf", "ce_plus_emd"):
    js(f"wl.{v}.accuracy", W, ["variants", v, "diagnostics", "accuracy"], f3)
    js(f"wl.{v}.within1", W, ["variants", v, "diagnostics", "within_one_interval"], f3)
    js(f"wl.{v}.argmax", W, ["variants", v, "diagnostics", "argmax_histogram"], lambda x: "[" + ", ".join(map(str, x)) + "]")
    js(f"wl.{v}.mean_max", W, ["variants", v, "diagnostics", "mean_max_weight"], f3)
    js(f"wl.{v}.confusion", W, ["variants", v, "diagnostics", "confusion_matrix_rows_true_cols_pred"], json.dumps)
    for r in ("[0.0,1.0]", "[0.0,0.2]", "[0.9,1.0]"):
        js(f"wl.{v}.val.{r}", W, ["variants", v, "qre_val_mse", r, "mse"], f2)
for ref in ("reference_bgnet_ceiling", "reference_perfect_member"):
    js(f"wl.{ref}.accuracy", W, [ref, "diagnostics", "accuracy"], f3)
    js(f"wl.{ref}.val.[0.0,1.0]", W, [ref, "qre_val_mse", "[0.0,1.0]", "mse"], f2)
js("wl.chance.uniform", W, ["chance", "uniform_accuracy"], f3)
js("wl.chance.majority", W, ["chance", "majority_class_accuracy"], f3)
js("wl.bgnet_val.[0.0,1.0]", W, ["bgnet_val_mse", "[0.0,1.0]", "mse"], f2)
js("wl.true_hist", W, ["val_true_histogram"], lambda x: "[" + ", ".join(map(str, x)) + "]")
for r in ("[0.0,1.0]", "[0.0,0.2]", "[0.9,1.0]"):
    js(f"wl.test.qre_ce.{r}", W, ["test", "qre_cross_entropy", r, "mse"], f2)

# ---------------------------------------------------------------- NZ member study (lab run 1)
MS = "results/fixes_v2/nz_wn/sanity/member_study/results.json"
for v in ("warm_start", "scratch", "sanity_10_epoch"):
    js(f"ms.{v}.pm_val", MS, [v, "perfect_member_val_mse", "[0.0,1.0]", "mse"], f2)
    for i in range(6):
        js(f"ms.{v}.ratio.{i}", MS, [v, "own_interval_val", i, "ratio"], f3)
        if v != "sanity_10_epoch":
            js(f"ms.{v}.best_epoch.{i}", MS, [v, "own_interval_val", i, "best_epoch"], str)
js("ms.selected", MS, ["selection", "best_by_perfect_member_val_mse_0_1"])
for m in ("qre_warm_start_squared_emd", "perfect_member", "bgnet", "bgnet_converged"):
    for r in ("[0.0,1.0]", "[0.0,0.2]", "[0.9,1.0]"):
        js(f"ms.test.{m}.{r}", MS, ["selection", "test", m, r, "mse"], f2)
js("ms.bgc.best_epoch", MS, ["bgnet_converged_training", "best_epoch"], str)
js("ms.bgc.epochs", MS, ["bgnet_converged_training", "epochs"], str)
js("ms.bgc_val", MS, ["bgnet_converged_val_mse", "[0.0,1.0]", "mse"], f2)
js("ms.bgnet10_val", MS, ["bgnet_val_mse", "[0.0,1.0]", "mse"], f2)
md("paper.pred_shape", "README.md", "T, Q, U, V, **W** at 850 hPa, (5, 36, 41)", "T, Q, U, V, W at 850 hPa, (5, 36, 41)")
md("paper.target", "README.md", "ERA5 → VCSN observations, 5 km, 11,491 points")
md("paper.split", "README.md", "| Split | 32 / 6 / 8 years |", "32 / 6 / 8 years")
md("nz.v2.split_text", "README.md", "**1961–1988 / 1989–1993 / 1994–2000** (28 / 5 / 7 years)", "28 / 5 / 7 years")
md("mt.collapse_text", "README.md", "100 % of validation days go to interval 3", "100 % of validation days in one interval")
md("in.top_member", "README.md", "the top one the top 7.7 %", "top 7.7 % of days")
md("in.res", "README.md", "4 × 4 block means of the 0.25° grid (1°)", "1°")
md("in.imd_res", "README.md", "IMD gridded rainfall (mm/day), 0.25°", "0.25°")
md("in.era5_two_machines", "README.md", "the 27 years present on both machines were byte-identical", "27 years")
md("nz.zenodo_v4", "README.md", "[Zenodo record 17957264](https://zenodo.org/records/17957264)", "17957264")
md("nz.zenodo_v7", "README.md", "[Zenodo record 20985924](https://zenodo.org/records/20985924)", "20985924")
md("nz.sel.members_cost", "docs/STATUS.md", "they raised val MSE by 6–53 %", "6–53 %")
md("nz.grid16", "README.md", "the CORDEX grid is 16 × 16", "16 × 16")

# ---------------------------------------------------------------- 10-repetition summaries (lab run 2)
REPS = {"nz": "results/fixes_v2/nz_wn/sanity/mass_study/repetitions/summary.json",
        "in": "results/india/qre_mass_v1/repetitions/summary.json"}
METRICS = {"mse_0_1": f2, "mse_0_0.2": f2, "mse_0.9_1": f2, "daily_total_mae": f3, "abs_total_rel_discrepancy": f3}
for ds, rel in REPS.items():
    s = load(rel)
    put(f"{ds}.rep.placement", s["placement"], s["placement"], rel, "placement")
    put(f"{ds}.rep.lambda", s["selected_lambda"], f"{s['selected_lambda']:g}", rel, "selected_lambda")
    put(f"{ds}.rep.n", s["stats"]["n_repetitions"], str(s["stats"]["n_repetitions"]), rel, "stats/n_repetitions")
    for m in ("bgnet", "qre", "control", "qre_mass"):
        for k, fmt in METRICS.items():
            st = s["stats"]["summary"][m][k]
            put(f"{ds}.rep.{m}.{k}.mean", st["mean"], fmt(st["mean"]), rel, f"stats/summary/{m}/{k}/mean")
            put(f"{ds}.rep.{m}.{k}.std", st["std"], fmt(st["std"]), rel, f"stats/summary/{m}/{k}/std")
            put(f"{ds}.rep.{m}.{k}.values", st["values"], json.dumps([round(x, 4) for x in st["values"]]), rel,
                f"stats/summary/{m}/{k}/values")
    for t, tests in s["stats"]["wilcoxon"].items():
        for k, w in tests.items():
            put(f"{ds}.rep.p.{t}.{k}", w["p_value"], p3(w["p_value"]), rel, f"stats/wilcoxon/{t}/{k}/p_value")
            put(f"{ds}.rep.med.{t}.{k}", w["median_difference"], f"{w['median_difference']:+.3g}", rel,
                f"stats/wilcoxon/{t}/{k}/median_difference")
md("rep.signed.in.qre", "docs/STATUS.md", "| QRE | −0.032 ± 0.010 | −0.022 ± 0.015 |", "−0.032 ± 0.010")
md("rep.signed.in.mass", "docs/STATUS.md", "| QRE + mass | −0.003 ± 0.008 | −0.022 ± 0.022 |", "−0.003 ± 0.008")
md("rep.signed.in.bgnet", "docs/STATUS.md", "| BGNet | −0.029 ± 0.017 | +0.007 ± 0.031 |", "−0.029 ± 0.017")
md("rep.signed.nz.qre", "docs/STATUS.md", "| QRE | −0.032 ± 0.010 | −0.022 ± 0.015 |", "−0.022 ± 0.015")
md("rep.signed.nz.mass", "docs/STATUS.md", "| QRE + mass | −0.003 ± 0.008 | −0.022 ± 0.022 |", "−0.022 ± 0.022")
md("rep.signed.nz.bgnet", "docs/STATUS.md", "| BGNet | −0.029 ± 0.017 | +0.007 ± 0.031 |", "+0.007 ± 0.031")
md("rep.in.bias_change", "docs/STATUS.md", "Signed, the bias goes from −3.2 % to −0.3 %.", "−3.2 % → −0.3 %")
md("rep.in.mse_cost", "docs/STATUS.md", "That is +1.7 % and +2.2 %.", "+1.7 % (MSE [0,1]), +2.2 % (MSE [0.9,1])")
# signed per-repetition values for the bias chart (computed directly from the ten rep_<k>.json files)
for ds, d in (("in", "results/india/qre_mass_v1/repetitions"), ("nz", "results/fixes_v2/nz_wn/sanity/mass_study/repetitions")):
    for m in ("bgnet", "qre", "qre_mass"):
        vals = [load(f"{d}/rep_{k}.json")["test"][m]["total_rel_discrepancy"] for k in range(10)]
        put(f"{ds}.signed.{m}.values", vals, json.dumps([round(x, 4) for x in vals]), f"{d}/rep_0.json … rep_9.json",
            f"test/{m}/total_rel_discrepancy")

# ---------------------------------------------------------------- placement studies (validation)
for ds, rel in (("nz", "results/fixes_v2/nz_wn/sanity/mass_study/placements/selection.json"),
                ("in", "results/india/qre_mass_v1/placements/selection.json")):
    s = load(rel)
    js(f"{ds}.sel.placement", rel, ["placement"])
    js(f"{ds}.sel.lambda", rel, ["selected_lambda"], lambda v: f"{v:g}")
    js(f"{ds}.sel.tolerance_met", rel, ["tolerance_met"], str)
    for i, c in enumerate(s["candidates"]):
        tag = f"{ds}.cand.{c['placement']}.{c['lambda_mass']:g}"
        put(f"{tag}.ratio", c["val_mse_ratio"], f4(c["val_mse_ratio"]), rel, f"candidates/{i}/val_mse_ratio")
        put(f"{tag}.val_mse", c["val_mse"], f3(c["val_mse"]), rel, f"candidates/{i}/val_mse")
        put(f"{tag}.control_mse", c["val_mse_control"], f3(c["val_mse_control"]), rel, f"candidates/{i}/val_mse_control")
        put(f"{tag}.mae", c["val_daily_mae_mm_per_day"], f4(c["val_daily_mae_mm_per_day"]), rel,
            f"candidates/{i}/val_daily_mae_mm_per_day")
        put(f"{tag}.control_mae", c["val_daily_mae_mm_per_day_control"], f4(c["val_daily_mae_mm_per_day_control"]), rel,
            f"candidates/{i}/val_daily_mae_mm_per_day_control")
        put(f"{tag}.ok", c["within_tolerance"], str(c["within_tolerance"]), rel, f"candidates/{i}/within_tolerance")
js("nz.check.qre", "results/fixes_v2/nz_wn/sanity/mass_study/placements/comparison.json",
   ["member_study_check", "qre", "max_relative_difference"], lambda v: f"{v:g}")

# ---------------------------------------------------------------- India data
IR = "results/india/qre_mass_v1/data_report.json"
IC = "results/india/qre_mass_v1/data_check.json"
js("in.files.era5", IR, ["era5_files"], str)
js("in.files.imd", IR, ["imd_files"], str)
js("in.imd_grid", IR, ["imd_grid"], lambda v: f"{v[0]} × {v[1]}")
js("in.cells", IR, ["imd_valid_cells"], comma)
js("in.input_grid", IR, ["input_grid"], lambda v: f"{v[0]} × {v[1]}")
js("in.days_per_file", IC, ["era5", "days_per_file"], str)
js("in.n_days", IR, ["n_days"], comma)
js("in.day_offset", IR, ["days", "day_offset"], str)
for s in ("train", "val", "test"):
    js(f"in.{s}_days", IR, ["split", s, "days"], comma)
    js(f"in.{s}_first", IR, ["split", s, "first"], lambda v: v[:4])
    js(f"in.{s}_last", IR, ["split", s, "last"], lambda v: v[:4])
js("in.zero_frac", IC, ["imd", "rain_train_years", "fraction_exact_zero"], pct1)
js("in.mean_rain", IC, ["imd", "rain_train_years", "mean_mm_per_day"], f2)
js("in.p99", IC, ["imd", "rain_train_years", "p99_mm_per_day"], lambda v: f"{v:.1f}")
js("in.p999", IC, ["imd", "rain_train_years", "p99_9_mm_per_day"], lambda v: f"{v:.1f}")
js("in.max_rain", IC, ["imd", "rain_train_years", "max_mm_per_day"], lambda v: f"{v:.1f}")
js("in.area_total", IR, ["cell_areas_km2", "total_valid"], lambda v: f"{v / 1e6:.2f} M")
js("in.area_min", IR, ["cell_areas_km2", "min"], lambda v: f"{v:.0f}")
js("in.area_max", IR, ["cell_areas_km2", "max"], lambda v: f"{v:.0f}")
SEG = "results/india/qre_mass_v1/segmentation.json"
js("in.seg.method", SEG, ["method"])
js("in.seg.train_days", SEG, ["train_days_per_interval"], lambda v: " / ".join(map(str, v)))
js("in.seg.val_days", SEG, ["val_days_per_interval"], lambda v: " / ".join(map(str, v)))
js("in.seg.test_days", SEG, ["test_days_per_interval"], lambda v: " / ".join(map(str, v)))
md("in.gmm.degenerate", "docs/STATUS.md", "47 of 100 runs degenerate")
md("in.gmm.top28", "docs/STATUS.md", "top interval = top 28 % of days", "top 28 % of days")
js("in.offset.recommended", "results/india/qre_mass_v1/day_offset/day_offset.json", ["recommended_day_offset"], str)
js("in.offset.score_diff", "results/india/qre_mass_v1/day_offset/day_offset.json", ["score_difference"], f4)
md("in.params", "README.md", "This also shrinks BGNet from 147 M to 13 M parameters.", "147 M → 13 M")
md("in.mass.R", "README.md", "with $R = 6371$ km", "6371 km")
js("in.mass.eps", "configs/india_v1.json", ["mass", "eps_mm_per_day"], lambda v: f"{v:g}")
js("in.study.lambdas", "configs/india_v1.json", ["study", "lambdas"], lambda v: ", ".join(f"{x:g}" for x in v))
js("in.sel.tolerance", "configs/india_v1.json", ["selection", "mse_tolerance"], lambda v: f"{100 * v:g} %")
js("in.study.reps", "configs/india_v1.json", ["study", "repetitions"], str)
js("in.kernel", "configs/india_v1.json", ["architecture", "kernel"], str)
js("in.padding", "configs/india_v1.json", ["architecture", "padding"])

# ---------------------------------------------------------------- India member study (lab run 2)
IM = "results/india/qre_mass_v1/member_study/results.json"
js("im.bgnet_val", IM, ["bgnet_val", "mse", "[0.0,1.0]"], f2)
for v in ("scratch", "warm_start"):
    js(f"im.{v}.pm_val", IM, [v, "perfect_member_val", "mse", "[0.0,1.0]"], f2)
    for i in range(6):
        js(f"im.{v}.ratio.{i}", IM, [v, "own_interval_val", i, "ratio"], f3)
        js(f"im.{v}.best_epoch.{i}", IM, [v, "own_interval_val", i, "best_epoch"], str)
        js(f"im.{v}.epochs.{i}", IM, [v, "own_interval_val", i, "epochs"], str)
    for i in range(6):
        js(f"im.{v}.train_days.{i}", IM, [v, "training", i, "train_days"], comma)
for i in (2, 4):
    js(f"im.scratch.still_improving.{i}", IM, ["scratch", "training", i, "still_improving_at_cap"], str)
js("im.selected", "results/india/qre_mass_v1/member_study/selection.json", ["selected_init"])
A = "results/india/qre_mass_v1/A/training_log.json"
js("in.A.bgnet.best_epoch", A, ["bgnet", "best_epoch"], str)
js("in.A.bgnet.epochs", A, ["bgnet", "epochs"], str)

# ---------------------------------------------------------------- hyperparameters (configs)
C2, CI = "configs/cordex_v2.json", "configs/india_v1.json"
g = lambda x: f"{x:g}"
js("cfg.nz.member_lr", C2, ["members", "learning_rate"], g)
js("cfg.nz.decay", C2, ["members", "decay_rate"], g)
js("cfg.nz.decay_steps", C2, ["members", "decay_steps"], comma)
js("cfg.nz.patience", C2, ["members", "patience"], str)
js("cfg.nz.batch", C2, ["members", "batch_size"], str)
js("cfg.nz.max_epochs", C2, ["members", "max_epochs"], str)
js("cfg.nz.max_epochs_hard", C2, ["members", "max_epochs_hard"], str)
js("cfg.nz.wn_lr", C2, ["weight_network", "learning_rate"], g)
js("cfg.nz.wn_patience", C2, ["weight_network", "patience"], str)
js("cfg.nz.wn_dense", C2, ["weight_network", "dense"], str)
js("cfg.nz.wn_dropout", C2, ["weight_network", "dropout"], g)
js("cfg.nz.wn_loss", C2, ["weight_network", "loss"])
js("cfg.nz.rain_threshold", C2, ["loss", "rain_threshold"], g)
js("cfg.nz.channels", C2, ["architecture", "channels"], lambda x: " / ".join(map(str, x)))
js("cfg.nz.kernel", C2, ["architecture", "kernel"], str)
js("cfg.nz.n_intervals", C2, ["n_intervals"], str)
js("cfg.nz.warm_lr", C2, ["member_study", "warm_start", "learning_rate"], g)
js("cfg.nz.ens_lr", C2, ["ensemble", "learning_rate"], g)
js("cfg.nz.ens_patience", C2, ["ensemble", "patience"], str)
js("cfg.nz.ens_max_epochs", C2, ["ensemble", "max_epochs"], str)
js("cfg.nz.sanity_member_epochs", "results/fixes_v2/nz_wn/sanity/run_info.json", ["config", "members", "max_epochs"], str)
js("cfg.nz.sanity_wn_epochs", "results/fixes_v2/nz_wn/sanity/run_info.json", ["config", "weight_network", "max_epochs"], str)
js("cfg.in.member_lr", CI, ["members", "learning_rate"], g)
js("cfg.in.warm_lr", CI, ["members", "warm_start", "learning_rate"], g)
js("cfg.in.patience", CI, ["members", "patience"], str)
js("cfg.in.max_epochs", CI, ["members", "max_epochs"], str)
js("cfg.in.bgnet_max_epochs", CI, ["bgnet", "max_epochs"], str)
js("cfg.in.wn_lr", CI, ["weight_network", "learning_rate"], g)
js("cfg.in.wn_patience", CI, ["weight_network", "patience"], str)
js("cfg.in.channels", CI, ["architecture", "channels"], lambda x: " / ".join(map(str, x)))
js("cfg.in.dense", CI, ["architecture", "dense"], lambda x: " / ".join(map(str, x)))
js("cfg.in.batch", CI, ["members", "batch_size"], str)
js("cfg.in.rain_threshold", CI, ["loss", "rain_threshold"], g)
js("cfg.in.ens_lr", CI, ["ensemble", "learning_rate"], g)
js("cfg.in.ens_patience", CI, ["ensemble", "patience"], str)
js("cfg.in.ens_max_epochs", CI, ["ensemble", "max_epochs"], str)
js("cfg.in.mass_mode", CI, ["mass", "mode"])
js("cfg.in.scaling", CI, ["mass", "lambda_scaling"])

# ---------------------------------------------------------------- compute / protocol
md("run.gpu", "results/lab_runs/2026-11-24_rtx4000ada/logs/gpu_check.log", "NVIDIA RTX 4000 Ada Generation", "NVIDIA RTX 4000 Ada")
md("run.tf", "results/lab_runs/2026-11-24_rtx4000ada/logs/gpu_check.log", "TensorFlow 2.10.1", "TensorFlow 2.10.1")
md("run.first", "docs/STATUS.md", "all 10 run_lab steps in about 25 min", "about 25 min (steps 0–9)")
md("run.second", "docs/STATUS.md", "run_lab steps 10–14 in 1 h 31 min", "1 h 31 min (steps 10–14)")
md("run.t13", "docs/STATUS.md", "13: 48 min 02 s", "48 min 02 s")
md("run.t14", "docs/STATUS.md", "14: 22 min 32 s", "22 min 32 s")
md("run.seeds", "docs/STATUS.md", "seed 42 + 1000 k", "42 + 1000 k")
md("run.p_min", "docs/STATUS.md", "p = 0.00195 = 2/1024 is the smallest possible with 10 pairs", "2/1024 = 0.00195")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(N, f, indent=1, ensure_ascii=False)
print(f"{len(N)} numbers -> {os.path.relpath(OUT, ROOT)}")
