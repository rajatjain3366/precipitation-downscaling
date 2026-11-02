"""
Data-match check: do the Zenodo files reproduce the data-only quantities of the
old run (results/final_cordex_qre, commit d3d5a3e)?

Produced results/baseline_rerun/data_match_check.json (all fields except
"test_target_1981_2000", the CRC32/MD5 record of the v7 test-target file,
which was added to that JSON separately).

It imports the loader and GMM code of the OLD commit, so it needs a worktree of it:
  git worktree add ../qre_d3d5a3e d3d5a3e
  python scripts/data_match_check.py ../qre_d3d5a3e results/final_cordex_qre results/baseline_rerun/data_match_check.json
  git worktree remove ../qre_d3d5a3e
Data: data/CORDEX_NZ (README section 2).
"""
import json
import os
import sys

import numpy as np

worktree, old_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, worktree)
sys.path.insert(0, os.path.join(worktree, "experiments"))

from src.data_adapter_cordex import load_cordex_dataset
from run_cordex_experiment import derive_gmm_quantile_ranges

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "CORDEX_NZ")
d = load_cordex_dataset(DATA, config="config_A", target_mode="land_masked", verbose=True)


def years_of(t):
    t = np.asarray(t)
    return np.array([x.year for x in t]) if t.dtype == object else t.astype("datetime64[Y]").astype(int) + 1970


# Same split as d3d5a3e's prepare_explicit_temporal_splits (train <= 1975, val 1976-1980),
# done here so the check does not need the 1981-2000 test target
yrs = years_of(d["coords"]["train_time"])
s = {"Y_train": d["y_train"][yrs <= 1975], "Y_val": d["y_train"][(yrs >= 1976) & (yrs <= 1980)],
     "D_land": d["y_train"].shape[1]}
report = {"train_days": int(len(s["Y_train"])), "val_days": int(len(s["Y_val"])), "D_land": int(s["D_land"]),
          "x_train_shape": list(d["x_train"].shape), "time_dtype": str(np.asarray(d["coords"]["train_time"]).dtype),
          "first_time": str(d["coords"]["train_time"][0]), "last_time": str(d["coords"]["train_time"][-1]),
          "old_dates": json.load(open(os.path.join(old_dir, "repetition_0", "metadata.json")))["dates"]}

# GMM intervals for the old seeds (deterministic given data + seed + sklearn version)
old_ranges = json.load(open(os.path.join(old_dir, "quantile_ranges.json")))
report["gmm"] = []
for k, seed in enumerate([42, 1042, 2042]):
    old_log = json.load(open(os.path.join(old_dir, f"repetition_{k}", "training_history", "training_log.json")))
    q, diag = derive_gmm_quantile_ranges(s["Y_train"], 6, seed)
    new_b = [q[0][0]] + [hi for _, hi in q]
    old_b = [old_ranges[f"rep_{k}"][0][0]] + [hi for _, hi in old_ranges[f"rep_{k}"]]
    report["gmm"].append({
        "seed": seed,
        "new_bounds": new_b, "old_bounds": old_b,
        "max_abs_bound_diff": float(np.max(np.abs(np.array(new_b) - np.array(old_b)))),
        "new_means": diag["means"], "old_means": old_log["gmm_diagnostics"]["means"],
        "max_rel_mean_diff": float(np.max(np.abs(np.array(diag["means"]) - np.array(old_log["gmm_diagnostics"]["means"]))
                                          / np.array(old_log["gmm_diagnostics"]["means"]))),
        "new_sample_counts": diag["sample_counts"], "old_sample_counts": old_log["gmm_diagnostics"]["sample_counts"],
    })


def fractions(y):
    return {"n_values": int(y.size), "exact_zero": float(np.mean(y == 0.0)),
            "lt_0.1": float(np.mean(y < 0.1)), "lt_0.5": float(np.mean(y < 0.5)),
            "negative": float(np.mean(y < 0.0)),
            "smallest_positive": float(y[y > 0].min()) if np.any(y > 0) else None}


# Rain-threshold question: land-cell target fractions
report["target_fractions_train_1961_1975"] = fractions(s["Y_train"])
if d["coords"]["test_time"] is not None and d["y_test"].shape[0] > 10:
    years2 = years_of(d["coords"]["test_time"])
    y_1961_1988 = np.concatenate([d["y_train"], d["y_test"][years2 <= 1988]], axis=0)
    report["target_fractions_train_1961_1988"] = fractions(y_1961_1988)
    report["n_days_1961_1988"] = int(len(y_1961_1988))
else:
    report["target_fractions_train_1961_1988"] = "needs the 1981-2000 target file"

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
