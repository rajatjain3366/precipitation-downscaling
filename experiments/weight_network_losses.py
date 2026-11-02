"""
weight_network_losses.py
========================
Weight Network loss study on saved members (fixes-v2, H1 follow-up).

The paper's EMD loss (Eq. 6) collapses the Weight Network onto one interval on
CORDEX NZ. This script, using the members / BGNet saved by run_cordex_v2.py:

  1. computes references that need no training (validation):
       - ceiling:   interval of BGNet's predicted daily total, one-hot weights
       - perfect:   true interval's member (Perfect-Member, an upper bound)
  2. evaluates the paper-EMD Weight Network saved by run_cordex_v2.py, then
     trains the other loss variants one after another, same seed and backbone;
     validation diagnostics + QRE validation MSE
  3. picks the variant with the lowest validation QRE MSE on [0, 1] and only
     then evaluates it (and BGNet) on the test split.

Uses the same flags as run_cordex_v2.py, which must have saved the models
first with the same flags, e.g.
  python experiments/run_cordex_v2.py      --sanity --set checkpoints.dir='"checkpoints/nz_v2_sanity"' ...
  python experiments/weight_network_losses.py --sanity --set checkpoints.dir='"checkpoints/nz_v2_sanity"' ...
"""
import copy
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
sys.path.insert(0, SCRIPT_DIR)

import run_cordex_v2 as R
from src import checkpoints
from src.builders import build_bgnet, build_omega
from src.ConvolutionalNetworks import BGCallWrapper
from src.quantiles import bin_y_var
from src.segmentation import assign_intervals, daily_totals

VARIANTS = {
    "emd_paper": {"loss": "emd"},  # loaded from the checkpoint run_cordex_v2.py trained
    "cross_entropy": {"loss": "cross_entropy"},
    "squared_emd": {"loss": "squared_emd"},
    "ce_plus_emd": {"loss": "ce_plus_emd", "ce_weight": 1.0, "emd_weight": 1.0},
}


def ensemble(member_preds, weights):
    """g(x) = sum_i w_i(x) f_i(x); member_preds (N, T, D), weights (T, N)."""
    return np.einsum("tn,ntd->td", weights, member_preds)


def main():
    args = R.parse_args()
    cfg = R.load_config(args)
    if cfg["checkpoints"]["dir"] is None:
        raise ValueError("set checkpoints.dir to the directory run_cordex_v2.py saved the models in")
    out_dir = os.path.join(cfg["save_dir"], "weight_network_losses")
    os.makedirs(out_dir, exist_ok=True)

    data = R.prepare_data(cfg)
    seg = R.derive_segmentation(cfg, data)
    if seg["method"] == "legacy":
        raise ValueError("needs segmentation.method = bayesian_gmm")
    s, N, D, aux, k = data["split"], cfg["n_intervals"], data["D"], data["aux"], 0
    seed = cfg["base_seed"]
    Xtr, Ytr, Xva, Yva, Xte, Yte = (s["train"]["x"], s["train"]["y"], s["val"]["x"], s["val"]["y"],
                                    s["test"]["x"], s["test"]["y"])
    thr = seg["thresholds_daily_total_mm"]
    b = seg["bounds"]
    q_ranges = [(b[i], b[i + 1]) for i in range(N)]
    train_totals = daily_totals(Ytr)
    Ltr = np.array(bin_y_var(Ytr, q_ranges, thresholds=thr), dtype=np.float32)
    Lva = np.array(bin_y_var(Yva, q_ranges, thresholds=thr), dtype=np.float32)
    val_label = np.argmax(Lva, axis=1)
    mode = cfg["prediction"]["mode"]
    ckdir = cfg["checkpoints"]["dir"]

    # --- saved members and BGNet (no training here) ---
    def load_bgnet(name, sig):
        if not checkpoints.exists(ckdir, name):
            raise FileNotFoundError(f"{name} not in {ckdir}; run run_cordex_v2.py with the same flags first")
        model = build_bgnet(cfg, D, aux)
        checkpoints.load(model, ckdir, name, sig, Xtr[:1])
        return BGCallWrapper(model, mean_mode=mode)

    members = [load_bgnet(f"rep{k}/member_{i}",
                          checkpoints.signature(cfg, f"member_{i}", seed,
                                                {"interval": list(q_ranges[i]), "thresholds": thr}))
               for i in range(N)]
    bgnet = load_bgnet(f"rep{k}/bgnet", checkpoints.signature(cfg, "bgnet", seed))
    mem_val = np.stack([R.predict(m, Xva) for m in members])  # (N, T_val, D)
    bg_val = R.predict(bgnet, Xva)

    results = {"thresholds_daily_total_mm": thr, "val_true_histogram": np.bincount(val_label, minlength=N).tolist()}
    wcfg = cfg["weight_network"]
    diag = lambda w: R.weight_network_diagnostics(w, val_label, N, wcfg["collapse_fraction"])
    val_mse = lambda pred: R.regime_mse(Yva, pred, train_totals)

    # --- references without training ---
    ceiling_label = assign_intervals(daily_totals(bg_val), thr)
    ceiling_w = np.eye(N, dtype=np.float32)[ceiling_label]
    results["reference_bgnet_ceiling"] = {"diagnostics": diag(ceiling_w), "qre_val_mse": val_mse(ensemble(mem_val, ceiling_w))}
    perfect = np.eye(N, dtype=np.float32)[val_label]
    results["reference_perfect_member"] = {"diagnostics": diag(perfect), "qre_val_mse": val_mse(ensemble(mem_val, perfect))}
    results["bgnet_val_mse"] = val_mse(bg_val)
    majority = float(np.max(np.bincount(val_label, minlength=N)) / len(val_label))
    results["chance"] = {"uniform_accuracy": 1.0 / N, "majority_class_accuracy": majority}
    for name in ("reference_bgnet_ceiling", "reference_perfect_member"):
        r = results[name]
        print(f"{name}: acc={r['diagnostics']['accuracy']:.3f} within1={r['diagnostics']['within_one_interval']:.3f} "
              f"QRE val MSE={ {k2: round(v['mse'], 2) for k2, v in r['qre_val_mse'].items()} }", flush=True)
    print("BGNet val MSE:", {k2: round(v["mse"], 2) for k2, v in results["bgnet_val_mse"].items()}, flush=True)
    R.save_json(results, os.path.join(out_dir, "results.json"))

    # --- loss variants, one training at a time, same seed ---
    results["variants"] = {}
    for vname, overrides in VARIANTS.items():
        vcfg = copy.deepcopy(cfg)
        vcfg["weight_network"].update(overrides)
        R.set_seed(seed)
        t0 = time.time()
        sig = checkpoints.signature(vcfg, "weight_network", seed, {"thresholds": thr})
        omega, info = R.train_or_load(
            vcfg, f"rep{k}/weight_network_{overrides['loss']}", sig, lambda: build_omega(vcfg, N), Xtr,
            lambda model: R.fit(model, R.weight_network_loss(vcfg), Xtr, Ltr, Xva, Lva, vcfg["weight_network"]))
        w_val = R.predict(omega, Xva)
        res = {"loss": overrides, "diagnostics": diag(w_val), "qre_val_mse": val_mse(ensemble(mem_val, w_val)),
               "training": {k2: v for k2, v in info.items() if k2 != "history"},
               "val_loss_history": info.get("history", {}).get("val_loss"), "seconds": time.time() - t0}
        results["variants"][vname] = res
        dg = res["diagnostics"]
        print(f"{vname}: acc={dg['accuracy']:.3f} within1={dg['within_one_interval']:.3f} "
              f"argmax={dg['argmax_histogram']} mean_max_w={dg['mean_max_weight']:.3f} collapsed={dg['collapsed']} "
              f"best_epoch={info['best_epoch']}/{info['epochs']} "
              f"QRE val MSE={ {k2: round(v['mse'], 2) for k2, v in res['qre_val_mse'].items()} }", flush=True)
        R.save_json(results, os.path.join(out_dir, "results.json"))

    # --- choose on validation QRE MSE [0,1]; only then look at test ---
    best = min(results["variants"], key=lambda v: results["variants"][v]["qre_val_mse"]["[0.0,1.0]"]["mse"])
    clearly_above_chance = results["variants"][best]["diagnostics"]["accuracy"] > majority + 0.10 \
        and not results["variants"][best]["diagnostics"]["collapsed"]
    results["selection"] = {"best_by_val_qre_mse_0_1": best, "clearly_above_chance": clearly_above_chance,
                            "rule": "accuracy > majority-class accuracy + 0.10 and not collapsed"}
    print(f"best by validation QRE MSE [0,1]: {best} (clearly above chance: {clearly_above_chance})", flush=True)

    if clearly_above_chance:
        vcfg = copy.deepcopy(cfg)
        vcfg["weight_network"].update(VARIANTS[best])
        omega = build_omega(vcfg, N)
        checkpoints.load(omega, ckdir, f"rep{k}/weight_network_{VARIANTS[best]['loss']}",
                         checkpoints.signature(vcfg, "weight_network", seed, {"thresholds": thr}), Xtr[:1])
        mem_te = np.stack([R.predict(m, Xte) for m in members])
        w_te = R.predict(omega, Xte)
        results["test"] = {"qre_" + best: R.regime_mse(Yte, ensemble(mem_te, w_te), train_totals),
                           "bgnet": R.regime_mse(Yte, R.predict(bgnet, Xte), train_totals)}
        print("TEST:", {m: {k2: round(v["mse"], 2) for k2, v in r.items()} for m, r in results["test"].items()})
    R.save_json(results, os.path.join(out_dir, "results.json"))


if __name__ == "__main__":
    main()
