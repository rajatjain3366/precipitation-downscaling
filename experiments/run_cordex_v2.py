"""
run_cordex_v2.py
================
QRE replication on CORDEX-ML-Bench (NZ), version 2 ("fixes-v2" branch).

Every fix relative to the first run (experiments/run_cordex_experiment.py,
results/final_cordex_qre/) is a config flag in configs/cordex_v2.json, so the
old behaviour stays reachable and ablations are a matter of --set overrides.

Rule: every statistic (predictor normalization, GMM intervals, thresholds,
empirical means) is fitted on the TRAINING split only; the test split is used
for the final evaluation only.

Usage:
  python experiments/run_cordex_v2.py --dry-run
  python experiments/run_cordex_v2.py --sanity                  # 1 repetition, few epochs
  python experiments/run_cordex_v2.py --repeats 3
  python experiments/run_cordex_v2.py --set data.standardize_predictors=false --save-dir results/fixes_v2/ablation_no_norm
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for p in [WORKSPACE_ROOT, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data_adapter_cordex import load_cordex_full_period
from src.preprocessing import (ChannelStandardizer, channel_stats, looks_standardized,
                               split_masks_by_years, time_to_years)
from src.ConvolutionalNetworks import BGCallWrapper
from src.GammaLoss import GammaLoss
from src.quantiles import OmegaLoss, ynetwork, bin_y_var
from src.builders import build_bgnet, build_bgnet_minus, build_omega, make_optimizer
from src import checkpoints
from src.useful_functions import data_between
from src.segmentation import (assign_intervals, bounds_to_thresholds, daily_totals,
                              gmm_quantile_bounds, partition_indices)
from run_cordex_experiment import derive_gmm_quantile_ranges

PREDICTOR_NAMES = ["q850", "t850", "u850", "v850"]
PAPER_BOUNDS_N6 = [0.0, 0.166, 0.392, 0.61, 0.812, 0.923, 1.0]  # Table 1 of the paper


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="QRE on CORDEX-ML-Bench, fixes-v2")
    p.add_argument("--config", default="configs/cordex_v2.json")
    p.add_argument("--dry-run", action="store_true", help="load data, report splits/statistics, no training")
    p.add_argument("--sanity", action="store_true", help="1 repetition with sanity.max_epochs epochs")
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY.PATH=JSON",
                   help="override a config value, e.g. --set members.patience=4")
    return p.parse_args()


def set_by_path(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node:
            raise KeyError(f"unknown config section '{k}' in --set {dotted}")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"unknown config key '{dotted}'")
    node[keys[-1]] = value


def load_config(args):
    path = args.config if os.path.isabs(args.config) else os.path.join(WORKSPACE_ROOT, args.config)
    with open(path) as f:
        cfg = json.load(f)
    for item in args.set:
        key, raw = item.split("=", 1)
        set_by_path(cfg, key, json.loads(raw))
    if args.repeats is not None:
        cfg["repeats"] = args.repeats
    if args.save_dir is not None:
        cfg["save_dir"] = args.save_dir
    if args.sanity:
        cfg["repeats"] = 1
        for section in ("members", "weight_network"):
            cfg[section]["max_epochs"] = cfg["sanity"][f"{section}_max_epochs"]
            cfg[section]["max_epochs_hard"] = cfg["sanity"][f"{section}_max_epochs"]  # no extension in sanity runs
        cfg["save_dir"] = os.path.join(cfg["save_dir"], "sanity")
    for k in ("save_dir",):
        if not os.path.isabs(cfg[k]):
            cfg[k] = os.path.normpath(os.path.join(WORKSPACE_ROOT, cfg[k]))
    if not os.path.isabs(cfg["data"]["data_dir"]):
        cfg["data"]["data_dir"] = os.path.normpath(os.path.join(WORKSPACE_ROOT, cfg["data"]["data_dir"]))
    ck = cfg["checkpoints"]["dir"]
    if ck is not None and not os.path.isabs(ck):
        cfg["checkpoints"]["dir"] = os.path.normpath(os.path.join(WORKSPACE_ROOT, ck))
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


# -----------------------------------------------------------------------------
# Data (Phase 1)
# -----------------------------------------------------------------------------
def prepare_data(cfg):
    dcfg = cfg["data"]
    full = load_cordex_full_period(dcfg["data_dir"], config=dcfg["predictor_config"],
                                   train_gcm=dcfg["gcm"], test_gcm=dcfg["gcm"])
    masks = split_masks_by_years(full["time"], dcfg["train_years"], dcfg["val_years"], dcfg["test_years"])
    x, y, t = full["x"], full["y"], full["time"]

    split = {name: {"x": x[m], "y": y[m], "time": t[m]} for name, m in masks.items()}
    years = time_to_years(t)
    report = {"total_days": int(len(t)), "first_day": str(t[0]), "last_day": str(t[-1])}
    for name, m in masks.items():
        report[f"{name}_days"] = int(m.sum())
        report[f"{name}_years"] = [int(years[m].min()), int(years[m].max())]
        report[f"{name}_fraction"] = float(m.sum() / len(t))

    # H2: are the CORDEX predictors already standardized? (training split only)
    raw_stats = channel_stats(split["train"]["x"], PREDICTOR_NAMES)
    report["raw_predictor_stats_train"] = raw_stats
    report["raw_predictors_already_standardized"] = looks_standardized(raw_stats)
    if dcfg["standardize_predictors"]:
        scaler = ChannelStandardizer().fit(split["train"]["x"])
        for name in split:
            split[name]["x"] = scaler.transform(split[name]["x"])
        report["standardizer"] = scaler.to_dict()
        report["standardized_stats"] = {name: channel_stats(split[name]["x"], PREDICTOR_NAMES) for name in split}

    # L2: how many target values are exactly zero / below the old 0.5 mm threshold?
    ytr = split["train"]["y"]
    report["target_train"] = {
        "fraction_exact_zero": float(np.mean(ytr == 0.0)),
        "fraction_below_0.1": float(np.mean(ytr < 0.1)),
        "fraction_le_0.5": float(np.mean(ytr <= 0.5)),
        "fraction_negative": float(np.mean(ytr < 0.0)),
        "rain_threshold_used": cfg["loss"]["rain_threshold"],
        "fraction_wet_at_threshold": float(np.mean(ytr > cfg["loss"]["rain_threshold"])),
        "mean": float(ytr.mean()), "max": float(ytr.max()),
    }

    aux = full["auxiliary"] if dcfg["use_elevation"] else None
    if aux is not None:
        land = aux[0, :, :, 0][full["mask"]]
        report["elevation_feature"] = {"shape": list(aux.shape), "land_min": float(land.min()),
                                       "land_max": float(land.max()), "land_mean": float(land.mean())}
    return {"split": split, "aux": aux, "coords": full["coords"], "mask": full["mask"],
            "D": y.shape[1], "report": report}


# -----------------------------------------------------------------------------
# Segmentation (Phase 3): computed ONCE on the training split, reused by every
# repetition, applied unchanged to validation and test days
# -----------------------------------------------------------------------------
def derive_segmentation(cfg, data):
    scfg, N = cfg["segmentation"], cfg["n_intervals"]
    if scfg["method"] == "legacy":
        return {"method": "legacy"}  # old behaviour: re-fitted per repetition in run_repetition()

    s = data["split"]
    train_totals = daily_totals(s["train"]["y"])
    if scfg.get("intervals_file"):
        with open(os.path.join(WORKSPACE_ROOT, scfg["intervals_file"])) as f:
            seg = json.load(f)
        if len(seg["bounds"]) != N + 1:
            raise ValueError(f"{scfg['intervals_file']} has {len(seg['bounds']) - 1} intervals, config wants {N}")
    else:
        t0 = time.time()
        bounds, per_run, n_degenerate = gmm_quantile_bounds(
            train_totals, N, n_runs=scfg["n_runs"], base_seed=scfg["seed"],
            weight_concentration_prior=scfg["weight_concentration_prior"])
        seg = {"method": "bayesian_gmm", "n_runs": scfg["n_runs"], "n_degenerate_runs": n_degenerate,
               "weight_concentration_prior": scfg["weight_concentration_prior"],
               "seed": scfg["seed"], "bounds": bounds,
               "per_run_bounds": per_run, "per_run_bound_std": np.std(per_run, axis=0).tolist(),
               "seconds": time.time() - t0}
    seg["method"] = "bayesian_gmm"
    seg["thresholds_daily_total_mm"] = bounds_to_thresholds(train_totals, seg["bounds"])
    seg["paper_bounds"] = PAPER_BOUNDS_N6 if N == 6 else None
    for name in ("train", "val", "test"):
        labels = assign_intervals(daily_totals(s[name]["y"]), seg["thresholds_daily_total_mm"])
        seg[f"{name}_days_per_interval"] = np.bincount(labels, minlength=N).tolist()
    seg["train_fraction_per_interval"] = (np.array(seg["train_days_per_interval"]) / len(train_totals)).tolist()
    assert sum(seg["train_days_per_interval"]) == len(train_totals), "segments must cover every training day once"
    save_json(seg, os.path.join(cfg["save_dir"], "segmentation.json"))
    return seg


# -----------------------------------------------------------------------------
# Training. Models come from src/builders.py: QRE members and the BGNet
# baseline share build_bgnet(), so they always have the same architecture.
# -----------------------------------------------------------------------------
class BestWeightsEarlyStopping(tf.keras.callbacks.Callback):
    """
    Early stopping on val_loss that keeps the best weights across several fit()
    calls. (TF 2.10's EarlyStopping only restores the best weights when it
    actually stops early, not when the epoch cap is reached.)
    """

    def __init__(self, patience):
        super().__init__()
        self.patience, self.best, self.best_epoch, self.best_weights = patience, np.inf, 0, None

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_loss"] < self.best:
            self.best, self.best_epoch, self.best_weights = logs["val_loss"], epoch + 1, self.model.get_weights()
        elif epoch + 1 - self.best_epoch >= self.patience:
            self.model.stop_training = True


def fit(model, loss, x, y, xv, yv, tcfg, metrics=None):
    """
    Adam + ExponentialDecay from the config, early stopping on val_loss, best
    weights restored. M3: if training is still improving when max_epochs is
    reached, it continues in blocks of extend_epochs up to max_epochs_hard.
    `metrics` are only logged (the India runner logs the loss terms separately).
    """
    model.compile(optimizer=make_optimizer(tcfg), loss=loss, metrics=metrics)
    stopper = BestWeightsEarlyStopping(tcfg["patience"])
    t0, hist, cap = time.time(), {}, tcfg["max_epochs"]
    while True:
        done = len(hist.get("loss", []))
        h = model.fit(x, y, validation_data=(xv, yv), initial_epoch=done, epochs=cap,
                      batch_size=tcfg["batch_size"], callbacks=[stopper], verbose=0).history
        for key, vals in h.items():
            hist.setdefault(key, []).extend(float(v) for v in vals)
        epochs = len(hist["loss"])
        still_improving = epochs == cap and stopper.best_epoch > epochs - tcfg["patience"]
        if not still_improving or cap >= tcfg["max_epochs_hard"]:
            break
        cap = min(cap + tcfg["extend_epochs"], tcfg["max_epochs_hard"])
    model.set_weights(stopper.best_weights)
    epochs = len(hist["loss"])
    return {"epochs": epochs, "best_epoch": stopper.best_epoch, "best_val_loss": float(stopper.best),
            "still_improving_at_cap": epochs >= cap and stopper.best_epoch > epochs - tcfg["patience"],
            "final_epoch_cap": cap, "seconds": time.time() - t0, "seconds_per_epoch": (time.time() - t0) / epochs,
            "history": hist}


def train_or_load(cfg, name, sig, build_fn, x_build, train_fn):
    """
    Loads checkpoints.dir/<name> if it exists (its signature must match `sig`),
    otherwise trains with train_fn(model) and saves. checkpoints.dir = null disables both.
    """
    ckdir = cfg["checkpoints"]["dir"]
    model = build_fn()
    if checkpoints.exists(ckdir, name):
        info = checkpoints.load(model, ckdir, name, sig, x_build[:1])
        print(f"  loaded {name} from checkpoint", flush=True)
        return model, dict(info, loaded_from_checkpoint=True)
    info = train_fn(model)
    if ckdir is not None:
        checkpoints.save(model, ckdir, name, sig, info)
    return model, info


def weight_network_loss(cfg):
    w = cfg["weight_network"]
    return OmegaLoss(num_classes=cfg["n_intervals"], class_reduction=w["emd_reduction"], kind=w["loss"],
                     ce_weight=w["ce_weight"], emd_weight=w["emd_weight"])


def weight_network_diagnostics(weights, labels, n, collapse_fraction):
    """Validation accuracy, confusion matrix (rows = true interval), argmax histogram, mean max weight."""
    pred = np.argmax(weights, axis=1)
    conf = np.zeros((n, n), dtype=int)
    np.add.at(conf, (labels, pred), 1)
    hist = np.bincount(pred, minlength=n)
    top_share = float(hist.max() / len(pred))
    return {"accuracy": float(np.mean(pred == labels)),
            "within_one_interval": float(np.mean(np.abs(pred - labels) <= 1)),
            "confusion_matrix_rows_true_cols_pred": conf.tolist(),
            "argmax_histogram": hist.tolist(), "true_histogram": np.bincount(labels, minlength=n).tolist(),
            "mean_max_weight": float(np.mean(weights.max(axis=1))),
            "largest_class_share": top_share,
            "collapsed": top_share > collapse_fraction}


def predict(fn, x, batch=256):
    return np.concatenate([np.asarray(fn(x[i:i + batch])) for i in range(0, len(x), batch)], axis=0)


# -----------------------------------------------------------------------------
# Evaluation (minimal until Phase 5): MSE on [0,1], [0,0.2], [0.9,1] using
# daily-total thresholds fitted on the TRAINING split
# -----------------------------------------------------------------------------
def regime_mse(y_true, y_pred, train_totals, regimes=((0.0, 1.0), (0.0, 0.2), (0.9, 1.0))):
    totals = daily_totals(y_true)
    out = {}
    for lo, hi in regimes:
        t_lo = -np.inf if lo == 0.0 else np.quantile(train_totals, lo)
        t_hi = np.inf if hi == 1.0 else np.quantile(train_totals, hi)
        m = (totals >= t_lo) & (totals < t_hi)
        out[f"[{lo:.1f},{hi:.1f}]"] = {"mse": float(np.mean((y_true[m] - y_pred[m]) ** 2)), "days": int(m.sum())}
    return out


# -----------------------------------------------------------------------------
# One repetition
# -----------------------------------------------------------------------------
def run_repetition(cfg, data, seg, k, return_models=False):
    seed = cfg["base_seed"] + 1000 * k
    set_seed(seed)
    s = data["split"]
    Xtr, Ytr, Xva, Yva, Xte, Yte = (s["train"]["x"], s["train"]["y"], s["val"]["x"],
                                    s["val"]["y"], s["test"]["x"], s["test"]["y"])
    D, aux, N = data["D"], data["aux"], cfg["n_intervals"]
    train_totals = daily_totals(Ytr)
    loss_fn = GammaLoss(epsilon=cfg["loss"]["epsilon"], y_thrs=cfg["loss"]["rain_threshold"])
    log = {"seed": seed}

    if seg["method"] == "legacy":
        # old behaviour: GMM per repetition, inclusive/rounded bounds, labels recomputed per split
        q_ranges, _ = derive_gmm_quantile_ranges(Ytr, N, seed)
        segments = [data_between(Ytr, ql, Xtr, qh) for ql, qh in q_ranges]
        Ltr = np.array(bin_y_var(Ytr, q_ranges), dtype=np.float32)
        Lva = np.array(bin_y_var(Yva, q_ranges), dtype=np.float32)
        prob_weights = [qh - ql for ql, qh in q_ranges]
    else:
        b, thr = seg["bounds"], seg["thresholds_daily_total_mm"]
        q_ranges = [(b[i], b[i + 1]) for i in range(N)]
        parts = partition_indices(train_totals, thr, N)
        segments = [(Xtr[idx], Ytr[idx]) for idx in parts]
        Ltr = np.array(bin_y_var(Ytr, q_ranges, thresholds=thr), dtype=np.float32)
        Lva = np.array(bin_y_var(Yva, q_ranges, thresholds=thr), dtype=np.float32)  # M1: training thresholds
        prob_weights = seg["train_fraction_per_interval"]
    log["q_ranges"] = q_ranges
    val_label = np.argmax(Lva, axis=1)
    val_parts = [np.flatnonzero(val_label == i) for i in range(N)]

    mode = cfg["prediction"]["mode"]
    own_val = cfg["members"]["validate_on_own_segment"]
    if own_val and seg["method"] == "legacy":
        raise ValueError("members.validate_on_own_segment needs segmentation.method = bayesian_gmm")
    members, wrappers, log["members"] = [], [], []
    for i, ((ql, qh), (Xs, Ys)) in enumerate(zip(q_ranges, segments)):
        # M2: early-stop each member on validation days of ITS interval (training thresholds)
        if own_val:
            if len(val_parts[i]) == 0:
                raise ValueError(f"no validation days in interval {i}; cannot early-stop member {i}")
            Xv, Yv = Xva[val_parts[i]], Yva[val_parts[i]]
        else:
            Xv, Yv = Xva, Yva
        sig = checkpoints.signature(cfg, f"member_{i}", seed,
                                    {"interval": [ql, qh], "thresholds": seg.get("thresholds_daily_total_mm")})
        m, info = train_or_load(cfg, f"rep{k}/member_{i}", sig, lambda: build_bgnet(cfg, D, aux), Xs,
                                lambda model: fit(model, loss_fn, Xs, Ys, Xv, Yv, cfg["members"]))
        info.update({"interval": [ql, qh], "train_days": int(len(Ys)), "val_days": int(len(Yv))})
        log["members"].append(info)
        members.append(m)
        wrappers.append(BGCallWrapper(m, mean_mode=mode))
        print(f"  member {i} [{ql:.3f},{qh:.3f}] n={len(Ys)} val={len(Yv)} best_epoch={info['best_epoch']}/"
              f"{info['epochs']} ({info['seconds_per_epoch']:.1f}s/epoch)", flush=True)

    wcfg = cfg["weight_network"]
    sig = checkpoints.signature(cfg, "weight_network", seed, {"thresholds": seg.get("thresholds_daily_total_mm")})
    omega, log["omega"] = train_or_load(cfg, f"rep{k}/weight_network_{wcfg['loss']}", sig,
                                        lambda: build_omega(cfg, N), Xtr,
                                        lambda model: fit(model, weight_network_loss(cfg), Xtr, Ltr, Xva, Lva, wcfg))
    print(f"  weight network best_epoch={log['omega']['best_epoch']}/{log['omega']['epochs']} "
          f"({log['omega']['seconds_per_epoch']:.1f}s/epoch)", flush=True)
    # H1 check on the validation set
    diag = weight_network_diagnostics(predict(omega, Xva), val_label, N, wcfg["collapse_fraction"])
    log["omega_diagnostics_val"] = diag
    print(f"  weight network val accuracy={diag['accuracy']:.3f} argmax histogram={diag['argmax_histogram']} "
          f"(true {diag['true_histogram']}) mean max weight={diag['mean_max_weight']:.3f} "
          f"collapsed={diag['collapsed']}", flush=True)

    # H3: the baseline is built by the same function as the members
    sig = checkpoints.signature(cfg, "bgnet", seed)
    bgnet, log["bgnet"] = train_or_load(cfg, f"rep{k}/bgnet", sig, lambda: build_bgnet(cfg, D, aux), Xtr,
                                        lambda model: fit(model, loss_fn, Xtr, Ytr, Xva, Yva, cfg["members"]))
    print(f"  bgnet best_epoch={log['bgnet']['best_epoch']}/{log['bgnet']['epochs']} "
          f"({log['bgnet']['seconds_per_epoch']:.1f}s/epoch)", flush=True)
    # M7: no try/except; if BGNet(-) fails the run fails
    sig = checkpoints.signature(cfg, "bgnet_minus", seed)
    bgnet_minus, log["bgnet_minus"] = train_or_load(
        cfg, f"rep{k}/bgnet_minus", sig, lambda: build_bgnet_minus(cfg, D), Xtr,
        lambda model: fit(model, loss_fn, Xtr, Ytr, Xva, Yva, cfg["members"]))
    print(f"  bgnet(-) best_epoch={log['bgnet_minus']['best_epoch']}/{log['bgnet_minus']['epochs']} "
          f"({log['bgnet_minus']['seconds_per_epoch']:.1f}s/epoch)", flush=True)

    cnn = {f"CNN_{i}": [w] for i, w in enumerate(wrappers)}
    models = {"qre": ynetwork(cnn, omega),
              "bgnet": BGCallWrapper(bgnet, mean_mode=mode),
              "bgnet_minus": BGCallWrapper(bgnet_minus, mean_mode=mode),
              "bagging": ynetwork(cnn, fixed_weights=[1.0 / N] * N),
              "probability": ynetwork(cnn, fixed_weights=prob_weights)}
    metrics = {}
    for split_name, (X, Y) in {"val": (Xva, Yva), "test": (Xte, Yte)}.items():
        metrics[split_name] = {name: regime_mse(Y, predict(fn, X), train_totals) for name, fn in models.items()}
    log["metrics"] = metrics

    # Paper's core assumption: each member beats the single BGNet on its own interval (validation)
    bg_val = predict(models["bgnet"], Xva)
    log["member_vs_bgnet_own_interval_val"] = []
    for i, idx in enumerate(val_parts):
        if len(idx) == 0:
            continue
        mem = float(np.mean((predict(wrappers[i], Xva[idx]) - Yva[idx]) ** 2))
        bg = float(np.mean((bg_val[idx] - Yva[idx]) ** 2))
        log["member_vs_bgnet_own_interval_val"].append(
            {"interval": i, "val_days": int(len(idx)), "member_mse": mem, "bgnet_mse": bg, "member_wins": mem < bg})
    if return_models:  # experiments/nz_mass.py builds the mass-conservation study on these exact models
        return log, {"members": members, "omega": omega, "bgnet": bgnet, "models": models}
    return log


def main():
    args = parse_args()
    cfg = load_config(args)
    os.makedirs(cfg["save_dir"], exist_ok=True)
    t0 = time.time()
    data = prepare_data(cfg)
    save_json(data["report"], os.path.join(cfg["save_dir"], "data_report.json"))
    print(json.dumps({k: v for k, v in data["report"].items() if k.endswith(("_days", "_years", "_fraction"))}, indent=1))
    print("raw predictors already standardized:", data["report"]["raw_predictors_already_standardized"])
    seg = derive_segmentation(cfg, data)
    if seg["method"] != "legacy":
        print("interval bounds (ours): ", np.round(seg["bounds"], 3).tolist())
        print("interval bounds (paper):", seg["paper_bounds"])
        print("train days per interval:", seg["train_days_per_interval"],
              "val:", seg["val_days_per_interval"], "test:", seg["test_days_per_interval"])
    if args.dry_run:
        return

    logs = []
    for k in range(cfg["repeats"]):
        print(f"=== repetition {k + 1}/{cfg['repeats']} ===", flush=True)
        t_rep = time.time()
        log = run_repetition(cfg, data, seg, k)
        log["seconds"] = time.time() - t_rep
        logs.append(log)
        save_json(log, os.path.join(cfg["save_dir"], f"repetition_{k}.json"))
        print(json.dumps(log["metrics"], indent=1), flush=True)
    save_json({"config": cfg, "total_seconds": time.time() - t0}, os.path.join(cfg["save_dir"], "run_info.json"))


if __name__ == "__main__":
    main()
