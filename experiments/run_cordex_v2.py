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
import copy
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
from src.ConvolutionalNetworks import BGNet, BGCallWrapper
from src.GammaLoss import GammaLoss
from src.quantiles import Omega, OmegaLoss, ynetwork, bin_y_var
from src.useful_functions import data_between
from src.segmentation import daily_totals
from run_cordex_experiment import derive_gmm_quantile_ranges

PREDICTOR_NAMES = ["q850", "t850", "u850", "v850"]


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
            cfg[section]["max_epochs"] = cfg["sanity"]["max_epochs"]
        cfg["save_dir"] = os.path.join(cfg["save_dir"], "sanity")
    for k in ("save_dir",):
        if not os.path.isabs(cfg[k]):
            cfg[k] = os.path.normpath(os.path.join(WORKSPACE_ROOT, cfg[k]))
    if not os.path.isabs(cfg["data"]["data_dir"]):
        cfg["data"]["data_dir"] = os.path.normpath(os.path.join(WORKSPACE_ROOT, cfg["data"]["data_dir"]))
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
# Models (legacy architecture until Phase 2)
# -----------------------------------------------------------------------------
def build_member(cfg, D, aux):
    a = cfg["architecture"]
    return BGNet(output_dim=D, coords=None, baseline=False, ch_attn=True, avg_pool=True, max_pool=True,
                 kernel_sizes=[a["kernel"]] * 3, num_channels=a["channels"], num_dense=a["dense"],
                 drop_out_rate=a["dropout"], aux_data=aux, aux_mode="embed",
                 n=cfg["data"]["input_hw"][0], m=cfg["data"]["input_hw"][1])


def build_weight_network(cfg, n):
    a = cfg["architecture"]
    return Omega(output_dim=n, kernels=[a["kernel"]] * 3, channels=a["channels"])


def fit(model, loss, x, y, xv, yv, tcfg):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tcfg["learning_rate"]), loss=loss)
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=tcfg["patience"],
                                            restore_best_weights=True)
    t0 = time.time()
    hist = model.fit(x, y, validation_data=(xv, yv), epochs=tcfg["max_epochs"],
                     batch_size=tcfg["batch_size"], callbacks=[stop], verbose=0).history
    epochs = len(hist["loss"])
    return {"epochs": epochs, "best_epoch": int(np.argmin(hist["val_loss"]) + 1),
            "best_val_loss": float(np.min(hist["val_loss"])), "hit_epoch_cap": epochs == tcfg["max_epochs"],
            "seconds": time.time() - t0, "seconds_per_epoch": (time.time() - t0) / epochs,
            "history": {k: [float(v) for v in vals] for k, vals in hist.items()}}


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
def run_repetition(cfg, data, k):
    seed = cfg["base_seed"] + 1000 * k
    set_seed(seed)
    s = data["split"]
    Xtr, Ytr, Xva, Yva, Xte, Yte = (s["train"]["x"], s["train"]["y"], s["val"]["x"],
                                    s["val"]["y"], s["test"]["x"], s["test"]["y"])
    D, aux, N = data["D"], data["aux"], cfg["n_intervals"]
    train_totals = daily_totals(Ytr)
    loss_fn = GammaLoss(epsilon=cfg["loss"]["epsilon"], y_thrs=cfg["loss"]["rain_threshold"])
    log = {"seed": seed}

    # Segmentation (legacy until Phase 3)
    q_ranges, gmm_diag = derive_gmm_quantile_ranges(Ytr, N, seed)
    log["q_ranges"] = q_ranges

    members, wrappers, log["members"] = [], [], []
    for i, (ql, qh) in enumerate(q_ranges):
        Xs, Ys = data_between(Ytr, ql, Xtr, qh)
        m = build_member(cfg, D, aux)
        info = fit(m, loss_fn, Xs, Ys, Xva, Yva, cfg["members"])
        info.update({"interval": [ql, qh], "train_days": int(len(Ys))})
        log["members"].append(info)
        members.append(m)
        wrappers.append(BGCallWrapper(m))
        print(f"  member {i} [{ql:.3f},{qh:.3f}] n={len(Ys)} best_epoch={info['best_epoch']}/{info['epochs']} "
              f"({info['seconds_per_epoch']:.1f}s/epoch)", flush=True)

    omega = build_weight_network(cfg, N)
    Ltr = np.array(bin_y_var(Ytr, q_ranges), dtype=np.float32)
    Lva = np.array(bin_y_var(Yva, q_ranges), dtype=np.float32)
    log["omega"] = fit(omega, OmegaLoss(num_classes=N), Xtr, Ltr, Xva, Lva, cfg["weight_network"])
    print(f"  weight network best_epoch={log['omega']['best_epoch']}/{log['omega']['epochs']}", flush=True)

    bgnet = build_member(cfg, D, aux)
    log["bgnet"] = fit(bgnet, loss_fn, Xtr, Ytr, Xva, Yva, cfg["members"])

    cnn = {f"CNN_{i}": [w] for i, w in enumerate(wrappers)}
    models = {"qre": ynetwork(cnn, omega), "bgnet": BGCallWrapper(bgnet)}
    metrics = {}
    for split_name, (X, Y) in {"val": (Xva, Yva), "test": (Xte, Yte)}.items():
        metrics[split_name] = {name: regime_mse(Y, predict(fn, X), train_totals) for name, fn in models.items()}
    log["metrics"] = metrics
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
    if args.dry_run:
        return

    logs = []
    for k in range(cfg["repeats"]):
        print(f"=== repetition {k + 1}/{cfg['repeats']} ===", flush=True)
        t_rep = time.time()
        log = run_repetition(cfg, data, k)
        log["seconds"] = time.time() - t_rep
        logs.append(log)
        save_json(log, os.path.join(cfg["save_dir"], f"repetition_{k}.json"))
        print(json.dumps(log["metrics"], indent=1), flush=True)
    save_json({"config": cfg, "total_seconds": time.time() - t0}, os.path.join(cfg["save_dir"], "run_info.json"))


if __name__ == "__main__":
    main()
