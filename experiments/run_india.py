"""
run_india.py
============
QRE with global mass-conservation regularization: ERA5 (q, t, u, v, w at
850 hPa) -> BGNet members -> Weight Network -> rainfall on the IMD 0.25 deg grid.

Same QRE code as the NZ fixes-v2 pipeline (run_cordex_v2.py): train-only
normalization, p*alpha*beta prediction, one builder for members and BGNet,
Bayesian-GMM intervals fitted once on training totals, training thresholds for
validation/test labels, members early-stopped on their own validation interval,
squared-EMD Weight Network, checkpoints. The only addition is the mass term
(src/mass_conservation.py), placed by mass.placement:

  "members"   L = NLL + lambda * L_mass in each member's training
  "ensemble"  a joint fine-tuning stage of g(x) = sum_i w_i(x) f_i(x),
              L = L_QRE + lambda * L_mass (Weight Network; members optional)
  "both"      both of the above with the same lambda

Stages (--stage):
  data   load, align, split, standardize, segment; writes reports only
  A      experiment A (lambda = 0): BGNet baseline, members, Weight Network and,
         if the placement has one, the ensemble stage with lambda = 0;
         validation + test metrics, Weight Network diagnostics, plots
  sweep  every lambda in mass.lambdas, VALIDATION only; selects lambda (selection.*)
  B      experiment B (selected lambda): test metrics, plots, comparison with A
  all    A, sweep, B

Studies after the first lab run (experiments/mass_study.py; config section "study"):
  members     member study: members from scratch and warm-started from BGNet,
              selected on Perfect-Member validation MSE [0,1] (as for NZ);
              writes member_study/results.json and selection.json
  placements  mass term in each placement (study.placements) with study.lambdas,
              each against its own lambda = 0 control; (placement, lambda)
              selected on validation only, test once (placements/)
  repeats     study.repetitions repetitions of BGNet, QRE, control and QRE + mass
              at the selected placement and lambda: mean +- std, paired Wilcoxon
              tests (repetitions/)

Every model is checkpointed in checkpoints.dir; rerunning a stage loads what
exists, so an interrupted run continues with the model it was training.

Usage:
  python experiments/run_india.py --config configs/india_v1.json --stage A
  python experiments/run_india.py --config configs/india_smoke.json --stage all
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for p in [WORKSPACE_ROOT, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import run_cordex_v2 as R  # noqa: E402
from src import checkpoints  # noqa: E402
from src import india_metrics as M  # noqa: E402
from src import india_plots as P  # noqa: E402
from src.builders import build_bgnet, build_omega  # noqa: E402
from src.ConvolutionalNetworks import BGCallWrapper  # noqa: E402
from src.GammaLoss import GammaLoss  # noqa: E402
from src.india_data import load_india  # noqa: E402
from src.mass_conservation import (JointEnsemble, MassConservation, MemberLossWithMass, bg_expected,  # noqa: E402
                                   member_metrics, weight_network_loss)
from src.preprocessing import ChannelStandardizer, channel_stats, split_masks_by_dates  # noqa: E402
from src.quantiles import bin_y_var  # noqa: E402
from src.segmentation import daily_totals, partition_indices  # noqa: E402

PREDICTOR_NAMES = ["q850", "t850", "u850", "v850", "w850"]
PLACEMENTS = ("members", "ensemble", "both")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="India QRE with global mass-conservation regularization")
    p.add_argument("--config", default="configs/india_v1.json")
    p.add_argument("--stage", choices=["data", "A", "sweep", "B", "all", "members", "placements", "repeats"],
                   default="all")
    p.add_argument("--save-dir", default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY.PATH=JSON",
                   help="override a config value, e.g. --set mass.placement='\"both\"'")
    return p.parse_args(argv)


def load_config(args):
    path = args.config if os.path.isabs(args.config) else os.path.join(WORKSPACE_ROOT, args.config)
    with open(path) as f:
        cfg = json.load(f)
    for item in args.set:
        key, raw = item.split("=", 1)
        R.set_by_path(cfg, key, json.loads(raw))
    if args.save_dir is not None:
        cfg["save_dir"] = args.save_dir
    for section, key in (("", "save_dir"), ("data", "data_dir"), ("checkpoints", "dir")):
        node = cfg[section] if section else cfg
        if node[key] is not None and not os.path.isabs(node[key]):
            node[key] = os.path.normpath(os.path.join(WORKSPACE_ROOT, node[key]))
    mcfg = cfg["mass"]
    if mcfg["placement"] not in PLACEMENTS:
        raise ValueError(f"mass.placement must be one of {PLACEMENTS}, got {mcfg['placement']!r}")
    if 0.0 not in [float(v) for v in mcfg["lambdas"]]:
        raise ValueError("mass.lambdas must contain 0 (experiment A)")
    if cfg["checkpoints"]["dir"] is None:
        raise ValueError("checkpoints.dir is required (stages reuse each other's models)")
    return cfg


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def prepare_data(cfg):
    dcfg = cfg["data"]
    full = load_india(dcfg)
    if list(full["x"].shape[1:3]) != list(dcfg["input_hw"]):
        raise ValueError(f"input grid {full['x'].shape[1:3]} != data.input_hw {dcfg['input_hw']}")
    if dcfg["use_elevation"]:
        raise ValueError("no elevation source is configured for India; set data.use_elevation=false")
    masks = split_masks_by_dates(full["dates"], cfg["split"])
    split = {n: {"x": full["x"][m], "y": full["y"][m], "dates": full["dates"][m]} for n, m in masks.items()}
    report = dict(full["report"])
    n_all = len(full["dates"])
    report["split"] = {n: {"days": int(m.sum()), "first": str(full["dates"][m][0]), "last": str(full["dates"][m][-1]),
                           "fraction": float(m.sum() / n_all)} for n, m in masks.items()}
    report["days_outside_splits"] = int(n_all - sum(int(m.sum()) for m in masks.values()))
    report["raw_predictor_stats_train"] = channel_stats(split["train"]["x"], PREDICTOR_NAMES)
    if dcfg["standardize_predictors"]:
        scaler = ChannelStandardizer().fit(split["train"]["x"])  # training split only
        for n in split:
            split[n]["x"] = scaler.transform(split[n]["x"])
        report["standardizer"] = scaler.to_dict()
    ytr = split["train"]["y"]
    report["target_train"] = {
        "fraction_exact_zero": float(np.mean(ytr == 0.0)),
        "fraction_in_0_0.1": float(np.mean((ytr > 0.0) & (ytr < 0.1))),
        "rain_threshold_used": cfg["loss"]["rain_threshold"],
        "fraction_wet_at_threshold": float(np.mean(ytr > cfg["loss"]["rain_threshold"])),
        "mean": float(ytr.mean()), "max": float(ytr.max()),
    }
    areas = full["areas_km2"]
    report["cell_areas_km2"] = {"min": float(areas.min()), "max": float(areas.max()), "total_valid": float(areas.sum())}
    mass = MassConservation(areas, mode=cfg["mass"]["mode"], eps=cfg["mass"]["eps_mm_per_day"])
    R.save_json(report, os.path.join(cfg["save_dir"], "data_report.json"))
    return {"split": split, "aux": None, "D": full["y"].shape[1], "mask": full["mask"], "grid_lat": full["imd_lat"],
            "grid_lon": full["imd_lon"], "areas": areas, "mass": mass, "report": report,
            "obs_label": "observed (IMD)", "region": "all-India"}


class Context:
    """Data, segmentation and shared objects of one run."""

    def __init__(self, cfg):
        self.cfg, self.seed, self.N = cfg, cfg["base_seed"], cfg["n_intervals"]
        self.data = prepare_data(cfg)
        self.seg = R.derive_segmentation(cfg, self.data)
        s = self.data["split"]
        self.X = {n: s[n]["x"] for n in s}
        self.Y = {n: s[n]["y"] for n in s}
        self.dates = {n: s[n]["dates"] for n in s}
        self.D, self.mass = self.data["D"], self.data["mass"]
        self.thr, b = self.seg["thresholds_daily_total_mm"], self.seg["bounds"]
        self.q_ranges = [(b[i], b[i + 1]) for i in range(self.N)]
        self.train_totals = daily_totals(self.Y["train"])
        self.parts = partition_indices(self.train_totals, self.thr, self.N)
        self.L = {n: np.array(bin_y_var(self.Y[n], self.q_ranges, thresholds=self.thr), dtype=np.float32) for n in s}
        self.labels = {n: np.argmax(self.L[n], axis=1) for n in s}
        self.val_parts = [np.flatnonzero(self.labels["val"] == i) for i in range(self.N)]
        self.nll = GammaLoss(epsilon=cfg["loss"]["epsilon"], y_thrs=cfg["loss"]["rain_threshold"])
        self.mode = cfg["prediction"]["mode"]
        self.out = cfg["save_dir"]


def summary(info):
    """Training info without the history, plus the logged loss terms at the best epoch."""
    if info is None:
        return None
    out = {k: v for k, v in info.items() if k != "history"}
    hist, best = info.get("history") or {}, info.get("best_epoch")
    if best:
        out["at_best_epoch"] = {k: v[best - 1] for k, v in hist.items() if len(v) >= best}
    return out


def scaled_lambda(cfg, lam, qre0, mass0):
    """lambda_eff = lambda * L_QRE / L_mass at the start of the stage (mass.lambda_scaling = initial_ratio)."""
    if lam == 0.0:
        return 0.0, None
    if cfg["mass"]["lambda_scaling"] == "none" or mass0 <= 1e-12:
        return float(lam), None
    ratio = qre0 / mass0
    return float(lam * ratio), float(ratio)


def mass_settings(cfg):
    return {k: v for k, v in cfg["mass"].items() if k not in ("lambdas", "placement")}


# -----------------------------------------------------------------------------
# Models (each one trained or loaded from checkpoints.dir)
# -----------------------------------------------------------------------------
def bgnet_cfg(cfg):
    b = copy.deepcopy(cfg)
    b["members"] = cfg["bgnet"]
    return b


def get_bgnet(c):
    """Single BGNet baseline, same builder as the members, plain NLL, trained to convergence."""
    bcfg = bgnet_cfg(c.cfg)
    R.set_seed(c.seed)
    model, info = R.train_or_load(
        bcfg, "bgnet", checkpoints.signature(bcfg, "bgnet", c.seed), lambda: build_bgnet(bcfg, c.D, c.data["aux"]), c.X["train"],
        lambda m: R.fit(m, c.nll, c.X["train"], c.Y["train"], c.X["val"], c.Y["val"], bcfg["members"],
                        metrics=member_metrics(c.nll, c.mass)))
    return model, info


def member_cfg(cfg):
    m = copy.deepcopy(cfg)
    if cfg["members"].get("init", "scratch") == "warm_start":
        m["members"].update(cfg["members"]["warm_start"])
    elif cfg["members"].get("init", "scratch") != "scratch":
        raise ValueError(f"members.init must be 'scratch' or 'warm_start', got {cfg['members']['init']!r}")
    return m


def get_members(c, lam_m, bgnet):
    """The N members; with lam_m > 0 each is trained with NLL + lambda_eff * L_mass on its own segment."""
    cfg, mcfg = c.cfg, member_cfg(c.cfg)
    init = mcfg["members"].get("init", "scratch")
    R.set_seed(c.seed)
    models, infos = [], []
    for i in range(c.N):
        Xs, Ys = c.X["train"][c.parts[i]], c.Y["train"][c.parts[i]]
        if len(c.val_parts[i]) == 0:
            raise ValueError(f"no validation days in interval {i}; cannot early-stop member {i}")
        Xv, Yv = c.X["val"][c.val_parts[i]], c.Y["val"][c.val_parts[i]]
        extra = {"interval": list(c.q_ranges[i]), "thresholds": c.thr, "init": init,
                 "lambda_mass": lam_m, "mass": mass_settings(cfg) if lam_m else None}
        if init == "warm_start":
            # the BGNet the members start from; a context may name its own (NZ: the 10-epoch BGNet)
            extra["init_from"] = (getattr(c, "warm_start_signature", None)
                                  or checkpoints.signature(bgnet_cfg(cfg), "bgnet", c.seed))

        def build():
            m = build_bgnet(mcfg, c.D, c.data["aux"])
            if init == "warm_start":
                m(c.X["train"][:1], training=False)
                m.set_weights(bgnet.get_weights())  # same builder, so the weights line up
            return m

        def train(m):
            lam_eff, ratio = 0.0, None
            if lam_m:
                n = cfg["mass"]["scaling_max_days"]
                p0 = tf.constant(R.predict(m, Xs[:n]))
                lam_eff, ratio = scaled_lambda(cfg, lam_m, float(c.nll(Ys[:n], p0)),
                                               float(c.mass(Ys[:n], bg_expected(p0))))
            loss = MemberLossWithMass(c.nll, c.mass, lam_eff) if lam_eff else c.nll
            info = R.fit(m, loss, Xs, Ys, Xv, Yv, mcfg["members"], metrics=member_metrics(c.nll, c.mass))
            info.update(lambda_mass=lam_m, lambda_effective=lam_eff, loss_ratio_nll_over_mass_at_init=ratio)
            return info

        sig = checkpoints.signature(mcfg, f"member_{i}", c.seed, extra)
        m, info = R.train_or_load(mcfg, f"members_lam{lam_m:g}/member_{i}", sig, build, Xs, train)
        info.update(interval=list(c.q_ranges[i]), train_days=int(len(Ys)), val_days=int(len(Yv)))
        models.append(m)
        infos.append(info)
        if "best_epoch" in info:
            print(f"  member {i} (lambda {lam_m:g}) n={len(Ys)} val={len(Yv)} best_epoch={info['best_epoch']}/"
                  f"{info['epochs']} ({info['seconds_per_epoch']:.1f}s/epoch)", flush=True)
    return models, infos


def get_omega(c):
    cfg = c.cfg
    R.set_seed(c.seed)
    sig = checkpoints.signature(cfg, "weight_network", c.seed, {"thresholds": c.thr})
    return R.train_or_load(
        cfg, f"weight_network_{cfg['weight_network']['loss']}", sig, lambda: build_omega(cfg, c.N), c.X["train"],
        lambda m: R.fit(m, weight_network_loss(cfg["weight_network"], c.N), c.X["train"], c.L["train"],
                        c.X["val"], c.L["val"], cfg["weight_network"]))


def member_preds(c, members, X):
    """(T, N, D) member predictions p*alpha*beta."""
    return np.stack([R.predict(BGCallWrapper(m, mean_mode=c.mode), X) for m in members], axis=1)


def get_ensemble_stage(c, lam_e, lam_m, members, omega):
    """
    Joint fine-tuning of g = sum_i w_i f_i with L_QRE + lambda_eff * L_mass, starting from
    the trained Weight Network (and members). Returns (omega, members, info).
    """
    cfg, ecfg = c.cfg, c.cfg["ensemble"]
    train_members = ecfg["train_members"]
    precompute = ecfg["precompute_member_predictions"] and not train_members
    tag = f"ensemble_lam{lam_e:g}" + (f"_members_lam{lam_m:g}" if lam_m else "")
    extra = {"training": ecfg, "weight_network": cfg["weight_network"], "members_training": member_cfg(cfg)["members"],
             "members_lambda": lam_m, "lambda_mass": lam_e, "mass": mass_settings(cfg), "thresholds": c.thr}
    sig = checkpoints.signature(cfg, "ensemble_omega", c.seed, extra)
    tuned = {"members": members}
    mcfg = member_cfg(cfg)

    def copy_member(m, i):
        new = build_bgnet(mcfg, c.D, c.data["aux"])
        new(c.X["train"][:1], training=False)
        new.set_weights(m.get_weights())
        return new

    def build():
        om = build_omega(cfg, c.N)
        om(c.X["train"][:1], training=False)
        om.set_weights(omega.get_weights())
        return om

    def train(om):
        R.set_seed(c.seed)
        mem = None if precompute else ([copy_member(m, i) for i, m in enumerate(members)] if train_members else members)
        joint = JointEnsemble(om, weight_network_loss(cfg["weight_network"], c.N), c.mass, 0.0,
                              members=mem, train_members=train_members, nll=c.nll)
        if precompute:
            x_tr = [c.X["train"], member_preds(c, members, c.X["train"])]
            x_va = [c.X["val"], member_preds(c, members, c.X["val"])]
        else:
            x_tr, x_va = [c.X["train"]], [c.X["val"]]
        lam_eff, ratio = 0.0, None
        if lam_e:
            n = cfg["mass"]["scaling_max_days"]
            _, qre0, mass0 = joint.loss_terms([t[:n] for t in x_tr], tf.constant(c.L["train"][:n]),
                                           tf.constant(c.Y["train"][:n]), training=False)
            lam_eff, ratio = scaled_lambda(cfg, lam_e, float(qre0), float(mass0))
        joint.lam = lam_eff
        info = R.fit(joint, None, x_tr, [c.L["train"], c.Y["train"]], x_va, [c.L["val"], c.Y["val"]], ecfg)
        info.update(lambda_mass=lam_e, lambda_effective=lam_eff, loss_ratio_qre_over_mass_at_init=ratio,
                    members_trained=bool(train_members))
        if train_members:
            for i, m in enumerate(mem):
                checkpoints.save(m, c.cfg["checkpoints"]["dir"], f"{tag}/member_{i}", dict(sig, member=i), {})
            tuned["members"] = mem
        return info

    om, info = R.train_or_load(cfg, f"{tag}/omega", sig, build, c.X["train"], train)
    if train_members and info.get("loaded_from_checkpoint"):
        loaded = []
        for i in range(c.N):
            m = build_bgnet(mcfg, c.D, c.data["aux"])
            checkpoints.load(m, c.cfg["checkpoints"]["dir"], f"{tag}/member_{i}", dict(sig, member=i), c.X["train"][:1])
            loaded.append(m)
        tuned["members"] = loaded
    print(f"  ensemble stage (lambda {lam_e:g}) best_epoch={info.get('best_epoch')}/{info.get('epochs')} "
          f"lambda_eff={info.get('lambda_effective')}", flush=True)
    return om, tuned["members"], info


def build_qre(c, lam, bgnet, omega):
    """QRE for one lambda: members (+ mass if placement has members), then the ensemble stage if any."""
    placement = c.cfg["mass"]["placement"]
    lam_m = float(lam) if placement in ("members", "both") else 0.0
    members, m_infos = get_members(c, lam_m, bgnet)
    final_omega, e_info = omega, None
    if placement in ("ensemble", "both") and not skip_lambda0_stage(c.cfg, lam):
        final_omega, members, e_info = get_ensemble_stage(c, float(lam), lam_m, members, omega)
    return members, final_omega, {"members": m_infos, "ensemble_stage": e_info}


def skip_lambda0_stage(cfg, lam):
    """
    mass.lambda0 = "finetune_control" (default): lambda = 0 also runs the ensemble stage (a fine-tuned
    control, so lambda = 0 and lambda > 0 get the same extra Weight Network training);
    "original": lambda = 0 is the trained QRE as it is, without that stage.
    """
    mode = cfg["mass"].get("lambda0", "finetune_control")
    if mode not in ("finetune_control", "original"):
        raise ValueError(f"mass.lambda0 must be 'finetune_control' or 'original', got {mode!r}")
    return float(lam) == 0.0 and mode == "original"


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def qre_predict(c, members, omega, split):
    F = member_preds(c, members, c.X[split])
    w = R.predict(omega, c.X[split])
    return np.einsum("tn,tnd->td", w, F), w, F


def evaluate_qre(c, members, omega, split):
    g, w, F = qre_predict(c, members, omega, split)
    ev = lambda pred: M.evaluate(c.Y[split], pred, c.train_totals, c.labels[split], c.N, c.data["areas"],
                                 c.cfg["mass"]["eps_mm_per_day"])
    perfect = np.einsum("tn,tnd->td", c.L[split], F)
    res = {"qre": ev(g), "perfect_member": ev(perfect),
           "weight_network": R.weight_network_diagnostics(w, c.labels[split], c.N,
                                                          c.cfg["weight_network"]["collapse_fraction"])}
    return res, g


def loss_scales(c, preds):
    """Both mass-loss forms (and the QRE-stage loss scales) on train / val, to compare their magnitudes."""
    out = {}
    clim = c.Y["train"].mean(axis=0, keepdims=True)
    for split in ("train", "val"):
        y = tf.constant(c.Y[split])
        cands = {"zero": np.zeros_like(c.Y[split]), "train_climatology": np.repeat(clim, len(c.Y[split]), axis=0)}
        cands.update({k: v[split] for k, v in preds.items()})
        out[split] = {name: {"mass_relative": float(c.mass(y, tf.constant(p), mode="relative")),
                             "mass_absolute_mm_per_day": float(c.mass(y, tf.constant(p), mode="absolute"))}
                      for name, p in cands.items()}
    out["eps_mm_per_day"] = c.mass.eps
    return out


def plots(c, out_dir, split, preds):
    obs, region = c.data["obs_label"], c.data["region"]
    P.daily_totals(out_dir, c.dates[split], c.Y[split], preds, c.data["areas"], title=f"({split})", obs_label=obs,
                   region=region)
    P.example_maps(out_dir, c.dates[split], c.Y[split], preds, c.data["mask"], c.data["grid_lat"], c.data["grid_lon"],
                   obs_label=obs)
    P.error_maps(out_dir, c.Y[split], preds, c.data["mask"], c.data["grid_lat"], c.data["grid_lon"])


def bgnet_eval(c, bgnet, split):
    pred = R.predict(BGCallWrapper(bgnet, mean_mode=c.mode), c.X[split])
    return M.evaluate(c.Y[split], pred, c.train_totals, c.labels[split], c.N, c.data["areas"],
                      c.cfg["mass"]["eps_mm_per_day"]), pred


# -----------------------------------------------------------------------------
# Stages
# -----------------------------------------------------------------------------
def stage_A(c):
    t0 = time.time()
    print("== experiment A (lambda_mass = 0) ==", flush=True)
    bgnet, bg_info = get_bgnet(c)
    omega, w_info = get_omega(c)
    members, final_omega, infos = build_qre(c, 0.0, bgnet, omega)
    out = os.path.join(c.out, "A")
    res, preds = {"lambda_mass": 0.0, "placement": c.cfg["mass"]["placement"]}, {"qre_A": {}, "bgnet": {}}
    for split in ("val", "test"):
        r, g = evaluate_qre(c, members, final_omega, split)
        r["bgnet"], preds["bgnet"][split] = bgnet_eval(c, bgnet, split)
        preds["qre_A"][split] = g
        res[split] = r
    print_summary("A", res)
    R.save_json(res, os.path.join(out, "metrics.json"))
    preds_train = {"qre_A": qre_predict(c, members, final_omega, "train")[0],
                   "bgnet": R.predict(BGCallWrapper(bgnet, mean_mode=c.mode), c.X["train"])}
    R.save_json(loss_scales(c, {k: {"train": preds_train[k], "val": preds[k]["val"]} for k in preds}),
                os.path.join(out, "loss_scales.json"))
    R.save_json({"bgnet": bg_info, "weight_network": w_info, **infos, "seconds": time.time() - t0},
                os.path.join(out, "training_log.json"))
    plots(c, os.path.join(out, "plots"), "test", {"QRE (A)": preds["qre_A"]["test"], "BGNet": preds["bgnet"]["test"]})
    return res


def stage_sweep(c):
    print("== lambda sweep (validation only) ==", flush=True)
    bgnet, _ = get_bgnet(c)
    omega, _ = get_omega(c)
    out = os.path.join(c.out, "sweep")
    path = os.path.join(out, "sweep.json")
    rows = {}
    for lam in [float(v) for v in c.cfg["mass"]["lambdas"]]:
        t0 = time.time()
        members, final_omega, infos = build_qre(c, lam, bgnet, omega)
        r, _ = evaluate_qre(c, members, final_omega, "val")
        rows[f"{lam:g}"] = {"lambda_mass": lam, "val": r, "seconds": time.time() - t0,
                            "training": {"members": [summary(i) for i in infos["members"]],
                                         "ensemble_stage": summary(infos["ensemble_stage"])}}
        q, t = r["qre"], r["qre"]["all_india_totals"]
        print(f"  lambda {lam:g}: val MSE {r3(v['mse'] for v in q['regime_mse'].values())} "
              f"daily total MAE {t['daily_mae_mm_per_day']:.4f} mm/day, rel {t['daily_mean_relative_error']:.4f}, "
              f"WN acc {r['weight_network']['accuracy']:.3f} collapsed={r['weight_network']['collapsed']}", flush=True)
        R.save_json(rows, path)
        del members
    sel = select_lambda(rows, c.cfg["selection"])
    R.save_json(sel, os.path.join(out, "selection.json"))
    print(f"  selected lambda_mass = {sel['selected_lambda']:g} ({sel['reason']})", flush=True)
    return rows, sel


def select_lambda(rows, scfg):
    """
    Validation only. Among lambda > 0 whose QRE MSE on scfg.mse_regime is within
    (1 + mse_tolerance) of lambda = 0, pick the lowest validation all-India daily-total
    error (scfg.total_metric). If none qualifies, pick the lambda > 0 with the lowest
    MSE and flag it.
    """
    key, tm, tol = scfg["mse_regime"], scfg["total_metric"], scfg["mse_tolerance"]
    mse = {k: r["val"]["qre"]["regime_mse"][key]["mse"] for k, r in rows.items()}
    tot = {k: r["val"]["qre"]["all_india_totals"][tm] for k, r in rows.items()}
    cands = [k for k in rows if rows[k]["lambda_mass"] > 0]
    if not cands:
        raise ValueError("mass.lambdas has no lambda > 0 to select")
    ok = [k for k in cands if mse[k] <= (1 + tol) * mse["0"]]
    if ok:
        best, reason = min(ok, key=lambda k: tot[k]), f"lowest val {tm} with val MSE{key} within {tol:.0%} of lambda=0"
    else:
        best, reason = min(cands, key=lambda k: mse[k]), f"no lambda>0 within {tol:.0%} of lambda=0 MSE; lowest MSE"
    return {"selected_lambda": rows[best]["lambda_mass"], "tolerance_met": bool(ok), "reason": reason,
            "rule": scfg, "val_mse": mse, f"val_{tm}": tot,
            "val_totals_improved_vs_lambda0": bool(tot[best] < tot["0"])}


def stage_B(c):
    sel_path = os.path.join(c.out, "sweep", "selection.json")
    if not os.path.exists(sel_path):
        raise FileNotFoundError(f"{sel_path} missing; run --stage sweep first")
    with open(sel_path) as f:
        lam = float(json.load(f)["selected_lambda"])
    print(f"== experiment B (lambda_mass = {lam:g}) ==", flush=True)
    bgnet, _ = get_bgnet(c)
    omega, _ = get_omega(c)
    mem_a, om_a, _ = build_qre(c, 0.0, bgnet, omega)
    mem_b, om_b, infos = build_qre(c, lam, bgnet, omega)
    out = os.path.join(c.out, "B")
    res = {"lambda_mass": lam, "placement": c.cfg["mass"]["placement"]}
    preds = {}
    for split in ("val", "test"):
        res[split], preds[f"B_{split}"] = evaluate_qre(c, mem_b, om_b, split)
    _, preds["A_test"] = evaluate_qre(c, mem_a, om_a, "test")
    bg_test, preds["bgnet_test"] = bgnet_eval(c, bgnet, "test")
    print_summary("B", res)
    R.save_json(res, os.path.join(out, "metrics.json"))
    R.save_json({"members": infos["members"], "ensemble_stage": infos["ensemble_stage"]},
                os.path.join(out, "training_log.json"))
    plots(c, os.path.join(out, "plots"), "test",
          {"QRE (A)": preds["A_test"], f"QRE+mass (B, lambda={lam:g})": preds["B_test"], "BGNet": preds["bgnet_test"]})
    a_path = os.path.join(c.out, "A", "metrics.json")
    with open(a_path) as f:
        res_a = json.load(f)
    comp = {"note": "test split; lambda chosen on validation only",
            "lambda_B": lam, "A": compact(res_a["test"]["qre"]), "B": compact(res["test"]["qre"]),
            "bgnet": compact(bg_test),
            "weight_network": {"A": res_a["test"]["weight_network"], "B": res["test"]["weight_network"]}}
    R.save_json(comp, os.path.join(c.out, "comparison.json"))
    return res


def compact(m):
    t = m["all_india_totals"]
    return {"mse": {k: v["mse"] for k, v in m["regime_mse"].items()}, "mae": m["spatial"]["mae"],
            "rmse": m["spatial"]["rmse"], "spatial_corr": m["spatial"]["spatial_corr_daily_mean"],
            "total_relative_discrepancy": t["total_relative_discrepancy"],
            "daily_total_mae_mm_per_day": t["daily_mae_mm_per_day"],
            "daily_mean_relative_error": t["daily_mean_relative_error"]}


def r3(values):
    """Rounded MSE list; None where a regime has no days (tiny splits)."""
    return [None if v is None else round(v, 3) for v in values]


def print_summary(name, res):
    for split in ("val", "test"):
        q = compact(res[split]["qre"])
        wn = res[split]["weight_network"]
        print(f"  {name} {split}: QRE MSE {r3(q['mse'].values())} "
              f"daily-total MAE {q['daily_total_mae_mm_per_day']:.4f} mm/day; WN acc {wn['accuracy']:.3f} "
              f"argmax {wn['argmax_histogram']} collapsed={wn['collapsed']}", flush=True)
        if "bgnet" in res[split]:
            b = compact(res[split]["bgnet"])
            print(f"  BGNet {split}: MSE {r3(b['mse'].values())} "
                  f"daily-total MAE {b['daily_total_mae_mm_per_day']:.4f} mm/day", flush=True)


# -----------------------------------------------------------------------------
# Studies after the first lab run: member study, placements, repetitions
# -----------------------------------------------------------------------------
def stage_members(c):
    """Members from scratch and warm-started from BGNet; select on Perfect-Member validation MSE [0,1]."""
    import mass_study as S
    study, ck = c.cfg["study"], c.cfg["checkpoints"]["dir"]
    out = os.path.join(c.out, "member_study")
    print("== India member study ==", flush=True)
    bgnet, _ = get_bgnet(c)
    bg_val = R.predict(BGCallWrapper(bgnet, mean_mode=c.mode), c.X["val"])
    ev = lambda pred: M.evaluate(c.Y["val"], pred, c.train_totals, c.labels["val"], c.N, c.data["areas"],
                                 c.cfg["mass"]["eps_mm_per_day"])
    res = {"bgnet_val": compact(ev(bg_val)), "rule": "lowest Perfect-Member validation MSE [0,1] (as the NZ member study)"}
    for init in study["member_inits"]:
        cm = S.view(c, ckdir=os.path.join(ck, "member_study", init), init=init)
        members, infos = get_members(cm, 0.0, bgnet)
        F = member_preds(cm, members, c.X["val"])
        pm = ev(np.einsum("tn,tnd->td", c.L["val"], F))
        own = []
        for i, idx in enumerate(c.val_parts):
            mem = float(np.mean((F[idx, i] - c.Y["val"][idx]) ** 2))
            bg = float(np.mean((bg_val[idx] - c.Y["val"][idx]) ** 2))
            own.append({"interval": i, "val_days": int(len(idx)), "member_mse": mem, "bgnet_mse": bg, "ratio": mem / bg,
                        "best_epoch": infos[i].get("best_epoch"), "epochs": infos[i].get("epochs")})
        res[init] = {"perfect_member_val": compact(pm), "own_interval_val": own,
                     "training": [summary(i) for i in infos]}
        print(f"  {init}: Perfect-Member val MSE {r3(v['mse'] for v in pm['regime_mse'].values())}, "
              f"member/BGNet own-interval ratio {[round(o['ratio'], 3) for o in own]}, "
              f"best_epoch {[o['best_epoch'] for o in own]}", flush=True)
        R.save_json(res, os.path.join(out, "results.json"))
    pm01 = {i: res[i]["perfect_member_val"]["mse"]["[0.0,1.0]"] for i in study["member_inits"]}
    best = min(pm01, key=pm01.get)
    sel = {"selected_init": best, "rule": res["rule"], "perfect_member_val_mse_0_1": pm01}
    R.save_json(sel, os.path.join(out, "selection.json"))
    print(f"  selected member init: {best}", flush=True)
    return sel


def selected_init(c):
    v = c.cfg["study"].get("member_init", "auto")
    if v != "auto":
        return v
    path = os.path.join(c.out, "member_study", "selection.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; run --stage members first (or set study.member_init)")
    with open(path) as f:
        return json.load(f)["selected_init"]


def rep_base(c, init):
    """
    Models of repetition k (seed base_seed + 1000 k). k = 0: BGNet and Weight Network of experiment A
    (checkpoints.dir), members of the member study, mass models in placements/<placement>; k >= 1:
    everything in repetitions/rep<k>/.
    """
    import mass_study as S
    ck = c.cfg["checkpoints"]["dir"]

    def models(k):
        if k == 0:
            ctx, mem_ctx = S.view(c, init=init), S.view(c, ckdir=os.path.join(ck, "member_study", init), init=init)
            mass_root = os.path.join(ck, "placements")
        else:
            d = os.path.join(ck, "repetitions", f"rep{k}")
            ctx = mem_ctx = S.view(c, ckdir=d, seed=c.seed + 1000 * k, init=init)
            mass_root = os.path.join(d, "placements")
        bgnet, _ = get_bgnet(ctx)
        omega, _ = get_omega(ctx)
        members, _ = get_members(mem_ctx, 0.0, bgnet)
        return {"ctx": ctx, "base": {"members": members, "omega": omega, "bgnet": bgnet, "warm_bgnet": bgnet},
                "mass_dir": lambda p: os.path.join(mass_root, p)}
    return models


def stage_placements(c):
    import mass_study as S
    init = selected_init(c)
    print(f"== placement study (members: {init}) ==", flush=True)
    return S.placement_study(c, rep_base(c, init)(0), c.cfg["study"], os.path.join(c.out, "placements"))


def stage_repeats(c):
    import mass_study as S
    init = selected_init(c)
    placement, lam = S.read_selection(os.path.join(c.out, "placements"))
    print(f"== repetitions: {placement}, lambda {lam:g}, members {init} ==", flush=True)
    return S.repetitions(c, rep_base(c, init), c.cfg["study"], os.path.join(c.out, "repetitions"), placement, lam)


def main(argv=None):
    # the study tables contain "λ"; on Windows a piped stdout (tee, tests) is cp1252 and would fail
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    cfg = load_config(args)
    os.makedirs(cfg["save_dir"], exist_ok=True)
    R.save_json({"config": cfg, "stage": args.stage, "argv": sys.argv}, os.path.join(cfg["save_dir"], f"run_{args.stage}.json"))
    if args.stage in ("members", "placements", "repeats") and "study" not in cfg:
        raise ValueError(f"--stage {args.stage} needs a 'study' section in the config")
    c = Context(cfg)
    if args.stage == "members":
        stage_members(c)
    elif args.stage == "placements":
        stage_placements(c)
    elif args.stage == "repeats":
        stage_repeats(c)
    if args.stage in ("A", "all"):
        stage_A(c)
    if args.stage in ("sweep", "all"):
        stage_sweep(c)
    if args.stage in ("B", "all"):
        stage_B(c)


if __name__ == "__main__":
    main()
