"""
member_study.py
===============
Member training study (fixes-v2). With the 10-epoch members, even Perfect-Member
(always the true interval's member) has about twice BGNet's validation MSE, so
the members, not the Weight Network, limit QRE. Using the BGNet and the
squared-EMD Weight Network saved by run_cordex_v2.py / weight_network_losses.py,
this script trains two new member sets (member_study.* in the config):

  scratch     from random init, early stopping on the member's own validation
              interval, to convergence (max 100 epochs)
  warm_start  initialised from the saved BGNet weights, fine-tuned on the
              member's own segment (lower LR, same early stopping)

For each set (and the saved 10-epoch members, as the "before" row): each
member's MSE on its own validation interval vs BGNet, Perfect-Member validation
MSE on [0,1], [0,0.2], [0.9,1], and the time taken. Once both sets exist, the
one with the lower Perfect-Member validation MSE [0,1] is paired with the
squared-EMD Weight Network and only then evaluated on test, next to BGNet.

The Weight Network does not depend on the members (it is trained on the inputs
and interval labels only), so it is reused, not retrained.

Uses the same flags as the run that saved the models, e.g.
  python experiments/member_study.py --sanity --set checkpoints.dir='"checkpoints/nz_v2_sanity"' ...
member_study.variants selects which sets to train or load in this call.
"""
import copy
import json
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
from src.GammaLoss import GammaLoss
from src.quantiles import bin_y_var
from src.segmentation import daily_totals, partition_indices
from weight_network_losses import ensemble


def main():
    args = R.parse_args()
    cfg = R.load_config(args)
    ckdir = cfg["checkpoints"]["dir"]
    if ckdir is None:
        raise ValueError("set checkpoints.dir to the directory run_cordex_v2.py saved the models in")
    out_dir = os.path.join(cfg["save_dir"], "member_study")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}

    data = R.prepare_data(cfg)
    seg = R.derive_segmentation(cfg, data)
    if seg["method"] == "legacy":
        raise ValueError("needs segmentation.method = bayesian_gmm")
    s, N, D, aux, k = data["split"], cfg["n_intervals"], data["D"], data["aux"], 0
    seed = cfg["base_seed"]
    Xtr, Ytr, Xva, Yva, Xte, Yte = (s["train"]["x"], s["train"]["y"], s["val"]["x"], s["val"]["y"],
                                    s["test"]["x"], s["test"]["y"])
    thr, b = seg["thresholds_daily_total_mm"], seg["bounds"]
    q_ranges = [(b[i], b[i + 1]) for i in range(N)]
    train_totals = daily_totals(Ytr)
    parts = partition_indices(train_totals, thr, N)
    val_label = np.argmax(np.array(bin_y_var(Yva, q_ranges, thresholds=thr)), axis=1)
    val_parts = [np.flatnonzero(val_label == i) for i in range(N)]
    perfect = np.eye(N, dtype=np.float32)[val_label]
    mode = cfg["prediction"]["mode"]
    loss_fn = GammaLoss(epsilon=cfg["loss"]["epsilon"], y_thrs=cfg["loss"]["rain_threshold"])

    def load(model, name, sig):
        if not checkpoints.exists(ckdir, name):
            raise FileNotFoundError(f"{name} not in {ckdir}; run the scripts that save it with the same flags first")
        checkpoints.load(model, ckdir, name, sig, Xtr[:1])
        return model

    bgnet_sig = checkpoints.signature(cfg, "bgnet", seed)
    bgnet = load(build_bgnet(cfg, D, aux), f"rep{k}/bgnet", bgnet_sig)
    bg_fn = BGCallWrapper(bgnet, mean_mode=mode)
    bg_val = R.predict(bg_fn, Xva)
    results["bgnet_val_mse"] = R.regime_mse(Yva, bg_val, train_totals)

    def evaluate(member_fns, infos):
        """Own-interval validation MSE vs BGNet per member, and Perfect-Member validation MSE."""
        mem_val = np.stack([R.predict(f, Xva) for f in member_fns])
        own = []
        for i, idx in enumerate(val_parts):
            mem = float(np.mean((mem_val[i, idx] - Yva[idx]) ** 2))
            bg = float(np.mean((bg_val[idx] - Yva[idx]) ** 2))
            row = {"interval": i, "val_days": int(len(idx)), "member_mse": mem, "bgnet_mse": bg, "ratio": mem / bg}
            if infos:
                row.update({key: infos[i][key] for key in ("epochs", "best_epoch", "still_improving_at_cap",
                                                           "seconds", "seconds_per_epoch") if key in infos[i]})
            own.append(row)
        return mem_val, {"own_interval_val": own, "perfect_member_val_mse": R.regime_mse(Yva, ensemble(mem_val, perfect),
                                                                                          train_totals)}

    def report(name, res):
        print(f"{name}: member/BGNet own-interval MSE ratio "
              f"{[round(r['ratio'], 3) for r in res['own_interval_val']]}, Perfect-Member val MSE "
              f"{ {k2: round(v['mse'], 2) for k2, v in res['perfect_member_val_mse'].items()} }", flush=True)

    # --- "before": the saved 10-epoch members ---
    before = [load(build_bgnet(cfg, D, aux), f"rep{k}/member_{i}",
                   checkpoints.signature(cfg, f"member_{i}", seed, {"interval": list(q_ranges[i]), "thresholds": thr}))
              for i in range(N)]
    _, results["sanity_10_epoch"] = evaluate([BGCallWrapper(m, mean_mode=mode) for m in before], None)
    report("sanity_10_epoch", results["sanity_10_epoch"])
    del before
    R.save_json(results, out_path)

    # --- new member sets, one training at a time ---
    study = cfg["member_study"]
    member_fns = {}
    for variant in study["variants"]:
        vcfg = copy.deepcopy(cfg)
        vcfg["members"].update(study[variant])
        init = {"scratch": "random", "warm_start": {"from": f"rep{k}/bgnet", "signature": bgnet_sig}}[variant]
        R.set_seed(seed)
        t0, fns, infos = time.time(), [], []
        for i in range(N):
            Xs, Ys = Xtr[parts[i]], Ytr[parts[i]]
            Xv, Yv = Xva[val_parts[i]], Yva[val_parts[i]]

            def build():
                model = build_bgnet(vcfg, D, aux)
                if variant == "warm_start":
                    model(Xtr[:1], training=False)
                    model.set_weights(bgnet.get_weights())  # same builder, so the weights line up
                return model

            sig = checkpoints.signature(vcfg, f"member_{i}", seed,
                                        {"interval": list(q_ranges[i]), "thresholds": thr, "init": init})
            m, info = R.train_or_load(vcfg, f"rep{k}/members_{variant}/member_{i}", sig, build, Xs,
                                      lambda model: R.fit(model, loss_fn, Xs, Ys, Xv, Yv, vcfg["members"]))
            info = {key: v for key, v in info.items() if key != "history"}
            infos.append(info)
            fns.append(BGCallWrapper(m, mean_mode=mode))
            if "best_epoch" in info:
                print(f"  {variant} member {i} n={len(Ys)} val={len(Yv)} best_epoch={info['best_epoch']}/"
                      f"{info['epochs']} ({info['seconds_per_epoch']:.1f}s/epoch)", flush=True)
        _, res = evaluate(fns, infos)
        res["settings"] = study[variant]
        res["init"] = "random" if variant == "scratch" else f"rep{k}/bgnet"
        res["training_seconds"] = float(sum(inf.get("seconds", 0.0) for inf in infos))
        res["wall_seconds_this_call"] = time.time() - t0
        results[variant] = res
        member_fns[variant] = fns
        report(variant, res)
        R.save_json(results, out_path)

    # --- choose on Perfect-Member validation MSE [0,1]; pair with the squared-EMD Weight Network; then test ---
    if not {"scratch", "warm_start"} <= set(member_fns):
        print("selection needs both member sets in member_study.variants; stopping here", flush=True)
        return
    best = min(("scratch", "warm_start"), key=lambda v: results[v]["perfect_member_val_mse"]["[0.0,1.0]"]["mse"])
    wcfg = copy.deepcopy(cfg)
    wcfg["weight_network"]["loss"] = "squared_emd"
    omega = load(build_omega(wcfg, N), f"rep{k}/weight_network_squared_emd",
                 checkpoints.signature(wcfg, "weight_network", seed, {"thresholds": thr}))
    w_val = R.predict(omega, Xva)
    sel = {"best_by_perfect_member_val_mse_0_1": best, "weight_network": "rep0/weight_network_squared_emd (reused)"}
    for v in ("scratch", "warm_start"):
        mem_val = np.stack([R.predict(f, Xva) for f in member_fns[v]])
        sel[f"qre_val_mse_{v}"] = R.regime_mse(Yva, ensemble(mem_val, w_val), train_totals)
    mem_te = np.stack([R.predict(f, Xte) for f in member_fns[best]])
    sel["test"] = {f"qre_{best}_squared_emd": R.regime_mse(Yte, ensemble(mem_te, R.predict(omega, Xte)), train_totals),
                   "perfect_member": R.regime_mse(
                       Yte, ensemble(mem_te, np.array(bin_y_var(Yte, q_ranges, thresholds=thr), dtype=np.float32)),
                       train_totals),
                   "bgnet": R.regime_mse(Yte, R.predict(bg_fn, Xte), train_totals)}
    results["selection"] = sel
    print(f"best member set by Perfect-Member val MSE [0,1]: {best}", flush=True)
    for v in ("scratch", "warm_start"):
        print(f"QRE val MSE ({v}):", {k2: round(x["mse"], 2) for k2, x in sel[f"qre_val_mse_{v}"].items()}, flush=True)
    print("TEST:", {m: {k2: round(x["mse"], 2) for k2, x in r.items()} for m, r in sel["test"].items()}, flush=True)
    R.save_json(results, out_path)


if __name__ == "__main__":
    main()
