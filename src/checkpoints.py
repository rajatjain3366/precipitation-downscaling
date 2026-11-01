"""
Weight checkpoints for trained models, so later experiments can reuse them.

Each checkpoint is <name>.weights.h5 plus <name>.json holding a "signature":
the config values that determine the trained weights (data, loss, architecture,
training settings, interval thresholds, seed). Loading a checkpoint whose
signature differs from the current one raises instead of silently reusing a
model trained under other settings.
"""
import json
import os


def signature(cfg, role, seed, extra=None):
    """Config subset that determines a model's trained weights."""
    data = {k: v for k, v in cfg["data"].items() if k != "data_dir"}
    sig = {"role": role, "seed": seed, "data": data, "loss": cfg["loss"],
           "architecture": cfg["architecture"], "n_intervals": cfg["n_intervals"]}
    if role.startswith("member") or role.startswith("bgnet"):
        sig["training"] = cfg["members"]
    if role.startswith("weight_network"):
        sig["training"] = cfg["weight_network"]
    if extra:
        sig.update(extra)
    return json.loads(json.dumps(sig))  # normalise tuples/floats as JSON would


def paths(ckpt_dir, name):
    base = os.path.join(ckpt_dir, name)
    return base + ".weights.h5", base + ".json"


def exists(ckpt_dir, name):
    return ckpt_dir is not None and all(os.path.exists(p) for p in paths(ckpt_dir, name))


def save(model, ckpt_dir, name, sig, info):
    w_path, j_path = paths(ckpt_dir, name)
    os.makedirs(os.path.dirname(w_path), exist_ok=True)
    model.save_weights(w_path)
    with open(j_path, "w") as f:
        json.dump({"signature": sig, "training_info": info}, f, indent=2, default=str)


def load(model, ckpt_dir, name, sig, build_input):
    """Builds `model` on build_input, loads the weights, returns the saved training info."""
    w_path, j_path = paths(ckpt_dir, name)
    with open(j_path) as f:
        meta = json.load(f)
    if meta["signature"] != sig:
        diff = sorted(k for k in set(sig) | set(meta["signature"]) if sig.get(k) != meta["signature"].get(k))
        raise ValueError(f"checkpoint {j_path} was trained with different settings (differs in: {diff}); "
                         f"use another checkpoints.dir or delete it")
    model(build_input, training=False)
    model.load_weights(w_path)
    return meta["training_info"]
