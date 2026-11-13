"""
Single source of truth for model construction.

QRE members and the single-BGNet baseline are both built by build_bgnet() from
the same config section, so their architectures cannot drift apart (fixes H3).
"""
import tensorflow as tf

from src.ConvolutionalNetworks import BGNet
from src.quantiles import Omega


def _arch(cfg):
    a = cfg["architecture"]
    return dict(
        num_channels=list(a["channels"]),
        kernel_sizes=[a["kernel"]] * len(a["channels"]),
        conv_padding=a["padding"],
        num_dense=list(a["dense"]),
        dense_activation=a["dense_activation"],
        drop_out_rate=a["dropout"],
        reduction_rate=a["attention_reduction"],
    )


def build_bgnet(cfg, output_dim, aux=None):
    """BGNet with channel attention (avg + max). Used for every QRE member AND the BGNet baseline."""
    a = _arch(cfg)
    use_elev = cfg["data"]["use_elevation"]
    if use_elev and aux is None:
        raise ValueError("use_elevation=true but no elevation array was given")
    in_h, in_w = cfg["data"]["input_hw"]
    return BGNet(
        output_dim=output_dim,
        coords=None,
        baseline=False,
        avg_pool=True,
        max_pool=True,
        aux_data=aux if use_elev else None,
        aux_mode="embed",
        n=in_h,
        m=in_w,
        **a,
    )


def build_bgnet_minus(cfg, output_dim):
    """BGNet(-), the Rampal et al. (2022) style network: BatchNorm/conv/max-pool, no attention, no elevation."""
    a = _arch(cfg)
    # optional: BGNet(-) pools after every conv, so on 16x16 inputs it needs "same" padding even when
    # the members use "valid" (configs/cordex_midterm_on_v2_split.json); absent = architecture.padding
    a["conv_padding"] = cfg["architecture"].get("bgnet_minus_padding", a["conv_padding"])
    return BGNet(output_dim=output_dim, coords=None, baseline=True, aux_data=None, **a)


def build_omega(cfg, n_intervals):
    a = cfg["architecture"]
    return Omega(
        output_dim=n_intervals,
        kernels=[a["kernel"]] * len(a["channels"]),
        channels=list(a["channels"]),
        padding=a["padding"],
        init_conv=a["omega_init_conv"],
        dense_units=cfg["weight_network"]["dense"],
        dropout=cfg["weight_network"]["dropout"],
    )


def make_optimizer(train_cfg):
    """Adam with ExponentialDecay(initial_lr, decay_steps, decay_rate); decay_rate=null -> constant LR."""
    lr = train_cfg["learning_rate"]
    if train_cfg.get("decay_rate") is not None:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=train_cfg["decay_steps"],
            decay_rate=train_cfg["decay_rate"],
        )
    return tf.keras.optimizers.Adam(learning_rate=lr)
