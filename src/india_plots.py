"""Plots for the India experiments (matplotlib, Agg backend; files only)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.india_metrics import area_mean_totals  # noqa: E402


def to_grid(values, mask):
    """(n_valid,) -> (n_lat, n_lon) with NaN outside the mask."""
    g = np.full(mask.shape, np.nan, dtype=np.float32)
    g[mask] = values
    return g


def daily_totals(out_dir, dates, y_true, preds, areas_km2, title="", obs_label="observed", region="domain"):
    """Time series and scatter of area-weighted all-India daily means. preds: {name: (T, D)}."""
    os.makedirs(out_dir, exist_ok=True)
    w = areas_km2 / areas_km2.sum()
    o = area_mean_totals(y_true, w)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(dates, o, color="black", lw=1.2, label=obs_label)
    for name, p in preds.items():
        ax.plot(dates, area_mean_totals(p, w), lw=0.9, label=name)
    ax.set_ylabel(f"{region} mean (mm/day)")
    ax.set_title(f"Area-weighted daily totals {title}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "daily_totals_timeseries.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    hi = max([o.max()] + [area_mean_totals(p, w).max() for p in preds.values()])
    for name, p in preds.items():
        ax.scatter(o, area_mean_totals(p, w), s=6, alpha=0.6, label=name)
    ax.plot([0, hi], [0, hi], color="black", lw=0.8)
    ax.set_xlabel("observed (mm/day)")
    ax.set_ylabel("predicted (mm/day)")
    ax.set_title(f"Daily totals {title}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "daily_totals_scatter.png"), dpi=120)
    plt.close(fig)


def example_maps(out_dir, dates, y_true, preds, mask, lat, lon, n_days=3, obs_label="observed"):
    """Observed and predicted maps on the wettest, 75th-percentile and median days."""
    os.makedirs(out_dir, exist_ok=True)
    tot = y_true.sum(axis=1)
    order = np.argsort(tot)
    days = [order[-1], order[int(0.75 * (len(order) - 1))], order[(len(order) - 1) // 2]][:n_days]
    cols = [obs_label] + list(preds)
    fig, axes = plt.subplots(len(days), len(cols), figsize=(3.2 * len(cols), 3.0 * len(days)), squeeze=False)
    extent = [lon[0], lon[-1], lat[0], lat[-1]]
    for r, t in enumerate(days):
        fields = [y_true[t]] + [p[t] for p in preds.values()]
        vmax = max(float(np.nanmax(f)) for f in fields) or 1.0
        for c, (name, f) in enumerate(zip(cols, fields)):
            im = axes[r, c].imshow(to_grid(f, mask), origin="lower", extent=extent, cmap="Blues", vmin=0, vmax=vmax)
            axes[r, c].set_title(f"{name}\n{str(dates[t])[:10]}", fontsize=8)
            axes[r, c].tick_params(labelsize=6)
        fig.colorbar(im, ax=axes[r, :].tolist(), shrink=0.8, label="mm/day")
    fig.savefig(os.path.join(out_dir, "example_maps.png"), dpi=110)
    plt.close(fig)


def error_maps(out_dir, y_true, preds, mask, lat, lon):
    """Per-cell mean error (bias) and RMSE for each model."""
    os.makedirs(out_dir, exist_ok=True)
    extent = [lon[0], lon[-1], lat[0], lat[-1]]
    fig, axes = plt.subplots(2, len(preds), figsize=(3.4 * len(preds), 6.0), squeeze=False)
    bias = {n: (p - y_true).mean(axis=0) for n, p in preds.items()}
    rmse = {n: np.sqrt(((p - y_true) ** 2).mean(axis=0)) for n, p in preds.items()}
    b = max(float(np.abs(v).max()) for v in bias.values()) or 1.0
    r = max(float(v.max()) for v in rmse.values()) or 1.0
    for c, name in enumerate(preds):
        im0 = axes[0, c].imshow(to_grid(bias[name], mask), origin="lower", extent=extent, cmap="RdBu", vmin=-b, vmax=b)
        axes[0, c].set_title(f"{name}: mean error", fontsize=8)
        im1 = axes[1, c].imshow(to_grid(rmse[name], mask), origin="lower", extent=extent, cmap="magma_r", vmin=0, vmax=r)
        axes[1, c].set_title(f"{name}: RMSE", fontsize=8)
        for ax in axes[:, c]:
            ax.tick_params(labelsize=6)
    fig.colorbar(im0, ax=axes[0, :].tolist(), shrink=0.8, label="mm/day")
    fig.colorbar(im1, ax=axes[1, :].tolist(), shrink=0.8, label="mm/day")
    fig.savefig(os.path.join(out_dir, "error_maps.png"), dpi=110)
    plt.close(fig)
