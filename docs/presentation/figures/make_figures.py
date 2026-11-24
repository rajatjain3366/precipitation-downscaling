"""
Charts for the end-term presentation, made only from numbers.json (extract_numbers.py),
which is made only from committed result files.

Run from the repository root:
  python docs/presentation/figures/extract_numbers.py
  python docs/presentation/figures/make_figures.py
Writes docs/presentation/figures/*.png.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "numbers.json"), encoding="utf-8"))
v = lambda k: N[k]["value"]

# one colour per model on every slide (same hex values as build_deck.js)
COL = {"bgnet": "#5B6770", "qre": "#D9822B", "control": "#F0C28A", "qre_mass": "#167D7F",
       "pm": "#7A5195", "grey": "#B8C0C8", "ink": "#1F2A36", "muted": "#6B7785", "bad": "#B23A48"}
LABEL = {"bgnet": "BGNet", "qre": "QRE", "control": "QRE (λ = 0 control)", "qre_mass": "QRE + mass"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": COL["muted"],
    "axes.labelcolor": COL["ink"], "xtick.color": COL["ink"], "ytick.color": COL["ink"],
    "text.color": COL["ink"], "axes.titleweight": "bold", "savefig.dpi": 220,
})
REG = [("[0.0,1.0]", "MSE [0, 1]"), ("[0.0,0.2]", "MSE [0, 0.2]"), ("[0.9,1.0]", "MSE [0.9, 1]")]


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def bar_labels(ax, bars, fmt="{:.2f}", pad=3, size=13, errs=None):
    """Value labels above each bar, or above its error bar when errs is given."""
    for i, b in enumerate(bars):
        top = b.get_height() + (errs[i] if errs is not None else 0)
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, top),
                    xytext=(0, pad), textcoords="offset points", ha="center", va="bottom", fontsize=size)


# ------------------------------------------------------------------ mid-term results
def midterm_results():
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 4.3))
    for ax, (r, title) in zip(axes, REG):
        key = r.strip("[]")
        ms = [v(f"mt.{m}.{key}.mean_mse") for m in ("qre", "single_bgnet")]
        sd = [v(f"mt.{m}.{key}.std_mse") for m in ("qre", "single_bgnet")]
        bars = ax.bar([0, 1], ms, yerr=sd, capsize=6, color=[COL["qre"], COL["bgnet"]], width=0.6)
        bar_labels(ax, bars, pad=4, errs=sd)
        ax.set_xticks([0, 1], ["QRE", "BGNet"])
        ax.set_title(title)
        ax.set_ylim(0, max(m + s for m, s in zip(ms, sd)) * 1.25)
    axes[0].set_ylabel("Test MSE (mm²/day²)")
    fig.suptitle("Test 1981–2000, mean ± std over 3 repetitions", fontsize=14, color=COL["muted"], y=1.02)
    fig.tight_layout()
    save(fig, "fig_midterm_results.png")


# ------------------------------------------------------------------ Weight Network collapse
def wn_collapse():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(6)
    true, arg = v("s.wn.true"), v("s.wn.argmax")
    ax.bar(x - 0.2, true, 0.4, color=COL["grey"], label="true interval of the day")
    b = ax.bar(x + 0.2, arg, 0.4, color=COL["bad"], label="Weight Network argmax")
    ax.text(3.0, arg[3] * 0.93, f"all {arg[3]:,} of {sum(true):,}\nvalidation days →", ha="right", va="top",
            fontsize=14, color=COL["bad"], fontweight="bold")
    ax.set_xticks(x, [f"{i + 1}" for i in x])
    ax.set_xlabel("Interval (1 = driest, 6 = wettest)")
    ax.set_ylabel("Validation days")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.55))
    save(fig, "fig_wn_collapse.png")


# ------------------------------------------------------------------ GMM segmentation in the mid-term run
def gmm_midterm():
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    shades = ["#DCE8F2", "#B7D0E5", "#8FB4D3", "#6895BE", "#4677A6", "#2B5A88"]
    for k in range(3):
        edges = [0.0] + [b[1] for b in v(f"mt.gmm.rep{k}.intervals")]
        for i in range(6):
            ax.barh(2 - k, edges[i + 1] - edges[i], left=edges[i], color=shades[i], edgecolor="white", height=0.6)
        ax.text(1.02, 2 - k, f"{v(f'mt.gmm.rep{k}.sample_sum'):,} days", va="center", fontsize=14,
                color=COL["bad"])
    ax.set_yticks([2, 1, 0], ["rep 0\n(seed 42)", "rep 1\n(seed 1042)", "rep 2\n(seed 2042)"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Quantile of training daily totals (interval bounds)")
    ax.set_title(f"Intervals refit in every repetition; segments hold more days than the {v('mt.train_days'):,} training days",
                 fontsize=13, loc="left")
    save(fig, "fig_gmm_midterm.png")


# ------------------------------------------------------------------ raw predictor scales
def predictor_scales():
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    names = ["q850", "t850", "u850", "v850"]
    means = [abs(v(f"nz.raw_mean.{n}")) for n in names]
    stds = [v(f"nz.raw_std.{n}") for n in names]
    x = np.arange(4)
    b1 = ax.bar(x - 0.2, means, 0.4, color=COL["qre"], label="|mean|")
    b2 = ax.bar(x + 0.2, stds, 0.4, color=COL["bgnet"], label="std")
    ax.set_yscale("log")
    for bars, vals in ((b1, means), (b2, stds)):
        for bb, val in zip(bars, vals):
            ax.annotate(f"{val:.3g}", (bb.get_x() + bb.get_width() / 2, bb.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=12)
    ax.set_xticks(x, ["q850\n(kg/kg)", "t850\n(K)", "u850\n(m/s)", "v850\n(m/s)"])
    ax.set_ylim(1e-4, 5e4)
    ax.set_ylabel("Raw value, training split (log)")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    save(fig, "fig_predictor_scales.png")


# ------------------------------------------------------------------ Weight Network loss study
def loss_study():
    plt.rcParams.update({"xtick.labelsize": 12, "ytick.labelsize": 12, "axes.titlesize": 14})
    variants = [("emd_paper", "EMD\n(Eq. 6)"), ("cross_entropy", "CE"),
                ("emd_squared_cdf", "Sq. EMD\n(ours)"), ("ce_plus_emd", "CE +\nEMD")]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.3))
    x = np.arange(len(variants))
    acc = [v(f"wl.{k}.accuracy") for k, _ in variants]
    cols = [COL["bad"], COL["grey"], COL["qre_mass"], COL["grey"]]
    bars = axes[0].bar(x, acc, color=cols, width=0.6)
    bar_labels(axes[0], bars[1:], "{:.3f}")
    axes[0].annotate(f"{acc[0]:.3f}", (0, acc[0] / 2), ha="center", va="center", fontsize=13, color="white",
                     fontweight="bold")
    axes[0].axhline(v("wl.chance.majority"), color=COL["muted"], ls="--", lw=1.5,
                    label=f"majority-class chance (val) {v('wl.chance.majority'):.3f}")
    axes[0].legend(frameon=False, loc="upper left", fontsize=12)
    axes[0].set_xticks(x, [l for _, l in variants])
    axes[0].set_ylim(0, 0.85)
    axes[0].set_title("Validation accuracy", loc="left")
    mse = [v(f"wl.{k}.val.[0.0,1.0]") for k, _ in variants]
    bars = axes[1].bar(x, mse, color=cols, width=0.6)
    bar_labels(axes[1], bars)
    axes[1].axhline(v("wl.reference_perfect_member.val.[0.0,1.0]"), color=COL["pm"], ls="--", lw=1.5,
                    label=f"Perfect Member (val) {v('wl.reference_perfect_member.val.[0.0,1.0]'):.2f}")
    axes[1].legend(frameon=False, loc="upper right", fontsize=12, handlelength=1.2)
    axes[1].set_xticks(x, [l for _, l in variants])
    axes[1].set_ylim(0, 190)
    axes[1].set_title("QRE validation MSE [0, 1]", loc="left")
    fig.tight_layout()
    save(fig, "fig_loss_study.png")
    plt.rcParams.update({"xtick.labelsize": 14, "ytick.labelsize": 14, "axes.titlesize": 16})


# ------------------------------------------------------------------ NZ member study
def member_study_nz():
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(6)
    series = [("sanity_10_epoch", "10-epoch members (sanity run)", COL["grey"]),
              ("scratch", "from scratch, to convergence", COL["bgnet"]),
              ("warm_start", "warm start from BGNet (selected)", COL["qre"])]
    for j, (k, lab, c) in enumerate(series):
        ax.bar(x + (j - 1) * 0.27, [v(f"ms.{k}.ratio.{i}") for i in x], 0.27, color=c, label=lab)
    ax.axhline(1.0, color=COL["ink"], lw=1.2)
    ax.text(5.45, 1.04, "= BGNet", ha="right", fontsize=12)
    ax.set_xticks(x, [f"{i + 1}" for i in x])
    ax.set_xlabel("Member interval (1 = driest, 6 = wettest)")
    ax.set_ylabel("Own-interval val MSE / BGNet's")
    ax.set_ylim(0, 2.7)
    ax.legend(frameon=False, loc="upper left", fontsize=12)
    save(fig, "fig_member_study_nz.png")


# ------------------------------------------------------------------ NZ before vs after
def nz_before_after():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, (r, title) in zip(axes, REG):
        key = r.strip("[]")
        mk = {"[0.0,1.0]": "mse_0_1", "[0.0,0.2]": "mse_0_0.2", "[0.9,1.0]": "mse_0.9_1"}[r]
        vals = [v(f"mt.qre.{key}.mean_mse"), v(f"mt.single_bgnet.{key}.mean_mse"),
                v(f"nz.rep.qre.{mk}.mean"), v(f"nz.rep.bgnet.{mk}.mean")]
        errs = [v(f"mt.qre.{key}.std_mse"), v(f"mt.single_bgnet.{key}.std_mse"),
                v(f"nz.rep.qre.{mk}.std"), v(f"nz.rep.bgnet.{mk}.std")]
        xs = [0, 1, 2.6, 3.6]
        bars = ax.bar(xs, vals, yerr=errs, capsize=5, width=0.8,
                      color=[COL["qre"], COL["bgnet"], COL["qre"], COL["bgnet"]],
                      alpha=1.0)
        for bb in bars[:2]:
            bb.set_hatch("//")
            bb.set_edgecolor("white")
        bar_labels(ax, bars, pad=4, size=12, errs=errs)
        ax.set_xticks([0.5, 3.1], ["Mid-term\n(3 reps)", "Final\n(10 reps)"])
        ax.set_title(title)
        ax.set_ylim(0, max(a + b for a, b in zip(vals, errs)) * 1.28)
    axes[0].set_ylabel("Test MSE")
    h = [plt.Rectangle((0, 0), 1, 1, color=COL["qre"]), plt.Rectangle((0, 0), 1, 1, color=COL["bgnet"])]
    fig.legend(h, ["QRE", "BGNet"], loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(1.0, 1.06))
    fig.tight_layout()
    save(fig, "fig_nz_before_after.png")


# ------------------------------------------------------------------ final results, NZ and India
def final_bars(ds, models, metrics, name, size=(12, 4.3)):
    fig, axes = plt.subplots(1, len(metrics), figsize=size)
    rng = np.random.default_rng(0)
    for ax, (mk, title, fmt) in zip(axes, metrics):
        means = [v(f"{ds}.rep.{m}.{mk}.mean") for m in models]
        stds = [v(f"{ds}.rep.{m}.{mk}.std") for m in models]
        x = np.arange(len(models))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=[COL[m] for m in models], width=0.62,
                      error_kw=dict(lw=1.5))
        for i, m in enumerate(models):
            vals = v(f"{ds}.rep.{m}.{mk}.values")
            ax.scatter(i + rng.uniform(-0.14, 0.14, len(vals)), vals, s=14, color=COL["ink"], alpha=0.55, zorder=3)
        lo = min(min(v(f"{ds}.rep.{m}.{mk}.values")) for m in models)
        hi = max(max(v(f"{ds}.rep.{m}.{mk}.values")) for m in models)
        span = hi - lo
        ax.set_ylim(lo - 0.8 * span, hi + 0.55 * span)
        for i, (mm, s) in enumerate(zip(means, stds)):
            ax.text(i, hi + 0.12 * span, fmt.format(mm), ha="center", fontsize=12)
        ax.set_xticks(x, [LABEL[m].replace(" (λ = 0 control)", "\ncontrol") for m in models], fontsize=12)
        ax.set_title(title, loc="left")
    fig.tight_layout()
    save(fig, name)


def india_bias():
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    models = ["bgnet", "qre", "qre_mass"]
    rng = np.random.default_rng(1)
    for i, m in enumerate(models):
        vals = np.array(v(f"in.signed.{m}.values"))
        ax.bar(i, vals.mean(), yerr=vals.std(ddof=1), capsize=6, color=COL[m], width=0.6)
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(vals)), vals, s=16, color=COL["ink"], alpha=0.55, zorder=3)
    ax.axhline(0, color=COL["ink"], lw=1.2)
    labels = {"bgnet": N["rep.signed.in.bgnet"]["display"], "qre": N["rep.signed.in.qre"]["display"],
              "qre_mass": N["rep.signed.in.mass"]["display"]}
    for i, m in enumerate(models):
        ax.text(i, 0.014, labels[m], ha="center", fontsize=13)
    ax.set_xticks(range(3), [LABEL[m] for m in models])
    ax.set_ylabel("Test: signed total relative discrepancy\n(predicted − observed) / observed")
    ax.set_ylim(-0.07, 0.026)
    save(fig, "fig_india_bias.png")


def india_member():
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    x = np.arange(6)
    for j, (k, lab, c) in enumerate([("scratch", "from scratch", COL["bgnet"]),
                                     ("warm_start", "warm start (selected)", COL["qre"])]):
        bars = ax.bar(x + (j - 0.5) * 0.38, [v(f"im.{k}.ratio.{i}") for i in x], 0.38, color=c, label=lab)
    ax.axhline(1.0, color=COL["ink"], lw=1.2)
    for i in x:
        ax.text(i, 0.06, f"{v(f'im.scratch.train_days.{i}')}\ndays", ha="center", fontsize=11, color="white",
                fontweight="bold")
    ax.set_xticks(x, [f"{i + 1}" for i in x])
    ax.set_xlabel("Member interval (1 = driest, 6 = wettest)")
    ax.set_ylabel("Own-interval val MSE / BGNet's")
    ax.set_ylim(0, 1.55)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    save(fig, "fig_india_member.png")


def placements():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
    styles = {"ensemble": ("o", COL["qre_mass"]), "members": ("s", COL["qre"]), "both": ("^", COL["pm"])}
    for ax, ds, title in ((axes[0], "nz", "NZ"), (axes[1], "in", "India")):
        lams = [0.1, 1, 3, 10]
        for p, (mk, c) in styles.items():
            ax.plot(lams, [v(f"{ds}.cand.{p}.{l:g}.ratio") for l in lams], marker=mk, color=c, lw=2, ms=8, label=p)
        ax.axhspan(0, 1.02, color="#E8F3F3", zorder=0)
        ax.axhline(1.02, color=COL["muted"], ls="--", lw=1.2, label="+2 % of the control")
        sp, sl = v(f"{ds}.sel.placement"), v(f"{ds}.sel.lambda")
        ax.plot([sl], [v(f"{ds}.cand.{sp}.{sl:g}.ratio")], marker="o", ms=20, mfc="none", mec=COL["ink"], mew=2)
        ax.annotate(f"selected: {sp}, λ = {sl:g}", (sl, v(f"{ds}.cand.{sp}.{sl:g}.ratio")),
                    xytext=(-30, 34) if ds == "nz" else (40, -34), textcoords="offset points",
                    ha="right" if ds == "nz" else "left", va="center", fontsize=12, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=COL["ink"], lw=1))
        ax.set_xscale("log")
        ax.set_xticks(lams, ["0.1", "1", "3", "10"])
        ax.set_xlabel("λ (multiple of the initial L_QRE / L_mass)")
        ax.set_title(f"{title}: validation MSE [0, 1] / its λ = 0 control", loc="left")
        ax.set_ylim(0.97, 1.6 if ds == "nz" else 1.2)
    axes[0].set_ylabel("MSE ratio")
    axes[0].legend(frameon=False, loc="upper left", title="placement", title_fontsize=12)
    fig.tight_layout()
    save(fig, "fig_placements.png")


def confusion():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, (k, title) in zip(axes, (("emd_paper", "EMD (paper Eq. 6): collapsed"),
                                     ("emd_squared_cdf", "Squared EMD (ours)"))):
        cm = np.array(v(f"wl.{k}.confusion"))
        ax.imshow(cm, cmap="Blues")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                        color="white" if cm[i, j] > cm.max() * 0.55 else COL["ink"])
        ax.set_xticks(range(6), [str(i + 1) for i in range(6)])
        ax.set_yticks(range(6), [str(i + 1) for i in range(6)])
        ax.set_xlabel("predicted interval")
        ax.set_ylabel("true interval")
        ax.set_title(title, loc="left", fontsize=14)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    save(fig, "fig_confusion.png")


def india_intervals():
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    days = [int(d) for d in N["in.seg.train_days"]["display"].split(" / ")]
    bars = ax.bar(range(6), days, color=["#DCE8F2", "#B7D0E5", "#8FB4D3", "#6895BE", "#4677A6", "#2B5A88"])
    bar_labels(ax, bars, "{:.0f}", size=11)
    ax.set_xticks(range(6), [str(i + 1) for i in range(6)], fontsize=12)
    ax.set_xlabel("Interval (paper's bounds)", fontsize=12)
    ax.set_ylim(0, 1050)
    ax.set_yticks([0, 500, 1000])
    ax.tick_params(axis="y", labelsize=11)
    ax.set_title("Training days per member", fontsize=12, loc="left")
    save(fig, "fig_india_intervals.png")


def crops():
    """Crops of the committed example-map plots (pixels only; nothing redrawn)."""
    from PIL import Image
    root = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    for ds, rel in (("india", "results/india/qre_mass_v1/placements/plots/example_maps.png"),
                    ("nz", "results/fixes_v2/nz_wn/sanity/mass_study/placements/plots/example_maps.png")):
        im = Image.open(os.path.join(root, rel))
        assert im.size == (1760, 990), im.size  # the crop boxes below assume this layout
        right = 420 if ds == "india" else 410  # NZ: the next panel's tick marks start just right of 410
        im.crop((185, 95, right, 352)).save(os.path.join(HERE, f"crop_{ds}_observed_map.png"))
        im.crop((185, 95, 1455, 352)).save(os.path.join(HERE, f"crop_{ds}_maps_row.png"))
        print("wrote", f"crop_{ds}_observed_map.png", f"crop_{ds}_maps_row.png")


if __name__ == "__main__":
    crops()
    midterm_results()
    wn_collapse()
    gmm_midterm()
    predictor_scales()
    loss_study()
    member_study_nz()
    nz_before_after()
    final_bars("nz", ["bgnet", "qre", "control", "qre_mass"],
               [("mse_0_1", "Test MSE [0, 1]", "{:.2f}"), ("mse_0.9_1", "Test MSE [0.9, 1]", "{:.2f}"),
                ("daily_total_mae", "Test daily-total MAE", "{:.3f}")], "fig_nz_final.png")
    india_bias()
    india_member()
    placements()
    confusion()
    india_intervals()
