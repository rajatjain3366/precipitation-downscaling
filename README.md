# Quantile-Regression-Ensemble (QRE) for Precipitation Downscaling

> **Bachelor Thesis Project (BTP)**
> **Based on the AAAI-24 paper**: *"Quantile-Regression-Ensemble: A Deep Learning Algorithm for Downscaling Extreme Precipitation"* (Bailie, Koh, Rampal & Gibson), Proceedings of the AAAI Conference on Artificial Intelligence 38(20), 21914–21922, 2024, [doi:10.1609/aaai.v38i20.30193](https://doi.org/10.1609/aaai.v38i20.30193) ([AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/30193))
> **This repository**: a reproduction on alternative data (CORDEX-ML-Bench, New Zealand), and an extension to India (ERA5 → IMD) with a global mass-conservation regularization (§8, implemented and smoke-tested; no India results yet)

---

## 1. Overview

Deep-learning downscaling models trained over the whole record tend to regress to the mean and underestimate extreme precipitation. QRE addresses this by:

1. **Segmentation**: splitting training days into $N$ intensity intervals by their domain-total precipitation $\mathbf{1}\cdot y$, with interval boundaries from a Bayesian Gaussian Mixture Model (averaged over 100 runs).
2. **Members**: training one Bernoulli-Gamma network (`BGNet`) per interval.
3. **Weight Network**: training a classifier $\omega_\theta$ with the Earth Mover's Distance loss to say which interval a day belongs to.
4. **Ensemble**: $g(x) = \sum_{i=1}^N \omega_i(x)\, f_i(x)$ (paper Eq. 7).

---

## 2. Data

The paper used ERA5 predictors and VCSN observations, which were not available to us. Following advice from co-author Neelesh Rampal, we use **CORDEX-ML-Bench** (New Zealand domain):

| Item | Description |
| --- | --- |
| Predictors $X$ | q850, t850, u850, v850 on a 16 × 16 grid (ACCESS-CM2 "perfect" predictors). w850 is not in the benchmark. |
| Target $Y$ | Daily precipitation `pr` (regional climate model output) on a 128 × 128 grid, flattened to the **2,804** land cells (orography `orog > 0` m in `Static_fields.nc`, which has no `sftlf`; `src/data_adapter_cordex.py:extract_land_mask`). The same mask, in the same row-major cell order, is used for the training targets, every evaluation and the mass-conservation cell areas. |
| Static input | Surface elevation `orog` from `Static_fields.nc` |
| Period | 1961–2000 (two files: 1961–1980 and 1981–2000), 365-day calendar |

Sources:

- [Zenodo record 17957264](https://zenodo.org/records/17957264) (v4, `NZ_domain.zip`, md5 `a3f586f28d78b3a4c53b7b8324db4e62`) for all files except the 1981–2000 target. v4 does not contain any test targets.
- `test/historical/target/pr_tasmax_ACCESS-CM2_1981-2000.nc` from [Zenodo record 20985924](https://zenodo.org/records/20985924) (v7), md5 `15b0007f910b0c72c632cc37409d17f1`. Every file that exists in v4 is byte-identical in v7 (same CRC32 and size).
- Check: the v4 1961–1975 data reproduces the GMM intervals of our first run exactly ([results/baseline_rerun/data_match_check.json](results/baseline_rerun/data_match_check.json), produced by `scripts/data_match_check.py`).

Expected layout under `data/CORDEX_NZ/`:

```text
train/ESD_pseudo_reality/predictors/ACCESS-CM2_1961-1980.nc
train/ESD_pseudo_reality/predictors/Static_fields.nc
train/ESD_pseudo_reality/target/pr_tasmax_ACCESS-CM2_1961-1980.nc
test/historical/predictors/perfect/ACCESS-CM2_1981-2000.nc
test/historical/target/pr_tasmax_ACCESS-CM2_1981-2000.nc
```

**Split (v2):** train 1961–1988 (10,220 days), validation 1989–1993 (1,825), test 1994–2000 (2,555), i.e. 70 / 12.5 / 17.5 %, the same proportions as the paper's 32 / 6 / 8 years. All statistics (predictor normalization, GMM intervals, thresholds, empirical means) are fitted on the training split only.

---

## 3. Model

### BGNet (member and baseline)

- 3×3 init conv, then 3 blocks of [conv → **channel attention (CBAM-style: average and max pooling, shared MLP, r = 1)** → conv], with 64 / 128 / 256 channels.
- Dense 512 (ReLU, dropout 0.2), then three heads with one unit per land cell:
  - $p$ = **P(rain)**, through a sigmoid;
  - $\log\alpha$ (Gamma shape) and $\log\beta$ (Gamma scale), through a modified tanh $3.5\tanh(x/2.5)$.
- Elevation enters as an extra input channel through a small conv embedding (as in the original code).

Loss (paper Eq. 1–2) with rain threshold $c$:

$$\mathcal{L}_{BG} = -\Big[\mathbb{1}_{y \le c}\ln(1-p) + \mathbb{1}_{y > c}\big(\ln p + (\alpha-1)\ln y - \alpha\ln\beta - \ln\Gamma(\alpha) - y/\beta\big)\Big]$$

Prediction (Eq. 3): $\mathbb{E}[y] = p\cdot\alpha\cdot\beta$.

QRE members and the single-BGNet baseline are built by the same function (`src/builders.py:build_bgnet`), so they always share the same architecture.

### Weight Network

The same conv backbone, then 2 × Dense 100 (dropout 0.2) and an $N$-way softmax. The paper trains it with the EMD loss (Eq. 5–6, $p = 1$): $\mathcal{L}_\omega = \sum_{k} |k - k'|\,\omega_k$, where $k'$ is the true interval. On CORDEX NZ this loss collapses the network onto one interval, so v2 uses the squared EMD on cumulative distributions of Hou et al. (2016) instead, $\sum_k (\mathrm{CDF}_\omega(k) - \mathrm{CDF}_y(k))^2$ (see §4 and §5). The other losses can be selected with `weight_network.loss`.

---

## 4. Deviations from the paper

| Item | Paper | Ours | Reason |
| --- | --- | --- | --- |
| Data | ERA5 → VCSN observations, 5 km, 11,491 points | CORDEX-ML-Bench GCM predictors → regional-model precipitation, 128 × 128, **2,804 land cells** | ERA5/VCSN not available to us; CORDEX recommended by a co-author. Absolute MSE is not comparable with the paper. |
| Predictors | T, Q, U, V, **W** at 850 hPa, (5, 36, 41) | T, Q, U, V at 850 hPa, (4, 16, 16) | W850 is not in the benchmark |
| Split | 32 / 6 / 8 years | **1961–1988 / 1989–1993 / 1994–2000** (28 / 5 / 7 years) | Same 70 / 12.5 / 17.5 % proportions over the 40 years with targets |
| Rain threshold in the loss | $y > 0$ | **$y > 0.1$ mm/day** | 94.3 % of CORDEX training values are $> 0$ (only 5.7 % are exactly 0, with positive values down to $6\times10^{-13}$). $y > 0$ would make almost every cell "wet" and force the Gamma part to fit numerical drizzle. 0.1 mm is the usual trace threshold. Configurable: `loss.rain_threshold`. |
| Conv kernel | 6 | **3, "same" padding** | Six valid 6 × 6 convolutions need at least 31 × 31 pixels; the CORDEX grid is 16 × 16 |
| Segment bounds | $l_i < \mathbf{1}\cdot y < h_i$ (strict on both sides) | $t_i \le \mathbf{1}\cdot y < t_{i+1}$ | Strict bounds on both sides would drop boundary days. Ours puts every day in exactly one interval. |
| Batch size | not stated | 91 (about one season) | Value used in the original code |
| Learning-rate decay | 0.7 every 6,000 steps | same; with batch 91 that is about every 53 epochs for the full training set | — |
| Max epochs | not stated | 100, extended in blocks of 50 (up to 300) while the validation loss still improves | Early stopping decides in practice |
| Member early stopping | not stated | on validation days of the member's own interval | A member is only used for its interval |
| Weight Network loss | EMD, Eq. 6: $\sum_k \lvert k-k'\rvert\,\omega_k$ | **Squared EMD on CDFs** (Hou et al. 2016): $\sum_k (\mathrm{CDF}_\omega(k) - \mathrm{CDF}_y(k))^2$ | Eq. 6 collapsed on NZ: validation argmax histogram **[0, 0, 0, 1825, 0, 0]**, mean max weight 1.000, best epoch 1. The loss is linear in $\omega$, so its gradient through the softmax vanishes once $\omega$ is one-hot. Cross-entropy, squared EMD and CE + EMD all fix the collapse, and their validation QRE MSE is within 0.5 of each other (§5), i.e. tied within noise. Squared EMD was chosen because it is the EMD family the paper cites, still penalises distant intervals more, and is smooth with a nonzero gradient. Options: `weight_network.loss = "emd"` (paper), `"cross_entropy"`, `"ce_plus_emd"`. |
| Repetitions | 40 | NZ: 1 sanity repetition only (the NZ study was reduced to "show the fixes work") | Compute |

---

## 5. Results

### First run (before fixes), `results/final_cordex_qre/`

Old split (train 1961–1975, val 1976–1980, test 1981–2000), 3 repetitions, produced by `experiments/run_cordex_experiment.py` with `configs/cordex_final.json`. These numbers come from the pipeline **before** the fixes. Its known problems:

- the Weight Network collapsed onto one class;
- the predictors were not normalized;
- the BGNet baseline had a different architecture from the members;
- predictions used a $p > 0.5$ threshold instead of $p\cdot\alpha\beta$.

| Model | MSE [0, 1] | MSE [0, 0.2] | MSE [0.9, 1] |
| --- | --- | --- | --- |
| QRE | 112.47 ± 9.45 | 27.86 ± 15.06 | 579.69 ± 80.66 |
| Single BGNet | 62.03 ± 0.22 | 4.28 ± 0.35 | 251.72 ± 3.78 |
| Bagging | 106.96 ± 5.82 | 43.22 ± 8.84 | 494.99 ± 19.88 |
| Probability | 105.48 ± 4.08 | 24.67 ± 2.85 | 550.61 ± 19.50 |

### After fixes (v2), `results/fixes_v2/`

NZ sanity run ([results/fixes_v2/nz/sanity/](results/fixes_v2/nz/sanity/)): 1 repetition, new split, all fixes. Epochs were cut to 10 for the members and both BGNets and 30 for the Weight Network, so **every model is undertrained** (all stopped at their epoch cap while still improving). Test set is 1994–2000, so the numbers are not comparable with the first run's 1981–2000 test set.

| Model (test, 1994–2000) | MSE [0, 1] | MSE [0, 0.2] | MSE [0.9, 1] |
| --- | --- | --- | --- |
| QRE | 124.13 | 15.23 | 627.19 |
| Single BGNet | 60.60 | 2.85 | 231.79 |
| BGNet(-) | 67.31 | 2.45 | 261.29 |
| Bagging | 134.36 | 17.35 | 674.44 |
| Probability | 133.96 | 14.47 | 682.15 |

**The Weight Network still collapses** even with the paper's training settings (LR 5e-5, decay 0.7/6000, patience 15, EMD as a sum): 100 % of validation days go to interval 3 and the mean max weight is 1.000, from epoch 1 on. At initialization its output is uniform (mean max weight 0.167), so this is not an initialization problem. The likely cause is the EMD loss of Eq. 6 itself, $\sum_k |k-k'|\,\omega_k$. It is linear in $\omega$, so its minimum is a one-hot vector, and its gradient through the softmax, $\omega_k(c_k - \mathcal{L})$, goes to zero once $\omega$ is one-hot. QRE therefore equals member 3 for every day. Diagnostics are in `results/fixes_v2/nz/sanity/repetition_0.json`. The loss study below fixes the collapse.

Members vs single BGNet on each member's own validation interval: the members win on the two lowest intervals (MSE ratio 0.90, 0.82) and lose on the others (1.06 to 2.49). With a fixed epoch cap the small high-intensity members get far fewer optimizer steps than BGNet (e.g. 14 vs 112 steps per epoch), so this comparison is biased against them.

### Weight Network loss study, `results/fixes_v2/nz_wn/sanity/weight_network_losses/`

Produced by `experiments/weight_network_losses.py`. It reuses the 6 members and BGNet saved from one sanity run: members and BGNet trained 10 epochs, each Weight Network trained 30 epochs, seed 42, same backbone for every variant. The variants were trained one at a time. **Everything below is on validation (1989–1993, 1825 days)** except the last table. True interval histogram: [234, 384, 388, 320, 260, 239]. Chance levels: uniform 0.167, majority class 0.213.

| Weight source | Accuracy | Within ±1 | Argmax histogram | Mean max ω | QRE val MSE [0,1] / [0,0.2] / [0.9,1] |
| --- | --- | --- | --- | --- | --- |
| Ceiling: interval of BGNet's predicted daily total (no training) | 0.627 | 0.984 | — | 1 (one-hot) | 117.88 / 2.36 / 583.0 |
| Perfect Member: true interval (no training) | 1.000 | 1.000 | — | 1 (one-hot) | 116.74 / 1.89 / 585.1 |
| EMD, Eq. 6 (paper) | 0.175 | 0.530 | [0, 0, 0, 1825, 0, 0] | 1.000 | 134.93 / 14.76 / 693.2 |
| **Cross-entropy** | 0.649 | 0.982 | [243, 395, 405, 326, 242, 214] | 0.683 | **116.55** / 2.02 / 580.2 |
| Squared EMD on CDFs | 0.656 | 0.980 | [229, 438, 349, 329, 267, 213] | 0.685 | 117.04 / 1.99 / 581.5 |
| CE + EMD (1 : 1) | 0.654 | 0.986 | [196, 426, 407, 322, 268, 206] | 0.768 | 116.96 / 2.03 / 581.8 |
| *Single BGNet (for reference)* | | | | | *60.82 / 2.42 / 229.9* |

Confusion matrices (rows are the true interval) are in `results.json`. The three new losses all give a near-diagonal matrix with almost all errors one interval away. They beat the BGNet-ceiling accuracy slightly, and none of them collapses.

Selection (validation QRE MSE [0, 1]) picked **cross-entropy**. The three new losses differ by less than 0.5 MSE, which is well within what one repetition can resolve, so the ranking among them is not meaningful. The config default was therefore set to squared EMD afterwards (see §4); the variant is named `emd_squared_cdf` in this run's `results.json`. Only after selection was the test set evaluated, for the cross-entropy variant that the rule picked:

| Model (test, 1994–2000) | MSE [0, 1] | MSE [0, 0.2] | MSE [0.9, 1] |
| --- | --- | --- | --- |
| QRE, cross-entropy Weight Network | 109.02 | 2.41 | 531.39 |
| QRE, EMD Weight Network (sanity run above) | 124.13 | 15.23 | 627.19 |
| Single BGNet | 60.60 | 2.85 | 231.79 |

What this shows:

- **The Weight Network is fixed.** The new loss removes the collapse and its accuracy is about 3× chance.
- **QRE still loses to BGNet, and the cause is now the members.** Even always picking the correct member (Perfect Member, 116.7) is nearly twice BGNet's validation MSE (60.8). The cross-entropy network is already at that bound. No weighting can close the gap with these members; it needs better-trained members.
- **Everything is undertrained.** Every new-loss Weight Network was still improving at the 30-epoch cap (best epochs 28–30). Members and BGNet were capped at 10 epochs.

Time per Weight Network epoch was 140–190 s in this run, against 85 s in the earlier sanity run, on the same CPU.

---

## 6. How to run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt netCDF4 pytest

# All smoke / unit tests (synthetic data, a few minutes on CPU)
python tests/run_smoke_tests.py

# v2 pipeline: data report + GMM intervals only
python experiments/run_cordex_v2.py --dry-run
# 1 repetition with reduced epochs (sanity.* in the config)
python experiments/run_cordex_v2.py --sanity
# Full run
python experiments/run_cordex_v2.py --repeats 3
# Any config value can be overridden, e.g. to restore an old behaviour:
python experiments/run_cordex_v2.py --sanity --set prediction.mode='"threshold"' --save-dir results/fixes_v2/ablation_threshold
```

The legacy runner `experiments/run_cordex_experiment.py` (with `configs/cordex_final.json`) is kept unchanged in behaviour, for reproducing the first run.

### Config flags that restore old behaviour (`configs/cordex_v2.json`)

| Flag | v2 value | Old behaviour |
| --- | --- | --- |
| `data.train_years` / `val_years` / `test_years` | 1961–1988 / 1989–1993 / 1994–2000 | 1961–1975 / 1976–1980 / 1981–2000 |
| `data.standardize_predictors` | `true` | `false` |
| `data.use_elevation` | `true` | `false` |
| `loss.rain_threshold` | `0.1` | `0.5` |
| `prediction.mode` | `"expected"` ($p\alpha\beta$) | `"threshold"` |
| `segmentation.method` | `"bayesian_gmm"` | `"legacy"` |
| `members.validate_on_own_segment` | `true` | `false` |
| `weight_network.loss` | `"squared_emd"` (also `"cross_entropy"`, `"ce_plus_emd"`) | `"emd"` (paper Eq. 6) |
| `mass.lambdas` / `mass.lambda0` | `[0, 0.01, 0.1, 1]` / `"original"` (λ = 0 is the v2 QRE) | no mass term: `[0]` |
| `weight_network.emd_reduction` | `"sum"` | `"mean"` |
| `architecture.*`, learning rates, decay, patience | paper values where possible | see `configs/cordex_final.json` |

---

## 7. Repository layout

```text
configs/
  cordex_v2.json               v2 configuration (all fixes)
  cordex_final.json            configuration of the first run
  cordex_midterm_on_v2_split.json  mid-term settings on the v2 split (approximation, not run)
  india_v1.json                India experiment (ERA5 -> IMD, mass conservation)
  india_smoke.json             India smoke test on data/India_sample/
docs/
  RUNBOOK.md                   every long run, in order, for running on another machine
  STATUS.md                    what is done, key numbers, next steps, open questions
experiments/
  run_cordex_v2.py             v2 pipeline (train-only statistics, flags for every fix)
  run_cordex_experiment.py     legacy runner (first run)
  weight_network_losses.py     Weight Network loss study on saved checkpoints
  member_study.py              NZ member study: warm start (B) / from scratch (A), BGNet retrain
  run_india.py                 India runner, stages data / A / sweep / B / all
  india_day_offset.py          ERA5 -> IMD day offset check (train + val only)
  nz_mass.py                   NZ: QRE vs QRE + mass conservation on the v2 models (§9)
scripts/
  download_era5_india.py       ERA5 CDS download + 4x4 coarsening (not run here)
  data_match_check.py          produced results/baseline_rerun/data_match_check.json
src/
  data_adapter_cordex.py       NetCDF loading, land mask, elevation, 1961-2000 loader
  india_data.py                IMD / ERA5 loading, grid alignment, day matching, cell areas
  grid_areas.py                cell areas from 1-D lat/lon centres (NZ target grid)
  mass_conservation.py         mass loss, member loss with mass, joint ensemble stage
  india_metrics.py, india_plots.py   India evaluation and plots
  preprocessing.py             year / date splits, training-split standardization
  segmentation.py              Bayesian-GMM intervals, thresholds, partition
  builders.py                  shared model builders (members = baseline)
  checkpoints.py               save/load weights with a config signature
  ConvolutionalNetworks.py     BGNet, BG_mean (p*alpha*beta or threshold)
  Modules.py                   conv backbone, CBAM-style channel attention, elevation embedding
  GammaLoss.py                 Bernoulli-Gamma loss
  quantiles.py                 Weight Network (Omega), EMD loss, bin_y_var, ensemble
tests/                         unit + smoke tests (tests/run_smoke_tests.py runs all)
results/
  final_cordex_qre/            first run (before fixes)
  baseline_rerun/              data-match check against the first run
  fixes_v2/                    v2 outputs
  india/smoke/                 India smoke-test outputs (code check only, not results; reference copy)
  india/smoke_run/             written by tests/run_smoke_tests.py (gitignored)
```

---

## 8. India: QRE with global mass-conservation regularization

**Status: implemented and tested on the sample files only. No India experiment has been run, and no India result exists yet.** The long runs are listed in [docs/RUNBOOK.md](docs/RUNBOOK.md).

The same QRE code as the NZ v2 pipeline is used: train-only normalization, $p\alpha\beta$ predictions, one builder for members and BGNet, Bayesian-GMM intervals fitted once on training totals, training thresholds for validation and test labels, members early-stopped on their own validation interval, the squared-EMD Weight Network and checkpoints. Everything is new code (`experiments/run_india.py`, `configs/india_*.json`, `results/india/`). The NZ code, configs and outputs are unchanged. There is one addition: a constraint on the all-India daily rainfall total. It does not model moisture-flux convergence or any other physics.

### Data pipeline (`src/india_data.py`)

| Item | India setting |
| --- | --- |
| Predictors $X$ | ERA5 q, t, u, v, w at 850 hPa, 4 × 4 block means of the 0.25° grid (1°). Sample: one 00 UTC field per day. Daily means (preferred) and hourly data are also supported; sub-daily data is averaged per day, optionally over 03–03 UTC (`data.aggregation_shift_hours = 3`). |
| Target $Y$ | IMD gridded rainfall (mm/day), 0.25°, 129 × 135, flattened to the **4,964** valid cells |
| Grid alignment | ERA5 latitude flipped to ascending. Each ERA5 coordinate is checked to be the mean of exactly 4 IMD coordinates, with the blocks contiguous. The partial last row (6.5° N) and column (99.75° E) are dropped, which gives **32 × 33** inputs. |
| Mask | IMD NaN cells are outside India. The mask is computed once and checked to be constant over time. NaN is never replaced by zero; only valid cells are used everywhere. |
| Days | ERA5 day $D$ + `data.day_offset` (0 or 1) predicts IMD day $D$ (IMD days are 24 h accumulations ending 08:30 IST). `experiments/india_day_offset.py` picks the offset by predictor–rainfall correlation on train + val days. Missing days are listed in `data_report.json` and are an error unless `data.allow_missing_days` is set. |
| Period, split | 1990–2019; train 1990–2010, val 2011–2014, test 2015–2019 (21 / 4 / 5 years = 70 / 13.3 / 16.7 %), configurable date ranges |
| Rain threshold | $y > 0$: IMD is gauge-based with true zeros (sample: 67.2 % of valid values are exactly 0, none in (0, 0.1)) |
| Elevation | Not used (`data.use_elevation = false`). No elevation source is configured for India. |

### Mass-conservation loss (`src/mass_conservation.py`)

For day $t$, valid mask $M$, cell area $A_i$, observed $P$ and predicted $\hat P$ (mm/day):

$$O_t = \sum_i M_i A_i P_{i,t}, \qquad S_t = \sum_i M_i A_i \hat P_{i,t}$$

$$L_\text{mass,abs} = \frac{1}{B}\sum_t \frac{|S_t - O_t|}{\sum_i M_i A_i} \quad\text{(area-mean mm/day, default)}, \qquad L_\text{mass,rel} = \frac{1}{B}\sum_t \frac{|S_t - O_t|}{O_t + \varepsilon'}$$

- **Cell areas:** $A_i = R^2\,\Delta\lambda\,(\sin\varphi_\text{north} - \sin\varphi_\text{south})$, in radians, with $R = 6371$ km. They are precomputed once with the mask as constant tensors. The valid cells total 3.50 M km², and the areas of the full grid sum exactly to the analytic box area (tested).
- **Implementation:** the code uses the normalized weights $w_i = M_iA_i / \sum M A$, so it works on area-mean totals. This is the same formula; the relative form's $\varepsilon'$ equals $\varepsilon\sum M A$, with $\varepsilon$ = `mass.eps_mm_per_day` (0.1).
- **Per-day averaging:** the loss is computed per day and averaged over the batch, never pooled into one total across days.
- **Masking:** masked cells are removed with `tf.where` before any arithmetic, so NaN there cannot reach the loss or its gradient.
- **Units and gradients:** targets are never normalized, so the loss is in physical units, and everything stays in TensorFlow so gradients flow.

**Why absolute is the default.** The relative form divides by the observed total, so a near-dry day weighs up to $1/\varepsilon$ = 10× while a 10 mm/day monsoon day weighs 0.1×. The days that carry most of the rainfall count least. Scales on the sample (January 1990, validation days):

| Prediction | $L_\text{mass,rel}$ | $L_\text{mass,abs}$ (mm/day) |
| --- | --- | --- |
| zero everywhere | 0.52 | 0.25 |
| training climatology | 1.66 | 0.28 |

The relative form ranks "no rain anywhere" far better than climatology, although their absolute total errors are almost equal. For conserving rainfall mass we therefore recommend the absolute form (`mass.mode = "absolute"`). `"relative"` remains available. The sample covers only dry-season days, so the teammate's run also writes both scales on the full data to `A/loss_scales.json`.

### Combining the losses and where to apply them

$L_\text{total} = L_\text{QRE} + \lambda_\text{mass} L_\text{mass}$, where $L_\text{QRE}$ is the loss of the stage being trained: the Bernoulli-Gamma NLL for members and the squared EMD for the Weight Network. Both terms are logged separately (`bg_nll` / `mass_loss`, `qre_loss` / `mass_loss` in each training history). With $\lambda = 0$ the mass term is skipped, and loss and gradients are exactly the original ones (tested).

| `mass.placement` | What is trained with the mass term | CPU cost of the λ sweep | Trade-offs |
| --- | --- | --- | --- |
| `"ensemble"` (**default**) | A joint fine-tuning stage of $g(x)=\sum_i \omega_i(x) f_i(x)$, starting from experiment A's Weight Network. Members are frozen by default, with their predictions precomputed; `ensemble.train_members = true` also fine-tunes them, keeping each member's NLL on its own days. | One short Weight Network fine-tune per λ; members and Weight Network are reused from A | Constrains exactly the quantity that is evaluated, the ensemble output. A and B differ only in this stage, because λ = 0 runs the same stage as the control. With frozen members it can only reweight members, so it fixes totals only as far as the members bracket the truth. Precomputed member predictions need about 1 GB of RAM for 21 training years. |
| `"members"` | Each member's training: NLL + λ·L_mass on its $p\alpha\beta$ | A full set of 6 members per λ | Changes the members themselves, but each member only sees its own intensity segment. The ensemble total is not constrained directly, since the Weight Network mixes the members afterwards. |
| `"both"` | Both of the above, with the same λ | As `"members"` plus the ensemble stage | The strongest constraint and the most expensive. Hardest to attribute. |

**λ values:** λ is a list (`mass.lambdas`, default [0, 0.01, 0.1, 1]), and `--stage sweep` trains all of them in one command. With `mass.lambda_scaling = "initial_ratio"`, each λ is multiplied by $L_\text{QRE}/L_\text{mass}$ measured at the start of the stage on training days. λ = 1 therefore means both terms start equally large, and the effective λ is logged.

**λ selection (validation only):** among λ > 0 whose validation QRE MSE on [0, 1] is within 2 % of λ = 0, take the one with the lowest validation all-India daily-total MAE. If none qualifies, take the λ > 0 with the lowest MSE and flag it (`sweep/selection.json`). The test split is never used for a choice.

### Experiments

- **A:** India QRE, λ = 0.
- **B:** India QRE with mass conservation at the selected λ.
- **Shared:** same data, preprocessing, split, seeds and evaluation; the conservation term is the only difference. Both are compared with a single BGNet (same builder, plain NLL, early stopping with patience 4 and at most 100 epochs).
- **Members:** trained to convergence (patience 4, at most 100 epochs). `members.init = "warm_start"` starts them from BGNet, as in the NZ member study.
- **Evaluation**, in physical units on the same mask (`src/india_metrics.py`):
  - MSE on the [0, 1], [0, 0.2] and [0.9, 1] quantile ranges of daily totals, per-interval MSE and the Perfect-Member bound;
  - MAE, RMSE and spatial correlation;
  - area-weighted all-India daily totals, with the absolute and relative total discrepancy and daily total-error statistics;
  - Weight Network diagnostics: accuracy, confusion matrix and argmax histogram.
- **Plots:** daily-total time series and scatter, example maps (observed / A / B) and spatial error maps.

### Deviations (India)

| Item | Paper | India here | Reason |
| --- | --- | --- | --- |
| Target | VCSN, NZ | IMD 0.25° gridded rainfall, India | End-evaluation dataset |
| Predictors | ERA5 q, t, u, v, w 850 hPa, (36, 41) | Same variables, 1° (4 × 4 block means), 32 × 33 | Grid of the provided ERA5 sample |
| Conv kernel | 6, valid | **6, valid (as in the paper)** | 32 × 33 ≥ 31 × 31, so the paper's architecture fits (NZ needed kernel 3). This also shrinks BGNet from 147 M to 13 M parameters. |
| Weight Network loss | EMD, Eq. 6 | Squared EMD on CDFs | Eq. 6 collapsed on NZ (§4) |
| Loss | BG NLL / EMD | + λ·L_mass (global mass conservation) | Extension of this project |
| Elevation input | yes | no | No India elevation source configured |
| Split | 32 / 6 / 8 years | 21 / 4 / 5 years (1990–2019) | Data planned for download |

Measured on this laptop's CPU (batch 91, India sizes): one BGNet training step takes 1.6 s and one Weight Network step 1.9 s. With kernel 3 "same" it would be 3.5 s and 2.7 s. Expected run times are in the RUNBOOK.

### How to run

```powershell
python experiments/india_day_offset.py --config configs/india_v1.json      # then set data.day_offset
python experiments/run_india.py --config configs/india_v1.json --stage A
python experiments/run_india.py --config configs/india_v1.json --stage sweep
python experiments/run_india.py --config configs/india_v1.json --stage B
# smoke test on the samples (about 30 s):
python experiments/run_india.py --config configs/india_smoke.json --stage all
```

---

## 9. NZ: QRE vs QRE + mass conservation

**Status: implemented and smoke-tested on the synthetic fixture only; not run on the real NZ data.** The run is RUNBOOK step 4.

Before the India experiment, the new model is compared with the old one on the NZ (CORDEX) data. `experiments/nz_mass.py` uses `configs/cordex_v2.json` and the v2 split (train 1961–1988, val 1989–1993, test 1994–2000):

1. **Models:** the members, Weight Network and BGNet are trained or loaded by `run_cordex_v2.run_repetition`, with the same checkpoint names, signatures and seeds as `experiments/run_cordex_v2.py`.
   - Members and BGNet early-stop with patience 4 and keep training while still improving, up to 300 epochs ("converged").
   - With `mass.lambda0 = "original"` (the NZ default), **λ = 0 is exactly the current NZ v2 QRE.** The runner checks that its λ = 0 test MSEs equal `run_repetition`'s, and a test compares them with a separate `run_cordex_v2.py` run to the last digit.
2. **Mass term:** the India module (`src/mass_conservation.py`), with the same formula, absolute mode by default, the same placements (`mass.placement`: `ensemble` default, `members`, `both`) and the λ list `[0, 0.01, 0.1, 1]` with initial-ratio scaling. λ is selected on validation only, with the same rule as India; the test split is evaluated once, afterwards.
3. **Cell areas:**
   - **Grid:** the target files carry real 1-D latitude/longitude on a regular lat/lon grid (128 × 128, 0.10733° spacing, uniform to 1.4e-4). It is CCAM output regridded to lat/lon, not a rotated-pole or projected grid. The `lat_bnds`/`lon_bnds` variables named in the file attributes are not in the files.
   - **Edges and areas:** `src/grid_areas.py` takes cell edges as the midpoints between centres and computes $A_i = R^2\,\Delta\lambda\,(\sin\varphi_\text{north} - \sin\varphi_\text{south})$, about 106 km² per cell.
   - **Mask:** areas are restricted to the 2,804 land cells (`orog > 0`), in the order of the target columns.

**Outputs** (`<save_dir>/`):

- `comparison.md` / `comparison.json`: one table with BGNet (converged), QRE (λ = 0) and QRE + mass (selected λ). Columns: MSE on [0, 1], [0, 0.2] and [0.9, 1], MAE, RMSE, spatial correlation, area-weighted daily-total error over the land cells, and Weight Network accuracy and collapse. The table also has the mid-term numbers from `results/final_cordex_qre/` as **reference rows**, which used a different split (train 1961–1975, test 1981–2000), the pipeline before the fixes, and 3 repetitions.
- `sweep/`: the validation results and the λ selection.
- `repetition_0.json`: the NZ v2 run, including bagging, probability and BGNet(-).
- `loss_scales.json`.
- `plots/`: the same plots as India, labelled "observed (CORDEX)".

**The mid-term model on the v2 split.** The mid-term runner (tag `midterm-eval`) hard-codes its split, so it cannot be rerun on the v2 split with a config alone. `configs/cordex_midterm_on_v2_split.json` is the closest approximation: it runs `run_cordex_v2.py` with every fix switched off and the mid-term hyperparameters. The differences that remain are listed in its `_note`, mainly that the BGNet baseline uses the members' architecture and that BGNet(-) is trained (the mid-term skipped it). It is not run.

```powershell
python experiments/nz_mass.py --set 'checkpoints.dir="checkpoints/nz_mass"' --save-dir results/fixes_v2/nz_mass
```

---

## 10. Scope

This is QRE implemented on alternative datasets with documented adjustments. It is not a numerical reproduction of the paper's ERA5 → VCSN experiment, and absolute MSE values are not comparable with the paper's tables.
