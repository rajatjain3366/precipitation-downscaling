# Quantile-Regression-Ensemble (QRE) for Precipitation Downscaling

> **Bachelor Thesis Project (BTP)**
> **Based on the AAAI-24 paper**: *"Quantile-Regression-Ensemble: A Deep Learning Algorithm for Downscaling Extreme Precipitation"* (Bailie, Koh, Rampal & Gibson)
> **This repository**: a reproduction on alternative data (CORDEX-ML-Bench, New Zealand)

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
|---|---|
| Predictors $X$ | q850, t850, u850, v850 on a 16 × 16 grid (ACCESS-CM2 "perfect" predictors). w850 is not in the benchmark. |
| Target $Y$ | Daily precipitation `pr` (regional climate model output) on a 128 × 128 grid, flattened to the **2,804** land cells (`sftlf > 0.5`) |
| Static input | Surface elevation `orog` from `Static_fields.nc` |
| Period | 1961–2000 (two files: 1961–1980 and 1981–2000), 365-day calendar |

Sources:

- [Zenodo record 17957264](https://zenodo.org/records/17957264) (v4, `NZ_domain.zip`, md5 `a3f586f28d78b3a4c53b7b8324db4e62`) for all files except the 1981–2000 target. v4 does not contain any test targets.
- `test/historical/target/pr_tasmax_ACCESS-CM2_1981-2000.nc` from [Zenodo record 20985924](https://zenodo.org/records/20985924) (v7), md5 `15b0007f910b0c72c632cc37409d17f1`. Every file that exists in v4 is byte-identical in v7 (same CRC32 and size).
- Check: the v4 1961–1975 data reproduces the GMM intervals of our first run exactly ([results/baseline_rerun/data_match_check.json](results/baseline_rerun/data_match_check.json)).

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

The same conv backbone, then 2 × Dense 100 (dropout 0.2) and an $N$-way softmax. The paper trains it with the EMD loss (Eq. 5–6, $p = 1$): $\mathcal{L}_\omega = \sum_{k} |k - k'|\,\omega_k$, where $k'$ is the true interval. On CORDEX NZ this loss collapses the network onto one interval, so v2 uses cross-entropy instead (see §4 and §5). The other losses can be selected with `weight_network.loss`.

---

## 4. Deviations from the paper

| Item | Paper | Ours | Reason |
|---|---|---|---|
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
| Weight Network loss | EMD, Eq. 6: $\sum_k \lvert k-k'\rvert\,\omega_k$ | **Cross-entropy** $-\log \omega_{k'}$ | Eq. 6 collapsed on NZ: validation argmax histogram **[0, 0, 0, 1825, 0, 0]**, mean max weight 1.000, best epoch 1. The loss is linear in $\omega$, so its gradient through the softmax vanishes once $\omega$ is one-hot. Cross-entropy, squared EMD on CDFs (Hou et al. 2016) and CE + EMD all fix the collapse. Cross-entropy was chosen by validation QRE MSE (§5). Old behaviour: `weight_network.loss = "emd"`. |
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
|---|---|---|---|
| QRE | 112.47 ± 9.45 | 27.86 ± 15.06 | 579.69 ± 80.66 |
| Single BGNet | 62.03 ± 0.22 | 4.28 ± 0.35 | 251.72 ± 3.78 |
| Bagging | 106.96 ± 5.82 | 43.22 ± 8.84 | 494.99 ± 19.88 |
| Probability | 105.48 ± 4.08 | 24.67 ± 2.85 | 550.61 ± 19.50 |

### After fixes (v2), `results/fixes_v2/`

NZ sanity run ([results/fixes_v2/nz/sanity/](results/fixes_v2/nz/sanity/)): 1 repetition, new split, all fixes. Epochs were cut to 10 for the members and both BGNets and 30 for the Weight Network, so **every model is undertrained** (all stopped at their epoch cap while still improving). Test set is 1994–2000, so the numbers are not comparable with the first run's 1981–2000 test set.

| Model (test, 1994–2000) | MSE [0, 1] | MSE [0, 0.2] | MSE [0.9, 1] |
|---|---|---|---|
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
|---|---|---|---|---|---|
| Ceiling: interval of BGNet's predicted daily total (no training) | 0.627 | 0.984 | — | 1 (one-hot) | 117.88 / 2.36 / 583.0 |
| Perfect Member: true interval (no training) | 1.000 | 1.000 | — | 1 (one-hot) | 116.74 / 1.89 / 585.1 |
| EMD, Eq. 6 (paper) | 0.175 | 0.530 | [0, 0, 0, 1825, 0, 0] | 1.000 | 134.93 / 14.76 / 693.2 |
| **Cross-entropy** | 0.649 | 0.982 | [243, 395, 405, 326, 242, 214] | 0.683 | **116.55** / 2.02 / 580.2 |
| Squared EMD on CDFs | 0.656 | 0.980 | [229, 438, 349, 329, 267, 213] | 0.685 | 117.04 / 1.99 / 581.5 |
| CE + EMD (1 : 1) | 0.654 | 0.986 | [196, 426, 407, 322, 268, 206] | 0.768 | 116.96 / 2.03 / 581.8 |
| *Single BGNet (for reference)* | | | | | *60.82 / 2.42 / 229.9* |

Confusion matrices (rows are the true interval) are in `results.json`. The three new losses all give a near-diagonal matrix with almost all errors one interval away. They beat the BGNet-ceiling accuracy slightly, and none of them collapses.

Selection (validation QRE MSE [0, 1]) picked **cross-entropy**. The three new losses differ by less than 0.5 MSE, which is well within what one repetition can resolve, so the ranking among them is not meaningful. Only after selection was the test set evaluated:

| Model (test, 1994–2000) | MSE [0, 1] | MSE [0, 0.2] | MSE [0.9, 1] |
|---|---|---|---|
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
|---|---|---|
| `data.train_years` / `val_years` / `test_years` | 1961–1988 / 1989–1993 / 1994–2000 | 1961–1975 / 1976–1980 / 1981–2000 |
| `data.standardize_predictors` | `true` | `false` |
| `data.use_elevation` | `true` | `false` |
| `loss.rain_threshold` | `0.1` | `0.5` |
| `prediction.mode` | `"expected"` ($p\alpha\beta$) | `"threshold"` |
| `segmentation.method` | `"bayesian_gmm"` | `"legacy"` |
| `members.validate_on_own_segment` | `true` | `false` |
| `weight_network.loss` | `"cross_entropy"` (also `"emd_squared_cdf"`, `"ce_plus_emd"`) | `"emd"` (paper Eq. 6) |
| `weight_network.emd_reduction` | `"sum"` | `"mean"` |
| `architecture.*`, learning rates, decay, patience | paper values where possible | see `configs/cordex_final.json` |

---

## 7. Repository layout

```text
configs/
  cordex_v2.json               v2 configuration (all fixes)
  cordex_final.json            configuration of the first run
experiments/
  run_cordex_v2.py             v2 pipeline (train-only statistics, flags for every fix)
  run_cordex_experiment.py     legacy runner (first run)
  weight_network_losses.py     Weight Network loss study on saved checkpoints
src/
  data_adapter_cordex.py       NetCDF loading, land mask, elevation, 1961-2000 loader
  preprocessing.py             year split, training-split standardization
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
```

---

## 8. Scope

This is QRE implemented on an alternative dataset with documented adjustments. It is not a numerical reproduction of the paper's ERA5 → VCSN experiment, and absolute MSE values are not comparable with the paper's tables.
