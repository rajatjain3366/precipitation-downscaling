# Phase 11: Final Experiment Design Audit Report

**Project**: Quantile-Regression-Ensemble (QRE) Reproduction (AAAI 2024)  
**Date**: September 16, 2026  
**Primary Reference**: AAAI 2024 Paper *"Quantile-Regression-Ensemble: A Deep Learning Algorithm for Downscaling Extreme Precipitation"* (Bailie, Koh, Rampal, Gibson)  
**Secondary Reference**: Official Repository Implementation (`C:\BTP`)  
**Alternative Dataset**: CORDEX-ML-Bench (New Zealand Domain, ACCESS-CM2 Downscaled Simulation)  

---

## 1. Executive Summary

This document establishes the definitive experimental audit comparing the AAAI 2024 research paper with the repository implementation. It categorizes all parameters into methodological constants, dataset-driven adaptations, and computational adjustments, and defines two structured experiment configurations:
1. **Configuration 1 ("Methodology-Faithful Alternative-Data Experiment")**: Full scientific fidelity on the author-recommended CORDEX-ML-Bench dataset ($N=6$, 40 repetitions, full 20-year train / 20-year test splits, Wilcoxon significance testing).
2. **Configuration 2 ("CPU-Feasible Validation Experiment")**: A scaled experiment executable on local CPU hardware ($N=3$ or $N=6$, 1–3 repetitions, 1–2 year temporal span) to evaluate downscaling performance, dynamic weighting behavior, and extreme precipitation metrics without multi-day compute bottlenecks.

---

## 2. Paper vs Repository Component Audit

The table below cross-references every experimental component from the paper against the repository code:

| Experimental Component | Paper Specification | Repository Implementation | Status |
| :--- | :--- | :--- | :--- |
| **Training Temporal Span** | 32 years (1979–2010, $\approx 11,688$ days) | Hardcoded `N_train = 1` year in `data_handling.py` (lines 158–159) | **DIFFERENT** |
| **Validation Temporal Span** | 6 years (2011–2016, $\approx 2,192$ days) | Not separated in `open_data`; relies on Keras validation split | **DIFFERENT** |
| **Testing Temporal Span** | 8 years (2017–2024, $\approx 2,922$ days) | Hardcoded `N_test = 1` year in `data_handling.py` | **DIFFERENT** |
| **Predictor Variables** | 5 channels at 850 hPa: $q, t, w, u, v$ | Attempted 5 channels, but `.w_200` hardcoded in `open_data` | **DIFFERENT** |
| **Predictor Adaptation (CORDEX)** | $w_{850}$ used in original ERA5 | $w_{850}$ unavailable in CORDEX; 4-channel core ($q, t, u, v$) selected | **CONFIRMED** |
| **Target Variable & Domain** | Daily precipitation on NZ VCSN ($\approx 11,471$ land stations, 5 km) | NZ VCSN in original; CORDEX target has $D_{\text{land}} = 2,418$ land cells ($10\text{ km}$) | **CONFIRMED** |
| **Target Land Masking** | Evaluated strictly on land grid cells | Derived dynamically from `static.nc` ($D_{\text{land}} = 2,418$) | **CONFIRMED** |
| **Number of Specialists ($N$)** | $N = 6$ intensity regimes | Hardcoded `'n_models': 2` in `run.py` metadata | **DIFFERENT** |
| **Number of Repetitions** | 40 independent runs with random seeds | Hardcoded `'repeats': 1` in `run.py` | **DIFFERENT** |
| **Specialist Architecture** | `BGNet` with Channel Attention Module (CAM) | `BGNet(baseline=False, ch_attn=True)` in `ConvolutionalNetworks.py` | **CONFIRMED** |
| **Specialist Loss Function** | Bernoulli-Gamma log-likelihood ($-\log \mathcal{L}_{\text{BG}}$) | `GammaLoss` in `GammaLoss.py` with parameter unpacking $(p, \alpha, \beta)$ | **CONFIRMED** |
| **Weight Network Architecture** | Conv2D + CAM + Dense + Softmax | `Omega` in `quantiles.py` | **CONFIRMED** |
| **Weight Network Loss** | Earth Mover's Distance / Wasserstein EMD | `OmegaLoss` in `quantiles.py` with distance matrix weighting ($p=1$) | **CONFIRMED** |
| **Ensemble Aggregation** | $g(x) = \sum_{i=1}^N \omega_i(x) f_i(x)$ | `ynetwork` in `quantiles.py` (broadcasting fix verified in Phase 4/10) | **CONFIRMED** |
| **Optimizer & Schedule** | Adam optimizer | Adam with ExponentialDecay ($\text{lr}_0 = 10^{-4}$ specialists, $5 \times 10^{-5}$ omega) | **CONFIRMED** |
| **Early Stopping** | Early stopping on validation loss | `EarlyStopping(patience=15)` for specialists, `patience=4` for omega | **CONFIRMED** |
| **Batch Size** | Unspecified in paper text | Uses default batch size 32 or whole slice | **UNKNOWN** |
| **Baselines Evaluated** | Single BGNet, Bagging (`cqre`), Probability (`pqre`), Mean Baseline | Implemented in `run.py` (`empirical_mean_baseline`, `cqre`, `pqre`) | **CONFIRMED** |
| **Evaluation Metrics** | Overall MSE, Regional MSE, Extreme Quantile MSE | `collect_metrics` in `useful_functions.py` with `(0, 1.0)`, `(0, 0.2)`, `(0.9, 1.0)` | **CONFIRMED** |
| **Statistical Significance** | Wilcoxon signed-rank test ($\alpha = 0.05$) across 40 runs | `signifcant_test` in `useful_functions.py` using `scipy.stats.wilcoxon` | **CONFIRMED** |

---

## 3. Parameter Categorization

### GROUP A — Must Remain Faithful to Paper Methodology (Scientific Constants)
These mathematical and algorithmic elements define the QRE paper contribution and must remain unmodified:
1. **QRE Formulation**: $g(x) = \sum_{i=1}^N \omega_i(x) f_i(x)$ with dynamic weights $\omega_i(x) \ge 0$ and $\sum_i \omega_i(x) = 1.0$.
2. **Loss Formulations**:
   - Specialists: Bernoulli-Gamma NLL `GammaLoss` with zero-inflation threshold ($y_{\text{thrs}} = 0.5\text{ mm/day}$ or continuous limit).
   - Weight Network: Earth Mover's Distance `OmegaLoss` penalizing ordinal misclassification between intensity bins.
3. **Model Architectures**:
   - `BGNet`: 3 convolutional blocks with Channel Attention Modules (CAM) + dense projection to $(p, \alpha, \beta)$ parameter vectors.
   - `Omega`: 3 convolutional blocks + CAM + dense layers + Softmax output over $N$ classes.
4. **Partitioning Principle**: Slicing the training set into ordered precipitation intensity regimes based on daily spatial sums $\sum_d Y_{t, d}$.
5. **Baselines**: Single BGNet without quantile segmentation, constant equal-weight ensemble (`cqre`), and fixed prior-probability ensemble (`pqre`).
6. **Evaluation Protocol**: Disaggregated evaluation over overall ($[0, 1.0]$), dry/light ($[0, 0.2]$), and extreme heavy-tail ($[0.9, 1.0]$) precipitation quantiles.

---

### GROUP B — Must Change Because of Alternative Dataset (CORDEX-ML-Bench)
These parameters are dictated by the physical structure of CORDEX-ML-Bench:
1. **Atmospheric Predictor Channels**: 4 channels ($q_{850}, t_{850}, u_{850}, v_{850}$). $w_{850}$ is unavailable in CORDEX and is formally omitted (Configuration A).
2. **Predictor Grid Dimension**: $16 \times 16$ spatial grid ($\approx 0.44^\circ$ resolution) instead of ERA5 $36 \times 41$.
3. **Target Grid Dimension & Mask**: $128 \times 128$ high-resolution grid downscaled to $D_{\text{land}} = 2,418$ land points (derived from `static.nc` orography).
4. **Target Precipitation Variable**: `pr` (daily total precipitation in $\text{mm/day}$ / $\text{kg m}^{-2}\text{ s}^{-1}$).
5. **Temporal Availability**: 
   - Historical Simulation (`ESD_pseudo-reality`): 1961–1980 ($20\text{ years} = 7,305\text{ daily timesteps}$).
   - Historical Evaluation (`historical/perfect`): 1981–2000 ($20\text{ years} = 7,305\text{ daily timesteps}$).

---

### GROUP C — Computational Feasibility Parameters
Parameters that can be adjusted to balance computational budget against statistical rigor:
1. **Number of Specialists ($N$)**:
   - Paper value: $N = 6$.
   - Feasibility alternative: $N = 3$ (smoke/quick validation) or $N = 6$ (comprehensive run).
2. **Number of Training Repetitions**:
   - Paper value: $40$ independent runs.
   - Feasibility alternative: $1$ to $5$ runs on local CPU; $40$ runs on GPU cluster.
3. **Training Sample Size (Years / Days)**:
   - Full CORDEX: $20$ years train ($7,305$ days), $20$ years test ($7,305$ days).
   - Feasibility subset: $1$ to $2$ years train ($365$–$730$ days), $1$ year test ($365$ days).

---

## 4. Specification of Two Experiment Configurations

```
+---------------------------------------------------------------------------------------------------------+
|                                    EXPERIMENT CONFIGURATION MATRIX                                      |
+------------------------------------+----------------------------------+---------------------------------+
| Dimension / Parameter              | Configuration 1 (Faithful CORDEX)| Configuration 2 (CPU Feasible)  |
+------------------------------------+----------------------------------+---------------------------------+
| Target Dataset                     | CORDEX-ML-Bench (New Zealand)    | CORDEX-ML-Bench (New Zealand)   |
| GCM Run                            | ACCESS-CM2                       | ACCESS-CM2                      |
| Atmospheric Predictors             | q850, t850, u850, v850 (4 ch)    | q850, t850, u850, v850 (4 ch)   |
| Vertical Velocity (w850)           | Omitted (unavailable)            | Omitted (unavailable)           |
| Target Variable                    | Daily Precipitation (pr)         | Daily Precipitation (pr)        |
| Target Representation              | Land-masked (D_land = 2,418)     | Land-masked (D_land = 2,418)    |
| Static Orography                   | (1, 128, 128, 1) from static.nc  | (1, 128, 128, 1) from static.nc |
| Number of Specialists (N)          | N = 6                            | N = 3 (or N = 6)                |
| Quantile Partition Method          | GMM / Spatial Sum Quantiles      | Spatial Sum Quantiles           |
| Training Period                    | 1961-01-01 to 1976-12-31 (16 yrs)| 1961-01-01 to 1962-12-31 (2 yrs)|
| Validation Period                  | 1977-01-01 to 1980-12-31 (4 yrs) | 1963-01-01 to 1963-12-31 (1 yr) |
| Test Period                        | 1981-01-01 to 2000-12-31 (20 yrs)| 1981-01-01 to 1981-12-31 (1 yr) |
| Training Timesteps                 | 5,844 days                       | 730 days                        |
| Test Timesteps                     | 7,305 days                       | 365 days                        |
| Number of Repetitions              | 40 runs                          | 3 runs (or 1 run)               |
| Optimizer                          | Adam with Exponential Decay      | Adam with Exponential Decay     |
| Specialist Initial LR              | 1e-4 (decay 0.7 every 6000 steps)| 1e-3 (or 1e-4)                  |
| Omega Initial LR                   | 5e-5 (decay 0.8 every 6000 steps)| 1e-3 (or 5e-5)                  |
| Early Stopping Patience            | Pat = 15 (Spec), Pat = 4 (Omega) | Pat = 15 (Spec), Pat = 4 (Omega)|
| Batch Size                         | 32                               | 16 or 32                        |
| Benchmark Baselines                | Single BGNet, Bagging, Prob, Mean| Single BGNet, Bagging, Prob     |
| Evaluation Metrics                 | MSE (Overall, [0, 0.2], [0.9, 1])| MSE (Overall, [0, 0.2], [0.9, 1])|
| Statistical Significance           | Wilcoxon signed-rank (p < 0.05)  | Mean +/- Std across repeats      |
| Compute Target                     | GPU Cluster (CUDA / A100)        | Local CPU (Intel / AMD64)       |
| Estimated Runtime                  | ~2-4 hours on GPU / ~3 days CPU  | ~15-30 minutes on CPU           |
+------------------------------------+----------------------------------+---------------------------------+
```

---

## 5. Evaluation Methodology & Extreme Precipitation Metrics

### How the Paper Evaluates Downscaling:
1. **Spatial Aggregation**: The domain is evaluated overall (all New Zealand land points) and stratified across geographic sub-regions (e.g., North Island, South Island, high elevation $> 400\text{ m}$).
2. **Quantile Stratification of Ground Truth ($Y_{\text{test}}$)**:
   - **Overall ($[0.00, 1.00]$)**: Average MSE across all test days and grid points.
   - **Dry / Low Rain ($[0.00, 0.20]$)**: MSE on test days within the lowest $20\%$ cumulative spatial rainfall.
   - **Extreme Heavy-Tail ($[0.90, 1.00]$)**: MSE strictly on the top $10\%$ most severe precipitation days.
   - **Out-of-Domain / Weak Learner Slices**: Evaluation within each specialist's assigned intensity regime.
3. **Statistical Hypothesis Testing**:
   - For each evaluation regime, a Wilcoxon signed-rank test is conducted across the 40 paired repetition MSEs:
     $$H_0: \text{MSE}_{\text{QRE}} \ge \text{MSE}_{\text{Baseline}} \quad \text{vs} \quad H_1: \text{MSE}_{\text{QRE}} < \text{MSE}_{\text{Baseline}}$$
   - If $p < 0.05$, QRE's improvement is reported as statistically significant.

---

## 6. Audit of Unknowns and Discrepancy Reconciliation

- **[CONFIRMED]** The mathematical structure of QRE (BGNet, GammaLoss, Omega, OmegaLoss, ynetwork) in the repository is fully faithful to the paper.
- **[CONFIRMED]** CORDEX-ML-Bench target $D_{\text{land}} = 2,418$ and 4-channel predictors ($q, t, u, v$) operate stably in the verified pipeline.
- **[DIFFERENT]** The repository defaults (`'n_models': 2`, `'repeats': 1`, `N_train = 1`, `cnst_base = True`) were temporary development configurations and must be replaced by the formal configuration file.
- **[UNKNOWN]** Exact batch size used in paper training (32 is standard and inferred from Keras defaults).

---

> [!IMPORTANT]
> **Next Step**: An experiment configuration artifact [EXPERIMENT_CONFIG.md](file:///c:/BTP/EXPERIMENT_CONFIG.md) has been created.
> No training has been started. No modifications have been made to `run.py`.
> Awaiting user review and approval before connecting the configuration.
