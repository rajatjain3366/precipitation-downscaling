# PHASE 15 — FINAL EXPERIMENT DESIGN

## 1. Executive Summary & Research Goal

The objective of this final BTP experiment is to execute a rigorous, scientifically defensible, and fully auditable implementation of the **Quantile-Regression-Ensemble (QRE)** downscaling algorithm introduced by Bailie et al. (AAAI 2024).

Because the original research datasets (ERA5 reanalysis predictors and VCSN rain-gauge observational target) are no longer publicly available, this experiment is conducted on the author-recommended **CORDEX-ML-Bench** dataset (New Zealand domain, ACCESS-CM2 simulation).

---

## 2. Scientific Classification of Experimental Components

```
+---------------------------------------------------------------------------------------------------+
|                                 EXPERIMENT TAXONOMY & AUDIT TRAIL                                 |
+------------------------------------+------------------------------------+-------------------------+
| [1] PAPER-FAITHFUL                 | [2] ALTERNATIVE-DATA               | [3] COMPUTATIONAL       |
|                                    |                                    |     REDUCTIONS          |
+------------------------------------+------------------------------------+-------------------------+
| - QRE ensemble formulation         | - CORDEX-ML-Bench dataset          | - N = 6 specialists     |
| - BGNet architecture (Ch-Attention)| - 4 atmospheric channels (850 hPa) | - 3 repetitions         |
| - GammaLoss (shape, scale, p0)     | - w850 unavailable & omitted       | - 730 train / 365 val / |
| - GMM spatial sum clustering       | - 16x16 predictor grid             |   365 test days         |
| - Omega network & OmegaLoss (EMD)  | - 128x128 full target domain       | - CPU execution         |
| - Dynamic Softmax aggregation      | - D_land = 2,418 valid points      | - Batch size = 16       |
| - Evaluation intervals [0,1],[0,0.2| - Static land-sea boolean mask     | - Early stopping        |
|   and [0.9, 1.0]                   |                                    |   patience: 5 (BGNet)   |
| - Wilcoxon signed-rank testing     |                                    |             3 (Omega)   |
+------------------------------------+------------------------------------+-------------------------+
```

---

## 3. Detailed Component Breakdown

### A. PAPER-FAITHFUL COMPONENTS
1. **QRE Formulation**: Non-linear ensemble where $N$ specialist deep networks $\hat{Y}_i(X)$ are dynamically weighted by an Earth Mover's Distance-trained weight network $\omega_i(X)$:
   $$\hat{Y}_{\text{QRE}}(X) = \sum_{i=1}^N \omega_i(X) \cdot \hat{Y}_i(X)$$
2. **BGNet Architecture**: Bernoulli-Gamma CNN architecture with Channel Attention (Squeeze-and-Excitation with GlobalAveragePooling and GlobalMaxPooling), producing 3 physical parameters per grid cell: shape ($\alpha$), scale ($\beta$), and probability of zero precipitation ($p_0$).
3. **GammaLoss**: Numerically stable zero-inflated Gamma negative log-likelihood loss:
   $$\mathcal{L}_{\text{Gamma}}(y; \alpha, \beta, p_0) = -\sum_{j} \left[ \mathbb{I}_{y_j = 0} \ln(p_{0,j}) + \mathbb{I}_{y_j > 0} \left( \ln(1 - p_{0,j}) + \alpha_j \ln \beta_j - \ln \Gamma(\alpha_j) + (\alpha_j - 1)\ln y_j - \beta_j y_j \right) \right]$$
4. **GMM Partitioning**: Daily spatial precipitation sum clustering via Gaussian Mixture Model (`n_components=N, n_init=100`) to derive empirical quantile boundaries $Q_{\text{ranges}}$.
5. **Omega Network & OmegaLoss**: Multi-layer CNN with dynamic Softmax output trained with Earth Mover's Distance (EMD) loss against the binned ground-truth quantile assignment.
6. **Evaluation Regimes**: Quantile-tail evaluation across test-truth spatial precipitation distribution:
   - Overall: $[0.0, 1.0]$
   - Low Precipitation: $[0.0, 0.2]$
   - Extreme Precipitation: $[0.9, 1.0]$

### B. ALTERNATIVE-DATA COMPONENTS
1. **Dataset**: CORDEX-ML-Bench New Zealand domain (ACCESS-CM2 global climate model simulation).
2. **Predictors**: 4 atmospheric channels at 850 hPa: specific humidity (`q850`), temperature (`t850`), zonal wind (`u850`), and meridional wind (`v850`).
3. **Unavailable Channel**: Vertical velocity at 850 hPa (`w850`) is not present in CORDEX-ML-Bench and is explicitly omitted without proxy substitution.
4. **Domain Resolution**: Predictor input grid is $16 \times 16$; target grid is $128 \times 128$ ($D_{\text{land}} = 2,418$ land points).

### C. COMPUTATIONAL REDUCTIONS
1. **Specialist Count ($N$)**: $N = 6$ (Full paper-faithful number of specialists).
2. **Repetitions**: 3 independent repetitions (`seeds: 42, 1042, 2042`).
3. **Temporal Period**: 730 training days (`1961-01-01` to `1962-12-31`), 365 validation days (`1963-01-01` to `1963-12-31`), and 365 test days (`1981-01-01` to `1981-12-31`).
4. **Hardware**: CPU execution (Intel AVX/AVX2 enabled, oneDNN optimized).

---

## 4. Runtime & Computational Feasibility Scaling Analysis

Based on Phase 14B empirical benchmarks (3 repetitions with $N=3$ took 323.59 seconds, avg ~107.8 s per repetition):

| Configuration | Specialists ($N$) | Repetitions | Total Models Trained | Estimated Runtime | Feasibility Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Config A (Selected)** | **6** | **3** | **24 models** | **~8.5 minutes** | **Feasible, robust, complete N=6 paper structure** |
| **Config B** | 6 | 10 | 80 models | ~28.8 minutes | Feasible but extended execution |
| **Config C** | 6 | 40 | 320 models | ~1.92 hours | Long CPU runtime |

**Design Choice**: **Configuration A ($N=6$, 3 repetitions)** is chosen for the final experiment because it implements the exact paper-faithful specialist count ($N=6$) with multi-repetition variance and Wilcoxon statistical testing within a verified ~8-10 minute CPU budget.

---

## 5. Comparative Model Set & Baseline Audit

| Model Name | Inclusion Status | Implementation Rationale |
| :--- | :---: | :--- |
| **QRE (Full Ensemble)** | **INCLUDED** | Full non-linear adaptive ensemble with Omega Softmax weights |
| **Single BGNet** | **INCLUDED** | Single unpartitioned BGNet trained across full training set |
| **Bagging Ensemble (`cqre`)** | **INCLUDED** | Uniform weighting ($\omega_i = 1/N$) across the $N=6$ specialists |
| **Probability Ensemble (`pqre`)** | **INCLUDED** | Prior probability weighting ($\omega_i = \Delta Q_i$) across specialists |
| **Empirical Mean** | **INCLUDED** | Spatial climatological mean precipitation vector |
| **BG-Net(-) Ablation** | **EXCLUDED** | **Technical Reason**: The original `bmodule` in `ConvolutionalNetworks.py` contains consecutive `MaxPooling2D((2,2))` layers without padding designed for $36 \times 41$ ERA5 inputs. On $16 \times 16$ CORDEX inputs, spatial dimensions reduce to $2 \times 2$, causing a $3 \times 3$ Conv2D layer on block 3 to fail. In accordance with strict guidelines, the original architecture is NOT modified merely to force a baseline to execute. |

---

## 6. Paper vs Final CORDEX Experiment Comparison Matrix

| Property | AAAI 2024 Paper | Final CORDEX Experiment | Scientific Classification |
| :--- | :--- | :--- | :--- |
| **Dataset Source** | ERA5 + VCSN (Observed rain gauges) | CORDEX-ML-Bench (ACCESS-CM2 GCM) | `ALTERNATIVE-DATA` |
| **Predictor Channels** | `q850`, `t850`, `w850`, `u850`, `v850` (5 ch) | `q850`, `t850`, `u850`, `v850` (4 ch) | `ALTERNATIVE-DATA` (`w850` omitted) |
| **Input Grid** | $36 \times 41$ | $16 \times 16$ | `ALTERNATIVE-DATA` |
| **Target Grid** | $36 \times 41$ ($D = 1,476$ points) | $128 \times 128$ ($D_{\text{land}} = 2,418$ points) | `ALTERNATIVE-DATA` |
| **Specialist Count ($N$)** | $N = 6$ | $N = 6$ | `PAPER-FAITHFUL` |
| **Loss Function** | GammaLoss + OmegaLoss (EMD) | GammaLoss + OmegaLoss (EMD) | `PAPER-FAITHFUL` |
| **Ensemble Aggregation** | Dynamic Softmax | Dynamic Softmax | `PAPER-FAITHFUL` |
| **Training Period** | 32 years (1979–2010) | 2 years (1961–1962) | `COMPUTATIONAL-REDUCTION` |
| **Validation Period** | 6 years (2011–2016) | 1 year (1963) | `COMPUTATIONAL-REDUCTION` |
| **Test Period** | 8 years (2017–2024) | 1 year (1981) | `COMPUTATIONAL-REDUCTION` |
| **Repetitions** | 40 | 3 | `COMPUTATIONAL-REDUCTION` |
| **Hardware** | GPU Cluster | CPU (oneDNN AVX2) | `COMPUTATIONAL-REDUCTION` |
| **Evaluation Metrics** | Overall, Low-rain, Extreme MSE/MAE | Overall, Low-rain, Extreme MSE/MAE | `PAPER-FAITHFUL` |
| **Statistical Test** | Paired Wilcoxon signed-rank test | Paired Wilcoxon signed-rank test | `PAPER-FAITHFUL` |
