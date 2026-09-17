# Phase 9: Real-Data BGNet Specialist Training Smoke Test Report

**Project**: Quantile-Regression-Ensemble (QRE) Reproduction (AAAI 2024)  
**Date**: September 16, 2026  
**Test Script**: [tests/test_real_data_specialist_smoke.py](file:///c:/BTP/tests/test_real_data_specialist_smoke.py)  
**Status**: **ALL SPECIALIST TRAINING & GRADIENT VERIFICATIONS PASSED**

> [!IMPORTANT]
> **CRITICAL REMINDER**: This test was performed **strictly for software, numerical stability, and gradient-update verification** on real CORDEX-ML-Bench data. It is **NOT** a full research reproduction, and loss values/MSE must **NOT** be compared against the published AAAI 2024 paper results.

---

## A. Dataset Subset
- **Dataset**: CORDEX-ML-Bench (*New Zealand Domain, ACCESS-CM2 climate model run*).
- **Predictor Variables**: Configuration A (4-channel core at 850 hPa: $q_{850}, t_{850}, u_{850}, v_{850}$).
- **Vertical Velocity ($w_{850}$)**: **[CONFIRMED]** Unavailable in CORDEX-ML-Bench and intentionally omitted.
- **Target Variable**: Daily precipitation (`pr`).
- **Target Representation**: Land-masked 1D vector corresponding to dynamic land grid points derived from `static.nc` ($D_{\text{land}} = 2,418$).
- **Temporal Slice**: Small 5-day training slice (`2000-01-01` to `2000-01-05`), 2-day test/validation slice (`2000-01-06` to `2000-01-07`).

---

## B. Number of Samples
- **Training Samples**: $N_{\text{train}} = 5$ daily timesteps.
- **Validation/Test Samples**: $N_{\text{val}} = 2$ daily timesteps.
- **Total Ingested Timesteps**: $7$ daily timesteps (small slice loaded without full-dataset memory footprint).

---

## C. Input Statistics
- **Input Tensor Dimensions**: `(5, 16, 16, 4)` (Batch, Height, Width, Channels).
- **Channel 0 ($q_{850}$)**: Range $[0.0039, 0.0116]\text{ kg/kg}$, Mean $0.0076\text{ kg/kg}$.
- **Channel 1 ($t_{850}$)**: Range $[273.68, 287.96]\text{ K}$, Mean $281.33\text{ K}$.
- **Channel 2 ($u_{850}$)**: Range $[-13.20, 21.05]\text{ m/s}$, Mean $4.18\text{ m/s}$.
- **Channel 3 ($v_{850}$)**: Range $[-15.76, 17.52]\text{ m/s}$, Mean $0.43\text{ m/s}$.
- **NaN / Infinite Checks**: **[CONFIRMED]** Zero NaNs, zero Infs (100% finite values).

---

## D. Target Statistics
- **Target Tensor Dimensions**: `(5, 2418)` (Batch, Land Grid Points).
- **Total Evaluated Points**: $5 \times 2,418 = 12,090$ point values.
- **Precipitation Value Range**: $[0.00, 58.71]\text{ mm/day}$.
- **Overall Mean $\pm$ Std**: $5.01 \pm 5.05\text{ mm/day}$.
- **Dry / Zero Target Count**: $0\text{ points}\ (0.00\%)$ in this active storm sample slice.
- **Wet / Positive Rain Points**: $12,090\text{ points}\ (100.00\%)$.
- **Positive Rain Mean**: $5.01\text{ mm/day}$.
- **Maximum Target Precipitation**: $58.71\text{ mm/day}$ (Real heavy-tail precipitation event present).
- **NaN / Infinite Checks**: **[CONFIRMED]** Zero NaNs, zero Infs.

---

## E. Intensity & Partition Information
- **Partitioning Algorithm**: `data_between(Y, q_low, X, q_high)` using cumulative daily spatial sums $\sum_{d} Y_{t, d}$.
- **Target Quantile Range**: Lower $50\%$ intensity regime ($q_{\text{low}} = 0.0, q_{\text{high}} = 0.50$).
- **Partitioned Sample Count**: $3$ daily timesteps selected out of $5$.
- **Partitioned Predictor Shape**: `(3, 16, 16, 4)`.
- **Partitioned Target Shape**: `(3, 2418)`.
- **Integrity Verifications**:
  - Subset non-empty: **[CONFIRMED]** (3 samples).
  - Target dimension preserved: **[CONFIRMED]** ($D = 2,418$).
  - Input dimension preserved: **[CONFIRMED]** ($16 \times 16 \times 4$).
  - No unintended duplication or NaNs: **[CONFIRMED]**.

---

## F. Specialist Configuration
- **Architecture**: `BGNet` with dynamic output dimension $D_{\text{land}} = 2,418$.
- **Output Heads**: 3 parallel output heads ($p$: rain probability, $\alpha$: Gamma shape, $\beta$: Gamma rate).
- **Total Trainable Parameter Tensors**: $34$ tensors ($4,272,398$ parameters, $\approx 16.3\text{ MB}$).

---

## G. Training Configuration
- **Device**: CPU (Intel / AMD64 on Windows).
- **Optimizer**: Adam ($\text{lr} = 0.001$).
- **Loss Function**: Original research `GammaLoss` ($-\log \mathcal{L}_{\text{Bernoulli-Gamma}}$).
- **Batch Size**: $4$.
- **Epochs**: $5$.

---

## H. Loss per Epoch
- **Initial Loss (Pre-training)**: $5.779322$
- **Epoch 1**: $5.773839$
- **Epoch 2**: $5.603584$
- **Epoch 3**: $5.300069$
- **Epoch 4**: $4.929462$
- **Epoch 5**: $4.736359$
- **Final Post-Training Loss**: $3.960381$
- **Total Training Time**: $\approx 0.65\text{ seconds}$ on CPU.

---

## I. Gradient Status
- **Initial Total Gradient Norm**: $13.240394$
- **Gradient Validity**: **[CONFIRMED]** Finite and non-zero across all 34 trainable weight tensors.
- **NaN / Inf Gradients**: None detected.

---

## J. Weight-Update Verification
- **Total Trainable Tensors**: $34$.
- **Tensors with Non-Zero Parameter Change ($\Delta W > 0$)**: $34 / 34$ ($100.0\%$).
- **Maximum Parameter Delta ($\max |\Delta W|$)**: $5.014853 \times 10^{-3} > 0.0$.
- **Status**: **[CONFIRMED]** Backpropagation successfully updated all convolutional, attention, dense, and parameter output layers.

---

## K. Prediction Shapes
- **Raw `BGNet(X_val)` Output Shape**: `(2, 2418, 3)` (Batch, Grid Points, $[p, \alpha, \beta]$).
- **`BGCallWrapper(X_val)` Output Shape**: `(2, 2418)` (Batch, Grid Points, expected precipitation $p \times \frac{\alpha}{\beta}$).
- **Shape Preservation**: **[CONFIRMED]** matches expected 2D spatial vector convention.

---

## L. Prediction Statistics
- **Rain Probability $p$ Range**: $[0.2591, 0.8083] \subseteq [0, 1]$ (**[CONFIRMED]** strictly in valid probability bounds).
- **Alpha ($\alpha$) Parameter Values**: Positive and finite.
- **Beta ($\beta$) Parameter Values**: Positive and finite.
- **Wrapper Expected Precipitation**: Range $[0.37, 7.85]\text{ mm/day}$, Mean $2.41\text{ mm/day}$.
- **Finiteness**: **[CONFIRMED]** 100% finite (no NaNs or Infs).

---

## M. Numerical Stability
- **Assessment**: **[CONFIRMED]** `GammaLoss` executed with complete numerical stability on real CORDEX-ML-Bench precipitation data.
- **Zero/Positive Handling**: No log-of-zero errors, no division-by-zero, and no underflow/overflow in the Gamma log-likelihood formulation.

---

## N. Basic MSE Sanity Check
- **Evaluation Dataset**: $2$-day validation slice ($2 \times 2,418 = 4,836$ point predictions).
- **Observed Sanity MSE**: $34.6752\text{ mm}^2/\text{day}^2$.
- **Methodological Note**: **[CONFIRMED]** Calculated solely as a software execution verification; must **NOT** be interpreted scientifically or compared against paper metrics.

---

## O. PASS / FAIL Summary Table

| Test / Verification Item | Target Criteria | Observed Result | Status |
| :--- | :--- | :--- | :--- |
| **Real Data Ingestion** | No NaNs, small slice loaded | `X: (5, 16, 16, 4), Y: (5, 2418)` | **PASS** |
| **Quantile Slicing** | Non-empty subset, shape preserved | `X_spec: (3, 16, 16, 4), Y_spec: (3, 2418)` | **PASS** |
| **Gradient Flow** | Finite, non-zero gradient norm | Grad Norm: `13.240394` (34 tensors) | **PASS** |
| **Specialist Training** | Stable loss computation over 5 epochs | Loss: $5.7793 \to 3.9604$ | **PASS** |
| **Weight Update** | All trainable tensors updated | 34 / 34 tensors updated ($\Delta W = 5.01 \times 10^{-3}$) | **PASS** |
| **Prediction Shapes** | BGNet `(2, 2418, 3)`, Wrapper `(2, 2418)` | Exact shape match | **PASS** |
| **Parameter Finiteness** | $p \in [0, 1]$, $\alpha > 0, \beta > 0$ finite | $p \in [0.259, 0.808]$, all finite | **PASS** |
| **Numerical Stability** | No NaNs or crash on real rain distribution | Zero NaNs across loss, grads, preds | **PASS** |

---

## P. Files Modified
- **New Test Script**: [tests/test_real_data_specialist_smoke.py](file:///c:/BTP/tests/test_real_data_specialist_smoke.py)
- **Report**: [PHASE9_REAL_DATA_SPECIALIST_SMOKE_TEST.md](file:///c:/BTP/PHASE9_REAL_DATA_SPECIALIST_SMOKE_TEST.md)
- **Original Source Files**: **Zero modifications**. Original research files ([ConvolutionalNetworks.py](file:///c:/BTP/ConvolutionalNetworks.py), [GammaLoss.py](file:///c:/BTP/GammaLoss.py), [quantiles.py](file:///c:/BTP/quantiles.py), [Modules.py](file:///c:/BTP/Modules.py), [data_handling.py](file:///c:/BTP/data_handling.py)) remain pristine.

---

## Q. Implementation Issues
- **None**. The CORDEX-ML-Bench real data seamlessly passed through the existing `data_between` quantile partitioning, `BGNet` forward pass, `GammaLoss` backpropagation, and `BGCallWrapper` inference.

---

## R. Scientific Limitations
1. **Alternative Dataset**: **[CONFIRMED]** Evaluated on CORDEX-ML-Bench (ACCESS-CM2 GCM downscaled over New Zealand), not the original ERA5 $\to$ VCSN observational dataset which is no longer public.
2. **Predictor Omission**: **[CONFIRMED]** w850 (vertical velocity) is absent from CORDEX-ML-Bench; Configuration A ($q_{850}, t_{850}, u_{850}, v_{850}$) is used.
3. **Smoke Test Scale**: **[CONFIRMED]** Tested on $N_{\text{train}} = 5$ days for 5 epochs on CPU for software verification only.

---

> [!NOTE]
> **Status**: Ready for review. Real-data specialist training is verified. Weight Network ($\Omega$) training has **NOT** been started. Execution is paused awaiting user approval.
