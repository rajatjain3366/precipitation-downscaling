# Phase 8: CORDEX Real-Data Adapter Report

**Project**: Quantile-Regression-Ensemble (QRE) Reproduction (AAAI 2024)  
**Date**: September 16, 2026  
**Implementation File**: [data_adapter_cordex.py](file:///c:/BTP/data_adapter_cordex.py)  
**Verification Script**: [tests/test_cordex_adapter.py](file:///c:/BTP/tests/test_cordex_adapter.py)  
**Status**: **ALL ADAPTER & REAL-DATA FORWARD PASS TESTS PASSED**

---

## 1. Selected Dataset & Exact Files Used

- **Selected Dataset**: CORDEX-ML-Bench (*WCRP-CORDEX Regional Climate Downscaling Benchmark, Zenodo Record 17957264 / 20985924*).
- **Target Domain**: New Zealand ($0.11^\circ$ resolution, $16 \times 16$ coarse predictor grid $\to 128 \times 128$ high-resolution target grid).
- **Exact Files Handled by Adapter**:
  1. `train/ESD_pseudo-reality/predictors/ACCESS-CM2_1961-1980.nc` (Historical predictors)
  2. `train/ESD_pseudo-reality/target/pr_tasmax_ACCESS-CM2_1961-1980.nc` (Historical target precipitation & temperature)
  3. `train/ESD_pseudo-reality/static.nc` (Static surface orography and land-sea mask)
  4. `test/historical/predictors/perfect/ACCESS-CM2_1981-2000.nc` (Evaluation predictors)
  5. `test/historical/target/pr_tasmax_ACCESS-CM2_1981-2000.nc` (Evaluation target precipitation)

---

## 2. Predictor Variable Selection & Omission of $w_{850}$

- **Selected Predictor Suite (Configuration A)**:
  - $q_{850}$: Specific humidity at 850 hPa
  - $t_{850}$: Air temperature at 850 hPa
  - $u_{850}$: Zonal wind component at 850 hPa
  - $v_{850}$: Meridional wind component at 850 hPa
- **Omission of Vertical Velocity ($w_{850}$)**:
  - **[CONFIRMED]** The original AAAI 2024 paper used 5 variables at 850 hPa ($q, t, w, u, v$).
  - **[CONFIRMED]** In CORDEX-ML-Bench, vertical velocity $w_{850}$ is **unavailable**.
  - **[CONFIRMED]** As instructed, geopotential height $z_{850}$ was **NOT substituted** for $w_{850}$. Configuration A strictly isolates the 4-channel core physical subset ($q, t, u, v$ at 850 hPa), explicitly recording the omission of $w_{850}$ as a dataset constraint.
- **Predictor Tensor Output**: `(batch, 16, 16, 4)`.

---

## 3. Target Spatial Representation & Dynamic Land-Mask Construction

- **Target Precipitation Variable**: `pr` (daily precipitation in mm/day).
- **Land Mask Derivation**:
  - Derived dynamically from `static.nc` using surface altitude (`orog > 0.0`) and land area fraction (`sftlf > 0.5`).
  - Shape: $(128, 128)$ boolean array.
  - **Time Invariance**: **[CONFIRMED]** Static land mask is fixed across all training and testing timesteps.
- **Dynamic Count of Valid Land Points ($D_{\text{land}}$)**:
  - Derived dynamically: $D_{\text{land}} = 2,418$ land cells (out of $128 \times 128 = 16,384$ total domain cells, $\approx 14.8\%$ land coverage).
  - Target Tensor Output: `(batch, D_land)` $\to$ `(batch, 2418)`.
- **Reconstructibility**:
  - `unflatten_target_cordex(y_flat, mask, H=128, W=128)` maps 1D predictions back to the full $(128, 128)$ 2D spatial field cleanly.

---

## 4. Static Topography Handling

- Extracted from `static.nc` (variable `orog`).
- Standardized across valid land cells: $(z - \mu_{\text{land}}) / \sigma_{\text{land}}$.
- Formatted as a 4D tensor of shape `(1, 128, 128, 1)` matching `BGNet`'s auxiliary elevation pathway.

---

## 5. Summary Table: Verified Tensor Shapes

| Data Element | Tensor Object | Verified Dimensions | Data Type | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **$X_{\text{train}}$ Predictors** | `np.ndarray` | `(T_train, 16, 16, 4)` | Float32 | 4-channel 850 hPa suite ($q, t, u, v$) |
| **$Y_{\text{train}}$ Targets** | `np.ndarray` | `(T_train, 2418)` | Float32 | Land-masked daily precipitation $pr$ (mm/day) |
| **$X_{\text{test}}$ Predictors** | `np.ndarray` | `(T_test, 16, 16, 4)` | Float32 | Out-of-sample evaluation predictors |
| **$Y_{\text{test}}$ Targets** | `np.ndarray` | `(T_test, 2418)` | Float32 | Out-of-sample evaluation targets |
| **Auxiliary Topography**| `np.ndarray` | `(1, 128, 128, 1)` | Float32 | Standardized orography |
| **Land Mask** | `np.ndarray` | `(128, 128)` | Boolean | Time-invariant land boundary |

---

## 6. BGNet Real-Data Forward-Pass Verification

We executed a forward pass through `BGNet` initialized with the adapted CORDEX dimensions ($H=16, W=16, C=4$, $D_{\text{land}} = 2,418$):

```
========================================================================================
BGNET FORWARD PASS VERIFICATION RESULTS
========================================================================================
- Input Tensor Shape:           (2, 16, 16, 4)
- Output Tensor Shape:          (2, 2418, 3) (p, alpha, beta per land point)
- Total Model Parameter Count:  4,268,950 parameters (~16.28 MB float32)
- Forward-Pass Runtime (CPU):   ~349.67 ms
- Probability Range (p):        [0.2491, 0.7380] (Valid Sigmoid in [0, 1])
- Alpha / Beta Values:          Finite, non-NaN, non-Inf
- BGCallWrapper Output:         (2, 2418)
- Verification Status:          PASS
```

---

## 7. Differences from Original Paper

| Aspect | Original Paper (AAAI 2024) | CORDEX Adapter Setup | Classification |
| :--- | :--- | :--- | :--- |
| **Data Nature** | Observational (ERA5 $\to$ VCSN rain gauges) | Model-as-Truth (ACCESS-CM2 $\to$ CCAM simulation) | **[CONFIRMED]** |
| **Predictor Channels** | $5$ channels ($q, t, w, u, v$ @ 850 hPa) | $4$ channels ($q, t, u, v$ @ 850 hPa; $w_{850}$ omitted) | **[CONFIRMED]** |
| **Spatial Grid (Predictors)**| $36 \times 41$ ($\sim 100\text{ km}$) | $16 \times 16$ ($\sim 200\text{ km}$) | **[CONFIRMED]** |
| **Spatial Grid (Target)** | $257 \times 241 \to 11,491$ land points | $128 \times 128 \to 2,418$ land points | **[CONFIRMED]** |
| **Target Variable** | `Rain_bc` (mm/day) | `pr` (mm/day) | **[CONFIRMED]** |

---

## 8. Rule Compliance & Current Status

- **No source code in `ConvolutionalNetworks.py`, `GammaLoss.py`, `quantiles.py`, `Modules.py`, or `run.py` was modified.**
- **No training was started.**
- **No QRE ensemble was trained.**
- The real-data adapter [data_adapter_cordex.py](file:///c:/BTP/data_adapter_cordex.py) is fully verified and self-contained.

---

I have stopped here as instructed. Please review this report and let me know how you would like to proceed.
