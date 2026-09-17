# Phase 7E: Dataset Selection & Minimal Real-Data Adapter Design

**Project**: Quantile-Regression-Ensemble (QRE) Reproduction (AAAI 2024)  
**Date**: September 16, 2026  
**Context**: Factual evaluation of author-recommended datasets (Dataset 1: Zenodo 13755688 vs Dataset 2: Zenodo 17957264 / 20985924), target spatial representations, predictor configurations, and conceptual design of a minimal data adapter.

---

## 1. Factual Criteria Comparison Table

| Evaluation Criterion | Dataset 1: AI-RCME (Zenodo 13755688) | Dataset 2: CORDEX-ML-Bench (Zenodo 17957264) |
| :--- | :--- | :--- |
| **1. Predictor Variables** | **[CONFIRMED]** $u, v, t, q$ at 850 & 500 hPa ($8$ channels total). | **[CONFIRMED]** $u, v, q, t, z$ at 850, 700, 500 hPa + static orography ($16$ channels total). |
| **2. 850 hPa Core Suite ($q, t, u, v$)** | **[CONFIRMED]** Available ($q_{850}, t_{850}, u_{850}, v_{850}$). | **[CONFIRMED]** Available ($q_{850}, t_{850}, u_{850}, v_{850}$). |
| **3. Vertical Velocity ($w_{850}$)** | **[CONFIRMED]** **UNAVAILABLE**. | **[CONFIRMED]** **UNAVAILABLE**. |
| **4. Precipitation Target** | **[CONFIRMED]** Daily precipitation variable `pr` (CCAM RCM, mm/day). | **[CONFIRMED]** Daily precipitation variable `pr` (CCAM RCM, mm/day) and `tasmax`. |
| **5. Target Spatial Structure** | **[CONFIRMED]** $12\text{ km}$ regular 2D grid covering NZ ($165^\circ\text{E} - 184^\circ\text{W}, 33^\circ\text{S} - 51^\circ\text{S}$). | **[CONFIRMED]** $0.11^\circ$ ($\sim 10\text{ km}$) regular 2D grid ($128 \times 128$ cells). |
| **6. Valid Land / Target Mask** | **[INFERRED]** Land mask stored in companion GitHub repository. | **[CONFIRMED]** Included in `static.nc` (surface orography and land/sea mask). |
| **7. Static Topography Availability**| **[INFERRED]** Stored in companion GitHub repository. | **[CONFIRMED]** `static.nc` directly packaged in domain folder. |
| **8. Temporal Continuity & Split** | **[CONFIRMED]** Historical (1960–2014) and Future SSP370 (2044–2099) in continuous multi-decade files. | **[CONFIRMED]** Separated into standardized benchmark splits: Train (1961–1980, 20 yrs), Test (1981–2000, 20 yrs). |
| **9. Train / Test Separation** | **[INFERRED]** Requires manual temporal slicing by user. | **[CONFIRMED]** Pre-split into dedicated directories (`train/ESD_pseudo-reality/`, `test/historical/`). |
| **10. Tensor Construction** | **[CONFIRMED]** Predictors $1.5^\circ \to$ Target $12\text{ km}$. | **[CONFIRMED]** Predictors $16 \times 16 \to$ Target $128 \times 128$ ($8\times$ resolution jump). |
| **11. QRE Code Compatibility** | **[CONFIRMED]** BGNet, GammaLoss, OmegaLoss, Omega, ynetwork mathematically compatible. | **[CONFIRMED]** BGNet, GammaLoss, OmegaLoss, Omega, ynetwork mathematically compatible. |
| **12. Additional Preprocessing** | **[INFERRED]** Slicing 850 hPa channels, manual time splitting, separate topography loading. | **[CONFIRMED]** Slicing 850 hPa channels from 16-channel array, loading `static.nc`. |
| **13. Documentation Quality** | **[CONFIRMED]** GRL 2024 paper text + GitHub repo README. | **[CONFIRMED]** WCRP-CORDEX benchmark documentation, data walkthrough notebooks, and evaluation suite. |

---

## 2. Predictor Configuration Analysis

The original paper utilized:
$$\mathbf{x} = (q_{850}, t_{850}, w_{850}, u_{850}, v_{850}) \quad (5\text{ channels at } 850\text{ hPa})$$

Since vertical velocity $w_{850}$ is confirmed **unavailable** in both author-recommended datasets, we evaluate two candidate configurations:

---

### Configuration A: 4-Channel Core Physical Subset ($q_{850}, t_{850}, u_{850}, v_{850}$)
- **Variables**: Specific humidity ($q$), temperature ($t$), zonal wind ($u$), meridional wind ($v$) at 850 hPa.
- **Channels**: 4 channels.
- **Methodological Status**: Explicit omission of unavailable $w_{850}$.
- **Rationale**: Isolates the downscaling task to horizontal thermodynamic and kinematic variables directly shared with the paper, without introducing unverified variable substitutions.
- **Input Tensor Shape**: `(batch, H, W, 4)`.

---

### Configuration B: Full Available Benchmark Predictor Suite
- **Dataset 1**: 8 channels ($u, v, t, q$ at 850 hPa and 500 hPa).
- **Dataset 2**: 16 channels ($u, v, q, t, z$ across 850, 700, 500 hPa + static orography).
- **Methodological Status**: Multi-level vertical atmospheric profile expansion.
- **Rationale**: Supplies the full multi-level tropospheric state as standardized in the respective benchmark protocols.
- **Input Tensor Shape**: `(batch, 16, 16, 16)` (for Dataset 2) or `(batch, 24, 28, 8)` (for Dataset 1).

---

## 3. Target Spatial Representation & Mask Analysis

Two representations are mathematically possible for high-resolution precipitation $Y$:

---

### Representation 1: Full 2D Grid ($D = 128 \times 128 = 16,384$)
- **Target Shape**: `(batch, 16384)` where every cell represents a point on the continuous $128 \times 128$ domain (land + surrounding ocean).
- **Characteristics**:
  - Requires no spatial masking.
  - Model predicts $(p, \alpha, \beta)$ parameters for all regional grid cells.
  - Parameter count: $\approx 15.0\text{M}$ weights (~57.4 MB), confirmed executable in Phase 7C.

---

### Representation 2: Land-Masked Station Representation ($D = D_{\text{land}}$)
- **Target Shape**: `(batch, D_land)` where $D_{\text{land}}$ is the exact number of valid land cells within the New Zealand land boundary.
- **Mask Derivation**:
  - Derived from `static.nc` (variable `orog > 0` or binary land-sea mask `sftlf`).
  - **Time Invariance**: **[CONFIRMED]** Topography and coastline boundaries are fixed across all timesteps $t \in [1, T]$.
  - **Flattening**: $Y_{\text{land}} = Y_{2D}[\text{land\_mask}]$ produces 1D vector `(D_land,)`.
  - **Unflattening**: $Y_{2D} = \text{zeros}(128, 128)$; $Y_{2D}[\text{land\_mask}] = Y_{\text{land}}$.
- **Analogy to Paper**: Directly mirrors the original paper's extraction of $11,491$ valid land stations from the raw $257 \times 241$ grid.

---

## 4. Conceptual Data Adapter Design

The data adapter will provide an isolated, standardized loading interface without modifying original research code:

```
========================================================================================
CONCEPTUAL ADAPTER INTERFACE
========================================================================================

load_dataset(
    data_dir,
    domain="NZ",
    config="config_A",        # "config_A" (4-channel) or "config_B" (16-channel)
    target_mode="land_masked",# "land_masked" or "full_grid"
    time_slice=None           # optional slice for small batches / debugging
)
    │
    ▼
Returns data_dict:
{
    'x_train': np.ndarray,      # Shape: (T_train, 16, 16, C)
    'x_test':  np.ndarray,      # Shape: (T_test, 16, 16, C)
    'y_train': np.ndarray,      # Shape: (T_train, D) where D = D_land or 16384
    'y_test':  np.ndarray,      # Shape: (T_test, D)
    'auxiliary': np.ndarray,    # Shape: (1, 128, 128, 1) standardized elevation
    'coords':  xr.Coordinates,  # Geographic metadata for unstacking / plotting
    'mask':    np.ndarray       # Boolean 2D array (128, 128) if land_masked
}
```

### Required Transformations:
1. **Predictors**:
   - For Configuration A: Slice channel indices corresponding to $q_{850}, t_{850}, u_{850}, v_{850}$.
   - Ensure channel dimension is trailing: `(time, lat, lon, channel)`.
2. **Target Precipitation**:
   - Extract variable `pr` (mm/day).
   - If `target_mode == "land_masked"`: Apply 2D boolean mask to extract 1D vector `(D_land,)`.
   - Preserve 2D coordinates for spatial unstacking and visualization.
3. **Auxiliary Elevation**:
   - Extract `orog` from `static.nc`.
   - Standardize over land: $(z - \mu_{\text{land}}) / \sigma_{\text{land}}$.
   - Format shape as `(1, 128, 128, 1)`.

---

## 5. Scientific Differences from Original Paper

1. **Observational vs. Simulated Ground Truth**:
   - Paper: ERA5 Reanalysis $\to$ VCSN weather station rain gauges ($5\text{ km}$).
   - Alternative Datasets: GCM ACCESS-CM2 $\to$ CCAM Regional Climate Model simulation ($10\text{ km}$).
   - **Documentation requirement**: Must be explicitly reported as evaluating the QRE algorithm on a Model-as-Truth climate downscaling benchmark.
2. **Omission of Vertical Velocity ($w_{850}$)**:
   - Paper: 5 variables at 850 hPa including vertical velocity.
   - Alternative: 4 variables at 850 hPa ($q, t, u, v$).
   - **Documentation requirement**: Must be explicitly reported as a 4-channel physical core configuration omitting unavailable $w_{850}$.

---

## 6. Exact Files & Data Volume Required for Real-Data Smoke Test

To execute a minimal real-data smoke test on actual NetCDF data:
- **Required Files**:
  - `train/ESD_pseudo-reality/predictors/ACCESS-CM2_1961-1980.nc` (~120 MB)
  - `train/ESD_pseudo-reality/target/pr_tasmax_ACCESS-CM2_1961-1980.nc` (~450 MB)
  - `train/ESD_pseudo-reality/static.nc` (< 1 MB)
- **Estimated Total Volume**: $\approx 570\text{ MB}$ (Extracted from `NZ_domain.zip` or direct download).
- **Avoided Download**: Avoids downloading complete multi-domain multi-GB archives (~30 GB).

---

## 7. Summary of Classifications

- **[CONFIRMED]** Neither alternative dataset contains vertical velocity $w_{850}$.
- **[CONFIRMED]** Both datasets contain the 4-channel core $q_{850}, t_{850}, u_{850}, v_{850}$.
- **[CONFIRMED]** Dataset 2 includes packaged static orography (`static.nc`) and pre-separated training/testing splits.
- **[CONFIRMED]** Time-invariant land masking is mathematically valid and reduces target dimension from $16,384$ to land cells.
- **[CONFIRMED]** No model code or data_handling.py changes have been made yet.

---

I have stopped here as instructed. Please review this report and let me know your decision on how to proceed.
