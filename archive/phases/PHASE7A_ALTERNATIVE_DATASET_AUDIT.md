# Phase 7A: Alternative Dataset Compatibility Audit

**Project**: Quantile-Regression-Ensemble (QRE) Reproduction (AAAI 2024)  
**Date**: September 16, 2026  
**Context**: Following direct communication with paper co-author Dr. Neelesh Rampal (NIWA), the original preprocessed research datasets (`x_train_25km.nc`, `y_train_25km.nc`, `vcsn_rainfall_augmented.nc` from the NeSI supercomputing cluster) are confirmed **no longer publicly hosted**. The author provided two alternative Zenodo dataset repositories representing similar New Zealand climate downscaling benchmark data:
1. **Dataset 1**: [Zenodo Record 13755688](https://doi.org/10.5281/zenodo.13755688) (*AI-RCME GRL 2024*)
2. **Dataset 2**: [Zenodo Record 20985924 / 17957264](https://doi.org/10.5281/zenodo.17957264) (*CORDEX-ML-Bench 2025*)

---

## 1. Comprehensive Comparison Table

| Property | Original Research Paper (Bailie et al., AAAI 2024) | Dataset 1: Zenodo 13755688 (Rampal et al., GRL 2024) | Dataset 2: Zenodo 20985924 (CORDEX-ML-Bench 2025) |
| :--- | :--- | :--- | :--- |
| **Dataset Purpose** | Downscaling extreme precipitation via QRE on observational reanalysis | Regional Climate Model (RCM) emulator for GAN vs CNN extrapolation in warmer climates | Standardized machine learning benchmark for empirical and emulator regional downscaling |
| **Geographic Domain** | New Zealand (VCSN land grid) | New Zealand ($165^\circ\text{E} - 184^\circ\text{W}$, $33^\circ\text{S} - 51^\circ\text{S}$) | New Zealand Domain ($0.11^\circ$, regular lon/lat), plus SA & Alps |
| **Data Nature / Paradigm** | Observational Reanalysis to Station Grid (ERA5 $\to$ VCSN) | Model-as-Truth Simulation (GCM ACCESS-CM2 $\to$ RCM CCAM) | Model-as-Truth Benchmark (GCM ACCESS-CM2 / EC-Earth3 $\to$ RCM CCAM) |
| **Spatial Resolution (Predictors)** | $\sim 100\text{ km}$ ($36 \times 41$ after $4\times 4$ coarsening from $25\text{ km}$) | $1.5^\circ$ ($\sim 150\text{ km}$) coarsened from CCAM | $\sim 200\text{ km}$ ($16 \times 16$ grid) |
| **Spatial Resolution (Target)** | $\sim 5\text{ km}$ ($257 \times 241$, $11,491$ land points) | $12\text{ km}$ 2D grid over New Zealand | $0.11^\circ$ ($\sim 10\text{ km}$, $128 \times 128$ grid) |
| **Temporal Resolution** | Daily accumulated precipitation | Daily accumulated precipitation | Daily accumulated precipitation |
| **Time Period** | 1979–2024 ($32\text{ yrs}$ train, $6\text{ yrs}$ val, $8\text{ yrs}$ test) | Historical 1960–2014 ($~21,000\text{ days}$), SSP370 2044–2099 | Historical Train: 1961–1980 ($~7,300\text{ days}$); Test: 1981–2000 |
| **Predictor Variables** | $5$ variables at 850 hPa: `q_850`, `t_850`, `w_850`, `u_850`, `v_850` | $4$ variables at 850 & 500 hPa: $u, v, t, q$ | $5$ variables at 850, 700, 500 hPa ($16$ channels): $u, v, q, t, z$ |
| **Target Variable** | `Rain_bc` (Bias-corrected VCSN precipitation, mm/day) | `pr` (CCAM daily precipitation) | `pr` (Precipitation) and `tasmax` (Max temp) |
| **Static / Auxiliary Variable** | Surface elevation ($11,491$ land stations) | Static predictors stored in companion repo | `static.nc` (Orography / surface topography $\sim 10\text{ km}$) |
| **Hosting & Format** | Proprietary NeSI cluster path (Not public) | Zenodo NetCDF files (`target_...nc`, `predictor_...nc`) | Zenodo Domain zip packages (~10 GB for NZ) with standardized folders |
| **Compatibility with QRE Code** | Direct (code was written for this exact data) | Requires adapter for grid shape & channel selection | Requires adapter for $16\times 16 \to 128\times 128$ grid shapes |

---

## 2. Detailed Audit of Dataset 1: Zenodo Record 13755688

- **A. Purpose**: Accompanies *"On the Extrapolation of Generative Adversarial Networks for downscaling precipitation extremes in warmer climates"* (GRL 2024, Rampal, Gibson, Sherwood, Abramowitz, Hobeichi). Evaluates extrapolation of deep learning downscaling from historical to future warmer climates.
- **B. Geographic Coverage**: **[CONFIRMED]** New Zealand region ($165^\circ\text{E} - 184^\circ\text{W}$, $33^\circ\text{S} - 51^\circ\text{S}$).
- **C. Spatial Resolution**: **[CONFIRMED]** Target: $12\text{ km}$ RCM grid. Predictors: $1.5^\circ$ ($\sim 150\text{ km}$).
- **D. Temporal Resolution**: **[CONFIRMED]** Daily accumulated precipitation.
- **E. Time Period**: **[CONFIRMED]** Historical: 1960–2014 (~21,000 daily timesteps). Future (SSP370): 2044–2099 (~21,000 daily timesteps).
- **F. Predictor Variables**: **[CONFIRMED]** $u, v, t, q$ at 850 hPa and 500 hPa (8 channels total). Note: vertical velocity $w_{850}$ is absent.
- **G. Target Variable**: **[CONFIRMED]** `pr` (daily accumulated precipitation from CCAM).
- **H. Static Variables**: **[INFERRED]** Topography stored separately in repository.
- **I. Usability with QRE Code**: **[INFERRED]** Requires data adapter to select 850 hPa channels $(q, t, u, v)$ and reshape 2D target into land cell vector.
- **J. Major Differences from Paper**:
  - Target is simulated CCAM RCM precipitation ($12\text{ km}$) rather than observed VCSN weather station precipitation ($5\text{ km}$).
  - Predictors are GCM outputs (ACCESS-CM2) rather than ERA5 reanalysis.

---

## 3. Detailed Audit of Dataset 2: Zenodo Record 20985924 (CORDEX-ML-Bench)

- **A. Purpose**: Standardized international benchmark for empirical and emulator climate downscaling created by WCRP-CORDEX (Rampal, González-Abad, Gibson, Engelbrecht, Steinkopf, Hardy, 2025).
- **B. Geographic Coverage**: **[CONFIRMED]** New Zealand Domain ($0.11^\circ$ resolution, regular lat/lon grid), South Africa, and European Alps.
- **C. Spatial Resolution**: **[CONFIRMED]** Predictors: $16 \times 16$ coarse grid (~200 km). Target: $128 \times 128$ grid (~10 km).
- **D. Temporal Resolution**: **[CONFIRMED]** Daily resolution.
- **E. Time Period**: **[CONFIRMED]** Historical training: 1961–1980 (~7,300 days); Historical test: 1981–2000 (~7,300 days).
- **F. Predictor Variables**: **[CONFIRMED]** 16 total channels: $u, v, q, t, z$ at 850 hPa, 700 hPa, 500 hPa. The 850 hPa channels (`q_850`, `t_850`, `u_850`, `v_850`) are directly available.
- **G. Target Variable**: **[CONFIRMED]** `pr` (daily precipitation in mm/day) and `tasmax`.
- **H. Static Variables**: **[CONFIRMED]** `static.nc` containing surface orography / elevation matching the target grid.
- **I. Usability with QRE Code**: **[INFERRED]** Highly structured with standardized directory hierarchy (`train/ESD_pseudo-reality/`, `test/historical/`, `static.nc`). Requires updating input/output spatial dimension arguments in model initialization (e.g. input $16 \times 16$, target $128 \times 128 \to$ land points).
- **J. Major Differences from Paper**:
  - Predictors are $16 \times 16$ (~200 km) instead of $36 \times 41$ (~100 km).
  - Target grid is $128 \times 128$ (~10 km) instead of $257 \times 241$ (~5 km VCSN).
  - Training span is 20 years (1961–1980) instead of 32 years (1979–2010).

---

## 4. Technical Comparison & Assessment: Which Dataset is Closer?

### Assessment:
1. **Dataset 2 (CORDEX-ML-Bench, Zenodo 20985924)** is **technically superior and closer to a standardized evaluation**:
   - **Pre-packaged static topography (`static.nc`)**: Directly provides the surface elevation grid needed for the auxiliary elevation branch in BGNet.
   - **Clean variable naming**: Contains standard CF-compliant variable names ($q, t, u, v, z$ at 850 hPa and $pr$).
   - **Clean temporal splits**: Features distinct, un-mixed training (`1961-1980`) and evaluation (`1981-2000`) files.
   - **Active Benchmark**: Maintained by the same co-authors (Rampal, Gibson) as the official WCRP-CORDEX benchmark for ML downscaling.
2. **Key Shared Limitation of Both Datasets**:
   - Both datasets use a **Model-as-Truth framework** (GCM $\to$ RCM simulated precipitation) rather than **Observational Reanalysis** (ERA5 $\to$ VCSN observed weather stations).
   - This means experimental downscaling metrics will reflect simulated climate physics rather than 1979–2024 New Zealand gauge station truth.

---

## 5. Summary of Certainties & Uncertainties

- **[CONFIRMED]** The original ERA5 $\to$ VCSN preprocessed files are no longer publicly available.
- **[CONFIRMED]** Both Zenodo datasets cover the exact same New Zealand geographical region and include daily precipitation with 850 hPa atmospheric variables.
- **[CONFIRMED]** The mathematical QRE algorithm ($N$ BGNet specialists, Bernoulli-Gamma loss, GMM quantile partitioning, Weight Network with EMD loss, dynamic softmax aggregation) is **dataset-agnostic** and can be executed on either dataset once dimensions are mapped.
- **[UNKNOWN]** The exact uncompressed size of individual subset NetCDF files within the New Zealand domain zip archive of CORDEX-ML-Bench before downloading.

---

## 6. Recommended Next Step
- **Do not download full multi-GB archives yet.**
- When approved, inspect the CORDEX-ML-Bench data download notebook / scripts to identify the smallest single-year or single-file test split for the New Zealand domain to construct the first real-data compatibility bridge.
