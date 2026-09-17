# Dataset Verification & Acquisition Report (Phase 5)

**Project**: Quantile-Regression-Ensemble (QRE) Reproduction (AAAI 2024)  
**Date**: September 15, 2026  
**Status**: **DATASET BLOCKED (Files Not Present Locally)**

---

## A. File Inventory

| Required File | Expected Path in Code | Local Presence | File Size | Status |
| :--- | :--- | :--- | :--- | :--- |
| `x_train_25km.nc` | `/nesi/project/niwa03712/group_shared/x_train_25km.nc` | **Absent** | N/A | **MISSING** |
| `x_test_25km.nc` | `/nesi/project/niwa03712/group_shared/x_test_25km.nc` | **Absent** | N/A | **MISSING** |
| `y_train_25km.nc` | `/nesi/project/niwa03712/group_shared/y_train_25km.nc` | **Absent** | N/A | **MISSING** |
| `y_test_25km.nc` | `/nesi/project/niwa03712/group_shared/y_test_25km.nc` | **Absent** | N/A | **MISSING** |
| `vcsn_rainfall_augmented.nc` | `/nesi/project/niwa03712/group_shared/vcsn_rainfall_augmented.nc` | **Absent** | N/A | **MISSING** |

---

## B. Expected NetCDF Metadata & Schema

```
========================================================================================
1. PREDICTORS (x_train_25km.nc / x_test_25km.nc)
========================================================================================
- Data Source:       ERA5 Reanalysis (ECMWF / Copernicus Climate Data Store)
- Target Level:      850 hPa isobaric surface
- Variable Names:    q_850 (Specific humidity, kg/kg)
                     t_850 (Air temperature, K)
                     w_850 (Vertical velocity, Pa/s)
                     u_850 (Zonal wind component, m/s)
                     v_850 (Meridional wind component, m/s)
- Spatial Domain:    New Zealand region (~144 x 164 at 25km resolution)
- Coarsened Grid:    36 x 41 (simulating ~100km coarse GCM) via 4x4 spatial mean pooling
- Temporal Range:    Train: 32 years (1979-2010, ~11,680 timesteps)
                     Val:    6 years (2011-2016, ~2,190 timesteps)
                     Test:   8 years (2017-2024, ~2,920 timesteps)
- Data Type:         Float32

========================================================================================
2. PRECIPITATION TARGET (y_train_25km.nc / y_test_25km.nc)
========================================================================================
- Data Source:       Virtual Climate Station Network (VCSN, NIWA New Zealand)
- Variable Name:     Rain_bc (Bias-corrected daily precipitation)
- Units:             mm / day
- Spatial Grid:      257 x 241 2D grid over New Zealand
- Valid Land Cells:  11,491 stations (after dropping ~50,446 ocean NaNs)
- Temporal Range:    Daily timesteps matching ERA5 predictors
- Data Type:         Float32

========================================================================================
3. AUXILIARY ELEVATION (vcsn_rainfall_augmented.nc)
========================================================================================
- Data Source:       Topography / Elevation grid matching VCSN domain
- Variable Name:     elevation
- Units:             meters
- Spatial Grid:      257 x 241
- Preprocessing:     Land standardization, ocean masked with land mean
- Output Shape:      (1, 257, 241, 1)
- Data Type:         Float32
```

---

## C. Expected vs Actual Local State

| Item | Paper Specification | Local Environment State | Verification Result |
| :--- | :--- | :--- | :--- |
| **ERA5 Predictor Files** | `x_train_25km.nc`, `x_test_25km.nc` | Not found in `C:\BTP` | **BLOCKED** |
| **VCSN Target Files** | `y_train_25km.nc`, `y_test_25km.nc` | Not found in `C:\BTP` | **BLOCKED** |
| **Auxiliary Elevation File**| `vcsn_rainfall_augmented.nc` | Not found in `C:\BTP` | **BLOCKED** |
| **Alternative Files** | None present in repository | Only Python source code and LICENSE | **CONFIRMED ABSENT** |

---

## D. Time Ranges
- **Training**: 32 years ($1979 - 2010$, $11,680$ daily observations).
- **Validation**: 6 years ($2011 - 2016$, $2,190$ daily observations).
- **Testing**: 8 years ($2017 - 2024$, $2,920$ daily observations).
- **Total Span**: 46 continuous years ($16,790$ days).

---

## E. Spatial Dimensions
- **Predictor Grid ($X$)**: Raw $\sim 144 \times 164 \xrightarrow{\text{4x4 Mean Pool}} 36 \times 41 \times 5$.
- **Target Grid ($Y$)**: Raw $257 \times 241 \xrightarrow{\text{Ocean Drop}} 11,491$ land station points.
- **Elevation Grid ($Z$)**: $257 \times 241 \to (1, 257, 241, 1)$.

---

## F. Predictor Channels
The 5 atmospheric channels at 850 hPa:
1. `q_850`: Specific humidity
2. `t_850`: Air temperature
3. `w_850`: Vertical wind velocity
4. `u_850`: Zonal wind component
5. `v_850`: Meridional wind component

---

## G. Target Variable
- **`Rain_bc`**: Daily precipitation in mm/day. Continuous, non-negative, zero-inflated (approx. 70% dry days, heavy-tailed positive distribution).

---

## H. Elevation Variable
- **`elevation`**: Surface altitude in meters. Processed to standardize land points and replace ocean NaNs with the land mean.

---

## I. Missing-Value Structure
- **Target $Y$**: Ocean grid points are filled with `NaN` in the raw 2D grid. Slicing with `.stack(z=['lat','lon']).dropna('z')` discards ocean cells, leaving a dense 1D vector of $11,491$ land points.
- **Auxiliary $Z$**: Ocean `NaN`s are identified with `is_ocean = np.isnan(aux_data.values)` and replaced with `np.mean(land_values)`.

---

## J. Root Cause of Missing Data & Acquisition Protocol

### Why the Data is Absent Locally:
1. The GitHub repository contains only source code scripts and no data assets due to size constraints.
2. The code contains original cluster file paths referencing the New Zealand eScience Infrastructure (NeSI) supercomputer directory:
   `/nesi/project/niwa03712/group_shared/`

### Official Data Acquisition Options:

1. **Option 1: Primary Authors / NIWA Shared Archive (Recommended)**
   - Request the preprocessed NetCDF files (`x_train_25km.nc`, `x_test_25km.nc`, `y_train_25km.nc`, `y_test_25km.nc`, `vcsn_rainfall_augmented.nc`) from corresponding author Thomas Bailie or Neelesh Rampal (NIWA / University of Auckland).
   - This ensures exact alignment with the paper's spatial cropping, coordinate grids, and bias-correction preprocessing.

2. **Option 2: Direct Raw Data Pipeline Extraction (Self-Reconstruction)**
   - **ERA5 Predictors**: Download raw hourly/daily ERA5 data at 850 hPa for variables $(Q, T, W, U, V)$ over New Zealand bounding box $[166^\circ\text{E} - 179^\circ\text{E}, 34^\circ\text{S} - 48^\circ\text{S}]$ via the Copernicus Climate Data Store (CDS) API (`cdsapi`).
   - **VCSN Target Data**: Obtain Virtual Climate Station Network daily precipitation and elevation from NIWA Datahub.
   - **Preprocessing**: Run domain-matching regridding to generate `x_train_25km.nc`, `y_train_25km.nc`, and `vcsn_rainfall_augmented.nc`.

---

## K. Scientific Integrity Summary
- **No unrelated datasets will be used.**
- **No fake synthetic data will be labeled as real experiments.**
- Until the exact preprocessed NetCDF files or raw ERA5/VCSN sources are placed in `C:\BTP\data\`, real-data training is **held**.
