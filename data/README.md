# DATASET DOCUMENTATION: CORDEX-ML-BENCH (NEW ZEALAND DOMAIN)

## 1. Dataset Overview & Source

- **Dataset Name**: CORDEX-ML-Bench (New Zealand Domain)
- **Scientific Role**: **Alternative Author-Recommended Benchmark Dataset**
- **Zenodo Repository**: [Zenodo Record 17957264](https://zenodo.org/records/17957264)
- **Recommendation Context**: Recommended directly by research paper co-author Neelesh Rampal because the original paper datasets (ERA5 atmospheric reanalysis and VCSN observational rain-gauge gridded target) are no longer publicly hosted or accessible.

> [!IMPORTANT]
> **NOT THE ORIGINAL ERA5/VCSN DATASET**:
> The final BTP experiment was performed on CORDEX-ML-Bench. This is a Global Climate Model (ACCESS-CM2) simulation downscaling task over New Zealand, not an observational rain-gauge reanalysis product. Metrics derived here must not be compared directly against the published ERA5/VCSN figures.

---

## 2. Directory Structure

The CORDEX dataset is structured inside `data/CORDEX_NZ_sample/` as follows:

```
data/CORDEX_NZ_sample/
├── train/
│   └── ESD_pseudo-reality/
│       ├── predictors/
│       │   └── ACCESS-CM2_1961-1980.nc      # Atmospheric predictors (q, t, u, v, z at 850, 700, 500 hPa)
│       ├── target/
│       │   └── pr_tasmax_ACCESS-CM2_1961-1980.nc # High-resolution surface targets (pr, tasmax)
│       └── static.nc                         # Surface orography (orog) and land fraction (sftlf)
└── test/
    └── historical/
        ├── predictors/
        │   └── perfect/
        │       └── ACCESS-CM2_1981-2000.nc  # Evaluation predictors (1981-2000)
        └── target/
            └── pr_tasmax_ACCESS-CM2_1981-2000.nc # Evaluation surface target (1981-2000)
```

---

## 3. Predictor Variables & Atmospheric Channels

| Variable Name | Description | Pressure Levels Available | Final Experiment Usage | Status |
| :--- | :--- | :---: | :---: | :--- |
| **`q`** | Specific Humidity | 850, 700, 500 hPa | Selected at 850 hPa (`q850`) | **USED** |
| **`t`** | Air Temperature | 850, 700, 500 hPa | Selected at 850 hPa (`t850`) | **USED** |
| **`u`** | Zonal Wind Velocity | 850, 700, 500 hPa | Selected at 850 hPa (`u850`) | **USED** |
| **`v`** | Meridional Wind Velocity | 850, 700, 500 hPa | Selected at 850 hPa (`v850`) | **USED** |
| **`z`** | Geopotential Height | 850, 700, 500 hPa | Available in raw NetCDF | Excluded from core 4-channel set |
| **`w`** | Vertical Wind Velocity | — | **UNAVAILABLE in CORDEX** | **EXPLICITLY OMITTED** |

---

## 4. Target Variable & Spatial Geometry

- **Target Variable**: Daily precipitation (`pr`), measured in mm/day.
- **Full Domain Grid**: $128 \times 128$ spatial grid over New Zealand (bounding coordinates: lat $[-51^\circ, -33^\circ]$, lon $[165^\circ, 184^\circ]$).
- **Land-Sea Mask**: Derived dynamically from `static.nc` (`sftlf > 0` or `orog > 0`).
- **Valid Land Points ($D_{\text{land}}$)**: **2,418 points** (~14.7% land coverage of bounding box).
- **Target Representation**: Flat 1D vector of length $D_{\text{land}} = 2,418$ containing only valid land grid points for numerical efficiency.
