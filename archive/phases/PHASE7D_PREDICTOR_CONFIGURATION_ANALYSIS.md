# Phase 7D: Dataset 1 Predictor Investigation & Experimental Configuration Analysis

**Project**: Quantile-Regression-Ensemble (QRE) Reproduction (AAAI 2024)  
**Date**: September 16, 2026  
**Context**: Technical investigation of the predictor variables, pressure levels, and channel layouts available in author-recommended alternative datasets (Dataset 1: Zenodo 13755688 vs Dataset 2: Zenodo 17957264 / 20985924) compared against the AAAI 2024 paper's specifications.

---

## 1. Direct Evidence on Predictor Availability

### A. Dataset 1: AI-RCME (Zenodo Record 13755688 / GRL 2024)
- **[CONFIRMED]** Source description explicitly states:
  > *"Daily-averaged large-scale prognostic variables, including zonal wind, meridional wind, temperature, and specific humidity, are employed as predictors at the 500mb and 850mb pressure levels."*
- **[CONFIRMED]** Variables available:
  - $u$ (zonal wind) at 850 hPa and 500 hPa
  - $v$ (meridional wind) at 850 hPa and 500 hPa
  - $t$ (temperature) at 850 hPa and 500 hPa
  - $q$ (specific humidity) at 850 hPa and 500 hPa
- **[CONFIRMED]** Vertical velocity ($w_{850}$ / $w_{500}$) is **absent**.
- **[CONFIRMED]** Geopotential height ($z_{850}$ / $z_{500}$) is **absent**.
- **[CONFIRMED]** Total predictor channels = $8$ ($4\text{ variables} \times 2\text{ levels}$).

---

### B. Dataset 2: CORDEX-ML-Bench (Zenodo Record 17957264 / 20985924)
- **[CONFIRMED]** Source documentation and companion repository explicitly state:
  > *"Atmospheric variables at 850 hPa, 700 hPa, 500 hPa (~200km): u - zonal wind component, v - meridional wind component, q - specific humidity, t - temperature, z - geopotential height. Static field: Orography."*
- **[CONFIRMED]** Variables available:
  - $u$ (zonal wind) at 850, 700, 500 hPa
  - $v$ (meridional wind) at 850, 700, 500 hPa
  - $q$ (specific humidity) at 850, 700, 500 hPa
  - $t$ (temperature) at 850, 700, 500 hPa
  - $z$ (geopotential height) at 850, 700, 500 hPa
  - Static surface orography / topography (`static.nc`)
- **[CONFIRMED]** Vertical velocity ($w_{850}$) is **absent**.
- **[CONFIRMED]** Geopotential height ($z_{850}$) is **present**.
- **[CONFIRMED]** Total predictor channels = $16$ ($5\text{ variables} \times 3\text{ levels} + 1\text{ static}$).

---

## 2. Requirement Comparison Table: Paper vs Dataset 1 vs Dataset 2

| Requirement / Variable | Original Paper (AAAI 2024) | Dataset 1: AI-RCME (Zenodo 13755688) | Dataset 2: CORDEX-ML-Bench (Zenodo 17957264) | Factual Status |
| :--- | :--- | :--- | :--- | :--- |
| **Atmospheric Source** | ERA5 Reanalysis | GCM ACCESS-CM2 | GCM ACCESS-CM2 / EC-Earth3 | **[CONFIRMED]** Model-as-truth in both |
| **Pressure Level** | 850 hPa only | 850 hPa & 500 hPa | 850 hPa, 700 hPa, 500 hPa | **[CONFIRMED]** Both contain 850 hPa |
| **Specific Humidity ($q_{850}$)** | **Required** | **Present** (`q` @ 850) | **Present** (`q` @ 850) | **[CONFIRMED]** Matched in both |
| **Air Temperature ($t_{850}$)** | **Required** | **Present** (`t` @ 850) | **Present** (`t` @ 850) | **[CONFIRMED]** Matched in both |
| **Zonal Wind ($u_{850}$)** | **Required** | **Present** (`u` @ 850) | **Present** (`u` @ 850) | **[CONFIRMED]** Matched in both |
| **Meridional Wind ($v_{850}$)** | **Required** | **Present** (`v` @ 850) | **Present** (`v` @ 850) | **[CONFIRMED]** Matched in both |
| **Vertical Velocity ($w_{850}$)** | **Required** | **UNAVAILABLE** | **UNAVAILABLE** | **[CONFIRMED]** Unavailable in both |
| **Geopotential Height ($z_{850}$)** | Not used | **UNAVAILABLE** | **Present** (`z` @ 850) | **[CONFIRMED]** Present in Dataset 2 only |
| **Static Topography Field** | Augmented VCSN | Stored in companion repo | `static.nc` in domain folder | **[CONFIRMED]** Direct in Dataset 2 |
| **Predictor Spatial Resolution** | $\sim 100\text{ km}$ ($36 \times 41$) | $1.5^\circ$ ($\sim 150\text{ km}$, $\sim 24 \times 28$) | $\sim 200\text{ km}$ ($16 \times 16$) | **[CONFIRMED]** Different grids |
| **Target Spatial Resolution** | $\sim 5\text{ km}$ ($257 \times 241$, $11,491$ land) | $12\text{ km}$ 2D grid | $\sim 10\text{ km}$ ($128 \times 128$) | **[CONFIRMED]** Both simulate high-res RCM |

---

## 3. Candidate Predictor Configurations

The following four candidate predictor configurations are possible for executing the QRE algorithm:

---

### Configuration 1: Exact Paper Predictor Set
- **Variables**: $q_{850}, t_{850}, w_{850}, u_{850}, v_{850}$ (5 channels).
- **Pressure Level**: 850 hPa only.
- **Availability**: **UNAVAILABLE in both Dataset 1 and Dataset 2**.
- **Reason**: Neither dataset includes vertical velocity ($w_{850}$).
- **Scientific Consequence**: Cannot be tested without generating/downloading raw ERA5 vertical velocity fields independently.

---

### Configuration 2A: 4-Channel 850 hPa Core Physical Subset
- **Variables**: $u_{850}, v_{850}, q_{850}, t_{850}$ (4 channels).
- **Pressure Level**: 850 hPa only.
- **Availability**: **Available in both Dataset 1 and Dataset 2**.
- **Deviation from Paper**: Omission of vertical velocity ($w_{850}$); 4 input channels instead of 5.
- **Scientific Consequence**: Retains *only* variables that directly match the paper's 850 hPa variable definitions, isolating the effect of downscaling horizontal atmospheric wind, temperature, and moisture without introducing non-paper variables.

---

### Configuration 2B: Alternative 5-Channel 850 hPa Configuration (with $z_{850}$)
- **Variables**: $u_{850}, v_{850}, q_{850}, t_{850}, z_{850}$ (5 channels).
- **Pressure Level**: 850 hPa only.
- **Availability**: **Available in Dataset 2 (CORDEX-ML-Bench)**.
- **Deviation from Paper**: Substitutes geopotential height ($z_{850}$) in place of vertical velocity ($w_{850}$).
- **Scientific Consequence**: Preserves the 5-channel tensor width at 850 hPa, but geopotential height measures isobaric surface height rather than vertical kinematic motion.

---

### Configuration 3: Full Multi-Level Benchmark Suite
- **Variables**:
  - For Dataset 1: $u, v, t, q$ at 850 hPa and 500 hPa (8 channels).
  - For Dataset 2: $u, v, q, t, z$ at 850, 700, 500 hPa + static orography (16 channels).
- **Pressure Levels**: Multi-level atmospheric column (850, 700, 500 hPa).
- **Availability**: **Available in respective dataset packages**.
- **Deviation from Paper**: Expands the single-level (850 hPa) representation to a multi-level atmospheric column.
- **Scientific Consequence**: Supplies the downscaling model with upper-level tropospheric dynamics (500 hPa steering winds and lapse rates), representing the standard CORDEX benchmark setup.

---

## 4. Key Unresolved Questions & Decisions Required Before Implementation

Before creating a data adapter or downloading data subsets, the following design decisions must be agreed upon:

1. **Choice of Dataset**:
   - Dataset 1 (AI-RCME): 8 channels ($u, v, t, q$ at 850 & 500 hPa), $1.5^\circ \to 12\text{ km}$ grid.
   - Dataset 2 (CORDEX-ML-Bench): 16 channels ($u, v, q, t, z$ at 850, 700, 500 hPa), $16 \times 16 \to 128 \times 128$ grid, includes packaged `static.nc`.
2. **Choice of Predictor Configuration**:
   - Configuration 2A (4-channel core: $u, v, q, t$ at 850 hPa).
   - Configuration 2B (Alternative 5-channel: $u, v, q, t, z$ at 850 hPa).
   - Configuration 3 (Full multi-level suite: 8 or 16 channels).
3. **Choice of Target Representation**:
   - Full 2D grid ($128 \times 128 = 16,384$ outputs).
   - Land-masked station vector ($\sim 3,500$ land cells).

---

## 5. Summary of Certainties & Uncertainties

- **[CONFIRMED]** Neither Dataset 1 nor Dataset 2 contains $w_{850}$ (vertical velocity).
- **[CONFIRMED]** Both datasets contain the four 850 hPa variables: $u_{850}, v_{850}, q_{850}, t_{850}$.
- **[CONFIRMED]** Dataset 2 contains geopotential height $z_{850}$ and static topography `static.nc`.
- **[CONFIRMED]** The mathematical QRE pipeline can execute with any of Configurations 2A, 2B, or 3 once configured in the data adapter.
- **[UNKNOWN]** Whether $w_{850}$ can be computed via diagnostic finite differences from $u, v, z$ without access to the original full model grid.

---

I have stopped here as instructed. Please review this analysis and let me know your guidance on the next step.
