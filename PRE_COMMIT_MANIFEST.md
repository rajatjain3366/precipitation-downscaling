# Pre-Commit Staging Audit & Manifest

**Repository**: `C:\BTP`  
**Target Branch**: `rajat`  
**Audit Date**: September 17, 2026  
**Staging Verdict**: **READY FOR GIT ADD**

---

## 1. Summary of Git Changes

```
Tracked Modified Files (2):
  - .gitignore
  - README.md

Existing Tracked Unmodified File (1):
  - LICENSE

Tracked Deleted Paths (9) [Moved to src/ or archive/]:
  - ConvolutionalNetworks.py  -> src/ConvolutionalNetworks.py
  - GammaLoss.py              -> src/GammaLoss.py
  - Modules.py                -> src/Modules.py
  - quantiles.py              -> src/quantiles.py
  - useful_functions.py       -> src/useful_functions.py
  - data_handling.py          -> archive/original_repo_reference/data_handling.py
  - new_baseline.py           -> archive/original_repo_reference/new_baseline.py
  - plot_functions.py         -> archive/original_repo_reference/plot_functions.py
  - run.py                    -> archive/original_repo_reference/run.py

Newly Added Files (101):
  - Root Documentation & Configs (9)
  - src/ (7)
  - experiments/ (1)
  - configs/ (3)
  - data/ (6)
  - results/ (23)
  - tests/ (6)
  - reports/ (4)
  - archive/ (42)

TOTAL FILES STAGED FOR ADD/MODIFY: 103 files (101 New + 2 Modified)
TOTAL TRACKED FILES IN REPO POST-COMMIT: 104 files (103 + 1 Unmodified LICENSE)
```

---

## 2. Category Count Reconciliation Table

| Category | Actual Files Staged | Previous Manifest Count | Discrepancy Reconciliation |
| :--- | :---: | :---: | :--- |
| **Root Files** | **11** (+ 1 `LICENSE`) | 10 (+ 1 `LICENSE`) | `PRE_COMMIT_MANIFEST.md` itself was omitted from the root count |
| **`src/`** | **7** | 7 | Exact match |
| **`experiments/`** | **1** | 1 | Exact match |
| **`configs/`** | **3** | 3 | Exact match |
| **`data/`** | **6** | 6 | Exact match |
| **`results/`** | **23** | 16 | Repetitions 1 & 2 files (5 files each) were summarized as single lines |
| **`tests/`** | **6** | 6 | Exact match |
| **`reports/`** | **4** | 4 | Exact match |
| **`archive/`** | **42** | 35 | Validation run repetitions 1 & 2 (5 files each) were summarized as single lines |
| **TOTAL STAGED** | **103** | **88 (Stated 102)** | **Fully Reconciled to Git Dry-Run (103 Files)** |

---

## 3. Exact File-by-File Staging Manifest (103 Files)

### A. Root Files (11 Added/Modified + 1 Tracked LICENSE)
1. `.gitignore` *(Modified)*
2. `README.md` *(Modified)*
3. `LICENSE` *(Existing Tracked, Unmodified)*
4. `requirements.txt` *(New)*
5. `PROJECT_STRUCTURE.md` *(New)*
6. `FINAL_BTP_FILES.md` *(New)*
7. `PROJECT_FILE_AUDIT.md` *(New)*
8. `ORGANIZATION_COMPLETE.md` *(New)*
9. `POST_RESTRUCTURE_INTEGRITY_REPORT.md` *(New)*
10. `GITHUB_AUDIT.md` *(New)*
11. `PRE_COMMIT_MANIFEST.md` *(New)*
12. `CLEANUP_CANDIDATES.md` *(New)*

### B. Core Research Source Code: `src/` (7 Files)
1. `src/__init__.py` (Package initializer exposing core QRE modules)
2. `src/ConvolutionalNetworks.py` (BGNet Bernoulli-Gamma CNN architecture & BGCallWrapper)
3. `src/data_adapter_cordex.py` (CORDEX-ML-Bench NetCDF data loader & land-mask processor)
4. `src/GammaLoss.py` (Zero-inflated Bernoulli-Gamma negative log-likelihood loss)
5. `src/Modules.py` (ChannelAttentionModule, SpatialAttentionModule, DownScaleModule, DenseModule)
6. `src/quantiles.py` (GMM quantile discretization, Omega Weight Network, OmegaLoss EMD, ynetwork)
7. `src/useful_functions.py` (Precipitation partitioning, coordinate slicing, numerical utilities)

### C. Experiment Runners: `experiments/` (1 File)
1. `experiments/run_cordex_experiment.py` (Standalone multi-repetition CORDEX experiment runner)

### D. Experiment Configurations: `configs/` (3 Files)
1. `configs/cordex_cpu.json` (N=3 CPU validation experiment configuration)
2. `configs/cordex_final.json` (Final N=6, 3 repetitions, 730/365/365 multi-year config)
3. `configs/cordex_full.json` (Full 1961-2000 multi-decade configuration)

### E. Dataset Documentation & Sample Fixture: `data/` (6 Files)
1. `data/README.md` (Dataset provenance, Zenodo link, coordinate & variable specifications)
2. `data/CORDEX_NZ_sample/train/ESD_pseudo-reality/static.nc` (133 KB)
3. `data/CORDEX_NZ_sample/train/ESD_pseudo-reality/predictors/ACCESS-CM2_1961-1980.nc` (308 KB)
4. `data/CORDEX_NZ_sample/train/ESD_pseudo-reality/target/pr_tasmax_ACCESS-CM2_1961-1980.nc` (2.62 MB)
5. `data/CORDEX_NZ_sample/test/historical/predictors/perfect/ACCESS-CM2_1981-2000.nc` (78 KB)
6. `data/CORDEX_NZ_sample/test/historical/target/pr_tasmax_ACCESS-CM2_1981-2000.nc` (658 KB)

### F. Final Experimental Evidence: `results/final_cordex_qre/` (23 Files)
1. `results/final_cordex_qre/aggregate_metadata.json` (8.4 KB)
2. `results/final_cordex_qre/aggregate_metrics.csv` (1.58 KB)
3. `results/final_cordex_qre/aggregate_metrics.json` (13.0 KB)
4. `results/final_cordex_qre/metadata.json` (8.4 KB)
5. `results/final_cordex_qre/metrics_summary.json` (6.8 KB)
6. `results/final_cordex_qre/metrics_table.csv` (1.58 KB)
7. `results/final_cordex_qre/quantile_ranges.json` (1.2 KB)
8. `results/final_cordex_qre/repetition_0/metadata.json` (2.15 KB)
9. `results/final_cordex_qre/repetition_0/metrics_summary.json` (4.2 KB)
10. `results/final_cordex_qre/repetition_0/metrics_table.csv` (1.1 KB)
11. `results/final_cordex_qre/repetition_0/quantile_ranges.json` (0.4 KB)
12. `results/final_cordex_qre/repetition_0/training_history/training_log.json` (4.0 KB)
13. `results/final_cordex_qre/repetition_1/metadata.json` (2.15 KB)
14. `results/final_cordex_qre/repetition_1/metrics_summary.json` (4.2 KB)
15. `results/final_cordex_qre/repetition_1/metrics_table.csv` (1.1 KB)
16. `results/final_cordex_qre/repetition_1/quantile_ranges.json` (0.4 KB)
17. `results/final_cordex_qre/repetition_1/training_history/training_log.json` (4.0 KB)
18. `results/final_cordex_qre/repetition_2/metadata.json` (2.15 KB)
19. `results/final_cordex_qre/repetition_2/metrics_summary.json` (4.2 KB)
20. `results/final_cordex_qre/repetition_2/metrics_table.csv` (1.1 KB)
21. `results/final_cordex_qre/repetition_2/quantile_ranges.json` (0.4 KB)
22. `results/final_cordex_qre/repetition_2/training_history/training_log.json` (4.0 KB)
23. `results/final_cordex_qre/training_history/training_log.json` (13.2 KB)

### G. Isolated Test Suite: `tests/` (6 Files)
1. `tests/generate_sample_cordex.py` (Sample fixture generator utility)
2. `tests/smoke_test_qre.py` (Synthetic end-to-end unit tests)
3. `tests/test_cordex_adapter.py` (CORDEX adapter verification)
4. `tests/test_qre_pipeline.py` (6-stage synthetic pipeline integration test)
5. `tests/test_real_data_specialist_smoke.py` (Real CORDEX single specialist smoke test)
6. `tests/test_real_data_qre_integration.py` (Real CORDEX multi-specialist integration test)

### H. Research Reports: `reports/` (4 Files)
1. `reports/DATASET_VERIFICATION.md` (Dataset provenance audit)
2. `reports/PAPER_REPO_DIFFERENCES.md` (Paper vs reference code discrepancy catalog)
3. `reports/PHASE15_FINAL_EXPERIMENT_DESIGN.md` (Formal taxonomy & design)
4. `reports/PHASE15_FINAL_EXPERIMENT_REPORT.md` (Final experiment report & synthesis)

### I. Historical Archive: `archive/` (42 Files)
- `archive/original_repo_reference/` (4 files: `data_handling.py`, `new_baseline.py`, `plot_functions.py`, `run.py`)
- `archive/phases/` (15 files: Phase 7A through Phase 14B development audit reports)
- `archive/validation_runs/cordex_qre_cpu/` (23 validation metric, metadata, and log files)

---

## 4. Files Strictly Excluded (Ignored)

- `.venv/` (Virtual environment)
- `__pycache__/` and `*.pyc` (Python bytecode)
- `.pytest_cache/`, `.mypy_cache/` (Test & lint caches)
- `.vscode/`, `.idea/` (IDE metadata)
- `Thumbs.db`, `.DS_Store` (OS metadata)
- `data/CORDEX_NZ_full/`, `*.tar.gz`, `*.zip`, external `*.nc` outside sample fixture

---

## 5. Architectural Implementation Verification

| Component | Code Location | Verified Implementation Details |
| :--- | :--- | :--- |
| **`BGNet`** | `src/ConvolutionalNetworks.py` | CNN with `DownScaleModule` (Conv2D + `ChannelAttentionModule` + Conv2D) and `DenseModule`. Predicts 3 heads: $\alpha$ (Shape, Softplus), $\beta$ (Scale, Softplus), $p_0$ (Rain Probability, Sigmoid). |
| **`ChannelAttentionModule`** | `src/Modules.py` | Squeeze-and-Excitation channel attention (GlobalAveragePooling2D + Dense layers + Sigmoid scaling). |
| **`GammaLoss`** | `src/GammaLoss.py` | Negative log-likelihood for zero-inflated Bernoulli-Gamma distribution over precipitation field $y$. |
| **`Omega`** | `src/quantiles.py` | Weight Network mapping large-scale atmospheric predictors $X$ to continuous Softmax weights $\omega_i(X)$ ($\sum \omega_i = 1$). |
| **`OmegaLoss`** | `src/quantiles.py` | 1D Wasserstein / Earth Mover's Distance (EMD) loss on cumulative distribution functions. |
| **`ynetwork`** | `src/quantiles.py` | Convex linear combination synthesis $\hat{Y}_{\text{QRE}}(X) = \sum_{i=1}^N \omega_i(X) \cdot \hat{Y}_i(X)$. |

---

## 6. Staging Metrics Summary

- **Total Staged Files (Add/Modify)**: **103 files**
- **Total Tracked Files Post-Commit**: **104 files**
- **Total Staging Size**: **4.23 MB** (4,438,023 bytes)
- **Largest Staged File**: `data/CORDEX_NZ_sample/.../pr_tasmax_ACCESS-CM2_1961-1980.nc` (**2.56 MB**)
- **Files $> 10\text{ MB}$**: **0**
- **Secrets Found**: **0**
- **Hardcoded Machine Paths in Active Code**: **0**

---

## Final Verdict

### **READY FOR GIT ADD**
