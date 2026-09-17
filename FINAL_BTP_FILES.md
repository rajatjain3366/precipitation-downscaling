# FINAL BTP PROJECT FILE CATEGORIZATION

This document categorizes all files in `C:\BTP` into functional categories to guide defense preparation, execution, thesis writing, and repository archiving.

---

## 1. Essential for Demonstration
*Files required to demonstrate the complete working pipeline and show live test inference:*
1. **`experiments/run_cordex_experiment.py`**: Executes dry-run audits, single-repetition smoke runs, or full multi-repetition experiments.
2. **`configs/cordex_final.json`**: Active final experiment configuration ($N=6$, 3 repetitions, 730/365/365 days).
3. **`src/data_adapter_cordex.py`**: Loads CORDEX NetCDF data, applies dynamic land-sea mask ($D_{\text{land}} = 2418$), and generates spatial precipitation splits.
4. **`data/CORDEX_NZ_sample/`**: Working NetCDF sample dataset spanning 1961–1963 and 1981.

---

## 2. Essential for Running
*Source code modules required to compile, instantiate, train, and aggregate the QRE model:*
1. **`experiments/run_cordex_experiment.py`**: Main runner orchestrator.
2. **`src/data_adapter_cordex.py`**: Dataset loader and tensor preprocessor.
3. **`src/ConvolutionalNetworks.py`**: `BGNet` architecture with Squeeze-and-Excitation Channel Attention.
4. **`src/GammaLoss.py`**: Bernoulli-Gamma zero-inflated negative log-likelihood loss function.
5. **`src/quantiles.py`**: `Omega` weight network, `OmegaLoss` (EMD), and `ynetwork` dynamic Softmax aggregation.
6. **`src/Modules.py`**: Custom Keras layers (`ChannelAttention`, `SpatialAttention`, `ConvBlock`).
7. **`src/useful_functions.py`**: Helper routines including `data_between` for GMM quantile slicing.
8. **`.venv/`**: Python 3.10 virtual environment with TensorFlow 2.10.1 (CPU-optimized AVX2).

---

## 3. Essential for Thesis & Report
*Key research design, discrepancy analysis, and evaluation reports supporting the BTP dissertation:*
1. **`README.md`**: Main project overview, architecture breakdown, and reproduction summary.
2. **`reports/PHASE15_FINAL_EXPERIMENT_REPORT.md`**: Final empirical results, metrics tables, Wilcoxon tests, and synthesis.
3. **`reports/PHASE15_FINAL_EXPERIMENT_DESIGN.md`**: Methodological taxonomy (Paper-Faithful vs Alternative-Data vs Computational-Reductions).
4. **`reports/PAPER_REPO_DIFFERENCES.md`**: Comprehensive discrepancy catalog between the AAAI 2024 paper and reference code.
5. **`reports/DATASET_VERIFICATION.md`**: Dataset availability audit and provenance summary.
6. **`data/README.md`**: Complete documentation of CORDEX-ML-Bench dataset coordinates and variables.
7. **`PROJECT_STRUCTURE.md`**: Repository layout and navigation map.
8. **`PROJECT_FILE_AUDIT.md`**: Complete file-by-file provenance audit.

---

## 4. Experimental Evidence
*Saved empirical outputs, metrics, logs, and metadata proving experimental execution:*
1. **`results/final_cordex_qre/aggregate_metrics.json`**: Aggregated MSE/MAE across overall, low-rain, and extreme regimes + Wilcoxon tests.
2. **`results/final_cordex_qre/aggregate_metrics.csv`**: Exported CSV metrics table across models and regimes.
3. **`results/final_cordex_qre/aggregate_metadata.json`**: Complete execution provenance, GMM parameters, dates, and runtimes.
4. **`results/final_cordex_qre/repetition_0/`**: Repetition 0 logs, parameters, and metrics (seed 42).
5. **`results/final_cordex_qre/repetition_1/`**: Repetition 1 logs, parameters, and metrics (seed 1042).
6. **`results/final_cordex_qre/repetition_2/`**: Repetition 2 logs, parameters, and metrics (seed 2042).
7. **`results/final_cordex_qre/quantile_ranges.json`**: GMM-derived quantile partition boundaries per repetition.
8. **`results/final_cordex_qre/training_history/training_log.json`**: Epoch-by-epoch loss trajectories for all 18 specialists and 3 Omega networks.
9. **`archive/validation_runs/cordex_qre_cpu/`**: Intermediate Phase 14A and 14B validation run evidence ($N=3$).

---

## 5. Historical Development
*Phase-by-phase development reports and earlier unit tests (preserved for full research auditability):*
1. **`archive/phases/PHASE14B_CPU_VALIDATION_REPORT.md`**: Multi-repetition CPU validation report ($N=3$).
2. **`archive/phases/PHASE14A_CONTROLLED_RUN_REPORT.md`**: Single-repetition controlled run report ($N=3$).
3. **`archive/phases/PHASE13_EXPERIMENT_RUNNER_IMPLEMENTATION.md`**: Runner design audit report.
4. **`archive/phases/PHASE12_EXPERIMENT_IMPLEMENTATION_AUDIT.md`**: Mathematical implementation audit report.
5. **`archive/phases/PHASE10_REAL_DATA_QRE_INTEGRATION.md`**: Real-data integration test report.
6. **`archive/phases/PHASE9_REAL_DATA_SPECIALIST_SMOKE_TEST.md`**: BGNet + GammaLoss specialist smoke test report.
7. **`archive/phases/PHASE8_CORDEX_DATA_ADAPTER.md`**: CORDEX adapter documentation.
8. **`archive/phases/PHASE7E_DATASET_SELECTION_AND_ADAPTER_DESIGN.md`**: Zenodo dataset selection report.
9. **`archive/phases/PHASE7D_PREDICTOR_CONFIGURATION_ANALYSIS.md`**: Predictor analysis (w850 omission rationale).
10. **`archive/phases/PHASE7A`–`7C` reports**, **`archive/phases/EXPERIMENT_CONFIG.md`**, **`archive/phases/PHASE12_CODE_CHANGE_PLAN.md`**.
11. **`tests/test_cordex_adapter.py`**, **`tests/test_real_data_specialist_smoke.py`**, **`tests/test_real_data_qre_integration.py`**.
12. **`tests/smoke_test_qre.py`**, **`tests/test_qre_pipeline.py`**, **`tests/generate_sample_cordex.py`**.

---

## 6. Original Repository Reference Source
*Original reference code (preserved in `archive/original_repo_reference/`):*
1. **`archive/original_repo_reference/run.py`**: Original ERA5/VCSN experiment runner.
2. **`archive/original_repo_reference/data_handling.py`**: Original ERA5/VCSN data handling routines.
3. **`archive/original_repo_reference/plot_functions.py`**: Original spatial plotting routines for ERA5 grid.
4. **`archive/original_repo_reference/new_baseline.py`**: Original baseline experimentation script.

---

## 7. Safe to Clean / Removed
*Temporary build and cache artifacts:*
1. **`__pycache__/`** and **`tests/__pycache__/`** (Cleaned).
