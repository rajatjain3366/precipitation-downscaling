# COMPLETE PROJECT FILE AUDIT

## Overview
This document contains a comprehensive, file-by-file audit of the entire `C:\BTP` project repository as of Phase 15 completion. Every file is classified according to its role in the final research deliverables, reproduction pipeline, and development history.

---

## Complete Repository Inventory

| File / Directory Path | Type | Purpose | Important? | Keep / Archive | Used by Final Experiment? | Reason / Dependency Context |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **`run_cordex_experiment.py`** | Python Script | Main experiment runner for CORDEX QRE pipeline (supports dry-run, CPU validation, and final N=6 runs) | **YES (CRITICAL)** | **Keep (Root/Experiments)** | **YES** | Primary executable script for Phase 14A, 14B, and Phase 15 final execution. |
| **`data_adapter_cordex.py`** | Python Module | Adapter for loading, preprocessing, land-masking, and structuring CORDEX-ML-Bench NetCDF data | **YES (CRITICAL)** | **Keep (Root/Src)** | **YES** | Imported directly by `run_cordex_experiment.py` for all data loading and splitting. |
| **`ConvolutionalNetworks.py`** | Python Module | Defines `BGNet`, `BGCallWrapper`, and convolutional blocks with channel attention | **YES (CRITICAL)** | **Keep (Root/Src)** | **YES** | Core paper-faithful architecture module imported by `run_cordex_experiment.py`. |
| **`GammaLoss.py`** | Python Module | Bernoulli-Gamma zero-inflated negative log-likelihood loss function | **YES (CRITICAL)** | **Keep (Root/Src)** | **YES** | Paper loss formulation imported by `run_cordex_experiment.py` for specialist training. |
| **`quantiles.py`** | Python Module | Defines `Omega` weight network, `OmegaLoss` (EMD), `ynetwork` gating, and `bin_y_var` | **YES (CRITICAL)** | **Keep (Root/Src)** | **YES** | Paper gating network and dynamic Softmax ensemble synthesis. |
| **`Modules.py`** | Python Module | Custom Keras layers (`ChannelAttention`, `SpatialAttention`, `ConvBlock`) | **YES (CRITICAL)** | **Keep (Root/Src)** | **YES** | Dependency of `ConvolutionalNetworks.py` and `quantiles.py`. |
| **`useful_functions.py`** | Python Module | Utility functions including `data_between` for quantile slicing | **YES (CRITICAL)** | **Keep (Root/Src)** | **YES** | Imported by `run_cordex_experiment.py` to partition datasets by GMM quantiles. |
| **`data_handling.py`** | Python Module | Original repository data processing scripts for ERA5 and VCSN | **YES** | **Keep / Archive** | **NO** | Reference implementation of original paper data loading; not used for CORDEX. |
| **`plot_functions.py`** | Python Module | Visualization and plotting utilities from original repository | **MEDIUM** | **Keep / Archive** | **NO** | Reference plotting routines for New Zealand spatial domain. |
| **`new_baseline.py`** | Python Module | Original repository baseline definitions | **LOW** | **Archive** | **NO** | Unused baseline experimentation script from original repository. |
| **`run.py`** | Python Script | Original repository execution runner for ERA5/VCSN dataset | **YES** | **Keep (Original Repo)** | **NO** | Kept for research integrity and original repository audit; not executed for CORDEX. |
| **`configs/cordex_final.json`** | Config (JSON) | Configuration for Phase 15 Final Experiment ($N=6$, 3 reps, 730/365/365 days) | **YES (CRITICAL)** | **Keep (Configs)** | **YES** | Exact configuration passed to `run_cordex_experiment.py` for final thesis results. |
| **`configs/cordex_cpu.json`** | Config (JSON) | Configuration for Phase 14A/14B validation runs ($N=3$, 3 reps) | **YES** | **Keep (Configs)** | **NO** | Validation configuration used in earlier testing phases. |
| **`configs/cordex_full.json`** | Config (JSON) | Full hypothetical GPU configuration ($N=6$, 40 reps, 20-year span) | **YES** | **Keep (Configs)** | **NO** | Research specification for full-scale cluster execution. |
| **`results/final_cordex_qre/`** | Directory | Complete final experiment output artifacts ($N=6$, 3 repetitions) | **YES (CRITICAL)** | **Keep (Results)** | **YES** | **Primary empirical evidence for BTP thesis**. |
| **`results/cordex_qre_cpu/`** | Directory | Intermediate Phase 14A and 14B CPU validation run artifacts ($N=3$) | **YES** | **Keep / Archive** | **NO** | Empirical evidence for pipeline validation and repetition scaling. |
| **`data/CORDEX_NZ_sample/`** | Directory | Local NetCDF files for CORDEX-ML-Bench New Zealand domain (1961-1963, 1981) | **YES (CRITICAL)** | **Keep (Data)** | **YES** | Local data source loaded by `data_adapter_cordex.py`. |
| **`tests/generate_sample_cordex.py`** | Python Script | Generates sample CORDEX NetCDF datasets with exact coordinate grids | **YES** | **Keep (Tests)** | **NO** | Dataset generation tool for local CORDEX testing and experiments. |
| **`tests/test_cordex_adapter.py`** | Test Script | Unit and forward pass test for `data_adapter_cordex.py` | **YES** | **Keep (Tests)** | **NO** | Phase 8 verification test. |
| **`tests/test_real_data_specialist_smoke.py`** | Test Script | Smoke test for training a single BGNet specialist on CORDEX data | **YES** | **Keep (Tests)** | **NO** | Phase 9 verification test. |
| **`tests/test_real_data_qre_integration.py`** | Test Script | End-to-end integration test for multi-specialist QRE pipeline | **YES** | **Keep (Tests)** | **NO** | Phase 10 verification test. |
| **`tests/smoke_test_qre.py`** | Test Script | Synthetic data smoke test for BGNet and GammaLoss | **MEDIUM** | **Archive (Tests)** | **NO** | Early structural smoke test. |
| **`tests/test_qre_pipeline.py`** | Test Script | Synthetic data multi-specialist integration test | **MEDIUM** | **Archive (Tests)** | **NO** | Early synthetic pipeline test. |
| **`PHASE15_FINAL_EXPERIMENT_REPORT.md`** | Report (MD) | Comprehensive synthesis report of the final experiment ($N=6$, 3 reps) | **YES (CRITICAL)** | **Keep (Reports)** | **YES** | **Primary final deliverable and thesis chapter summary**. |
| **`PHASE15_FINAL_EXPERIMENT_DESIGN.md`** | Report (MD) | Formal design document for Phase 15 final experiment | **YES (CRITICAL)** | **Keep (Reports)** | **YES** | Methodological design and taxonomy breakdown. |
| **`PHASE14B_CPU_VALIDATION_REPORT.md`** | Report (MD) | Validation report for multi-repetition CPU experiment ($N=3$) | **YES** | **Keep (Reports)** | **NO** | Milestone report for Phase 14B. |
| **`PHASE14A_CONTROLLED_RUN_REPORT.md`** | Report (MD) | Validation report for single-repetition controlled run | **YES** | **Keep (Reports)** | **NO** | Milestone report for Phase 14A. |
| **`PHASE13_EXPERIMENT_RUNNER_IMPLEMENTATION.md`** | Report (MD) | Implementation audit and documentation for `run_cordex_experiment.py` | **MEDIUM** | **Archive (Reports)** | **NO** | Phase 13 engineering report. |
| **`PHASE12_EXPERIMENT_IMPLEMENTATION_AUDIT.md`** | Report (MD) | Comprehensive architectural audit of all codebase components | **YES** | **Keep (Reports)** | **NO** | Detailed mathematical and code audit. |
| **`PHASE12_CODE_CHANGE_PLAN.md`** | Report (MD) | Change plan for runner implementation | **LOW** | **Archive (Reports)** | **NO** | Intermediate planning document. |
| **`PHASE11_FINAL_EXPERIMENT_DESIGN.md`** | Report (MD) | Initial experimental design document | **MEDIUM** | **Archive (Reports)** | **NO** | Precursor to Phase 15 design. |
| **`PHASE10_REAL_DATA_QRE_INTEGRATION.md`** | Report (MD) | Phase 10 integration test report | **MEDIUM** | **Archive (Reports)** | **NO** | Milestone report for Phase 10. |
| **`PHASE9_REAL_DATA_SPECIALIST_SMOKE_TEST.md`** | Report (MD) | Phase 9 specialist smoke test report | **MEDIUM** | **Archive (Reports)** | **NO** | Milestone report for Phase 9. |
| **`PHASE8_CORDEX_DATA_ADAPTER.md`** | Report (MD) | Documentation for CORDEX data adapter implementation | **YES** | **Keep (Reports)** | **NO** | Detailed reference for CORDEX adapter. |
| **`PHASE7E_DATASET_SELECTION_AND_ADAPTER_DESIGN.md`** | Report (MD) | Alternative dataset selection analysis | **YES** | **Keep (Reports)** | **NO** | Documents Zenodo dataset selection rationale. |
| **`PHASE7D_PREDICTOR_CONFIGURATION_ANALYSIS.md`** | Report (MD) | Predictor channel availability analysis (justifies omission of `w850`) | **YES** | **Keep (Reports)** | **NO** | Critical scientific justification for 4-channel set. |
| **`PHASE7C_REAL_DATA_SMOKE_TEST.md`** | Report (MD) | Phase 7C smoke test report | **LOW** | **Archive (Reports)** | **NO** | Milestone report for Phase 7C. |
| **`PHASE7B_DATA_INSPECTION.md`** | Report (MD) | Inspection report of Zenodo NetCDF files | **LOW** | **Archive (Reports)** | **NO** | Milestone report for Phase 7B. |
| **`PHASE7A_ALTERNATIVE_DATASET_AUDIT.md`** | Report (MD) | Initial audit of author-recommended alternative datasets | **LOW** | **Archive (Reports)** | **NO** | Milestone report for Phase 7A. |
| **`PAPER_REPO_DIFFERENCES.md`** | Report (MD) | Exhaustive discrepancy catalog between AAAI 2024 paper & GitHub repo | **YES (CRITICAL)** | **Keep (Reports)** | **NO** | **Core BTP audit document explaining all structural nuances**. |
| **`DATASET_VERIFICATION.md`** | Report (MD) | Verification of original vs alternative datasets | **YES** | **Keep (Reports)** | **NO** | Summary of dataset availability and provenance. |
| **`EXPERIMENT_CONFIG.md`** | Report (MD) | Configuration parameter reference | **MEDIUM** | **Archive (Reports)** | **NO** | Intermediate config reference. |
| **`README.md`** | Markdown Document | Repository root overview (to be comprehensively updated) | **YES (CRITICAL)** | **Keep (Root)** | **NO** | Main entry point for readers and evaluators. |
| **`LICENSE`** | Text File | Repository open-source license (MIT) | **YES** | **Keep (Root)** | **NO** | Standard repository licensing. |
| **`.gitignore`** | Git Configuration | Specifies ignored files and directories for version control | **YES** | **Keep (Root)** | **NO** | Git tracking rules. |
| **`__pycache__/`** | Cache Directory | Compiled Python `.pyc` bytecode files | **SAFE TO CLEAN** | **Clean / Re-generate** | **NO** | Auto-generated bytecode; safe to clear. |
| **`.venv/`** | Environment | Python virtual environment with TensorFlow 2.10.1 and dependencies | **CRITICAL (DO NOT MOVE)** | **Keep (Root)** | **YES** | Execution environment runtime. |
