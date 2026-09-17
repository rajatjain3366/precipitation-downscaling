# PROJECT REPOSITORY STRUCTURE & NAVIGATION MAP

## 1. Quick Navigation: START HERE

For readers, examiners, and evaluators reviewing the BTP implementation, follow this recommended reading path:

```
1. README.md                                    -> Project overview, methodology, setup, and execution
2. reports/PHASE15_FINAL_EXPERIMENT_REPORT.md   -> Final empirical results, metrics table, and synthesis
3. reports/PHASE15_FINAL_EXPERIMENT_DESIGN.md   -> Detailed design and paper-vs-CORDEX taxonomy
4. reports/PAPER_REPO_DIFFERENCES.md            -> Catalog of differences between paper and reference repo
5. configs/cordex_final.json                    -> Configuration for the final N=6 experiment
6. experiments/run_cordex_experiment.py         -> Main executable training and evaluation runner
7. src/data_adapter_cordex.py                   -> CORDEX dataset loader and land-masking engine
8. results/final_cordex_qre/                    -> Final empirical evidence (JSON metrics, CSV tables)
```

---

## 2. Directory Tree & File Catalog

```
C:\BTP\
│
├── README.md                              # [CRITICAL] Primary repository entry point & overview
├── PROJECT_STRUCTURE.md                   # [CRITICAL] This document (repository map & navigation)
├── PROJECT_FILE_AUDIT.md                  # Complete audit table of all 60+ repository files
├── CLEANUP_CANDIDATES.md                  # Review list of temporary/intermediate files
├── FINAL_BTP_FILES.md                     # Categorized file manifest for BTP defense
├── LICENSE                                # MIT open-source license
├── .gitignore                             # Git exclusion configuration
│
├── src/                                   # [ACTIVE SOURCE CODE]
│   ├── ConvolutionalNetworks.py           # [MODEL] BGNet architecture with Channel Attention
│   ├── data_adapter_cordex.py             # [DATA] CORDEX NetCDF loader, land-masker, and preprocessor
│   ├── GammaLoss.py                       # [LOSS] Bernoulli-Gamma negative log-likelihood loss
│   ├── quantiles.py                       # [GATING] Omega network, OmegaLoss (EMD), and ynetwork gating
│   ├── Modules.py                         # [LAYERS] Custom Keras attention blocks & conv modules
│   └── useful_functions.py                # [UTILS] Quantile data slicing (`data_between`)
│
├── experiments/                           # [EXPERIMENT RUNNERS]
│   └── run_cordex_experiment.py           # [RUNNER] Standalone executable experiment runner
│
├── configs/                               # [EXPERIMENT CONFIGURATIONS]
│   ├── cordex_final.json                  # [CRITICAL] Final Phase 15 config (N=6, 3 reps, 730/365/365 days)
│   ├── cordex_cpu.json                    # Phase 14A/14B CPU validation config (N=3, 3 reps)
│   └── cordex_full.json                   # Full theoretical GPU specification (N=6, 40 reps, 20 yrs)
│
├── data/                                  # [DATASETS & METADATA]
│   ├── README.md                          # [CRITICAL] CORDEX-ML-Bench dataset guide & coordinate metadata
│   └── CORDEX_NZ_sample/                  # Local NetCDF datasets (ACCESS-CM2 1961-1963, 1981)
│
├── results/                               # [EXPERIMENT RESULTS & EVIDENCE]
│   └── final_cordex_qre/                  # [CRITICAL] Final Phase 15 Empirical Evidence (N=6, 3 reps)
│       ├── aggregate_metadata.json        # Provenance, parameters, and runtime metrics
│       ├── aggregate_metrics.json         # Summary MSE/MAE across all regimes + Wilcoxon tests
│       ├── aggregate_metrics.csv          # Exported tabular metrics across regimes
│       ├── repetition_0/                  # Run 0 individual metrics, logs, and metadata (seed 42)
│       ├── repetition_1/                  # Run 1 individual metrics, logs, and metadata (seed 1042)
│       ├── repetition_2/                  # Run 2 individual metrics, logs, and metadata (seed 2042)
│       ├── quantile_ranges.json           # Derived GMM quantile partition boundaries
│       └── training_history/              # Epoch-by-epoch loss trajectories
│
├── reports/                               # [THESIS & RESEARCH REPORTS]
│   ├── PHASE15_FINAL_EXPERIMENT_REPORT.md # [CRITICAL] Final comprehensive synthesis report (N=6)
│   ├── PHASE15_FINAL_EXPERIMENT_DESIGN.md # [CRITICAL] Final experiment design and taxonomy matrix
│   ├── PAPER_REPO_DIFFERENCES.md          # [CRITICAL] Audit of paper vs GitHub repository differences
│   └── DATASET_VERIFICATION.md            # Dataset availability and provenance audit
│
├── tests/                                 # [TEST SUITE]
│   ├── generate_sample_cordex.py          # NetCDF data generator for CORDEX coordinates
│   ├── test_cordex_adapter.py             # Phase 8 unit tests for data adapter
│   ├── test_real_data_specialist_smoke.py # Phase 9 smoke test for BGNet + GammaLoss
│   ├── test_real_data_qre_integration.py  # Phase 10 multi-specialist integration test
│   ├── smoke_test_qre.py                  # Synthetic data single-specialist smoke test
│   └── test_qre_pipeline.py               # Synthetic data multi-specialist pipeline test
│
└── archive/                               # [HISTORICAL MILESTONES & REFERENCE]
    ├── phases/                            # Phase 7A through Phase 14B milestone reports
    │   ├── PHASE14B_CPU_VALIDATION_REPORT.md
    │   ├── PHASE14A_CONTROLLED_RUN_REPORT.md
    │   ├── PHASE13_EXPERIMENT_RUNNER_IMPLEMENTATION.md
    │   ├── PHASE12_EXPERIMENT_IMPLEMENTATION_AUDIT.md
    │   └── ... (All development phase reports)
    ├── validation_runs/                   # Intermediate validation run evidence
    │   └── cordex_qre_cpu/                # Phase 14A/14B validation run outputs
    └── original_repo_reference/           # Original repository reference files
        ├── run.py                         # Original ERA5/VCSN experiment runner
        ├── data_handling.py               # Original ERA5/VCSN data handling
        ├── plot_functions.py              # Original spatial plotting utilities
        └── new_baseline.py                # Original baseline scratch script
```

---

## 3. Dependency Graph for Final Experiment

```
experiments/run_cordex_experiment.py (Main Execution Entry Point)
  ├── configs/cordex_final.json (Experiment Parameters)
  ├── data/CORDEX_NZ_sample/ (NetCDF Meteorological Files)
  ├── src/data_adapter_cordex.py (Data Loading & Land Masking)
  ├── src/ConvolutionalNetworks.py (BGNet & BGCallWrapper)
  │     └── src/Modules.py (ChannelAttention & ConvBlock)
  ├── src/GammaLoss.py (Bernoulli-Gamma Likelihood)
  ├── src/quantiles.py (Omega Network, OmegaLoss, ynetwork)
  │     └── src/Modules.py (ChannelAttention & ConvBlock)
  └── src/useful_functions.py (data_between)
```
