# Phase 13: CORDEX QRE Experiment Runner Implementation Report

**Project**: Quantile-Regression-Ensemble (QRE) Reproduction (AAAI 2024)  
**Date**: September 16, 2026  
**Implementation Script**: [run_cordex_experiment.py](file:///c:/BTP/run_cordex_experiment.py)  
**Configurations**: [configs/cordex_cpu.json](file:///c:/BTP/configs/cordex_cpu.json), [configs/cordex_full.json](file:///c:/BTP/configs/cordex_full.json)  
**Status**: **IMPLEMENTATION & DRY-RUN AUDIT PASSED**

> [!IMPORTANT]
> **NO MODEL TRAINING WAS PERFORMED IN PHASE 13.**  
> In accordance with instructions, this phase exclusively created the standalone runner script `run_cordex_experiment.py` and JSON configuration specifications. All verifications were executed strictly in `--dry-run` schema and validation mode without training neural networks or modifying original research files.

---

## 1. Files Created
1. [run_cordex_experiment.py](file:///c:/BTP/run_cordex_experiment.py): Standalone CLI & Python experiment runner for CORDEX-ML-Bench QRE.
2. [configs/cordex_cpu.json](file:///c:/BTP/configs/cordex_cpu.json): CPU validation experiment configuration with explicit computational reductions.
3. [configs/cordex_full.json](file:///c:/BTP/configs/cordex_full.json): Methodology-faithful alternative-data configuration ($N=6$, 40 repetitions, 20-year span).
4. [PHASE13_EXPERIMENT_RUNNER_IMPLEMENTATION.md](file:///c:/BTP/PHASE13_EXPERIMENT_RUNNER_IMPLEMENTATION.md): This documentation report.

---

## 2. Files Modified
- **Zero original source files modified**.
- Original research files ([run.py](file:///c:/BTP/run.py), [ConvolutionalNetworks.py](file:///c:/BTP/ConvolutionalNetworks.py), [GammaLoss.py](file:///c:/BTP/GammaLoss.py), [quantiles.py](file:///c:/BTP/quantiles.py), [Modules.py](file:///c:/BTP/Modules.py), [data_handling.py](file:///c:/BTP/data_handling.py)) remain strictly untouched.

---

## 3. Runner Architecture

The new runner [run_cordex_experiment.py](file:///c:/BTP/run_cordex_experiment.py) is organized into modular functional stages:

```
[CLI / JSON Config Loader]
          │
          ▼
[load_cordex_dataset] ──► [prepare_explicit_temporal_splits] (Train / Val / Test)
                                      │
          ┌───────────────────────────┴───────────────────────────┐
          ▼                                                       ▼
[derive_gmm_quantile_ranges]                             [Train Single BGNet Baseline]
          │                                                       │
          ├──────────────────────────────────┐                    ▼
          ▼                                  ▼           [Train BG-Net(-) Ablation]
[Train N Specialists (BGNet)]    [Train Weight Network (Omega)]
          │                                  │
          └────────────────┬─────────────────┘
                           ▼
              [Assemble QRE ynetwork]
                           │
                           ▼
          [Generate Bagging / Prob Ensembles]
                           │
                           ▼
          [evaluate_test_quantiles ([0,1], [0,0.2], [0.9,1])]
                           │
                           ▼
         [Export metrics_summary.json, metadata.json]
```

---

## 4. Configuration System

Two distinct JSON configurations are formally defined:
- **`cordex_cpu.json` (`mode: cpu_feasible`)**:
  - Specialists: $N = 3$ (**COMPUTATIONAL_REDUCTION**)
  - Repetitions: $1$ (**COMPUTATIONAL_REDUCTION**)
  - Training Span: 2 years train, 1 year val, 1 year test (**COMPUTATIONAL_REDUCTION**)
  - Batch Size: 16 (**IMPLEMENTATION_CHOICE**)
  - Specialist LR: $0.001$, Omega LR: $0.001$
- **`cordex_full.json` (`mode: faithful_full`)**:
  - Specialists: $N = 6$ (**PAPER_FAITHFUL**)
  - Repetitions: $40$ (**PAPER_FAITHFUL**)
  - Training Span: 16 years train (1961–1976), 4 years val (1977–1980), 20 years test (1981–2000) (**PAPER_FAITHFUL**)
  - Batch Size: 32 (**IMPLEMENTATION_CHOICE**)
  - Specialist LR: $10^{-4}$ (decay 0.7), Omega LR: $5 \times 10^{-5}$ (decay 0.8) (**PAPER_FAITHFUL**)

---

## 5. Explicit Temporal Split Implementation

The runner strictly prohibits `validation_split=0.2` inside `model.fit()`. Instead, `prepare_explicit_temporal_splits()` constructs explicit chronological numpy arrays:
- $X_{\text{train}}, Y_{\text{train}}$: First 80% chronological span of training block.
- $X_{\text{val}}, Y_{\text{val}}$: Trailing 20% chronological span of training block.
- $X_{\text{test}}, Y_{\text{test}}$: Independent historical evaluation block.
- All models receive explicit `validation_data=(X_val, Y_val)` with early stopping.

---

## 6. GMM Quantile Partitioning Implementation

- Computes cumulative spatial sums: `daily_spatial_sum[t] = sum(Y_train[t, :])`.
- Fits `GaussianMixture(n_components=N, n_init=100, random_state=seed_k)`.
- Derives boundaries $Q_{\text{ranges}} = [(q_0, q_1), \dots, (q_{N-1}, 1.0)]$ using floating-point tolerance `daily_spatial_sum <= y_upper + 1e-1`.
- Validates monotonicity ($q_{i} > q_{i-1}$), zero NaNs/Infs, and asserts non-empty training partitions ($T_{\text{spec}} \ge 1$).

---

## 7. Specialist Training Design

- Slices $X_{\text{spec}}, Y_{\text{spec}}$ via `data_between(Y_train, ql, X_train, qh)`.
- Instantiates fresh `BGNet(output_dim=D_land, baseline=False, ch_attn=True)`.
- Compiles with Adam and `GammaLoss(epsilon=0.0001, y_thrs=0.5)`.
- Trains with `validation_data=(X_val, Y_val)` and `EarlyStopping(monitor='val_loss', patience=bg_patience, restore_best_weights=True)`.

---

## 8. Weight Network Design

- Generates training targets: `Y_omega_train = bin_y_var(Y_train, Q_ranges)`.
- Generates validation targets independently: `Y_omega_val = bin_y_var(Y_val, Q_ranges)`.
- Instantiates `Omega(output_dim=N)` with Conv2D + Channel Attention + Dense + Softmax.
- Compiles with Adam and `OmegaLoss(num_classes=N)`.
- Trains with `validation_data=(X_val, Y_omega_val)` and `EarlyStopping(patience=omega_patience)`.

---

## 9. Baseline Models Design

The runner natively trains and evaluates 5 comparative baselines:
1. **Single BGNet**: Unpartitioned deep BGNet trained on full $X_{\text{train}}, Y_{\text{train}}$.
2. **BG-Net(-)**: Standard deep baseline without attention (`baseline=True`).
3. **Bagging Ensemble (`cqre`)**: Equal-weighted specialist ensemble ($\omega_i = 1/N$).
4. **Probability Ensemble (`pqre`)**: Prior-probability weighted ensemble ($\omega_i = q_{i+1} - q_i$).
5. **Empirical Mean**: Mean precipitation baseline.

---

## 10. Native CORDEX Evaluation Protocol

Operates directly on 1D land-masked vectors $Y \in \mathbb{R}^{T \times 2418}$:
- **Overall**: $[0.0, 1.0]$ (100% of test days).
- **Low Precipitation**: $[0.0, 0.20]$ (Days in lowest 20% spatial rainfall).
- **Extreme Heavy-Tail**: $[0.90, 1.00]$ (Days in top 10% most severe storm events).
- Computes MSE across all points. For multi-repetition runs, computes mean, standard deviation, and Wilcoxon signed-rank test ($\alpha = 0.05$).

---

## 11. Reproducibility & Random Seeding

- Repetition $k \in \{0, \dots, R-1\}$ seeds:
  $$\text{Seed}_k = \text{base\_seed} + k \times 1000$$
- Explicitly passed to Python `random`, `np.random`, `tf.random`, and `GaussianMixture`.
- Fresh network weights instantiated for every repetition.

---

## 12. Artifact Structure

When executed, outputs will be written to `results/cordex_qre/`:
- `metadata.json`: Machine-readable dataset, hardware, and parameter classifications.
- `metrics_summary.json`: MSE values for QRE and all baselines across $[0, 1]$, $[0, 0.2]$, $[0.9, 1]$.
- `quantile_ranges.json`: Derived GMM interval boundaries for each repetition.
- `training_history/`: Per-epoch loss logs.
- `predictions/`: Array dumps of model predictions.

---

## 13. Dry-Run Audit Results

All 4 dry-run commands were executed and **PASSED**:
1. `python -m py_compile run_cordex_experiment.py` $\to$ **Exit Code 0**
2. `python run_cordex_experiment.py --mode cpu_feasible --dry-run` $\to$ **Exit Code 0**
3. `python run_cordex_experiment.py --mode faithful_full --dry-run` $\to$ **Exit Code 0**
4. `python run_cordex_experiment.py --config configs/cordex_cpu.json --dry-run` $\to$ **Exit Code 0**

---

## 14. Known Limitations & Scientific Disclaimers

1. **Alternative Dataset**: Evaluated on CORDEX-ML-Bench ACCESS-CM2 simulation, not original non-public ERA5/VCSN data.
2. **Predictor Omission**: $w_{850}$ is omitted due to absence in CORDEX; 4-channel physical core is used.
3. **Batch Size**: Batch size 16 (CPU) / 32 (Full) is an implementation choice, not a paper-confirmed value.

---

> [!IMPORTANT]
> **EXPLICIT CONFIRMATION**:  
> **NO MODEL TRAINING WAS PERFORMED IN PHASE 13.**  
> The codebase is fully prepared for execution in Phase 14 upon approval.
