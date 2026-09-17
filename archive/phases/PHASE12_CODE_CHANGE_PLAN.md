# Phase 12: Code Change Plan

**Target**: Structured implementation plan for the CORDEX-ML-Bench QRE Experiment Runner.

---

## 1. File-by-File Change Plan

| File | Action | Nature of Changes | Rationale |
| :--- | :--- | :--- | :--- |
| `run_cordex_experiment.py` | **[NEW]** | Create standalone experiment runner | Decouples CORDEX experiment from legacy `run.py`, supporting Config 1 & Config 2 without touching original repository files. |
| `data_adapter_cordex.py` | **[PRESERVE]** | No changes needed (already verified in Phase 8) | Ingests 4-channel predictors, dynamic land mask $D_{\text{land}} = 2,418$, and static orography cleanly. |
| `quantiles.py` | **[PRESERVE]** | No changes needed (verified in Phase 4 & 10) | `Omega`, `OmegaLoss`, `ynetwork`, and `bin_y_var` are fully functional. |
| `ConvolutionalNetworks.py` | **[PRESERVE]** | No changes needed (verified in Phase 4 & 9) | `BGNet` and `BGCallWrapper` are fully functional. |
| `GammaLoss.py` | **[PRESERVE]** | No changes needed (verified in Phase 4 & 9) | `GammaLoss` is numerically stable. |
| `Modules.py` | **[PRESERVE]** | No changes needed | `ChannelAttentionModule` and `DownScaleModule` are operational. |
| `run.py` | **[PRESERVE]** | **Zero modifications** | Preserved as historical repository reference. |

---

## 2. Detailed Structure of `run_cordex_experiment.py`

1. **Configuration Loader**: Reads parameter settings from `EXPERIMENT_CONFIG.md` or CLI flags (`--mode`, `--n_models`, `--repeats`, `--train_years`, `--test_years`).
2. **Dataset Setup**: Calls `load_cordex_dataset` from `data_adapter_cordex.py` to ingest $X_{\text{train}}, Y_{\text{train}}, X_{\text{test}}, Y_{\text{test}}$.
3. **Seeded Quantile Segmentation**: Fits `GaussianMixture(n_components=N, random_state=seed)` on cumulative daily spatial precipitation sums and extracts monotonic intervals $Q_{\text{ranges}}$.
4. **Specialist Training Loop**:
   - Loops over $N$ intensity regimes.
   - Slices $X_q, Y_q$ with `data_between`.
   - Trains `BGNet(output_dim=D_land, baseline=False)` with `GammaLoss` and early stopping.
5. **Weight Network Training**:
   - Generates one-hot targets $Y_\omega$ via `bin_y_var(Y_train, Q_ranges)`.
   - Trains `Omega(output_dim=N)` with `OmegaLoss` and early stopping.
6. **Baselines Training**:
   - Trains Single `BGNet` baseline on full unsegmented dataset.
   - Builds Bagging (`cqre`, fixed weights $1/N$) and Probability (`pqre`, fixed weights $q_2 - q_1$).
   - Computes Empirical Mean baseline.
7. **Disaggregated Evaluation**:
   - Computes test-set MSE overall ($[0.0, 1.0]$), dry/light rain ($[0.0, 0.2]$), and extreme tail ($[0.9, 1.0]$).
   - If `repeats > 1`, calculates mean $\pm$ std across repetitions and Wilcoxon signed-rank test.
8. **Artifact Export**:
   - Writes `results/metrics_summary.json` and `results/metrics_table.csv`.
   - Saves final model checkpoints if enabled.
