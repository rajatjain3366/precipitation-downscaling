# Paper vs Repository Discrepancy Log

This document tracks all identified differences between the research paper:
> **"Quantile-Regression-Ensemble: A Deep Learning Algorithm for Downscaling Extreme Precipitation"**  
> *Thomas Bailie, Yun Sing Koh, Neelesh Rampal, Peter B. Gibson (AAAI 2024)*  
and the official repository code in `C:\BTP`.

---

### Discrepancy 1: Number of Specialist Models ($N$)
- **Difference:** Default number of quantile specialist models in the ensemble.
- **Evidence from paper:** The paper uses $N = 6$ specialist models (Figure 4, Section 4.2 / 5.1).
- **Evidence from repository:** `run.py` (line 502) sets `'n_models': 2` in `meta_data()`.
- **Impact:** With $N=2$, the ensemble only splits into 2 bins instead of the 6 intensity regimes analyzed in the paper.
- **Decision:** Use $N=6$ for reproduction experiments, and $N=2$ (or 3) strictly for minimal smoke tests.
- **Reason:** $N=6$ is required to reproduce the paper's 6 specialist regimes (weak learners) and the Weight Network's 6-class softmax output.

---

### Discrepancy 2: Dataset Temporal Split and Span
- **Difference:** Dataset years allocated to training, validation, and testing.
- **Evidence from paper:** 32 years for training (1979–2010), 6 years for validation (2011–2016), and 8 years for testing (2017–2024), totaling 46 years (16,790 daily samples).
- **Evidence from repository:** `data_handling.py` (lines 158-159) hardcodes `N_train = 1` and `N_test = 1` year.
- **Impact:** Code only loads 365 daily samples for training and 365 for testing by default.
- **Decision:** Make the temporal split configurable (e.g. via experiment configuration) and preserve the 32/6/8 year split for reproduction.
- **Reason:** Hardcoded 1 year was used for quick local debugging by the authors.

---

### Discrepancy 3: Computational Device and Execution Context
- **Difference:** Execution device context (`GPU:0` vs CPU).
- **Evidence from paper:** Experiments were executed on an NVIDIA A100 GPU cluster.
- **Evidence from repository:** `run.py` (line 417) hardcodes `with tf.device('GPU:0'):` inside `exe_run()`.
- **Impact:** On a CPU-only environment, hardcoded `GPU:0` will trigger runtime device placement errors or CUDA library warnings.
- **Decision:** Dynamically select `GPU:0` if available, otherwise gracefully fallback to CPU.
- **Reason:** Ensures execution succeeds on any machine without modifying model or training logic.

---

### Discrepancy 4: Specialist Training Bypass in `run_file()`
- **Difference:** `run_file()` in `run.py` bypasses specialist training by forcing a constant baseline.
- **Evidence from paper:** QRE trains independent deep BGNet specialist models on quantile-segmented subsets.
- **Evidence from repository:** `run.py` (line 576) calls `quantile_alg(..., cnst_base=True)`.
- **Impact:** Setting `cnst_base=True` replaces all neural specialist models with `ConstantBaseline` predicting empirical means (`empirical_mean_baseline`), preventing actual QRE training.
- **Decision:** Make `cnst_base` configurable; set `cnst_base=False` when training full QRE.
- **Reason:** The repository author had temporarily switched on `cnst_base=True` to benchmark against the naive constant ensemble baseline.

---

### Discrepancy 5: Hardcoded NeSI Cluster Data Paths
- **Difference:** Absolute file paths for input NetCDF data.
- **Evidence from paper:** Public ERA5 reanalysis and NIWA VCSN station data.
- **Evidence from repository:** `data_handling.py` (lines 150-155) references `/nesi/project/niwa03712/group_shared/...`.
- **Impact:** Code fails with `FileNotFoundError` outside the NeSI cluster environment.
- **Decision:** Replace hardcoded paths with configurable data directory paths or environment variables.
- **Reason:** NeSI paths are specific to the New Zealand eScience Infrastructure supercomputing cluster.

---

### Discrepancy 6: Variable Selection in `open_data()`
- **Difference:** Dataset variable opening in `data_handling.py`.
- **Evidence from paper:** Input atmospheric predictors consist of 5 variables at 850 hPa: `T850`, `Q850`, `U850`, `V850`, `W850`.
- **Evidence from repository:** `data_handling.py` (lines 150-151) accesses `.w_200` directly from the dataset (`xr.open_dataset(...).w_200`), yet `prep_predictors` (line 111) attempts to slice `data.sel(channel=['q_850', 't_850', 'w_850', 'u_850', 'v_850'])`.
- **Impact:** Accessing `.w_200` loses the multi-channel dimension needed by `prep_predictors`.
- **Decision:** Ensure `open_data()` loads the multi-variable DataArray / Dataset properly matching the 5 channels.
- **Reason:** `.w_200` was likely an ad-hoc test variable leftover.

---

### Discrepancy 7: Missing Model Save Directory Paths
- **Difference:** Save path strings in `run.py` and `data_handling.py`.
- **Evidence from paper:** Trained model weights and training histories are stored for evaluation across 40 repetitions.
- **Evidence from repository:** `exe_run()` (line 408) has `model_path = f''` and `save()` (line 73) has `directory = f''`.
- **Impact:** Calling `model.save(model_path)` or `np.savetxt` with empty paths raises I/O errors when saving is enabled.
- **Decision:** Supply explicit, sanitized directory paths when saving is enabled.
- **Reason:** Incomplete path strings left in repository.

---

### Discrepancy 8: Ensemble Repetitions
- **Difference:** Number of independent training repetitions for statistical significance testing.
- **Evidence from paper:** 40 repetitions per model to compute mean $\pm$ standard deviation and Wilcoxon signed-rank significance.
- **Evidence from repository:** `run.py` (line 508) default is `'repeats': 1`.
- **Impact:** Single run cannot compute standard deviations or valid Wilcoxon $p$-values.
- **Decision:** Use `repeats=40` for full reproduction on GPU cluster; allow `repeats=1` or `2` for local smoke/validation tests.
- **Reason:** Computational feasibility on CPU vs A100 GPU.

---

### Discrepancy 9 / Implementation Fix: QRE Aggregation Shape Alignment
- **Difference:** Tensor shape convention in `ynetwork` aggregation.
- **Evidence from paper:** $g(x) = \sum_{i=1}^N \omega_i(x) f_i(x)$, where $f_i(x) \in \mathbb{R}^{B \times D}$ and $\omega(x) \in \mathbb{R}^{B \times N}$.
- **Evidence from repository:** `BGCallWrapper` produced `(B, D, 1)` which caused `ynetwork` to assemble a 4D tensor `(B, N, D, 1)` that failed to broadcast against weights `(B, N, 1)`.
- **Impact:** `ynetwork` raised `InvalidArgumentError: Incompatible shapes: [B, N, 1] vs [B, N, D, 1]` on eager tensor multiplication.
- **Decision:** Ensure `BGCallWrapper` and `ynetwork` standardize specialist predictions to `(B, N, D)` and weights to `(B, N, 1)`.
- **Reason:** Mathematically neutral fix ensuring standard NumPy/TensorFlow broadcasting $\mathbb{R}^{B \times N \times 1} \odot \mathbb{R}^{B \times N \times D} \to \mathbb{R}^{B \times N \times D} \xrightarrow{\sum_N} \mathbb{R}^{B \times D}$.

