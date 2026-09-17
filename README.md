# Quantile-Regression-Ensemble (QRE) for Precipitation Downscaling

> **Bachelor Thesis Project (BTP)**  
> **Based on AAAI 2024 Research Paper**: *"Quantile-Regression-Ensemble: A Deep Learning Algorithm for Downscaling Extreme Precipitation"* (Bailie, Rampal, Gibson, & Robinson)  
> **Repository Implementation**: Alternative-Data Reproduction on CORDEX-ML-Bench  

---

## 1. Project Overview & Research Context

Statistical downscaling of global climate model (GCM) projections to local surface precipitation is critical for regional climate impact assessments. Standard convolutional neural networks trained with Mean Squared Error (MSE) loss suffer from severe underprediction of extreme precipitation events due to the heavily right-skewed, zero-inflated nature of rainfall distributions.

The **Quantile-Regression-Ensemble (QRE)** framework (AAAI 2024) addresses this limitation by:
1. Decomposing precipitation intensity into $N$ distinct regimes via **Gaussian Mixture Model (GMM)** clustering of daily spatial cumulative precipitation sums.
2. Training $N$ independent **Bernoulli-Gamma specialist CNNs (`BGNet`)** on the partitioned subsets using zero-inflated Gamma likelihood loss (`GammaLoss`).
3. Training a **Weight Network ($\Omega$)** using **Earth Mover's Distance (`OmegaLoss`)** to assign dynamic, continuous Softmax weights $\omega_i(X)$ based on large-scale atmospheric predictors.
4. Dynamically synthesizing local precipitation fields via convex linear combination:
   $$\hat{Y}_{\text{QRE}}(X) = \sum_{i=1}^N \omega_i(X) \cdot \hat{Y}_i(X)$$

---

## 2. Dataset Scope & Provenance

### Original Dataset Unavailability
The original research paper utilized ERA5 atmospheric reanalysis as predictors and the New Zealand Virtual Climate Station Network (VCSN) rain-gauge observational analysis as target. As verified in early research audits, these exact proprietary datasets are no longer publicly hosted or accessible.

### Alternative Author-Recommended Benchmark (CORDEX-ML-Bench)
In accordance with direct guidance from paper co-author Neelesh Rampal, this reproduction was implemented on **CORDEX-ML-Bench** ([Zenodo Record 17957264](https://zenodo.org/records/17957264)):
- **Predictors ($X$)**: 4 atmospheric channels at 850 hPa: specific humidity (`q850`), temperature (`t850`), zonal wind (`u850`), and meridional wind (`v850`) on a $16 \times 16$ grid.
- **Unavailable Variable**: Vertical wind velocity (`w850`) is not present in CORDEX-ML-Bench and is explicitly omitted without proxy substitution.
- **Target ($Y$)**: Daily precipitation (`pr`) over New Zealand on a $128 \times 128$ grid, represented as a 1D vector of $D_{\text{land}} = 2,418$ valid land points.

---

## 3. Core Architecture & Mathematics

```
                                  [ Atmospheric Predictors: X (16x16x4) ]
                                             │
                      ┌──────────────────────┼──────────────────────┐
                      │                      │                      │
                      ▼                      ▼                      ▼
             ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
             │  Specialist 0   │    │  Specialist 1   │    │  Specialist N-1 │
             │     (BGNet)     │    │     (BGNet)     │    │     (BGNet)     │
             └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
                      │                      │                      │
                      │ ŷ_0(X)               │ ŷ_1(X)               │ ŷ_{N-1}(X)
                      │                      │                      │
                      │     ┌────────────────┴────────────────┐     │
                      │     │  Weight Network: Ω (OmegaLoss)  │     │
                      │     │    Dynamic Softmax: ω_i(X)      │     │
                      │     └────────────────┬────────────────┘     │
                      │                      │                      │
                      ▼                      ▼                      ▼
               [ Weighted Convex Synthesis: Ŷ_QRE = Σ ω_i(X) · ŷ_i(X) ]
                                             │
                                             ▼
                             [ Downscaled Precipitation Field ]
```

### A. BGNet Architecture & GammaLoss
Each specialist employs a Squeeze-and-Excitation channel-attention CNN (`BGNet`) to predict 3 physical Bernoulli-Gamma parameters per grid cell: shape ($\alpha$), scale ($\beta$), and probability of zero precipitation ($p_0$). The network is optimized via:
$$\mathcal{L}_{\text{Gamma}}(y; \alpha, \beta, p_0) = -\sum_{j} \left[ \mathbb{I}_{y_j = 0} \ln(p_{0,j}) + \mathbb{I}_{y_j > 0} \left( \ln(1 - p_{0,j}) + \alpha_j \ln \beta_j - \ln \Gamma(\alpha_j) + (\alpha_j - 1)\ln y_j - \beta_j y_j \right) \right]$$

### B. Omega Weight Network & Earth Mover's Distance
The gating network ($\Omega$) outputs continuous Softmax probabilities over the $N$ specialists ($\sum_i \omega_i = 1$, $\omega_i \ge 0$). It is trained using 1D Wasserstein / Earth Mover's Distance (`OmegaLoss`) to preserve the physical ordering of precipitation intensity:
$$\mathcal{L}_{\text{EMD}}(\omega, y_{\text{bin}}) = \frac{1}{N-1} \sum_{k=1}^{N-1} \left| \sum_{j=1}^k \omega_j - \sum_{j=1}^k y_{\text{bin}, j} \right|$$

---

## 4. Final Experimental Results ($N=6$, 3 Repetitions)

The final experiment was executed with $N=6$ specialists across 3 independent repetitions (seeds `42`, `1042`, `2042`) on the multi-year CORDEX chronological split (730 train / 365 val / 365 test days):

| Model | Status | Overall MSE $[0.0, 1.0]$ | Low Rain MSE $[0.0, 0.2]$ | Extreme Rain MSE $[0.9, 1.0]$ |
| :--- | :---: | :---: | :---: | :---: |
| **QRE ($N=6$, Ours)** | **COMPLETED** | $\mathbf{25.3174 \pm 0.0108}$ | $\mathbf{23.9449 \pm 0.0145}$ | $\mathbf{26.9649 \pm 0.0053}$ |
| **Single BGNet** | **COMPLETED** | $25.2870 \pm 0.0230$ | $23.9341 \pm 0.0326$ | $26.9346 \pm 0.0151$ |
| **Bagging (`cqre`)** | **COMPLETED** | $25.1203 \pm 0.0616$ | $23.7435 \pm 0.0802$ | $26.8136 \pm 0.0408$ |
| **Probability (`pqre`)** | **COMPLETED** | $25.1458 \pm 0.0425$ | $23.7829 \pm 0.0540$ | $26.8144 \pm 0.0305$ |
| **Empirical Mean** | **COMPLETED** | $24.8954 \pm 0.0000$ | $23.3897 \pm 0.0000$ | $26.7409 \pm 0.0000$ |
| **BG-Net(-)** | **SKIPPED** | — | — | — |

*Note: In accordance with rigorous scientific reporting, results are stated factually without ranking models.*

---

## 5. How to Run & Reproduce

### Environment Setup
Activate the configured virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Dry-Run Verification (0 Compute Cost)
Audit dataset shapes, coordinates, GMM partitioning, and model parameter counts:
```powershell
python experiments/run_cordex_experiment.py --config configs/cordex_final.json --dry-run
```

### Execute Final Experiment ($N=6$, 3 Repetitions)
Run the complete multi-specialist training, baseline evaluation, and metric export pipeline:
```powershell
python experiments/run_cordex_experiment.py --config configs/cordex_final.json
```

All empirical results, metadata, logs, and tables are saved automatically to `results/final_cordex_qre/`.

---

## 6. Project Navigation & Documentation

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md): Detailed repository layout, dependency graph, and "START HERE" guide.
- [reports/PHASE15_FINAL_EXPERIMENT_REPORT.md](reports/PHASE15_FINAL_EXPERIMENT_REPORT.md): Comprehensive final experimental report and synthesis.
- [reports/PHASE15_FINAL_EXPERIMENT_DESIGN.md](reports/PHASE15_FINAL_EXPERIMENT_DESIGN.md): Methodological taxonomy (Paper-Faithful vs Alternative-Data vs Reductions).
- [reports/PAPER_REPO_DIFFERENCES.md](reports/PAPER_REPO_DIFFERENCES.md): Discrepancy catalog between the AAAI 2024 paper and reference code.
- [data/README.md](data/README.md): Detailed CORDEX-ML-Bench dataset and coordinate documentation.
- [FINAL_BTP_FILES.md](FINAL_BTP_FILES.md): Categorized file manifest for thesis defense.

---

## 7. Reproduction & Scope Disclaimer

> [!IMPORTANT]
> This study implements the QRE algorithm on an **author-recommended alternative dataset (CORDEX-ML-Bench)** with necessary computational adjustments for feasibility. It does not represent an exact numerical reproduction of the original ERA5 $\to$ VCSN experiment.
