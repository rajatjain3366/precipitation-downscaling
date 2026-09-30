# Quantile-Regression-Ensemble (QRE) for Precipitation Downscaling

> **Bachelor Thesis Project (BTP)**  
> **Based on AAAI 2024 Research Paper**: *"Quantile-Regression-Ensemble: A Deep Learning Algorithm for Downscaling Extreme Precipitation"* (Bailie, Rampal, Gibson, & Robinson)  
> **Repository Implementation**: Alternative-Data Reproduction on CORDEX-ML-Bench  

---

## 1. Project Overview & Research Context

Statistical downscaling of Global Climate Model (GCM) projections to local surface precipitation is critical for regional climate impact assessments and disaster preparedness. Standard convolutional neural networks trained with Mean Squared Error (MSE) loss suffer from severe underprediction of extreme precipitation events due to the heavily right-skewed, zero-inflated nature of rainfall distributions ("regression to the mean").

The **Quantile-Regression-Ensemble (QRE)** framework (AAAI 2024) addresses this limitation by:
1. **Regime Partitioning**: Decomposing precipitation intensity into $N$ distinct regimes via **Gaussian Mixture Model (GMM)** clustering of daily spatial cumulative precipitation sums.
2. **Specialist Networks**: Training $N$ independent **Bernoulli-Gamma specialist CNNs (`BGNet`)** on the partitioned subsets using zero-inflated Gamma likelihood loss (`GammaLoss`).
3. **Dynamic Gating Network**: Training a **Weight Network ($\Omega$)** using **Earth Mover's Distance (`OmegaLoss`)** to assign dynamic, continuous Softmax weights $\omega_i(X)$ based on large-scale atmospheric predictors.
4. **Dynamic Convex Synthesis**: Synthesizing local precipitation fields via dynamic weighted aggregation:
   $$\hat{Y}_{\text{QRE}}(X) = \sum_{i=1}^N \omega_i(X) \cdot \hat{Y}_i(X)$$

---

## 2. Dataset Scope & Provenance

### Original Dataset Context
The original research paper utilized proprietary ERA5 atmospheric reanalysis as predictors and the New Zealand Virtual Climate Station Network (VCSN) rain-gauge observational analysis as target. Because these proprietary datasets are no longer publicly hosted, this reproduction was implemented on the author-recommended benchmark.

### Author-Recommended Benchmark (CORDEX-ML-Bench)
In accordance with direct guidance from paper co-author Neelesh Rampal, this implementation uses **CORDEX-ML-Bench** ([Zenodo Record 17957264](https://zenodo.org/records/17957264)):
- **Predictors ($X$)**: 4 atmospheric channels at 850 hPa: specific humidity (`q850`), temperature (`t850`), zonal wind (`u850`), and meridional wind (`v850`) on a $16 \times 16$ grid ($\sim 100 - 200\text{ km}$ nominal GCM scale).
- **Target ($Y$)**: Daily precipitation (`pr`) over New Zealand on a $128 \times 128$ grid ($\sim 12\text{ km}$ regional downscaled grid), represented as a 1D vector of $D_{\text{land}} = 2,418$ valid land points.
- **Topography**: Surface elevation (`orog`) and land-sea mask (`sftlf`) from `Static_fields.nc`.

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

The expected precipitation prediction per grid cell is calculated as:
$$\mathbb{E}[y] = p_0 \cdot \exp(\alpha) \cdot \exp(\beta)$$

### B. Omega Weight Network & Earth Mover's Distance
The gating network ($\Omega$) outputs continuous Softmax probabilities over the $N$ specialists ($\sum_i \omega_i = 1$, $\omega_i \ge 0$). It is trained using 1D Wasserstein / Earth Mover's Distance (`OmegaLoss`) to preserve the physical ordering of precipitation intensity:
$$\mathcal{L}_{\text{EMD}}(\omega, y_{\text{bin}}) = \frac{1}{N-1} \sum_{k=1}^{N-1} \left| \sum_{j=1}^k \omega_j - \sum_{j=1}^k y_{\text{bin}, j} \right|$$

---

## 4. Final Experimental Results ($N=6$, 3 Repetitions)

Evaluated on the full 20-year multi-decadal CORDEX test set (7,300 test days across seeds `42`, `1042`, `2042`):

| Model | Overall MSE $[0.0, 1.0]$ | Low Rain MSE $[0.0, 0.2]$ | Extreme Rain MSE $[0.9, 1.0]$ | Overall MAE $[0.0, 1.0]$ |
| :--- | :---: | :---: | :---: | :---: |
| **QRE ($N=6$, Ours)** | $112.47 \pm 9.45$ | $27.86 \pm 15.06$ | $579.69 \pm 80.66$ | $4.56 \pm 0.27$ |
| **Single BGNet** | $62.03 \pm 0.22$ | $4.28 \pm 0.35$ | $251.72 \pm 3.78$ | $3.47 \pm 0.03$ |
| **Bagging (`cqre`)** | $106.96 \pm 5.82$ | $43.22 \pm 8.84$ | $494.99 \pm 19.88$ | $5.83 \pm 0.51$ |
| **Probability (`pqre`)** | $105.48 \pm 4.08$ | $24.67 \pm 2.85$ | $550.61 \pm 19.50$ | $4.88 \pm 0.22$ |
| **Empirical Mean** | $155.40 \pm 0.00$ | $42.81 \pm 0.00$ | $751.48 \pm 0.00$ | $6.46 \pm 0.00$ |

*Note: In accordance with rigorous scientific reporting, results are stated factually without ranking models.*

---

## 5. How to Run & Reproduce

### Environment Setup
Activate the virtual environment and install dependencies:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Dry-Run Verification (0 Compute Cost)
Audit dataset shapes, coordinates, GMM partitioning, and parameter counts:
```powershell
python experiments/run_cordex_experiment.py --config configs/cordex_final.json --dry-run
```

### Execute Final Experiment ($N=6$, 3 Repetitions)
Run the complete multi-specialist training, baseline evaluation, and metric export pipeline:
```powershell
python experiments/run_cordex_experiment.py --config configs/cordex_final.json
```

All empirical results, metadata, logs, and tables are saved automatically to `results/final_cordex_qre/`.

### Inspect Dataset Samples
Run the dataset inspection script to print NetCDF summaries, variable shapes, and sample values:
```powershell
python test.py
```

---

## 6. Project Directory Structure

```
precipitation-downscaling/
├── configs/
│   └── cordex_final.json         # Active BTP experiment configuration (N=6, 3 seeds)
├── data/
│   └── CORDEX_NZ/                # CORDEX-ML-Bench NetCDF climate files
│       ├── train/                # 1961–1980 predictors, static fields, and targets
│       └── test/                 # 1981–2000 multi-decadal evaluation files
├── experiments/
│   └── run_cordex_experiment.py  # Complete end-to-end training and evaluation runner
├── results/
│   └── final_cordex_qre/         # Benchmark results (metrics_table.csv, JSONs, logs)
├── src/
│   ├── ConvolutionalNetworks.py  # BGNet architecture and parameter unpacking
│   ├── GammaLoss.py              # Zero-inflated Bernoulli-Gamma loss function
│   ├── Modules.py                # Squeeze-and-Excitation channel attention layers
│   ├── quantiles.py              # GMM binning, Omega weight network & EMD loss
│   └── data_adapter_cordex.py    # NetCDF loader, normalization & land-masking
├── tests/                        # Automated unit and integration test suite
├── requirements.txt              # Validated environment dependencies
└── test.py                       # Dataset viewing and sample inspection script
```

---

## 7. Reproduction & Scope Disclaimer

> [!IMPORTANT]
> This study implements the QRE algorithm on an **author-recommended alternative dataset (CORDEX-ML-Bench)** with necessary computational adjustments for feasibility. It does not represent an exact numerical reproduction of the original ERA5 $\to$ VCSN experiment.
