| Model | MSE [0,1] | MSE [0,0.2] | MSE [0.9,1] | Daily-total MAE (mm/day) | Abs. total rel. discrepancy |
| --- | --- | --- | --- | --- | --- |
| BGNet | 179.51 ± 3.74 | 58.03 ± 0.72 | 387.11 ± 9.19 | 1.040 ± 0.016 | 0.029 ± 0.017 |
| QRE | 179.07 ± 3.36 | 57.52 ± 0.45 | 384.58 ± 8.13 | 1.026 ± 0.014 | 0.032 ± 0.010 |
| QRE (λ = 0 control) | 179.07 ± 3.36 | 57.52 ± 0.45 | 384.58 ± 8.13 | 1.026 ± 0.014 | 0.032 ± 0.010 |
| QRE + mass (members, λ = 0.1) | 182.09 ± 3.41 | 57.79 ± 0.46 | 392.91 ± 8.60 | 1.027 ± 0.013 | 0.007 ± 0.005 |

| Test | MSE [0,1] | MSE [0,0.2] | MSE [0.9,1] | Daily-total MAE (mm/day) | Abs. total rel. discrepancy |
| --- | --- | --- | --- | --- | --- |
| QRE vs BGNet | p = 0.00195 * (Δmed -0.345) | p = 0.00977 * (Δmed -0.48) | p = 0.00195 * (Δmed -2.03) | p = 0.0371 * (Δmed -0.015) | p = 0.846 (Δmed +0.00181) |
| QRE + mass (members, λ = 0.1) vs QRE | p = 0.00195 * (Δmed +3.12) | p = 0.0195 * (Δmed +0.322) | p = 0.00195 * (Δmed +8.9) | p = 0.77 (Δmed +0.00399) | p = 0.00195 * (Δmed -0.025) |
| QRE + mass (members, λ = 0.1) vs QRE (λ = 0 control) | p = 0.00195 * (Δmed +3.12) | p = 0.0195 * (Δmed +0.322) | p = 0.00195 * (Δmed +8.9) | p = 0.77 (Δmed +0.00399) | p = 0.00195 * (Δmed -0.025) |

Test split, 10 repetitions (seed base_seed + 1000 k), mean ± std (ddof 1). * = significant at 5% (paired two-sided Wilcoxon signed-rank; lower is better; Δmed = median of first − second). paired two-sided Wilcoxon signed-rank tests at 5%; needs at least 6 repetitions.
