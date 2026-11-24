| Model | MSE [0,1] | MSE [0,0.2] | MSE [0.9,1] | Daily-total MAE (mm/day) | Abs. total rel. discrepancy |
| --- | --- | --- | --- | --- | --- |
| BGNet | 50.51 ± 1.05 | 2.24 ± 0.05 | 185.45 ± 4.02 | 1.132 ± 0.025 | 0.024 ± 0.020 |
| QRE | 53.51 ± 0.66 | 2.21 ± 0.05 | 195.81 ± 3.44 | 1.131 ± 0.015 | 0.023 ± 0.014 |
| QRE (λ = 0 control) | 53.59 ± 0.71 | 2.23 ± 0.07 | 196.24 ± 4.60 | 1.136 ± 0.025 | 0.023 ± 0.023 |
| QRE + mass (ensemble, λ = 10) | 53.19 ± 0.56 | 2.24 ± 0.05 | 195.10 ± 3.20 | 1.116 ± 0.013 | 0.027 ± 0.015 |

| Test | MSE [0,1] | MSE [0,0.2] | MSE [0.9,1] | Daily-total MAE (mm/day) | Abs. total rel. discrepancy |
| --- | --- | --- | --- | --- | --- |
| QRE vs BGNet | p = 0.00195 * (Δmed +3.1) | p = 0.695 (Δmed -0.00938) | p = 0.00195 * (Δmed +11.1) | p = 0.922 (Δmed +0.00168) | p = 0.922 (Δmed -0.000922) |
| QRE + mass (ensemble, λ = 10) vs QRE | p = 0.00195 * (Δmed -0.375) | p = 0.232 (Δmed +0.0329) | p = 0.0273 * (Δmed -0.628) | p = 0.00195 * (Δmed -0.0138) | p = 0.557 (Δmed +0.00232) |
| QRE + mass (ensemble, λ = 10) vs QRE (λ = 0 control) | p = 0.00195 * (Δmed -0.306) | p = 0.625 (Δmed +0.0199) | p = 0.275 (Δmed -0.787) | p = 0.00195 * (Δmed -0.0147) | p = 0.432 (Δmed +0.00716) |

Test split, 10 repetitions (seed base_seed + 1000 k), mean ± std (ddof 1). * = significant at 5% (paired two-sided Wilcoxon signed-rank; lower is better; Δmed = median of first − second). paired two-sided Wilcoxon signed-rank tests at 5%; needs at least 6 repetitions.
