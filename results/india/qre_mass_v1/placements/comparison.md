| Model | MSE [0,1] | MSE [0,0.2] | MSE [0.9,1] | MAE | Daily-total MAE (mm/day) | Total rel. discrepancy |
| --- | --- | --- | --- | --- | --- | --- |
| BGNet | 176.91 | 57.36 | 383.60 | 6.39 | 1.022 | -0.043 |
| QRE | 176.90 | 57.58 | 382.71 | 6.40 | 1.040 | -0.037 |
| QRE (λ = 0 control) | 176.90 | 57.58 | 382.71 | 6.40 | 1.040 | -0.037 |
| QRE + mass (members, λ = 0.1) | 179.88 | 57.30 | 387.45 | 6.56 | 1.027 | -0.006 |
| Perfect-Member (QRE members) | 175.48 | 57.11 | 380.09 | 6.37 | 0.837 | -0.036 |

Placement and λ selected on validation only: lowest val daily_mae_mm_per_day among (placement, lambda > 0) with val MSE[0.0,1.0] within 2% of their control (tolerance met: True). The control is the QRE itself (no extra stage at lambda = 0).
