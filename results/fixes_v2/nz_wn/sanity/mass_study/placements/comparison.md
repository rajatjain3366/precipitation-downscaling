| Model | MSE [0,1] | MSE [0,0.2] | MSE [0.9,1] | MAE | Daily-total MAE (mm/day) | Total rel. discrepancy |
| --- | --- | --- | --- | --- | --- | --- |
| BGNet | 50.87 | 2.30 | 186.33 | 3.05 | 1.131 | 0.006 |
| QRE | 52.03 | 2.13 | 190.46 | 3.12 | 1.099 | -0.032 |
| QRE (λ = 0 control) | 52.17 | 2.14 | 190.01 | 3.15 | 1.109 | -0.014 |
| QRE + mass (ensemble, λ = 10) | 51.98 | 2.22 | 190.52 | 3.12 | 1.096 | -0.030 |
| Perfect-Member (QRE members) | 50.31 | 1.89 | 187.43 | 3.07 | 0.798 | -0.014 |

Placement and λ selected on validation only: lowest val daily_mae_mm_per_day among (placement, lambda > 0) with val MSE[0.0,1.0] within 2% of their control (tolerance met: True). The control is fine-tuned lambda = 0 control (same ensemble stage, mass term off).
