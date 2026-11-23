| Model | Test years | MSE [0,1] | MSE [0,0.2] | MSE [0.9,1] | MAE | RMSE | Spatial corr | Daily-total MAE (mm/day) | Total rel. discrepancy | WN accuracy | WN collapsed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mid-term QRE (reference) | 1981–2000 | 112.47 ± 9.45 | 27.86 ± 15.06 | 579.69 ± 80.66 | 4.56 | n/a | n/a | n/a | n/a | n/a | n/a |
| Mid-term BGNet (reference) | 1981–2000 | 62.03 ± 0.22 | 4.28 ± 0.35 | 251.72 ± 3.78 | 3.47 | n/a | n/a | n/a | n/a | n/a | n/a |
| BGNet (converged) | 1994–2000 | 49.51 | 2.27 | 178.15 | 3.03 | 7.04 | 0.734 | 1.123 | 0.013 | n/a | n/a |
| QRE (λ = 0, no fine-tune) | 1994–2000 | 57.85 | 2.21 | 221.60 | 3.32 | 7.61 | 0.697 | 1.201 | -0.028 | 0.632 | False |
| QRE (λ = 0, fine-tuned) | 1994–2000 | 57.83 | 2.31 | 220.71 | 3.35 | 7.60 | 0.696 | 1.207 | -0.010 | 0.627 | False |
| QRE + mass (λ = 1) | 1994–2000 | 57.71 | 2.41 | 221.04 | 3.33 | 7.60 | 0.697 | 1.195 | -0.020 | 0.636 | False |

"QRE (λ = 0, no fine-tune)" is the current NZ v2 QRE. "QRE (λ = 0, fine-tuned)" gets the same extra Weight Network fine-tune as QRE + mass with the mass term off: compare QRE + mass with it to see the effect of the mass term alone.

Mid-term rows: results/final_cordex_qre: pipeline before the fixes, split train 1961-1975 / val 1976-1980 / test 1981-2000 (7,300 test days), mean of 3 repetitions. Not the same test years as the rows below.

λ selected on validation only, against the fine-tuned λ = 0 control (lowest val daily_mae_mm_per_day with val MSE[0.0,1.0] within 2% of lambda=0). Daily-total errors are area-weighted over the land cells.
