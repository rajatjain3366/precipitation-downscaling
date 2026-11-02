# Project status (2026-11-02, branch `main`)

`fixes-v2` was merged into `main` (PR #2, merge commit `1c0062b`). The code shown at the mid-term is tagged `midterm-eval`.

## Done

**NZ (CORDEX-ML-Bench), QRE replication with fixes**
- All Phase 1–4 fixes are implemented and committed: train-only normalization, $p\alpha\beta$ predictions, one builder for members and BGNet, fixed segment bounds and ties, training thresholds for labels, own-interval member validation, paper LRs, decay and patience, and Bayesian-GMM intervals fitted once. Each fix has a config flag that restores the old behaviour.
- Checkpoint and resume support: `src/checkpoints.py`, with config signatures so a model trained under other settings cannot be loaded by mistake.
- **Weight Network collapse found and fixed.** The paper's EMD loss (Eq. 6) put every validation day in one interval: argmax histogram [0, 0, 0, 1825, 0, 0]. Cross-entropy, squared EMD and CE + EMD all fix it, with accuracy 0.65–0.66 against a majority-class chance of 0.21. The default is now squared EMD; the three new losses are tied within noise.
- **Member study implemented** (`experiments/member_study.py`): warm start from BGNet (B), from scratch to convergence (A), and an optional BGNet retrain to convergence. It is sample-tested but **not run on real data**.

**India (ERA5 → IMD), QRE with global mass-conservation regularization**
- **Data pipeline:** multi-file IMD and ERA5, latitude flip, 4 × 4 block alignment check with partial edge blocks dropped (32 × 33 inputs), constant IMD mask with NaN never zero-filled (4,964 cells), configurable day offset with missing days reported, sub-daily ERA5 averaged to days, spherical cell areas.
- **Mass-conservation loss**, absolute or relative, applied to members, to a joint ensemble stage, or both. λ is a list, swept in one command and selected on validation only.
- **Runner** `experiments/run_india.py` with stages data / A / sweep / B / all; every model is checkpointed and resumable.
- **Supporting scripts:** a day-offset check script and an ERA5 download and coarsening script (not run).
- **Tests:** 19 India tests plus the existing ones. All 11 smoke-test scripts pass on this laptop (`tests/run_smoke_tests.py`, 2.7 min).
- **No India experiment has been run, and no India result exists.** `results/india/smoke/` holds smoke-test output on 31 sample days with a tiny network; it is not a result.

## Key numbers (NZ, test 1994–2000, one sanity repetition; MSE on [0,1] / [0,0.2] / [0.9,1])

All models below are undertrained: members and BGNet ran 10 epochs, the Weight Networks 30.

| Model | MSE |
|---|---|
| Single BGNet (10 epochs) | **60.60 / 2.85 / 231.79** |
| QRE, cross-entropy Weight Network | **109.02 / 2.41 / 531.39** |
| QRE, paper EMD Weight Network (collapsed) | 124.13 / 15.23 / 627.19 |
| BGNet(-) | 67.31 / 2.45 / 261.29 |

- **Validation reference:** the Perfect-Member bound is 116.74 against BGNet's 60.82 on [0, 1]. Even always choosing the right member doubles BGNet's MSE, so the remaining gap is in the members, not the Weight Network. The member study is meant to test this.
- **First run before the fixes** (old split, 3 repetitions): QRE 112.47 ± 9.45, BGNet 62.03 ± 0.22.
- **New results since the last status:** none from long runs. Only the sample-tested code above.

## Where things are

| What | Where |
|---|---|
| How to run everything (for the teammate) | [docs/RUNBOOK.md](RUNBOOK.md) |
| Method, deviations, results | [README.md](../README.md) (§4 deviations, §5 NZ results, §8 India) |
| NZ v2 configs / outputs | `configs/cordex_v2.json`, `results/fixes_v2/` |
| NZ loss study | `results/fixes_v2/nz_wn/sanity/weight_network_losses/results.json` |
| India configs | `configs/india_v1.json` (real), `configs/india_smoke.json` (samples) |
| India code | `src/india_data.py`, `src/mass_conservation.py`, `src/india_metrics.py`, `src/india_plots.py`, `experiments/run_india.py`, `experiments/india_day_offset.py`, `scripts/download_era5_india.py` |
| Tests | `tests/test_india.py`, `tests/run_smoke_tests.py` |
| Saved NZ models (not in git, ~1.2 GB) | `checkpoints/nz_v2_sanity/rep0/` on Kartik's laptop |

## Decisions made

1. **Weight Network loss: squared EMD** (Hou et al. 2016) instead of the paper's Eq. 6, which collapsed. It is tied with cross-entropy within noise and is the EMD family the paper cites.
2. **India mass loss: absolute form by default** (area-mean mm/day), not relative.
   - **Reason:** the relative form divides by the observed total, so dry days weigh up to 10× and wet days about 0.1×. On the sample it rates "no rain anywhere" (0.52) far better than climatology (1.66), although their absolute errors are about equal (0.25 vs 0.28 mm/day).
   - **Restore relative:** `mass.mode = "relative"`.
3. **India placement default: `"ensemble"`**, a joint fine-tuning of the Weight Network with members frozen.
   - It constrains the evaluated output directly.
   - A and B differ only in that stage, since λ = 0 runs the same stage.
   - The λ sweep then costs one short fine-tune per λ instead of six members per λ.
   - `"members"` and `"both"` are implemented as alternatives.
4. **λ scaling:** each λ is multiplied by $L_\text{QRE}/L_\text{mass}$ at the start of the stage, so λ = 1 means both terms start equally large.
5. **λ selection:** the λ > 0 with the lowest validation daily-total MAE among those within 2 % of λ = 0's validation MSE on [0, 1]. Validation only.
6. **India architecture: kernel 6, "valid"**, as in the paper; 32 × 33 inputs allow it. NZ needed kernel 3. This cuts BGNet from 147 M to 13 M parameters.
7. **India split:** 1990–2019, train 1990–2010, val 2011–2014, test 2015–2019 (70 / 13.3 / 16.7 %).
8. **No elevation input for India.** No source is configured, and no data was fabricated.
9. **No long runs on the laptop.** The teammate runs them following the RUNBOOK.

## Next steps

1. **Teammate:** RUNBOOK steps 0–3 (setup, smoke tests, NZ member study B → A → BGNet retrain).
2. **Teammate:** RUNBOOK step 4 (download ERA5 1990-01 to 2020-01 and IMD 1990–2019). **Multi-year ERA5 is required.** The sample has 31 days, which only shows the code runs.
3. **Teammate:** RUNBOOK steps 5–8 (day-offset check → India A → λ sweep → India B). Set `members.init` from the NZ member-study result first.
4. **Kartik:** read the returned results; update the README results sections and this file.
5. **Cleanup, on Kartik's go-ahead only.** The candidates are the scratchpad test folders and `data/CORDEX_NZ_sample`, plus these unused NZ files:
   - `data/CORDEX_NZ/test/historical/predictors/imperfect` (226 MB)
   - `data/CORDEX_NZ/test/historical/predictors/perfect/EC-Earth3_1981-2000.nc` (113 MB)

## Open questions for the professor

1. Is **QRE with a global mass-conservation regularization** (the all-India daily total only, with no moisture-flux or other physics) the intended scope for the end evaluation?
2. **Absolute vs relative** mass loss. We recommend absolute because relative over-weights dry days. Is that acceptable, or should both be reported?
3. **Placement:** is constraining the ensemble output in a fine-tuning stage acceptable as "QRE + mass conservation", or should the members themselves be trained with it (`"members"`, 4× the cost per λ)?
4. **ERA5 product:** daily mean (our preference) or the 00 UTC snapshot of the sample? And is the IMD day convention (24 h ending 08:30 IST) to be matched by the empirical offset check, as implemented?
5. **Period and split:** is 1990–2019 with 21 / 4 / 5 years sufficient, or should a longer record be used?
6. **Expectations:** the NZ results suggest QRE may not beat a single BGNet on overall MSE, since even the Perfect-Member bound was about 2× BGNet with undertrained members. Is the expected benefit on extremes ([0.9, 1]) only, or overall?
7. **Elevation:** should an elevation static input be added for India, and if so, from which source?
8. **Deviations from the paper:** the Weight Network loss change, the NZ kernel 3 and the India kernel 6 are documented in the README. Are these deviations acceptable to report as they are?

## Files added or modified in this phase (since commit `177807b`)

- **New:**
  - `experiments/member_study.py`, `experiments/run_india.py`, `experiments/india_day_offset.py`
  - `scripts/download_era5_india.py`
  - `src/india_data.py`, `src/mass_conservation.py`, `src/india_metrics.py`, `src/india_plots.py`
  - `configs/india_v1.json`, `configs/india_smoke.json`
  - `tests/test_india.py`
  - `results/india/smoke/` (smoke output only)
  - `docs/RUNBOOK.md`, `docs/STATUS.md`
- **Modified:**
  - `configs/cordex_v2.json`: squared-EMD default and the `member_study` section
  - `src/quantiles.py`: loss kind `squared_emd`
  - `experiments/weight_network_losses.py`
  - `experiments/run_cordex_v2.py`: `fit(..., metrics=None)`, logging only
  - `src/preprocessing.py`: `split_masks_by_dates`
  - `tests/test_builders.py`, `tests/run_smoke_tests.py`
  - `README.md`
