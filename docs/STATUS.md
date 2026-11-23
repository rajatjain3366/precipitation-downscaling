# Project status (2026-11-13, branch `main`)

`fixes-v2` was merged into `main` (PR #2, merge commit `1c0062b`). The code shown at the mid-term is tagged `midterm-eval`.

## Done

### NZ (CORDEX-ML-Bench), QRE replication with fixes

- All Phase 1–4 fixes are implemented and committed: train-only normalization, $p\alpha\beta$ predictions, one builder for members and BGNet, fixed segment bounds and ties, training thresholds for labels, own-interval member validation, paper LRs, decay and patience, and Bayesian-GMM intervals fitted once. Each fix has a config flag that restores the old behaviour.
- Checkpoint and resume support: `src/checkpoints.py`, with config signatures so a model trained under other settings cannot be loaded by mistake.
- **Weight Network collapse found and fixed.** The paper's EMD loss (Eq. 6) put every validation day in one interval: argmax histogram [0, 0, 0, 1825, 0, 0]. Cross-entropy, squared EMD and CE + EMD all fix it, with accuracy 0.65–0.66 against a majority-class chance of 0.21. The default is now squared EMD; the three new losses are tied within noise.
- **Member study implemented** (`experiments/member_study.py`): warm start from BGNet (B), from scratch to convergence (A), and an optional BGNet retrain to convergence. It is sample-tested but **not run on real data**.
- **NZ: QRE vs QRE + mass conservation implemented** (`experiments/nz_mass.py`, README §9), as the professor asked before the India experiment. It compares BGNet (converged), QRE (λ = 0) and QRE + mass (λ selected on validation) on the v2 split, with the mid-term numbers as reference rows. Details:
  - **Same models:** it uses the same members, Weight Network, BGNet and seeds as `run_cordex_v2.py`, and the India mass-loss module.
  - **λ = 0 is exactly the current NZ QRE:** on the fixture its test MSEs equal a separate `run_cordex_v2.py` run to the last digit (tested).
  - **Cell areas:** computed from the target grid's real 1-D lat/lon (a regular grid, not rotated or projected) on the 2,804 land cells.
  - **Land mask:** `orog > 0`, since `Static_fields.nc` has no `sftlf`. The README said `sftlf > 0.5`; that was wrong and is corrected. The same mask is used for training, evaluation and the mass loss.
  - **Status:** smoke-tested on the synthetic fixture; **not run on real data**.
- **Mid-term settings on the v2 split:** the mid-term runner hard-codes its split, so it cannot be rerun on the v2 split with a config alone. `configs/cordex_midterm_on_v2_split.json` is a labelled approximation using the v2 flags; it is checked on the fixture and not run.

### India (ERA5 → IMD), QRE with global mass-conservation regularization

- **Data pipeline:** multi-file IMD and ERA5, latitude flip, 4 × 4 block alignment check with partial edge blocks dropped (32 × 33 inputs), constant IMD mask with NaN never zero-filled (4,964 cells), configurable day offset with missing days reported, sub-daily ERA5 averaged to days, spherical cell areas.
- **Mass-conservation loss**, absolute or relative, applied to members, to a joint ensemble stage, or both. λ is a list, swept in one command and selected on validation only.
- **Runner** `experiments/run_india.py` with stages data / A / sweep / B / all; every model is checkpointed and resumable.
- **Supporting scripts:** a day-offset check script and an ERA5 download and coarsening script (not run).
- **Tests:** 19 India tests and 6 NZ mass-conservation tests plus the existing ones. All 12 smoke-test steps pass on this laptop (`tests/run_smoke_tests.py`).
- **No India experiment has been run, and no India result exists.** `results/india/smoke/` holds smoke-test output on 31 sample days with a tiny network; it is not a result.

## Key numbers (NZ, test 1994–2000, one sanity repetition; MSE on [0,1] / [0,0.2] / [0.9,1])

All models below are undertrained: members and BGNet ran 10 epochs, the Weight Networks 30.

| Model | MSE |
| --- | --- |
| Single BGNet (10 epochs) | **60.60 / 2.85 / 231.79** |
| QRE, cross-entropy Weight Network | **109.02 / 2.41 / 531.39** |
| QRE, paper EMD Weight Network (collapsed) | 124.13 / 15.23 / 627.19 |
| BGNet(-) | 67.31 / 2.45 / 261.29 |

- **Validation reference:** the Perfect-Member bound is 116.74 against BGNet's 60.82 on [0, 1]. Even always choosing the right member doubles BGNet's MSE, so the remaining gap is in the members, not the Weight Network. The member study is meant to test this.
- **First run before the fixes** (old split, 3 repetitions): QRE 112.47 ± 9.45, BGNet 62.03 ± 0.22.
- **New results since the last status:** none from long runs. Only the sample-tested code above.

## Where things are

| What | Where |
| --- | --- |
| How to run everything | [docs/RUNBOOK.md](RUNBOOK.md); on the lab PC, [docs/LAB_SETUP.md](LAB_SETUP.md) and `python scripts/run_lab.py` |
| Method, deviations, results | [README.md](../README.md) (§4 deviations, §5 NZ results, §8 India, §9 NZ mass conservation) |
| NZ v2 configs / outputs | `configs/cordex_v2.json`, `results/fixes_v2/` |
| NZ loss study | `results/fixes_v2/nz_wn/sanity/weight_network_losses/results.json` |
| NZ QRE vs QRE + mass conservation | `experiments/nz_mass.py`, `src/grid_areas.py`; outputs will be in `results/fixes_v2/nz_mass/` |
| India configs | `configs/india_v1.json` (real), `configs/india_smoke.json` (samples) |
| India code | `src/india_data.py`, `src/mass_conservation.py`, `src/india_metrics.py`, `src/india_plots.py`, `experiments/run_india.py`, `experiments/india_day_offset.py`, `scripts/download_era5_india.py` |
| Tests | `tests/test_india.py`, `tests/test_nz_mass.py`, `tests/run_smoke_tests.py` |
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
7. **India data and split:** JJAS only, 1971–2016 (the teammate's dataset), split as in the paper's 32 / 6 / 8 years: train 1971–2002, val 2003–2008, test 2009–2016 (3,904 / 732 / 976 days).
8. **No elevation input for India.** No source is configured, and no data was fabricated.
9. **No long runs on the laptop.** They run on the college lab PC with `scripts/run_lab.py`, following the RUNBOOK (decided 2026-11-23; before that, the teammate was to run them).
10. **NZ mass study reports two λ = 0 rows.** "QRE (λ = 0, no fine-tune)" is the current NZ QRE exactly, as required. "QRE (λ = 0, fine-tuned)" gets the same extra Weight Network fine-tune as λ > 0 with the mass term off, so it is the fair control for QRE + mass. λ is selected against the fine-tuned control (same training on both sides), the same rule as India, where λ = 0 is the fine-tuned control (`mass.lambda0 = "finetune_control"`).
11. **India intervals: the paper's published bounds** (0 / 0.166 / 0.392 / 0.61 / 0.812 / 0.923 / 1, `segmentation.method = "fixed"`). The Bayesian GMM on the 3,904 JJAS training totals was unstable (47 of 100 runs degenerate) and gave no extremes member (top interval = top 28 % of days). Training days per member are now 648 / 882 / 851 / 789 / 433 / 301; the extremes member has 301 training, 69 validation and 85 test days. The GMM remains an option. NZ is unchanged (GMM).
12. **India ERA5 inputs are daily means** over UTC days (all 24 hourly fields), confirmed in the script that made them, `scripts/download_era5_india_jjas.py` (`resample('1D').mean()`). The 00:00 timestamps are bin labels and the `stepType: instant` attributes come from the hourly data.

## First lab run (2026-11-23, RTX 4000 Ada, all 10 run_lab steps in about 25 min)

The result files are committed: the results at their usual paths, and the logs in `results/lab_runs/2026-11-23_rtx4000ada/`. The step times were 2–3 min for each NZ member-study step, 10 min 50 s for `nz_mass` and 2 min 36 s for India A.

- **NZ step 3, test MSE [0,1] / [0,0.2] / [0.9,1]:**

  | Model | [0,1] | [0,0.2] | [0.9,1] |
  | --- | --- | --- | --- |
  | QRE (warm start) | 52.03 | 2.13 | 190.46 |
  | Converged BGNet | 50.87 | 2.30 | 186.33 |
  | Perfect-Member | 50.31 | 1.89 | 187.43 |

- **NZ step 4:** QRE λ = 0 = 57.85 and BGNet = 49.51. The difference from step 3 is expected: step 4 trains its own v2 models (members from scratch with the v2 settings, its own BGNet, in `checkpoints/nz_mass`), not the member-study models. The two BGNets are separate trainings (other epoch cap, GPU); their 2.7 % gap is training variation, and the repetitions (step 13) will measure it. λ = 1 had no effect against the fine-tuned control.
- **India:**
  - A and B (λ = 0.01, ensemble placement) are identical to the 4th digit.
  - The members are near-copies of BGNet (per-interval MSE within about 1 %), so QRE ≈ BGNet ≈ Perfect-Member.
  - All three under-predict all-India totals by about 4 %.
  - **Why the India members ≈ BGNet:** the warm-start members stopped at best epochs 2 / 3 / 2 / 7 / 6 / 4 (of 6–11 epochs, patience 4).
    - They start from the converged BGNet (62 of 66 epochs) with a lower learning rate (5e-5). That is about 10–70 gradient steps per member, so they barely leave BGNet.
    - NZ's warm-start members start from the 10-epoch BGNet instead. They trained 28–47 epochs and beat it on their own intervals (ratios 0.69–0.87).
    - The India member study (step 10) adds members from scratch.
- **Follow-up (steps 10–14, implemented 2026-11-23):**
  - India member study (scratch vs warm start);
  - mass placements `members` / `ensemble` / `both` with λ ∈ {0, 0.1, 1, 3, 10}, NZ on the member-study models;
  - 10 repetitions with mean ± std and paired Wilcoxon tests. See the RUNBOOK §10–14.

## Next steps

1. **Done (2026-11-23):** lab PC set up (Linux, RTX 4000 Ada, CUDA 11.2 / cuDNN 8.1), smoke tests passed, RUNBOOK steps 0–9 run in about 25 min; results committed (see above).
2. **Kartik, lab PC:** the first run's result files are now in git, and the same files exist untracked on the lab PC, so a plain `git pull` would refuse to overwrite them. Run `git fetch origin && git checkout origin/main -- results && git pull` (the committed files are the lab's own, unchanged).
3. **Kartik, lab PC:** `python -m pytest -m slow tests/test_mass_study.py`, then `python scripts/run_lab.py` (runs steps 10–14; 0–9 are skipped as done).
4. **Kartik:** read the returned results; update the README results sections and this file.

Cleanup is done (2026-11-23): `_to_delete/` (old ERA5 files, 1.22 GB), the empty `data/India/raw_temp/` and the scratchpad test folders were removed. Kept on purpose, not cleanup candidates: `data/CORDEX_NZ_sample` (the smoke suite regenerates it) and the unused NZ predictors `data/CORDEX_NZ/test/historical/predictors/imperfect` (226 MB) and `.../perfect/EC-Earth3_1981-2000.nc` (113 MB), which were slow to download.

## Open questions for the professor

1. Is **QRE with a global mass-conservation regularization** (the all-India daily total only, with no moisture-flux or other physics) the intended scope for the end evaluation?
2. **Absolute vs relative** mass loss. We recommend absolute because relative over-weights dry days. Is that acceptable, or should both be reported?
3. **Placement:** is constraining the ensemble output in a fine-tuning stage acceptable as "QRE + mass conservation", or should the members themselves be trained with it (`"members"`, 4× the cost per λ)?
4. **IMD day convention:** the ERA5 inputs are daily means over UTC days; is matching the IMD day (24 h ending 08:30 IST) by the empirical offset check, as implemented, acceptable?
5. **Season:** the India data covers June–September only (1971–2016, 32 / 6 / 8 years). Is a monsoon-only experiment what is intended?
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
  - `scripts/download_era5_india_jjas.py` (the ERA5 JJAS script, unchanged; run on both machines), `scripts/slice_imd_jjas.py` (the teammate's IMD JJAS script, unchanged), `scripts/check_india_data.py`
  - NZ mass study (2026-11-13): `experiments/nz_mass.py`, `src/grid_areas.py`, `tests/test_nz_mass.py`, `configs/cordex_midterm_on_v2_split.json`
  - Lab runs (2026-11-23): `scripts/run_lab.py`, `docs/LAB_SETUP.md`, `tests/test_run_lab.py`
  - Post-lab studies (2026-11-23): `experiments/mass_study.py`, `tests/test_mass_study.py`; `run_india.py` stages members / placements / repeats; `nz_mass.py` stages placements / repeats; `study` sections in the configs; run_lab steps 10–14
  - GPU fixes from the first lab-PC test (2026-11-23): `src/gpu_setup.py` (memory growth, `GPU_RTOL`), `tests/conftest.py`, `tests/gpu_tol.py`, `tests/test_gpu_tol.py`
- **Modified:**
  - `configs/cordex_v2.json`: squared-EMD default and the `member_study` section
  - `src/quantiles.py`: loss kind `squared_emd`
  - `experiments/weight_network_losses.py`
  - `experiments/run_cordex_v2.py`: `fit(..., metrics=None)`, logging only
  - `src/preprocessing.py`: `split_masks_by_dates`
  - `tests/test_builders.py`, `tests/run_smoke_tests.py`
  - NZ mass study: `configs/cordex_v2.json` (`mass`, `ensemble`, `selection` sections), `run_cordex_v2.run_repetition(return_models=...)`, `run_india.py` (`mass.lambda0`, elevation input, dataset-aware plot labels), `src/builders.py` (optional `bgnet_minus_padding`)
  - `README.md`
