# RUNBOOK: long runs for the QRE project

This file lists every long training run in order: the exact command, the expected time, where the outputs go, what to check afterwards and what to send back. The code is finished and tested on small samples, so nothing here should need code changes. If something fails, send the console log instead of fixing it.

**Time estimates:**
- They are measured or extrapolated on the author's laptop CPU (TensorFlow 2.10 CPU build), which was running on battery and so possibly throttled.
- On another CPU they can differ by 2× either way. A GPU should be much faster, but that has not been measured.
- "Expected" assumes early stopping ends training before the epoch cap; "worst case" means every model runs to its cap.

---

## 0. Setup

```bash
git clone https://github.com/rajatjain3366/precipitation-downscaling.git
cd precipitation-downscaling
git checkout fixes-v2
python3.10 -m venv .venv                 # Python 3.10 (3.10.11 used); TF 2.10 needs 3.7-3.10
source .venv/bin/activate                # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt netCDF4==1.7.4 pytest
pip install cdsapi                       # only for the ERA5 download (step 4)
```

On Windows, use **Git Bash** (then the commands below work as written) or **PowerShell 7** (`pwsh`). Windows PowerShell 5.1 drops the inner quotes of arguments such as `--set 'checkpoints.dir="..."'`. In PowerShell 7, single commands work as written, but the shared flag string `$NZ` used in steps 0b and 1–3 must be an array:

```powershell
$NZ = @('--sanity', '--set', 'segmentation.intervals_file="results/fixes_v2/dry_run/segmentation.json"',
        '--set', 'sanity.members_max_epochs=10', '--set', 'sanity.weight_network_max_epochs=30',
        '--set', 'checkpoints.dir="checkpoints/nz_v2_sanity"', '--save-dir', 'results/fixes_v2/nz_wn')
python experiments/member_study.py @NZ --set 'member_study.variants=["warm_start"]' 2>&1 | tee results/fixes_v2/nz_wn/member_B.log
```

(that is, write `@NZ` where the bash commands say `$NZ`).

Every run below writes its console output to a log with `tee`. Keep the logs, because they are part of what you send back.

### Data folders

```text
data/
  CORDEX_NZ/                       NZ data, layout in README §2 (Zenodo v4 NZ_domain.zip + v7 1981-2000 target)
  India_sample/                    the two sample files (get them from Kartik; needed by the smoke tests)
    RF25_ind1990_rfp25.nc
    india_era5_1990_01_coarse_4x.nc
  India/
    imd/RF25_ind1990_rfp25.nc ... RF25_ind2019_rfp25.nc      IMD, one file per year (30 files)
    era5_raw/                      ERA5 0.25 deg as downloaded (written by the download script)
    era5/                          ERA5 4x4-coarsened files: this is what the runner reads
checkpoints/                       created by the runs (not in git)
```

- **IMD:** IMD Pune gridded rainfall, 0.25° × 0.25°, NetCDF, one file per year, **1990–2019**. The file names follow the sample (`RF25_ind{YEAR}_rfp25.nc`) and the variable is `RAINFALL`. Source: the IMD Pune data portal (imdpune.gov.in, "Gridded Rainfall (0.25 x 0.25) NetCDF"). Check the page for the current download link.
- **ERA5:** **1990-01-01 to 2020-01-01**. The extra January 2020 is only needed if the day-offset check picks offset 1.
  - Variables: specific humidity, temperature, u and v wind, vertical velocity, at **850 hPa**.
  - Area: N 38.5, W 66.5, S 6.5, E 100.
  - **Daily means preferred.** Use step 4, which also coarsens the data exactly like the sample.
- **NZ checkpoints:** the NZ member study needs the models already trained on Kartik's laptop (`checkpoints/nz_v2_sanity/rep0/`): `bgnet`, `member_0` to `member_5` and `weight_network_squared_emd`, about 1.1 GB. Ask Kartik for them and unpack them to `checkpoints/nz_v2_sanity/rep0/`. If that isn't possible, run step 0b.

### 0a. Smoke tests (about 3 min, required)

```bash
python tests/run_smoke_tests.py 2>&1 | tee smoke_tests.log
```

All 11 lines must say PASS. The India steps need `data/India_sample/`.

### 0b. Only if the NZ checkpoints were not copied: retrain them (about 1.5–2.5 h)

```bash
NZ='--sanity --set segmentation.intervals_file="results/fixes_v2/dry_run/segmentation.json" --set sanity.members_max_epochs=10 --set sanity.weight_network_max_epochs=30 --set checkpoints.dir="checkpoints/nz_v2_sanity" --save-dir results/fixes_v2/nz_wn'
python experiments/run_cordex_v2.py $NZ 2>&1 | tee nz_step0b.log
```

This reproduces the 10-epoch members and BGNet and the squared-EMD Weight Network. The numbers will differ slightly from Kartik's, so say in your report that you retrained them.

---

## 1–3. NZ member study

All three steps use the same flags. Run them in this order and one at a time.

```bash
NZ='--sanity --set segmentation.intervals_file="results/fixes_v2/dry_run/segmentation.json" --set sanity.members_max_epochs=10 --set sanity.weight_network_max_epochs=30 --set checkpoints.dir="checkpoints/nz_v2_sanity" --save-dir results/fixes_v2/nz_wn'
```

| Step | Command | Expected CPU time | Output |
|---|---|---|---|
| 1. B: members warm-started from BGNet | `python experiments/member_study.py $NZ --set 'member_study.variants=["warm_start"]' 2>&1 \| tee results/fixes_v2/nz_wn/member_B.log` | 20–50 min (worst 3.2 h) | `results/fixes_v2/nz_wn/sanity/member_study/results.json` |
| 2. A: members from scratch, to convergence | `python experiments/member_study.py $NZ 2>&1 \| tee results/fixes_v2/nz_wn/member_A.log` | 1.5–3.2 h (worst 3.2 h) | same file, plus the selection and the test table |
| 3. BGNet retrained to convergence | `python experiments/member_study.py $NZ --set member_study.retrain_bgnet=true 2>&1 \| tee results/fixes_v2/nz_wn/member_bgnet.log` | 1–3 h (worst 3.2 h) | same file, plus a `bgnet_converged` column |

Notes:
- **Step 2** reloads B from its checkpoints, trains A, picks the better set by validation Perfect-Member MSE and only then evaluates on test.
- **Step 3** reloads A and B and reports everything against both the 10-epoch BGNet and the converged one.

What to check after each step (in `results.json` and the log):
- **Training:** each member line shows `best_epoch` < `epochs`, i.e. early stopping, not the cap. `still_improving_at_cap` should be false.
- **Members vs BGNet:** `own_interval_val[*].ratio` is each member's MSE on its own validation interval divided by BGNet's. Below 1 means the member wins.
- **Perfect-Member:** `perfect_member_val_mse` should be well below the 10-epoch members' value of 116.7 ([0, 1]) and ideally close to BGNet's 60.8.
- **After step 2:** `selection.best_by_perfect_member_val_mse_0_1` gives the chosen set, and `selection.test` its test MSE next to BGNet.

**Decision for India:** if the chosen set is `warm_start`, set `"init": "warm_start"` in `configs/india_v1.json` under `members` before step 6. If it is `scratch`, keep the default.

---

## 4. India data download (network-bound; can run in parallel with steps 1–3)

```bash
python scripts/download_era5_india.py --years 1990 2020 --product daily_mean --out data/India 2>&1 | tee era5_download.log
```

- **Checks:** before the long download, open the CDS form of "ERA5 pressure levels daily statistics" (`derived-era5-pressure-levels-daily-statistics`), click "Show API request" and compare its keys with `request()` in the script. Also accept the dataset licence on the CDS website.
- **Size:** one request per month, 361 requests in all. CDS queue times vary from minutes to hours per request. The raw files are roughly 4 GB in total and the coarse files roughly 0.3 GB.
- **What it does:** it downloads to `data/India/era5_raw/` and writes the coarse files to `data/India/era5/`. Months that are already downloaded or coarsened are skipped.
- **The sample grid:** it is one **00 UTC snapshot** per day, while the default product above is a **daily mean**. If you want the sample's product instead, use `--product hourly --hours 00:00`. Either way, the day-offset check (step 5) must run on the files you actually downloaded.
- **Interrupted download:** run the same command again.
- **Leftover January 2020:** the script downloads all of 2020 because `--years` takes whole years. Delete the raw and coarse files for February–December 2020 if space matters; they are outside `data.period` and ignored.

IMD files go in `data/India/imd/` by hand, as described in the data folders above.

---

## 5. India: day-offset check (a few minutes)

```bash
python experiments/india_day_offset.py --config configs/india_v1.json 2>&1 | tee day_offset.log
```

Output: `results/india/qre_mass_v1/day_offset/day_offset.json`. It uses training and validation days only.

Check:
- `recommended_day_offset` and `score_difference`. Set `"day_offset"` in `configs/india_v1.json` to the recommended value. If the difference is below 0.005, keep 0 and say so in your report.
- If the loader reports missing days, the downloads are incomplete. Fix that before step 6; don't set `allow_missing_days` without asking.

**After this step, don't edit `configs/india_v1.json` again.** Steps 6–8 reuse each other's checkpoints, and a changed setting makes the loader raise "trained with different settings". That error is intended: it prevents mixing models trained under different configs. If a change is really needed, also set a new `checkpoints.dir` and `save_dir`, and start again from step 6.

---

## 6–8. India experiments

First create the folder for the logs: `mkdir -p results/india/qre_mass_v1`. `tee` does not create folders.

| Step | Command | Expected CPU time | Output (`results/india/qre_mass_v1/`) |
|---|---|---|---|
| 6. Experiment A (λ = 0) | `python experiments/run_india.py --config configs/india_v1.json --stage A 2>&1 \| tee results/india/qre_mass_v1/stage_A.log` | 4–12 h | `data_report.json`, `segmentation.json`, `A/` (metrics, loss scales, training log, plots) |
| 7. λ sweep (validation only) | `python experiments/run_india.py --config configs/india_v1.json --stage sweep 2>&1 \| tee results/india/qre_mass_v1/stage_sweep.log` | 1–7 h (3 λ > 0, each a short Weight Network fine-tune) | `sweep/sweep.json`, `sweep/selection.json` |
| 8. Experiment B (selected λ) | `python experiments/run_india.py --config configs/india_v1.json --stage B 2>&1 \| tee results/india/qre_mass_v1/stage_B.log` | 10–30 min (everything is loaded; evaluation and plots only) | `B/`, `comparison.json` |

(`--stage all` runs 6–8 in one go if you prefer.)

**Where step 6's time goes.** These are extrapolated from measured step times of 1.6 s per BGNet step and 1.9 s per Weight Network step at batch 91. With 21 training years, one epoch is 85 steps:
- single BGNet: about 2.5 min per epoch, patience 4, at most 100 epochs, so 0.5–4 h;
- members: about 2.5 min per epoch summed over all six, so 1–4 h;
- Weight Network: about 3 min per epoch, patience 15, at most 300 epochs. Expect 1.5–5 h; on NZ it was still improving at 30 epochs;
- ensemble stage with λ = 0: about 3 min per epoch, patience 5, at most 50 epochs, so 0.3–2.5 h.

**RAM:** the ensemble stage keeps all member predictions for the training days in memory, about 1 GB for 21 years. With less than about 8 GB of RAM, add `--set ensemble.precompute_member_predictions=false` to **all** India commands; it is slower. It is part of the checkpoint signature, so use the same value in steps 6–8.

What to check:

- **After step 6:**
  - **`data_report.json`:**
    - `days.missing_era5_days` and `days.missing_imd_days` should be empty;
    - `era5_unit_mismatches` should be `{}`;
    - `input_grid` should be `[32, 33]`, with `dropped_partial_blocks` lat [6.5] and lon [99.75];
    - `split.*.days` should be about 7670 / 1461 / 1826;
    - `target_train.fraction_exact_zero` should be large (the sample has 0.67).
  - **`segmentation.json`:** `n_degenerate_runs` should be small, and every entry of `val_days_per_interval` and `test_days_per_interval` should be above 0.
  - **`A/metrics.json`:**
    - `val.weight_network.collapsed` must be **false** (no interval gets more than 70 % of days), and `accuracy` should be clearly above the largest share in `true_histogram`;
    - compare `qre` with `bgnet` on `regime_mse`, and on `all_india_totals.daily_mae_mm_per_day` and `total_relative_discrepancy`;
    - `perfect_member` is the upper bound for QRE.
  - **`A/training_log.json`:** for every model, `best_epoch` < `epochs` and `still_improving_at_cap` false.
  - **`A/loss_scales.json`:** the two mass-loss forms on the real data.
  - **`A/plots/`:** look at all four PNGs.
- **After step 7:**
  - in `sweep/sweep.json`, for each λ: the Weight Network is not collapsed, the validation MSE barely changes and the daily-total MAE goes down as λ grows;
  - in `sweep/selection.json`: the selected λ and `tolerance_met`.
- **After step 8:** in `comparison.json`, A against B against BGNet on test. Check that the totals discrepancy is lower for B and how much MSE it costs. Look at the plots in `B/plots/`.

---

## Sending results back

Send these folders zipped; they are small, only JSON, PNG and logs:

- `results/fixes_v2/nz_wn/` (member study results and logs)
- `results/india/qre_mass_v1/` (everything, including the `.log` files)
- `smoke_tests.log`, `day_offset.log`, `era5_download.log`
- the final `configs/india_v1.json` you used (`results/india/qre_mass_v1/run_*.json` also records it)
- the machine: CPU or GPU model, RAM and OS

Don't send the checkpoints unless asked. NZ models are about 157 MB each; India BGNets are about 53 MB and Weight Networks about 20 MB.

---

## If a run is interrupted

Run **exactly the same command** again.

- Every finished model is saved in `checkpoints/...` and is loaded instead of retrained; the log shows `loaded ... from checkpoint`.
- Only the model that was training when the run stopped starts again from epoch 0. There are no mid-model checkpoints.
- Don't change the config or flags between the interrupted run and its restart.
- If you see "trained with different settings", a setting differs from the one the checkpoint was trained with. Restore the setting, or use a new `checkpoints.dir` and `save_dir`.
- The ERA5 download also resumes by rerunning it, skipping finished months.

---

## Optional: Kaggle GPU (not tested)

The code was only run on CPU with TensorFlow 2.10. Kaggle images ship a newer TensorFlow whose default Keras is Keras 3, and this code uses `tf.keras` (Keras 2). Try in this order, and run the smoke tests before any long run:

1. **Make a dataset:** create a Kaggle Dataset with `data/India/` (the coarse `era5/` files and `imd/`) and, for the smoke tests, `data/India_sample/`. Add the NZ data and NZ checkpoints as a second dataset if you run the NZ study there.
2. **Notebook:** use a GPU accelerator with Internet on.
   ```bash
   !git clone -b fixes-v2 https://github.com/rajatjain3366/precipitation-downscaling.git
   %cd precipitation-downscaling
   !pip install netCDF4 tf-keras
   import os; os.environ["TF_USE_LEGACY_KERAS"] = "1"   # tf.keras -> Keras 2; before importing TF
   !mkdir -p data && ln -s /kaggle/input/<your-dataset> data/India
   !TF_USE_LEGACY_KERAS=1 python tests/run_smoke_tests.py
   ```
   If `tf-keras` does not match Kaggle's TensorFlow version, install the `tf-keras` release with the same minor version, for example `tf-keras==2.17.*` for TF 2.17.
3. **Runs:** use the same commands with `TF_USE_LEGACY_KERAS=1` in front, and write outputs to `/kaggle/working`: add `--save-dir /kaggle/working/results/india/qre_mass_v1 --set 'checkpoints.dir="/kaggle/working/checkpoints/india"'` to **every** India command.
4. **Session limit:** Kaggle sessions stop after about 12 h. Save `/kaggle/working/checkpoints` as an output dataset, attach it in the next session, copy it back to the same path and rerun the command. The run resumes as described above.
5. **If the smoke tests fail on Kaggle,** stay on CPU and send the failing output.
