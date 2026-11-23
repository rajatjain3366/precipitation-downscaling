# Lab PC setup: running all trainings with `scripts/run_lab.py`

This guide sets up a fresh Windows or Linux PC, copies the data, checks the GPU, runs every training step of [RUNBOOK.md](RUNBOOK.md) with one command, and copies the results back. The RUNBOOK remains the reference for what each step does and what to check in its outputs.

**Total time:** about 9–32 h on a CPU (RUNBOOK estimates, steps 1–9), plus 1.5–2.5 h if the NZ checkpoints have to be retrained (step 0). A GPU should be faster but has not been measured.

---

## 1. Software

### Git and Python 3.10.11

The code was developed and tested with **Python 3.10.11, 64-bit** and TensorFlow 2.10.1 (`requirements.txt`). Use the same Python version; TensorFlow 2.10 runs on Python 3.7–3.10 only, so 3.11 or newer will not work.

- **Windows:**
  - Git for Windows: <https://git-scm.com/download/win>. It also installs Git Bash.
  - Python 3.10.11, "Windows installer (64-bit)": <https://www.python.org/downloads/release/python-31011/>. Without admin rights, choose "Install for me only". The `py` launcher then finds it as `py -3.10`.
- **Linux:**
  - Ubuntu 22.04 ships Python 3.10: `sudo apt install git python3.10 python3.10-venv`.
  - On other distributions, or without root, install 3.10.11 with [pyenv](https://github.com/pyenv/pyenv): `pyenv install 3.10.11`.

### Repository and virtual environment

Windows (PowerShell or Git Bash):

```powershell
git clone https://github.com/rajatjain3366/precipitation-downscaling.git
cd precipitation-downscaling
py -3.10 -m venv .venv
.venv\Scripts\activate                 # Git Bash: source .venv/Scripts/activate
python --version                       # must print Python 3.10.x
python -m pip install --upgrade pip
pip install -r requirements.txt netCDF4==1.7.4 pytest
```

If PowerShell refuses to run the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call the venv's Python directly as `.venv\Scripts\python.exe`.

Linux:

```bash
git clone https://github.com/rajatjain3366/precipitation-downscaling.git
cd precipitation-downscaling
python3.10 -m venv .venv               # pyenv: ~/.pyenv/versions/3.10.11/bin/python -m venv .venv
source .venv/bin/activate
python --version                       # must print Python 3.10.x
python -m pip install --upgrade pip
pip install -r requirements.txt netCDF4==1.7.4 pytest
```

All commands below assume the venv is active and the current folder is the repository.

---

## 2. Data

None of the data is in git (`data/` and `checkpoints/` are ignored). Copy these folders into the repository, with exactly these names:

```text
precipitation-downscaling/
  data/
    CORDEX_NZ/                     NZ data: the five files listed in README §2 (Zenodo)
      train/ESD_pseudo_reality/predictors/ACCESS-CM2_1961-1980.nc
      train/ESD_pseudo_reality/predictors/Static_fields.nc
      train/ESD_pseudo_reality/target/pr_tasmax_ACCESS-CM2_1961-1980.nc
      test/historical/predictors/perfect/ACCESS-CM2_1981-2000.nc
      test/historical/target/pr_tasmax_ACCESS-CM2_1981-2000.nc
    India_sample/                  needed only by the smoke tests
      RF25_ind1990_rfp25.nc
      india_era5_1990_01_coarse_4x.nc
    India/
      ERA5_full/era5_1971_coarse.nc ... era5_2016_coarse.nc   46 files
      IMD_jjas/RF25_ind1971_jjas.nc ... RF25_ind2016_jjas.nc   46 files
  checkpoints/
    nz_v2_sanity/rep0/             optional: the NZ base models from Kartik's laptop (about 1.1 GB)
```

- **NZ data:** from Kartik's copy, or downloaded as described in README §2. The other files in Kartik's `data/CORDEX_NZ/` (the `imperfect` predictors, `EC-Earth3_1981-2000.nc`) are not used and need not be copied.
- **India data and sample:** from Kartik's copy. Keep the folder names as shown; on Linux they are case-sensitive (`IMD_jjas`, not `IMD_JJAS`).
- **NZ checkpoints:** copy `checkpoints/nz_v2_sanity/rep0/` from Kartik's laptop if you can. It must contain `bgnet`, `member_0` to `member_5` and `weight_network_squared_emd`, each as a `.json` and a `.weights.h5` file. If it is missing or incomplete, `run_lab.py` retrains these models first (step 0, 1.5–2.5 h), and the member-study numbers will differ slightly from Kartik's.
- **Files that are in git:** the NZ segmentation that steps 0–3 need (`results/fixes_v2/dry_run/segmentation.json`) and the mid-term reference results (`results/final_cordex_qre/`) come with the clone.

`python scripts/run_lab.py --dry-run` marks each missing data folder with "data folder missing". A real run refuses to start a step whose data folder is missing.

---

## 3. GPU

### Check

```bash
nvidia-smi                              # the NVIDIA driver works and shows the GPU
python scripts/run_lab.py --check-gpu   # does TensorFlow see it?
```

The second command prints the TensorFlow version, whether it was built with CUDA, and the number of GPUs TensorFlow sees. It ends with either "TensorFlow sees a GPU; training will use it" or "TensorFlow sees NO GPU; training runs on the CPU". `run_lab.py` runs the same check at the start of every run and writes it to `logs/gpu_check.log`.

### If TensorFlow sees no GPU

TensorFlow 2.10.1 needs **CUDA 11.2 and cuDNN 8.1**. It finds them through the system path (Windows `PATH`, Linux `LD_LIBRARY_PATH`). It is also the last TensorFlow version that uses a GPU on native Windows. Without admin rights, the simplest route is a conda environment instead of the venv:

```bash
conda create -n qre python=3.10
conda activate qre
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install -r requirements.txt netCDF4==1.7.4 pytest
# Linux only, in every new shell:
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python scripts/run_lab.py --check-gpu
```

If this does not work either, run on the CPU: all estimates in the RUNBOOK are CPU times. A GPU was never tested with this code, so run the smoke tests (section 4) on it first, and say in your report whether a GPU was used. GPU results can differ slightly from CPU results.

**On a shared PC:** TensorFlow reserves all GPU memory by default. Set `TF_FORCE_GPU_ALLOW_GROWTH=true` (PowerShell: `$env:TF_FORCE_GPU_ALLOW_GROWTH = "true"`) before starting so that others can still use the GPU. To stop the runs from using a GPU at all, set `CUDA_VISIBLE_DEVICES=-1`.

---

## 4. Smoke tests (about 3 min, required)

```bash
python tests/run_smoke_tests.py 2>&1 | tee smoke_tests.log
```

All 12 lines must say PASS. If any fails, don't start the long runs; send `smoke_tests.log`.

---

## 5. Running

### The plan

```bash
python scripts/run_lab.py --dry-run
```

This prints every step with its command, its time estimate, its log files and whether it will run or be skipped. Nothing is run or written.

| Step | Name | What | Estimate (CPU) |
| --- | --- | --- | --- |
| 0 | `nz_checkpoints` | NZ base models (RUNBOOK 0b), only if `checkpoints/nz_v2_sanity/rep0` is incomplete | 1.5–2.5 h |
| 1 | `nz_member_B` | NZ member study B: members warm-started from BGNet | 20–50 min |
| 2 | `nz_member_A` | NZ member study A: members from scratch, then the selection | 1.5–3.2 h |
| 3 | `nz_member_bgnet` | NZ member study: BGNet retrained to convergence | 1–3 h |
| 4 | `nz_mass` | NZ: QRE vs QRE + mass conservation | 3–14 h |
| 5 | `india_data_check` | India data check; `data_check.json` must say `"ok": true` | minutes |
| 6 | `india_day_offset` | India day-offset check | minutes |
| 7 | `india_A` | India experiment A (λ = 0) | 2–7 h |
| 8 | `india_sweep` | India λ sweep (validation only) | 0.5–3.5 h |
| 9 | `india_B` | India experiment B (selected λ) | 10–30 min |

### The run

```bash
python scripts/run_lab.py
```

- **Order:** the steps run one after another. If one fails, the runner stops; later steps depend on earlier ones.
- **Logs:** each step's output appears on the screen and is written to `logs/<step>.log`, and also to the log file the RUNBOOK names (for example `results/india/qre_mass_v1/stage_A.log`). Each attempt is appended with a time stamp and its exit code.
- **Summary:** at the end, and after every step, the status of every step is written to `logs/summary.txt` and `logs/summary.json`: finished (with its duration), already done, failed, not run or not selected.
- **Quoting:** the runner passes the arguments directly, so it works the same from PowerShell 5.1, PowerShell 7, cmd, Git Bash or a Linux shell. The quoting problems described in the RUNBOOK don't apply.

### Interruptions and restarts

- **To resume,** run the same command again. Finished steps are skipped. The step that was interrupted starts again, and its finished models load from `checkpoints/` instead of being retrained.
- **When a step counts as finished:** the runner recorded it as finished (`logs/<step>.done`) and its outputs are still there. Outputs alone are not enough: `results/india/qre_mass_v1/data_check.json`, for example, is committed from Kartik's laptop, but the check has to run on this PC's copy of the data. Step 0 is the exception: copied NZ checkpoints count as finished.
- **`--from-step N`** (a number or a name, e.g. `--from-step india_A`) starts at step N; earlier steps are not run. `--to-step N` stops after step N, e.g. `--to-step 4` for the NZ part only.
- **`--force`** reruns the selected steps even if they are finished.
- **Keep the PC awake.**
  - **Windows:** Settings → System → Power → Sleep: Never (or `powercfg /change standby-timeout-ac 0`). Ask the lab admin whether the PC logs users out or restarts at night.
  - **Linux:** start the runner inside `tmux` or `screen`, or with `nohup python scripts/run_lab.py > run_lab.out 2>&1 &`, so that closing the terminal or SSH session doesn't stop it.

### The two decisions the RUNBOOK makes by hand

The runner applies them by the RUNBOOK's rules and passes them to steps 7–9 as `--set` overrides. `configs/india_v1.json` is not edited, so don't edit it either.

| Setting | Rule | Override |
| --- | --- | --- |
| `members.init` | The member set step 2 selected (`selection.best_by_perfect_member_val_mse_0_1` in the member-study `results.json`) | `--members-init scratch` or `warm_start` |
| `data.day_offset` | `recommended_day_offset` from step 6, but 0 if `score_difference` is below 0.005 | `--day-offset 0` or `1` |

The values, and why they were chosen, are saved in `logs/decisions.json` when step 7 first runs. Every later run reuses them, because steps 7–9 must share one configuration; their checkpoints refuse other settings. To change them afterwards, start the India experiments again: delete `checkpoints/india/qre_mass_v1/`, `results/india/qre_mass_v1/A`, `sweep`, `B` and `logs/decisions.json`, or ask Kartik first.

**Less than about 8 GB of RAM:** add `--india-set ensemble.precompute_member_predictions=false` (RUNBOOK steps 7–9). It is saved with the decisions and applies to all three steps.

### Other options

| Option | Effect |
| --- | --- |
| `--check-gpu` | Only the GPU check |
| `--no-gpu-check` | Skip the GPU check at the start |
| `--require-gpu` | Stop if TensorFlow sees no GPU |
| `--logs DIR` | Another log folder (default `logs`) |

---

## 6. After the run

1. **Summary:** read `logs/summary.txt`. Every step should say `finished` or `already done`.
2. **Checks:** go through the "what to check" lists of the RUNBOOK for steps 1–9.
3. **Copy back:** zip these folders. Without the checkpoints they are small (JSON, PNG, logs):
   - `logs/`, including `decisions.json`, `summary.txt` and `gpu_check.log`
   - `results/fixes_v2/nz_wn/`
   - `results/fixes_v2/nz_mass/`
   - `results/india/qre_mass_v1/`
   - `smoke_tests.log`, `day_offset.log` and, if step 0 ran, `nz_step0b.log`

   Windows PowerShell:

   ```powershell
   Compress-Archive -Path logs, results/fixes_v2/nz_wn, results/fixes_v2/nz_mass, results/india/qre_mass_v1, smoke_tests.log, day_offset.log -DestinationPath lab_results.zip
   ```

   Linux or Git Bash:

   ```bash
   zip -r lab_results.zip logs results/fixes_v2/nz_wn results/fixes_v2/nz_mass results/india/qre_mass_v1 smoke_tests.log day_offset.log
   ```

   Add `nz_step0b.log` to the list if it exists. Copy `lab_results.zip` to a USB drive or cloud storage, and unzip it at the root of the repository on Kartik's laptop: the paths inside it are the repository paths.
4. **Machine details:** with the zip, send the CPU and GPU model, the RAM, the OS and whether TensorFlow used the GPU (`logs/gpu_check.log`).
5. **Checkpoints:** keep `checkpoints/` on the lab PC, or on an external drive, until the results have been read. Don't send them unless asked. NZ models are about 157 MB each; India BGNets about 53 MB and Weight Networks about 20 MB.
