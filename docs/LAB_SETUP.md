# Lab PC setup: running all trainings with `scripts/run_lab.py`

This guide sets up the college lab PC, copies the data, checks the GPU, runs every training step of [RUNBOOK.md](RUNBOOK.md) with one command, and copies the results back. The main path is **Linux with an NVIDIA GPU and no admin rights**. Windows is covered briefly at the end. The RUNBOOK remains the reference for what each step does and what to check in its outputs.

**Total time:** about 9–32 h on a CPU (RUNBOOK estimates, steps 1–9), plus 1.5–2.5 h if the NZ checkpoints have to be retrained (step 0). On the lab GPU (RTX 4000 Ada), steps 0–9 took about 25 min. Steps 10–14 (RUNBOOK §10–14) are estimated at about 2–3.5 h on that GPU.

**GPU required by default:** `run_lab.py` checks at startup whether TensorFlow sees a GPU and stops if it does not. Add `--allow-cpu` to run on the CPU anyway.

---

## 1. Software (Linux, no admin rights)

TensorFlow 2.10.1 (`requirements.txt`) needs **Python 3.7–3.10** and, for the GPU, **CUDA 11.2 and cuDNN 8.1**. The lab's system CUDA, if there is one, is probably a different version. Miniconda in your home folder provides the right ones without admin rights.

### 1.1 Miniconda in `~/miniconda3`

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh   # or: curl -O <same URL>
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash      # adds conda to ~/.bashrc
exec bash                             # reload the shell; the prompt now starts with (base)
```

The environment with CUDA, cuDNN and TensorFlow takes a few GB in your home folder, and the data and checkpoints need more. Check your quota or free space first with `df -h ~` (some labs also have `quota -s`).

### 1.2 Environment with CUDA 11.2 and cuDNN 8.1

```bash
conda create -n qre python=3.10
conda activate qre
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
```

- **Python version:** `python=3.10` installs the newest 3.10.x, which is fine. The code was tested with 3.10.11; use `python=3.10.11` to match exactly.
- **Terms of Service:** if conda stops and asks you to accept the Terms of Service of its default channels, either accept as it prompts, or use conda-forge only: `conda create -n qre -c conda-forge --override-channels python=3.10`.
- **`LD_LIBRARY_PATH` is needed in every new shell, including inside tmux.** It tells TensorFlow where the conda CUDA and cuDNN libraries are. To have it set automatically by `conda activate qre`, run this once, with the environment active:

  ```bash
  mkdir -p $CONDA_PREFIX/etc/conda/activate.d
  echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
  ```

### 1.3 Repository and Python packages

```bash
cd ~
git clone https://github.com/rajatjain3366/precipitation-downscaling.git
cd precipitation-downscaling
pip install -r requirements.txt netCDF4==1.7.4 pytest
```

`netCDF4` reads the `.nc` files and `pytest` runs the smoke tests; neither is in `requirements.txt`. All commands below assume the `qre` environment is active, `LD_LIBRARY_PATH` is set and the current folder is the repository.

### 1.4 GPU check (must show a GPU)

```bash
nvidia-smi
python scripts/run_lab.py --check-gpu
```

The check prints:

- the GPU name and driver version from `nvidia-smi`;
- the TensorFlow version and whether it was built with CUDA;
- the number of GPUs TensorFlow sees, and any CUDA library it could not load.

It ends with "TensorFlow sees a GPU; training will use it" (exit code 0) or "TensorFlow sees NO GPU" (exit code 1). `run_lab.py` repeats this check at every start and writes it to `logs/gpu_check.log`.

### 1.5 If no GPU is shown

Work through these in order and rerun `python scripts/run_lab.py --check-gpu` after each:

1. **Library path not set.** This is the most common cause. It shows up as "Could not load dynamic library 'libcudart.so.11.0'" or 'libcudnn.so.8', and the check warns when `$CONDA_PREFIX/lib` is not on `LD_LIBRARY_PATH`. Fix:

   ```bash
   conda activate qre
   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
   ls $CONDA_PREFIX/lib/libcudart.so.11.0 $CONDA_PREFIX/lib/libcudnn.so.8   # both must exist
   ```

   If the `ls` fails, the conda install of `cudatoolkit` / `cudnn` did not finish; run it again.
2. **Wrong Python.** `which python` must point into `~/miniconda3/envs/qre/`, and `pip show tensorflow` must say 2.10.1.
3. **Driver too old for CUDA 11.2.** `nvidia-smi` prints the driver version. CUDA 11.2 needs at least 450.80.02, and 460.27.03 or newer is the safe choice. Newer drivers are fine; they also run CUDA 11.2 programs. Updating the driver needs admin rights, so ask the lab admin. Until then, only `--allow-cpu` works.
4. **GPU hidden.** `echo $CUDA_VISIBLE_DEVICES` should print nothing or GPU numbers such as `0`. `-1` or an empty-but-set value hides all GPUs; `unset CUDA_VISIBLE_DEVICES` fixes it.
5. **No GPU in `nvidia-smi` at all:** the driver or the GPU is the problem, not this setup. Ask the lab admin.

If TensorFlow sees the GPU, but training or the smoke tests fail with **`CUDNN_STATUS_INTERNAL_ERROR`** or **`CUDNN_STATUS_EXECUTION_FAILED`** (often in `bg_net/conv2d`):

6. **GPU memory held by another process.** By default TensorFlow reserves all GPU memory, so a second process, such as the training subprocess of a test, finds none left.
   - The code now sets `TF_FORCE_GPU_ALLOW_GROWTH=true` itself (`src/gpu_setup.py`; also `run_lab.py`, the smoke runner and the tests), so each process takes only what it needs. Check that nothing sets it to `false`: `echo $TF_FORCE_GPU_ALLOW_GROWTH` should print `true` or nothing.
   - `nvidia-smi` lists the processes using the GPU and their memory. Other users' jobs can fill it too; then choose a free GPU with `CUDA_VISIBLE_DEVICES`, or wait.
7. **cuDNN 8.1 on an Ada GPU** (RTX 40 series, RTX 4000/6000 Ada, L4, L40; compute capability 8.9). TensorFlow 2.10 was built against CUDA 11.2 and cuDNN 8.1, which predate Ada. It loads the libraries by major version only (`libcudart.so.11.0`, `libcudnn.so.8`), so the last 11.x / 8.x releases, CUDA 11.8 and cuDNN 8.9, which support Ada, can replace them. Try them in a copy of the environment, so the working one stays as it is:

   ```bash
   conda create -n qre118 --clone qre
   conda activate qre118
   conda install -c conda-forge cudatoolkit=11.8 "cudnn=8.9.*=*cuda11*"
   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
   python scripts/run_lab.py --check-gpu
   python tests/run_smoke_tests.py 2>&1 | tee smoke_tests.log
   ```

   - The `*cuda11*` part selects a cuDNN build for CUDA 11; the cuDNN 8.9 builds for CUDA 12 would not load. If conda cannot find it, `conda search -c conda-forge cudnn` lists the builds; choose an 8.9 one with `cuda11` in its build string.
   - From then on, use `qre118` instead of `qre` everywhere (also inside tmux).
   - In the report, give the versions actually used: `conda list | grep -E "cudatoolkit|cudnn"`.

If none of this helps, run on the CPU with `--allow-cpu`. All RUNBOOK estimates are CPU times. Send `logs/gpu_check.log` with the results.

**Very new GPUs** (newer than TensorFlow 2.10, which dates from 2022) may make the first start take several minutes while TensorFlow compiles its GPU code. This happens once per process, not per epoch. If cuDNN errors appear on such a GPU, see item 7.

### 1.6 Sharing the GPU

TensorFlow reserves all GPU memory by default. The code turns that off with `TF_FORCE_GPU_ALLOW_GROWTH=true` (see item 6), so each run takes only the memory it needs and others can still use the GPU. Exporting it yourself, as in 4.2, does no harm. `nvidia-smi` shows who else is using it. On a PC with several GPUs, choose one with, for example, `export CUDA_VISIBLE_DEVICES=1`.

---

## 2. Data

None of the data is in git (`data/` and `checkpoints/` are ignored). Copy these folders into the repository, with exactly these names. Linux is case-sensitive: `IMD_jjas`, not `IMD_JJAS`.

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

- **What to copy:** all of these, using a USB drive or `scp` / `rsync` over the network:
  - `data/CORDEX_NZ/` (the five files above)
  - **`data/India_sample/`**: the smoke tests need it (`test_india.py` and the India runner and day-offset smoke steps), even though no training uses it
  - `data/India/ERA5_full/` and `data/India/IMD_jjas/`
  - `checkpoints/nz_v2_sanity/rep0/`, if available
- **From the Windows laptop** (PowerShell has `scp`), first `ssh <user>@<lab-pc> "mkdir -p ~/precipitation-downscaling/data/India ~/precipitation-downscaling/checkpoints/nz_v2_sanity"`, then:

  ```powershell
  scp -r data\CORDEX_NZ data\India_sample <user>@<lab-pc>:~/precipitation-downscaling/data/
  scp -r data\India\ERA5_full data\India\IMD_jjas <user>@<lab-pc>:~/precipitation-downscaling/data/India/
  scp -r checkpoints
z_v2_sanity
ep0 <user>@<lab-pc>:~/precipitation-downscaling/checkpoints/nz_v2_sanity/
  ```

- **From Linux:** `rsync -av --progress data/CORDEX_NZ data/India_sample <user>@<lab-pc>:~/precipitation-downscaling/data/`, and likewise for `data/India/ERA5_full`, `data/India/IMD_jjas` and the checkpoints.
- **Check on the lab PC:** `ls data/India_sample data/India/ERA5_full | head` and `ls data/India/IMD_jjas | wc -l` (46).
- **NZ data:** from Kartik's copy, or downloaded as in README §2. The other files in Kartik's `data/CORDEX_NZ/` (the `imperfect` predictors, `EC-Earth3_1981-2000.nc`) are not used and need not be copied.
- **NZ checkpoints:** copy `checkpoints/nz_v2_sanity/rep0/` if you can. It must contain `bgnet`, `member_0` to `member_5` and `weight_network_squared_emd`, each as a `.json` and a `.weights.h5` file. If it is missing or incomplete, `run_lab.py` retrains these models first (step 0), and the member-study numbers will differ slightly from Kartik's.
- **Already in git:** the NZ segmentation that steps 0–3 need (`results/fixes_v2/dry_run/segmentation.json`) and the mid-term reference results (`results/final_cordex_qre/`).

`python scripts/run_lab.py --dry-run` marks each missing data folder with "data folder missing". A real run refuses to start a step whose data folder is missing.

---

## 3. Smoke tests (a few minutes, required)

```bash
python tests/run_smoke_tests.py 2>&1 | tee smoke_tests.log
```

All 13 lines must say PASS. If any fails, don't start the long runs; send `smoke_tests.log`. On the GPU this is also the first test that the code runs there at all.

- **India steps:** they need `data/India_sample/` (section 2).
- **Comparisons on a GPU:** the tests that compare two evaluations of the same model (for example "λ = 0 reproduces the original objective") are exact on a CPU and use a relative tolerance of 1e-5 on a GPU (`tests/gpu_tol.py`), as `nz_mass.py` does.
- **cuDNN errors:** see 1.5, items 6 and 7.

---

## 4. Running

### 4.1 The plan

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
| 10 | `india_member_study` | India member study: scratch vs warm start, selected on validation | GPU 2–5 min |
| 11 | `nz_placements` | NZ: mass placements × λ on the member-study models | GPU 20–40 min |
| 12 | `india_placements` | India: mass placements × λ | GPU 5–25 min |
| 13 | `nz_repetitions` | NZ: 10 repetitions, mean ± std, Wilcoxon tests | GPU 1–1.5 h |
| 14 | `india_repetitions` | India: the same | GPU 30–70 min |

The GPU estimates for steps 10–14 come from the measured times of the first lab run (RUNBOOK §10–14). The dry run also prints each finished step's measured duration. Before step 10, run the slow integration tests once: `python -m pytest -m slow tests/test_mass_study.py` (RUNBOOK §10–14).

### 4.2 Start inside tmux

tmux keeps the run going after you log out or lose the SSH connection. If `tmux` is not installed, install it into the environment without admin rights: `conda install -c conda-forge tmux`.

```bash
tmux new -s qre
# inside tmux: a new shell, so activate the environment and set the library path again
conda activate qre
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH   # not needed if you set up the activate.d hook (1.2)
export TF_FORCE_GPU_ALLOW_GROWTH=true                        # shared GPU (1.6)
cd ~/precipitation-downscaling
python scripts/run_lab.py --check-gpu
python scripts/run_lab.py
```

- **Detach:** press **Ctrl+B, then D**. The run continues, and you can log out.
- **Reattach:** `tmux attach -t qre`.
- **Watch without attaching:** `cat logs/summary.txt` after each step, or `tail -f logs/<step>.log` for the current step.
- **Stop:** reattach and press Ctrl+C. The runner stops the current step and records it as interrupted.

### 4.3 Fallback: nohup

If tmux cannot be used:

```bash
conda activate qre
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
export TF_FORCE_GPU_ALLOW_GROWTH=true
cd ~/precipitation-downscaling
nohup python scripts/run_lab.py > run_lab.out 2>&1 &
echo $! > run_lab.pid
tail -f run_lab.out                 # Ctrl+C here only stops tail, not the run
```

To stop the run, use `kill -INT $(cat run_lab.pid)`. That is the equivalent of Ctrl+C: the runner then also stops the step it started. A plain `kill` would leave that step running on its own.

Some lab PCs end all of a user's processes at logout, tmux included. If a run dies whenever you log out, ask the lab admin. Rerunning the same command resumes it.

### 4.4 What the runner does

- **Order:** the steps run one after another. If one fails, the runner stops; later steps depend on earlier ones.
- **Logs:** each step's output appears on the screen and is written to `logs/<step>.log`, and also to the log file the RUNBOOK names (for example `results/india/qre_mass_v1/stage_A.log`). Each attempt is appended with a time stamp and its exit code.
- **Summary:** at the end, and after every step, the status of every step is written to `logs/summary.txt` and `logs/summary.json`: finished (with its duration), already done, failed, not run or not selected. The summary also records whether a GPU was used.
- **No GPU:** the runner stops before any step, unless `--allow-cpu` is given.

### 4.5 Interruptions and restarts

- **To resume,** run the same command again: after a reboot, a crash, Ctrl+C or a failed step you have fixed. Finished steps are skipped. The step that was interrupted starts again, and its finished models load from `checkpoints/` instead of being retrained.
- **When a step counts as finished:** the runner recorded it as finished (`logs/<step>.done`) and its outputs are still there. Outputs alone are not enough: `results/india/qre_mass_v1/data_check.json`, for example, is committed from Kartik's laptop, but the check has to run on this PC's copy of the data. Step 0 is the exception: copied NZ checkpoints count as finished.
- **`--from-step N`** (a number or a name, e.g. `--from-step india_A`) starts at step N; earlier steps are not run. `--to-step N` stops after step N, e.g. `--to-step 4` for the NZ part only.
- **`--force`** reruns the selected steps even if they are finished.
- **Stay on one device.** Don't switch between GPU and CPU (`--allow-cpu`) partway through. Resuming works across devices, but a mix makes the results harder to interpret. If it happens, say so in the report.

### 4.6 The two decisions the RUNBOOK makes by hand

The runner applies them by the RUNBOOK's rules and passes them to steps 7–9 as `--set` overrides. `configs/india_v1.json` is not edited, so don't edit it either.

| Setting | Rule | Override |
| --- | --- | --- |
| `members.init` | The member set step 2 selected (`selection.best_by_perfect_member_val_mse_0_1` in the member-study `results.json`) | `--members-init scratch` or `warm_start` |
| `data.day_offset` | `recommended_day_offset` from step 6, but 0 if `score_difference` is below 0.005 | `--day-offset 0` or `1` |

The values, and why they were chosen, are saved in `logs/decisions.json` when step 7 first runs. Every later run reuses them, because steps 7–9 must share one configuration; their checkpoints refuse other settings. To change them afterwards, start the India experiments again: delete `checkpoints/india/qre_mass_v1/`, `results/india/qre_mass_v1/A`, `sweep`, `B` and `logs/decisions.json`, or ask Kartik first.

**Less than about 8 GB of RAM:** add `--india-set ensemble.precompute_member_predictions=false` (RUNBOOK steps 7–9). It is saved with the decisions and applies to all three steps.

### 4.7 Options

| Option | Effect |
| --- | --- |
| `--dry-run` | Print the plan and the commands only |
| `--check-gpu` | Only the GPU check |
| `--allow-cpu` | Run even if TensorFlow sees no GPU |
| `--from-step`, `--to-step` | Limit the steps (number or name) |
| `--force` | Rerun finished steps |
| `--members-init`, `--day-offset`, `--india-set` | Override the decisions for steps 7–9 (4.6) |
| `--logs DIR` | Another log folder (default `logs`) |

---

## 5. GPU results are not bit-identical to CPU results

A GPU computes the same models with different kernels and summation orders. Its results therefore differ from CPU runs in the last digits. They also differ between two GPU runs, because cuDNN may choose different algorithms. Over hundreds of training epochs these small differences grow, so trained models and metrics will not match Kartik's CPU numbers exactly, though they should be close. This is expected and not an error; say in the report that the runs used a GPU.

One consequence for the RUNBOOK checks:

- **Step 4 check:** it requires λ = 0 to reproduce the NZ v2 QRE test MSEs. On a CPU this is exact. On a GPU, `lambda0_reproduces_nz_v2_qre_exactly` in `comparison.json` may be `false`, which is fine as long as `lambda0_max_relative_difference` is at most 1e-5. Above that, the run stops with "lambda = 0 does not reproduce the NZ v2 QRE test MSEs".

---

## 6. After the run

1. **Summary:** read `logs/summary.txt`. Every step should say `finished` or `already done`.
2. **Checks:** go through the "what to check" lists of the RUNBOOK for steps 1–9.
3. **Copy back:** pack these folders. Without the checkpoints they are small (JSON, PNG, logs):
   - `logs/`, including `decisions.json`, `summary.txt` and `gpu_check.log`
   - `results/fixes_v2/nz_wn/`
   - `results/fixes_v2/nz_mass/`
   - `results/india/qre_mass_v1/` (with steps 10–14: `member_study/`, `placements/`, `repetitions/`; NZ steps 11 and 13 are in `results/fixes_v2/nz_wn/sanity/mass_study/`)
   - `smoke_tests.log`, `day_offset.log` and, if step 0 ran, `nz_step0b.log`

   ```bash
   cd ~/precipitation-downscaling
   tar czf lab_results.tar.gz logs results/fixes_v2/nz_wn results/fixes_v2/nz_mass results/india/qre_mass_v1 \
       smoke_tests.log day_offset.log $(ls nz_step0b.log 2>/dev/null)
   ```

   Then fetch it from the laptop, in PowerShell at the root of the repository, and unpack it there. Windows 10 and 11 include `tar`, and the paths inside the archive are the repository paths:

   ```powershell
   scp <user>@<lab-pc>:~/precipitation-downscaling/lab_results.tar.gz .
   tar -xzf lab_results.tar.gz
   ```

   A USB drive works as well.
4. **Machine details:** with the archive, send the CPU and GPU model, the driver version, the RAM and the OS. `logs/gpu_check.log` has the GPU, driver and TensorFlow details.
5. **Checkpoints:** keep `checkpoints/` on the lab PC until the results have been read. Don't send them unless asked. NZ models are about 157 MB each; India BGNets about 53 MB and Weight Networks about 20 MB.

---

## Appendix: Windows

On a Windows PC, the same steps work with these changes:

```powershell
git clone https://github.com/rajatjain3366/precipitation-downscaling.git
cd precipitation-downscaling
py -3.10 -m venv .venv                 # Python 3.10.11 64-bit from python.org
.venv\Scripts\activate
pip install -r requirements.txt netCDF4==1.7.4 pytest
python scripts/run_lab.py --check-gpu
```

- **GPU:** TensorFlow 2.10 is the last version that uses a GPU on native Windows. It also needs CUDA 11.2 and cuDNN 8.1, on `PATH`; with conda, the same `conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0` provides them, and no `LD_LIBRARY_PATH` is needed.
- **Without a GPU:** run with `--allow-cpu`.
- **Instead of tmux:** keep the PC awake (Settings → System → Power → Sleep: Never) and leave the window open.
- **Copying back:** `Compress-Archive -Path logs, results/fixes_v2/nz_wn, results/fixes_v2/nz_mass, results/india/qre_mass_v1, smoke_tests.log, day_offset.log -DestinationPath lab_results.zip`.
