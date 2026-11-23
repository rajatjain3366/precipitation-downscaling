"""Run every RUNBOOK training step in order, unattended, on the lab PC.

Steps (docs/RUNBOOK.md):
  0  nz_checkpoints    NZ base models (RUNBOOK 0b); only if checkpoints/nz_v2_sanity/rep0 is incomplete
  1  nz_member_B       NZ member study B: members warm-started from BGNet
  2  nz_member_A       NZ member study A: members from scratch, then the selection
  3  nz_member_bgnet   NZ member study: BGNet retrained to convergence
  4  nz_mass           NZ: QRE vs QRE + mass conservation
  5  india_data_check  India data check (data_check.json must say "ok": true)
  6  india_day_offset  India day-offset check
  7  india_A           India experiment A (lambda = 0)
  8  india_sweep       India lambda sweep (validation only)
  9  india_B           India experiment B (selected lambda)

Usage:
  python scripts/run_lab.py                 # run everything not finished yet
  python scripts/run_lab.py --dry-run       # print the plan and the commands only
  python scripts/run_lab.py --from-step 7   # start at step 7 (number or name)
  python scripts/run_lab.py --check-gpu     # only print whether TensorFlow sees a GPU

Resuming: a step counts as finished when this runner recorded it as finished
(logs/<step>.done) and its expected outputs are still there; finished steps are
skipped, so after an interruption just run the same command again. A step that
was interrupted is restarted, and its models resume from their checkpoints.
Step 0 is the exception: it counts as finished whenever the NZ checkpoints are
present, because they may have been copied from another machine. Outputs alone
are not enough, because some (e.g. results/india/qre_mass_v1/data_check.json)
are committed from the author's laptop and must be redone on this machine.

The RUNBOOK's two manual decisions are applied automatically, by the RUNBOOK's
rules, and passed to steps 7-9 as --set overrides (configs/india_v1.json is not
edited):
  members.init     the member set chosen in step 2 (selection.best_by_perfect_member_val_mse_0_1)
  data.day_offset  recommended_day_offset from step 6, or 0 if score_difference < 0.005
They are saved in logs/decisions.json the first time step 7, 8 or 9 runs and
reused on every later run, so steps 7-9 always share one configuration. Override
with --members-init / --day-offset; --india-set adds further overrides (e.g.
ensemble.precompute_member_predictions=false on PCs with < 8 GB RAM).

Every step's output goes to the console, to logs/<step>.log and to the log
file the RUNBOOK names (e.g. results/india/qre_mass_v1/stage_A.log). The
summary of what finished is printed at the end and written to logs/summary.txt
and logs/summary.json.
"""
import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

NZ_FLAGS = ["--sanity",
            "--set", 'segmentation.intervals_file="results/fixes_v2/dry_run/segmentation.json"',
            "--set", "sanity.members_max_epochs=10",
            "--set", "sanity.weight_network_max_epochs=30",
            "--set", 'checkpoints.dir="checkpoints/nz_v2_sanity"',
            "--save-dir", "results/fixes_v2/nz_wn"]
NZ_CKPT_DIR = "checkpoints/nz_v2_sanity/rep0"
NZ_CKPT_NAMES = ["bgnet"] + [f"member_{i}" for i in range(6)] + ["weight_network_squared_emd"]
MEMBER_RESULTS = "results/fixes_v2/nz_wn/sanity/member_study/results.json"
INDIA_OUT = "results/india/qre_mass_v1"
INDIA_CFG = ["--config", "configs/india_v1.json"]
DAY_OFFSET_MIN_DIFFERENCE = 0.005  # RUNBOOK step 6: below this, keep day_offset = 0


def jload(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------------- output checks
# Each returns None when the step's outputs are complete, else what is missing.

def nz_checkpoints_present(root):
    missing = [n for n in NZ_CKPT_NAMES
               if not all((root / NZ_CKPT_DIR / f"{n}{ext}").exists() for ext in (".json", ".weights.h5"))]
    return f"missing in {NZ_CKPT_DIR}: {', '.join(missing)}" if missing else None


def member_results_have(*keys, bgnet_converged=False):
    def check(root):
        r = jload(root / MEMBER_RESULTS)
        if r is None:
            return f"{MEMBER_RESULTS} missing"
        missing = [k for k in keys if k not in r]
        if bgnet_converged and "bgnet_converged" not in r.get("selection", {}).get("test", {}):
            missing.append("selection.test.bgnet_converged")
        return f"{MEMBER_RESULTS} lacks {', '.join(missing)}" if missing else None
    return check


def files_exist(*rel):
    def check(root):
        missing = [p for p in rel if not (root / p).exists()]
        return f"missing: {', '.join(missing)}" if missing else None
    return check


def data_check_ok(root):
    path = f"{INDIA_OUT}/data_check.json"
    r = jload(root / path)
    if r is None:
        return f"{path} missing"
    return None if r.get("ok") is True else f'{path} says "ok": {r.get("ok")} (see its "problems" entries)'


# ----------------------------------------------------------------------------- steps
@dataclass
class Step:
    num: int
    name: str
    title: str
    args: Callable[[dict], List[str]]        # decisions -> argv (without the interpreter)
    runbook_log: Optional[str]               # the log path the RUNBOOK uses, relative to the repo
    outputs: Callable[[Path], Optional[str]]  # None when complete, else what is missing
    estimate: str
    needs_marker: bool = True                # False: outputs alone mean finished (step 0)
    needs_decisions: bool = False            # True: gets the members.init / day_offset overrides
    data: List[str] = field(default_factory=list)  # data folders it needs


def india_sets(dec):
    sets = [f'members.init="{dec["members_init"]}"', f'data.day_offset={dec["day_offset"]}'] + dec["india_set"]
    return [a for s in sets for a in ("--set", s)]


STEPS = [
    Step(0, "nz_checkpoints", "NZ base models (RUNBOOK 0b, only if not copied)",
         lambda d: ["experiments/run_cordex_v2.py"] + NZ_FLAGS, "nz_step0b.log",
         nz_checkpoints_present, "1.5-2.5 h", needs_marker=False, data=["data/CORDEX_NZ"]),
    Step(1, "nz_member_B", "NZ member study B: warm-started members",
         lambda d: ["experiments/member_study.py"] + NZ_FLAGS + ["--set", 'member_study.variants=["warm_start"]'],
         "results/fixes_v2/nz_wn/member_B.log", member_results_have("warm_start"),
         "20-50 min (worst 3.2 h)", data=["data/CORDEX_NZ"]),
    Step(2, "nz_member_A", "NZ member study A: members from scratch + selection",
         lambda d: ["experiments/member_study.py"] + NZ_FLAGS,
         "results/fixes_v2/nz_wn/member_A.log", member_results_have("scratch", "warm_start", "selection"),
         "1.5-3.2 h", data=["data/CORDEX_NZ"]),
    Step(3, "nz_member_bgnet", "NZ member study: BGNet retrained to convergence",
         lambda d: ["experiments/member_study.py"] + NZ_FLAGS + ["--set", "member_study.retrain_bgnet=true"],
         "results/fixes_v2/nz_wn/member_bgnet.log",
         member_results_have("bgnet_converged_training", "selection", bgnet_converged=True),
         "1-3 h", data=["data/CORDEX_NZ"]),
    Step(4, "nz_mass", "NZ: QRE vs QRE + mass conservation",
         lambda d: ["experiments/nz_mass.py", "--set", 'checkpoints.dir="checkpoints/nz_mass"',
                    "--save-dir", "results/fixes_v2/nz_mass"],
         "results/fixes_v2/nz_mass/nz_mass.log",
         files_exist("results/fixes_v2/nz_mass/comparison.json", "results/fixes_v2/nz_mass/comparison.md"),
         "3-14 h", data=["data/CORDEX_NZ"]),
    Step(5, "india_data_check", "India data check",
         lambda d: ["scripts/check_india_data.py"] + INDIA_CFG,
         f"{INDIA_OUT}/data_check.log", data_check_ok, "a few minutes",
         data=["data/India/ERA5_full", "data/India/IMD_jjas"]),
    Step(6, "india_day_offset", "India day-offset check",
         lambda d: ["experiments/india_day_offset.py"] + INDIA_CFG, "day_offset.log",
         files_exist(f"{INDIA_OUT}/day_offset/day_offset.json"), "a few minutes",
         data=["data/India/ERA5_full", "data/India/IMD_jjas"]),
    Step(7, "india_A", "India experiment A (lambda = 0)",
         lambda d: ["experiments/run_india.py"] + INDIA_CFG + ["--stage", "A"] + india_sets(d),
         f"{INDIA_OUT}/stage_A.log", files_exist(f"{INDIA_OUT}/A/metrics.json", f"{INDIA_OUT}/A/training_log.json"),
         "2-7 h", needs_decisions=True, data=["data/India/ERA5_full", "data/India/IMD_jjas"]),
    Step(8, "india_sweep", "India lambda sweep (validation only)",
         lambda d: ["experiments/run_india.py"] + INDIA_CFG + ["--stage", "sweep"] + india_sets(d),
         f"{INDIA_OUT}/stage_sweep.log", files_exist(f"{INDIA_OUT}/sweep/sweep.json", f"{INDIA_OUT}/sweep/selection.json"),
         "0.5-3.5 h", needs_decisions=True, data=["data/India/ERA5_full", "data/India/IMD_jjas"]),
    Step(9, "india_B", "India experiment B (selected lambda)",
         lambda d: ["experiments/run_india.py"] + INDIA_CFG + ["--stage", "B"] + india_sets(d),
         f"{INDIA_OUT}/stage_B.log", files_exist(f"{INDIA_OUT}/B/metrics.json", f"{INDIA_OUT}/comparison.json"),
         "10-30 min", needs_decisions=True, data=["data/India/ERA5_full", "data/India/IMD_jjas"]),
]


# ----------------------------------------------------------------------------- decisions
class DecisionError(RuntimeError):
    pass


def derive_members_init(root):
    r = jload(root / MEMBER_RESULTS)
    best = (r or {}).get("selection", {}).get("best_by_perfect_member_val_mse_0_1")
    if best not in ("scratch", "warm_start"):
        return None, f"no selection in {MEMBER_RESULTS} (step 2 not finished)"
    return best, f"step 2 selected '{best}' ({MEMBER_RESULTS})"


def derive_day_offset(root):
    path = f"{INDIA_OUT}/day_offset/day_offset.json"
    r = jload(root / path)
    if r is None or "recommended_day_offset" not in r:
        return None, f"{path} missing (step 6 not finished)"
    rec, diff = int(r["recommended_day_offset"]), float(r["score_difference"])
    if diff < DAY_OFFSET_MIN_DIFFERENCE:
        return 0, f"recommended {rec}, but score_difference {diff:.4f} < {DAY_OFFSET_MIN_DIFFERENCE}: keep 0 ({path})"
    return rec, f"recommended {rec}, score_difference {diff:.4f} ({path})"


def resolve_decisions(root, args, logs, preview=False):
    """The overrides for steps 7-9. Reuses logs/decisions.json once it exists."""
    saved = jload(logs / "decisions.json")
    if saved is not None:
        given = {"members_init": args.members_init, "day_offset": args.day_offset}
        for key, val in given.items():
            if val != "auto" and str(val) != str(saved[key]):
                raise DecisionError(
                    f"{key} = {val} differs from {saved[key]} in logs/decisions.json, which steps 7-9 already use. "
                    "Changing it needs a fresh start: a new checkpoints.dir and save_dir, or delete "
                    "checkpoints/india/qre_mass_v1, the India results and logs/decisions.json.")
        if args.india_set and args.india_set != saved["india_set"]:
            raise DecisionError(f"--india-set {args.india_set} differs from {saved['india_set']} in logs/decisions.json")
        return saved
    if args.members_init != "auto":
        mi, mi_why = args.members_init, "given with --members-init"
    else:
        mi, mi_why = derive_members_init(root)
    if args.day_offset != "auto":
        do, do_why = int(args.day_offset), "given with --day-offset"
    else:
        do, do_why = derive_day_offset(root)
    if mi is None or do is None:
        if preview:
            return {"members_init": mi or "<from step 2>", "day_offset": "<from step 6>" if do is None else do,
                    "india_set": args.india_set, "why": {"members_init": mi_why, "day_offset": do_why}}
        raise DecisionError("cannot decide " + "; ".join(w for v, w in ((mi, mi_why), (do, do_why)) if v is None)
                            + ". Run the earlier steps or pass --members-init / --day-offset.")
    dec = {"members_init": mi, "day_offset": do, "india_set": list(args.india_set),
           "why": {"members_init": mi_why, "day_offset": do_why},
           "decided_at": now()}
    if not preview:
        write_json(logs / "decisions.json", dec)
    return dec


# ----------------------------------------------------------------------------- helpers
def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def fmt_seconds(s):
    s = int(round(s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def show(cmd):
    return shlex.join(cmd)


def finished(step, root, logs):
    """(True, info) if the step is finished, else (False, reason)."""
    missing = step.outputs(root)
    if not step.needs_marker:
        return (missing is None), (missing or "outputs present")
    marker = jload(logs / f"{step.name}.done")
    if marker is None:
        return False, "not finished by this runner yet" + (f" ({missing})" if missing else "")
    if missing:
        return False, f"was finished, but outputs are gone: {missing}"
    return True, f"finished {marker.get('finished_at')} in {fmt_seconds(marker.get('seconds', 0))}"


def pick_step(value, steps):
    for s in steps:
        if value == str(s.num) or value == s.name:
            return s.num
    raise SystemExit(f"unknown step '{value}'; use a number 0-{steps[-1].num} or one of: "
                     + ", ".join(s.name for s in steps))


def gpu_check(logs=None):
    code = ("import os; os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2'); import tensorflow as tf; "
            "g = tf.config.list_physical_devices('GPU'); "
            "print('TensorFlow', tf.__version__, '| built with CUDA:', tf.test.is_built_with_cuda()); "
            "print('GPUs seen by TensorFlow:', len(g)); [print('  ', d.name, d.device_type) for d in g]")
    print("== GPU check", flush=True)
    try:
        p = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=600)
        out = (p.stdout + (p.stderr if p.returncode else "")).strip()
        n = None
        for line in p.stdout.splitlines():
            if line.startswith("GPUs seen by TensorFlow:"):
                n = int(line.split(":")[1])
    except (OSError, subprocess.SubprocessError) as e:
        out, n = f"GPU check failed: {e}", None
    msg = out + "\n" + ("-> TensorFlow sees a GPU; training will use it." if n else
                        "-> TensorFlow sees NO GPU; training runs on the CPU (see docs/LAB_SETUP.md, GPU)."
                        if n == 0 else "-> could not import TensorFlow; is the venv active?")
    print(msg, flush=True)
    if logs is not None:
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "gpu_check.log").write_text(f"{now()}\n{msg}\n", encoding="utf-8")
    return n


def run_step(step, cmd, root, logs):
    """Run cmd, copying its output to the console, logs/<step>.log and the RUNBOOK log. Returns the exit code."""
    targets = [logs / f"{step.name}.log"] + ([root / step.runbook_log] if step.runbook_log else [])
    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)
    files = [open(t, "a", encoding="utf-8") for t in targets]
    header = f"\n===== {now()}  step {step.num} {step.name}\n$ {show(cmd)}\n"
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    try:
        for f in files:
            f.write(header)
            f.flush()
        proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        try:
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace")
                sys.stdout.write(line)
                sys.stdout.flush()
                for f in files:
                    f.write(line)
                    f.flush()
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            for f in files:
                f.write(f"===== {now()}  interrupted (Ctrl+C)\n")
            raise
        for f in files:
            f.write(f"===== {now()}  exit code {code}\n")
        return code
    finally:
        for f in files:
            f.close()


def write_summary(rows, logs, started, gpus):
    lines = [f"run_lab summary  (started {started}, written {now()})",
             f"GPU: {'none (CPU)' if gpus == 0 else ('unknown' if gpus is None else f'{gpus} seen by TensorFlow')}", ""]
    for r in rows:
        lines.append(f"  {r['step']:>2}  {r['name']:<17} {r['status']:<15} {r['detail']}")
    text = "\n".join(lines) + "\n"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "summary.txt").write_text(text, encoding="utf-8")
    write_json(logs / "summary.json", {"started": started, "written": now(), "gpus": gpus, "steps": rows})
    return text


# ----------------------------------------------------------------------------- main
def parse_args(argv):
    p = argparse.ArgumentParser(description="Run every RUNBOOK training step in order (see the module docstring).")
    p.add_argument("--from-step", default=None, help="first step to consider (number 0-9 or name); earlier steps are not run")
    p.add_argument("--to-step", default=None, help="last step to consider (number or name)")
    p.add_argument("--dry-run", action="store_true", help="only print the plan and the commands; run and write nothing")
    p.add_argument("--force", action="store_true",
                   help="rerun the selected steps even if finished (models still load from their checkpoints)")
    p.add_argument("--check-gpu", action="store_true", help="only run the GPU check")
    p.add_argument("--no-gpu-check", action="store_true", help="skip the GPU check at the start")
    p.add_argument("--require-gpu", action="store_true", help="stop if TensorFlow sees no GPU")
    p.add_argument("--members-init", choices=["auto", "scratch", "warm_start"], default="auto",
                   help="members.init for steps 7-9 (default: the set selected in step 2)")
    p.add_argument("--day-offset", choices=["auto", "0", "1"], default="auto",
                   help="data.day_offset for steps 7-9 (default: from step 6 by the RUNBOOK rule)")
    p.add_argument("--india-set", action="append", default=[], metavar="KEY.PATH=JSON",
                   help="extra --set for steps 7-9, e.g. ensemble.precompute_member_predictions=false")
    p.add_argument("--logs", default="logs", help="log folder, relative to the repository (default: logs)")
    return p.parse_args(argv)


def main(argv=None, root=ROOT, steps=None, py=None):
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    args = parse_args(argv)
    root, steps, py = Path(root), steps or STEPS, py or PY
    logs = root / args.logs
    if args.check_gpu:
        return 0 if gpu_check(logs) else 1
    first = pick_step(args.from_step, steps) if args.from_step is not None else steps[0].num
    last = pick_step(args.to_step, steps) if args.to_step is not None else steps[-1].num
    selected = [s for s in steps if first <= s.num <= last]
    if not selected:
        raise SystemExit("--from-step is after --to-step: nothing to run")

    if args.dry_run:
        print(f"DRY RUN: nothing is run or written. Repository: {root}\nPython: {py}\n")
        dec_shown = False
        for s in steps:
            if s not in selected:
                print(f"[{s.num}] {s.name}: not selected (--from-step/--to-step)\n")
                continue
            done, info = finished(s, root, logs)
            state = f"skip, {info}" if done and not args.force else f"RUN ({info})" if not done else "RUN (--force)"
            dec = {}
            if s.needs_decisions:
                try:
                    dec = resolve_decisions(root, args, logs, preview=True)
                except DecisionError as e:
                    print(f"  ! {e}")
                    dec = {"members_init": "<error>", "day_offset": "<error>", "india_set": args.india_set}
                if not dec_shown and not done:
                    print(f"  decisions for steps 7-9: members.init={dec['members_init']}, "
                          f"data.day_offset={dec['day_offset']}"
                          + (f"  ({'; '.join(dec['why'].values())})" if "why" in dec else ""))
                    dec_shown = True
            missing_data = [d for d in s.data if not (root / d).is_dir()]
            print(f"[{s.num}] {s.name}: {s.title}  -- {state}")
            print(f"    estimate: {s.estimate}")
            print(f"    $ {show([py] + s.args(dec))}")
            print(f"    log: {args.logs}/{s.name}.log" + (f" and {s.runbook_log}" if s.runbook_log else ""))
            if missing_data:
                print(f"    ! data folder missing: {', '.join(missing_data)}")
            print()
        return 0

    started = now()
    print(f"run_lab {started}  repository {root}\nPython {sys.version.split()[0]} ({py})", flush=True)
    if sys.version_info[:2] != (3, 10):
        print("! warning: the code was tested with Python 3.10 (TensorFlow 2.10 needs 3.7-3.10)", flush=True)
    gpus = None
    if not args.no_gpu_check:
        gpus = gpu_check(logs)
        if args.require_gpu and not gpus:
            print("--require-gpu: stopping, TensorFlow sees no GPU", flush=True)
            return 2

    # refuse to start if a step that will run lacks its data
    todo = [s for s in selected if args.force or not finished(s, root, logs)[0]]
    missing = sorted({d for s in todo for d in s.data if not (root / d).is_dir()})
    if missing:
        print(f"! data folders missing: {', '.join(missing)} (docs/LAB_SETUP.md, data). Nothing was run.", flush=True)
        return 2

    rows, code = [], 0
    try:
        for s in steps:
            row = {"step": s.num, "name": s.name, "title": s.title}
            rows.append(row)
            if code != 0:
                row.update(status="not run", detail="an earlier step failed")
                continue
            if s not in selected:
                done, info = finished(s, root, logs)
                row.update(status="not selected", detail=("finished earlier: " + info) if done else info)
                continue
            done, info = finished(s, root, logs)
            if done and not args.force:
                row.update(status="already done", detail=info)
                print(f"\n== [{s.num}] {s.name}: already done ({info}), skipping", flush=True)
                continue
            dec = {}
            if s.needs_decisions:
                try:
                    dec = resolve_decisions(root, args, logs)
                except DecisionError as e:
                    print(f"\n! [{s.num}] {s.name}: {e}", flush=True)
                    row.update(status="FAILED", detail=str(e))
                    code = 1
                    continue
                print(f"\n   decisions (logs/decisions.json): members.init={dec['members_init']}, "
                      f"data.day_offset={dec['day_offset']}, extra={dec['india_set']}", flush=True)
            cmd = [py] + s.args(dec)
            print(f"\n== [{s.num}] {s.name}: {s.title}  (estimate {s.estimate})\n$ {show(cmd)}", flush=True)
            if s.needs_marker:
                (logs / f"{s.name}.done").unlink(missing_ok=True)
            t0 = time.time()
            rc = run_step(s, cmd, root, logs)
            secs = time.time() - t0
            missing_out = s.outputs(root)
            if rc == 0 and missing_out is None:
                if s.needs_marker:
                    write_json(logs / f"{s.name}.done", {"finished_at": now(), "seconds": secs, "command": cmd})
                row.update(status="finished", detail=f"in {fmt_seconds(secs)}")
                print(f"== [{s.num}] {s.name}: finished in {fmt_seconds(secs)}", flush=True)
            else:
                why = f"exit code {rc}" if rc != 0 else f"exit code 0 but {missing_out}"
                row.update(status="FAILED", detail=f"{why} after {fmt_seconds(secs)}; see {args.logs}/{s.name}.log")
                print(f"== [{s.num}] {s.name}: FAILED ({why}). Later steps are not run.", flush=True)
                code = 1
            write_summary(rows, logs, started, gpus)
    except KeyboardInterrupt:
        rows[-1].update(status="interrupted", detail="Ctrl+C; rerun the same command to resume")
        for s in steps[len(rows):]:
            rows.append({"step": s.num, "name": s.name, "title": s.title, "status": "not run", "detail": "interrupted"})
        code = 130
    print("\n" + write_summary(rows, logs, started, gpus), flush=True)
    if code == 1:
        print("Fix the problem and run the same command again; finished steps are skipped.", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
