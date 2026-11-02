"""
Runs every smoke test in dependency order and prints a PASS/FAIL table.

Usage:  python tests/run_smoke_tests.py
Exit code is non-zero if any script fails.

Order matters: generate_sample_cordex.py writes the synthetic
data/CORDEX_NZ_sample fixture that the test_real_data_* scripts read.
"""
import os
import subprocess
import sys
import time

TESTS_DIR = os.path.abspath(os.path.dirname(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
INDIA_SMOKE_OUT = "results/india/smoke_run"

STEPS = [
    ["-m", "pytest", "-q", os.path.join(TESTS_DIR, "test_segmentation.py")],
    ["-m", "pytest", "-q", os.path.join(TESTS_DIR, "test_builders.py")],
    [os.path.join(TESTS_DIR, "test_cordex_adapter.py")],
    [os.path.join(TESTS_DIR, "generate_sample_cordex.py")],
    [os.path.join(TESTS_DIR, "test_real_data_specialist_smoke.py")],
    [os.path.join(TESTS_DIR, "test_real_data_qre_integration.py")],
    [os.path.join(TESTS_DIR, "smoke_test_qre.py")],
    [os.path.join(TESTS_DIR, "test_qre_pipeline.py")],
    # India (needs data/India_sample/): unit tests, then the runner and the day-offset check on the samples.
    # Outputs go to the gitignored results/india/smoke_run/ (they contain timings), so a test run leaves
    # the committed reference outputs in results/india/smoke/ untouched.
    ["-m", "pytest", "-q", os.path.join(TESTS_DIR, "test_india.py")],
    [os.path.join(WORKSPACE_ROOT, "experiments", "run_india.py"), "--config", "configs/india_smoke.json",
     "--stage", "all", "--save-dir", INDIA_SMOKE_OUT],
    [os.path.join(WORKSPACE_ROOT, "experiments", "india_day_offset.py"), "--config", "configs/india_smoke.json",
     "--save-dir", INDIA_SMOKE_OUT],
]

# Scripts that catch their own exceptions and only print FAIL lines
FAIL_MARKERS = ("FAILED", "FAIL:", "RESULT: FAIL", "Traceback")


def main():
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2")
    results = []
    for step in STEPS:
        script = next((s for s in step if s.endswith(".py")), step[-1])
        name = os.path.basename(script) + (" " + " ".join(step[step.index(script) + 1:]) if script != step[-1] else "")
        t0 = time.time()
        proc = subprocess.run([sys.executable] + step, cwd=WORKSPACE_ROOT, env=env,
                              capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        ok = proc.returncode == 0 and not any(m in proc.stdout for m in FAIL_MARKERS)
        results.append((name, ok, time.time() - t0))
        if not ok:
            print(f"----- {name} output (tail) -----")
            print("\n".join(out.splitlines()[-40:]))

    print("\n" + "=" * 60)
    for name, ok, dt in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<40} {dt:6.1f}s")
    print("=" * 60)
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == "__main__":
    main()
