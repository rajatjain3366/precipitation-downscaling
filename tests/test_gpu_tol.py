"""GPU helpers: memory growth is on for every process; comparisons exact on CPU, rtol 1e-5 on GPU."""
import os
import subprocess
import sys

import numpy as np
import pytest

import gpu_tol
from src.gpu_setup import GPU_RTOL

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_importing_src_sets_memory_growth_unless_already_set():
    code = "import os, src; print(os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH'))"
    env = {k: v for k, v in os.environ.items() if k != "TF_FORCE_GPU_ALLOW_GROWTH"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=WORKSPACE_ROOT)
    assert out.stdout.strip() == "true"
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"  # an explicit choice is kept
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=WORKSPACE_ROOT)
    assert out.stdout.strip() == "false"


def test_exact_on_cpu(monkeypatch):
    monkeypatch.setattr(gpu_tol, "on_gpu", lambda: False)
    gpu_tol.assert_close({"a": [1.0, 2]}, {"a": [1.0, 2]})
    with pytest.raises(AssertionError):
        gpu_tol.assert_close(1.0, 1.0 + 2e-6)
    with pytest.raises(AssertionError):
        gpu_tol.assert_arrays_close([1.0, 0.0], [1.0, 5.8e-11])


def test_rtol_on_gpu(monkeypatch):
    monkeypatch.setattr(gpu_tol, "on_gpu", lambda: True)
    # the lab PC's case: relative 2e-6, absolute 5.8e-11
    gpu_tol.assert_close({"mse": [10.0, None], "days": 5}, {"mse": [10.0 * (1 + 2e-6), None], "days": 5})
    gpu_tol.assert_arrays_close(np.array([1e-5, 0.0, 3.0]), np.array([1e-5 + 5.8e-11, 5.8e-11, 3.0]))
    with pytest.raises(AssertionError):
        gpu_tol.assert_close(10.0, 10.0 * (1 + 10 * GPU_RTOL))
    with pytest.raises(AssertionError):
        gpu_tol.assert_close({"days": 5}, {"days": 6})           # integers stay exact
    with pytest.raises(AssertionError):
        gpu_tol.assert_close({"a": 1.0}, {"b": 1.0})             # structure stays exact
    with pytest.raises(AssertionError):
        gpu_tol.assert_arrays_close([3.0], [3.0 * (1 + 10 * GPU_RTOL)])
