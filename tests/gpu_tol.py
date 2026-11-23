"""Comparisons that are exact on a CPU and allow GPU floating-point noise.

On a GPU, two evaluations of the same model (in one process or two) can differ in the
last digits, because cuDNN may pick other kernels. Checks that are exact on a CPU then
use a relative tolerance of src.gpu_setup.GPU_RTOL (1e-5), the one nz_mass.py uses.
"""
import functools
import math

import numpy as np

from src.gpu_setup import GPU_RTOL


@functools.lru_cache(maxsize=1)
def on_gpu():
    """True when TensorFlow sees a GPU (the code under test then runs on it)."""
    import tensorflow as tf

    return bool(tf.config.list_physical_devices("GPU"))


def assert_close(a, b, path="value"):
    """a == b on a CPU; on a GPU, equal structure and numbers within GPU_RTOL (relative)."""
    if not on_gpu():
        assert a == b, f"{path}: {a!r} != {b!r}"
        return
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), f"{path}: keys {sorted(a)} != {sorted(b)}"
        for k in a:
            assert_close(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b), f"{path}: lengths {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        ok = a == b or (a is not None and b is not None and math.isclose(a, b, rel_tol=GPU_RTOL, abs_tol=0.0))
        assert ok, f"{path}: {a!r} vs {b!r} (relative tolerance {GPU_RTOL:g})"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def assert_arrays_close(a, b, path="array"):
    """Exact on a CPU; on a GPU within GPU_RTOL relative to each value, with an absolute floor of
    GPU_RTOL times the largest magnitude of b (so entries near zero do not fail on noise)."""
    a, b = np.asarray(a), np.asarray(b)
    if not on_gpu():
        np.testing.assert_array_equal(a, b, err_msg=path)
        return
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    np.testing.assert_allclose(a, b, rtol=GPU_RTOL, atol=GPU_RTOL * scale, err_msg=path)
