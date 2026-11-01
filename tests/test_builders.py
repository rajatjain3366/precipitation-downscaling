"""
Phase 2 checks: shared builder, BGNet(-) on 16x16, BG_mean modes.
Run with:  python -m pytest tests/test_builders.py -q
"""
import json
import os
import sys

import numpy as np
import pytest
import tensorflow as tf

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.builders import build_bgnet, build_bgnet_minus, build_omega
from src.ConvolutionalNetworks import BG_mean

D = 50


@pytest.fixture(scope="module")
def cfg():
    with open(os.path.join(WORKSPACE_ROOT, "configs", "cordex_v2.json")) as f:
        c = json.load(f)
    c["architecture"]["channels"] = [8, 16, 32]  # small, so the test stays fast
    c["architecture"]["dense"] = [16]
    return c


@pytest.fixture(scope="module")
def inputs():
    x = tf.random.normal((2, 16, 16, 4))
    aux = np.random.rand(1, 128, 128, 1).astype(np.float32)
    return x, aux


def test_member_and_baseline_are_identical(cfg, inputs):
    x, aux = inputs
    member, baseline = build_bgnet(cfg, D, aux), build_bgnet(cfg, D, aux)
    assert member(x).shape == baseline(x).shape == (2, D, 3)
    assert member.count_params() == baseline.count_params()
    assert [w.shape for w in member.weights] == [w.shape for w in baseline.weights]


def test_bgnet_minus_runs_on_16x16(cfg, inputs):
    x, _ = inputs
    assert build_bgnet_minus(cfg, D)(x).shape == (2, D, 3)


def test_omega_softmax(cfg, inputs):
    x, _ = inputs
    w = build_omega(cfg, 6)(x).numpy()
    assert w.shape == (2, 6) and np.allclose(w.sum(axis=1), 1.0, atol=1e-5)


def test_elevation_required_when_enabled(cfg):
    with pytest.raises(ValueError):
        build_bgnet(cfg, D, aux=None)


def test_bg_mean_modes():
    p = tf.constant([[0.3], [0.8]])
    log_a = tf.math.log(tf.constant([[2.0], [2.0]]))
    log_b = tf.math.log(tf.constant([[3.0], [3.0]]))
    expected = BG_mean(p, log_a, log_b, mode="expected").numpy().ravel()
    threshold = BG_mean(p, log_a, log_b, mode="threshold").numpy().ravel()
    assert np.allclose(expected, [0.3 * 6, 0.8 * 6])  # Eq. 3: p * alpha * beta
    assert np.allclose(threshold, [0.0, 6.0])         # original repo rule
