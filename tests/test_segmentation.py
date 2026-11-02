"""
Unit tests for quantile segmentation (data_between) and Weight Network labels
(bin_y_var) with tied daily totals.

Run with:  python -m pytest tests/test_segmentation.py -v

The legacy functions are marked xfail(strict=True): they document bugs H5
(threshold = sum over tied samples) and L1 (inclusive bounds -> overlapping
segments). If one of them starts passing, pytest reports XPASS as a failure.
"""
import os
import sys

import numpy as np
import pytest

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.useful_functions import data_between
from src.quantiles import bin_y_var


def tied_dataset():
    # 10 days x 2 grid cells. Daily totals: 1,2,3,4,5,5,7,8,9,10 (two days tie at 5).
    totals = np.array([1, 2, 3, 4, 5, 5, 7, 8, 9, 10], dtype=np.float32)
    y = np.stack([totals / 2, totals / 2], axis=1)
    q_ranges = [(0.0, 0.5), (0.5, 1.0)]
    return y, totals, q_ranges


@pytest.mark.xfail(strict=True, reason="L1: legacy data_between uses inclusive bounds on both sides")
def test_legacy_data_between_partitions_days():
    y, _, q_ranges = tied_dataset()
    sizes = [len(data_between(y, ql, None, qh)) for ql, qh in q_ranges]
    assert sum(sizes) == len(y)


@pytest.mark.xfail(strict=True, reason="H5: legacy bin_y_var sums the totals of tied boundary days")
def test_legacy_bin_y_var_threshold_with_ties():
    y, totals, q_ranges = tied_dataset()
    labels = np.argmax(np.array(bin_y_var(y, q_ranges)), axis=1)
    # Boundary total is 5 -> days with totals > 5 must not be in class 0
    assert np.all(labels[totals > 5] == 1)


# ---------------------------------------------------------------------------
# Fixed functions (Phase 3)
# ---------------------------------------------------------------------------
from src.segmentation import (assign_intervals, bounds_to_thresholds, daily_totals,
                              gmm_quantile_bounds, partition_indices)


def test_thresholds_one_value_per_boundary_with_ties():
    y, totals, q_ranges = tied_dataset()
    thresholds = [5.0]
    labels = np.argmax(np.array(bin_y_var(y, q_ranges, thresholds=thresholds)), axis=1)
    assert np.all(labels[totals < 5] == 0)
    assert np.all(labels[totals >= 5] == 1)  # both tied days on the boundary go up, none duplicated


def test_partition_covers_every_day_exactly_once():
    y, totals, _ = tied_dataset()
    parts = partition_indices(daily_totals(y), [3.0, 5.0, 8.0], 4)
    assert sum(len(p) for p in parts) == len(y)
    assert len(np.unique(np.concatenate(parts))) == len(y)


def test_labels_match_partition():
    y, _, q_ranges = tied_dataset()
    thresholds = [5.0]
    labels = np.argmax(np.array(bin_y_var(y, q_ranges, thresholds=thresholds)), axis=1)
    parts = partition_indices(daily_totals(y), thresholds, 2)
    assert [int(np.sum(labels == i)) for i in range(2)] == [len(p) for p in parts]


def test_validation_uses_training_thresholds():
    # A validation set with a different distribution must keep the training boundary
    train_totals = np.arange(1, 101, dtype=float)
    thresholds = bounds_to_thresholds(train_totals, [0.0, 0.5, 1.0])
    val_totals = np.array([10.0, 20.0, 30.0, 40.0])  # all below the training median
    assert np.all(assign_intervals(val_totals, thresholds) == 0)


def test_gmm_bounds_are_increasing_and_averaged():
    rng = np.random.default_rng(0)
    totals = np.concatenate([rng.normal(10, 2, 300), rng.normal(40, 5, 200), rng.normal(90, 10, 100)])
    bounds, per_run, _ = gmm_quantile_bounds(np.clip(totals, 0, None), 3, n_runs=5, base_seed=0)
    assert bounds[0] == 0.0 and bounds[-1] == 1.0
    assert np.all(np.diff(bounds) > 0)
    assert np.allclose(bounds, np.mean(per_run, axis=0))


def test_gmm_raises_when_components_collapse():
    totals = np.random.default_rng(0).normal(1000.0, 10.0, 400)  # one Gaussian: DP prior prunes components
    with pytest.raises(ValueError, match="non-increasing"):
        gmm_quantile_bounds(totals, 6, n_runs=4, base_seed=0)
