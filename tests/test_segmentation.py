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
