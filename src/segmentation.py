"""
Quantile segmentation for QRE (paper: "Data Segmentation", Algorithm 1).

All thresholds are fitted on the TRAINING split and then applied unchanged to
validation and test days. Every day gets exactly one interval index:

    interval i  <=>  t_i <= 1.y < t_{i+1},   t_0 = -inf, t_N = +inf

so boundary days go to the upper interval and no day is dropped or duplicated.
(The paper's strict  l_i < 1.y < h_i  would drop days that sit exactly on a
boundary; we keep them, as the replication brief requires.)
"""
import numpy as np
from sklearn.mixture import BayesianGaussianMixture


def daily_totals(y):
    """1.y for each day: (T, D) -> (T,) in float64 (avoids float32 rounding ties)."""
    return np.sum(np.asarray(y, dtype=np.float64), axis=-1)


def gmm_quantile_bounds(totals, n_intervals, n_runs=100, base_seed=0, max_iter=1000,
                        weight_concentration_prior=None):
    """
    Paper procedure: fit a Bayesian GMM with N components to the daily totals,
    convert each sorted component mean mu_i to its quantile index q_i in the
    training totals, set q_N = 1, and average the boundaries over n_runs seeds.

    A run is "degenerate" when its bounds are not strictly increasing: the
    Dirichlet-process prior pruned components, which then sit at (almost) the
    same mean. Degenerate runs are left out of the average; if fewer than half
    the runs are usable this raises. weight_concentration_prior=None is the
    sklearn default (1/N); a large value (e.g. 1e3) keeps all N components
    active on unimodal data.

    Returns (bounds, per_run, n_degenerate) with bounds = [0, q_1, ..., q_{N-1}, 1].
    """
    x = np.asarray(totals, dtype=np.float64).reshape(-1, 1)
    per_run, n_degenerate = [], 0
    for r in range(n_runs):
        gmm = BayesianGaussianMixture(n_components=n_intervals, max_iter=max_iter, random_state=base_seed + r,
                                      weight_concentration_prior=weight_concentration_prior)
        gmm.fit(x)
        means = np.sort(gmm.means_.ravel())
        q = [float(np.mean(x.ravel() <= m)) for m in means]
        b = [0.0] + q[:-1] + [1.0]
        if np.any(np.diff(b) <= 0):
            n_degenerate += 1
            continue
        per_run.append(b)

    if len(per_run) < max(1, n_runs // 2):
        raise ValueError(f"{n_degenerate}/{n_runs} Bayesian GMM runs gave non-increasing bounds "
                         f"(components pruned); try a larger weight_concentration_prior or a smaller N")
    per_run = np.array(per_run)
    bounds = per_run.mean(axis=0)
    return bounds.tolist(), per_run.tolist(), n_degenerate


def bounds_to_thresholds(totals, bounds):
    """Interior quantile bounds q_1..q_{N-1} -> daily-total thresholds t_1..t_{N-1} (training data)."""
    return [float(np.quantile(np.asarray(totals, dtype=np.float64), q)) for q in bounds[1:-1]]


def assign_intervals(totals, thresholds):
    """Interval index in [0, N-1] for every day; one value per threshold, no sums over ties."""
    return np.searchsorted(np.asarray(thresholds, dtype=np.float64),
                           np.asarray(totals, dtype=np.float64), side="right")


def one_hot(labels, n_intervals):
    out = np.zeros((len(labels), n_intervals), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1.0
    return out


def partition_indices(totals, thresholds, n_intervals):
    """List of index arrays, one per interval; asserts they partition all days."""
    labels = assign_intervals(totals, thresholds)
    parts = [np.flatnonzero(labels == i) for i in range(n_intervals)]
    assert sum(len(p) for p in parts) == len(labels), "segments must partition the days"
    return parts
