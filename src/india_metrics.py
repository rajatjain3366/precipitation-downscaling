"""
Evaluation for the India experiments, in physical units (mm/day) on the IMD
valid cells only (the same mask everywhere).

- QRE metrics: MSE on the [0,1], [0,0.2], [0.9,1] quantile ranges of daily
  totals (thresholds from the training split), MSE per interval
- MAE, RMSE, spatial correlation (per day across cells, then averaged; and of
  the time-mean fields)
- area-weighted all-India daily totals: area-mean mm/day and volume km^3/day,
  total discrepancy over the period, daily total-error statistics
"""
import numpy as np


def regime_mse(y_true, y_pred, train_totals, regimes=((0.0, 1.0), (0.0, 0.2), (0.9, 1.0))):
    """Same definition as the NZ runner: quantiles of the unweighted daily total over cells."""
    totals = y_true.sum(axis=1, dtype=np.float64)
    out = {}
    for lo, hi in regimes:
        t_lo = -np.inf if lo == 0.0 else np.quantile(train_totals, lo)
        t_hi = np.inf if hi == 1.0 else np.quantile(train_totals, hi)
        m = (totals >= t_lo) & (totals < t_hi)
        mse = float(np.mean((y_true[m] - y_pred[m]) ** 2)) if m.any() else None
        out[f"[{lo:.1f},{hi:.1f}]"] = {"mse": mse, "days": int(m.sum())}
    return out


def interval_mse(y_true, y_pred, labels, n_intervals):
    out = []
    for i in range(n_intervals):
        m = labels == i
        out.append({"interval": i, "days": int(m.sum()),
                    "mse": float(np.mean((y_true[m] - y_pred[m]) ** 2)) if m.any() else None})
    return out


def spatial_scores(y_true, y_pred):
    err = y_pred - y_true
    corr = []
    for t in range(len(y_true)):
        a, b = y_true[t], y_pred[t]
        if a.std() > 0 and b.std() > 0:
            corr.append(float(np.corrcoef(a, b)[0, 1]))
    mt, mp = y_true.mean(axis=0), y_pred.mean(axis=0)
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mse": float(np.mean(err ** 2)), "bias": float(np.mean(err)),
            "spatial_corr_daily_mean": float(np.mean(corr)) if corr else None,
            "spatial_corr_days_used": len(corr),
            "spatial_corr_of_time_mean": float(np.corrcoef(mt, mp)[0, 1]) if mt.std() > 0 and mp.std() > 0 else None}


def area_mean_totals(y, weights):
    """Area-weighted all-India mean of each day (mm/day); weights = A_i / sum A."""
    return y.astype(np.float64) @ weights


def total_scores(y_true, y_pred, areas_km2, eps=0.1):
    """Daily all-India totals, observed vs predicted. 1 mm over 1 km^2 = 1e-6 km^3."""
    w = areas_km2 / areas_km2.sum()
    o, s = area_mean_totals(y_true, w), area_mean_totals(y_pred, w)
    to_km3 = areas_km2.sum() * 1e-6
    d = s - o
    rel = np.abs(d) / (o + eps)
    return {
        "observed_mean_mm_per_day": float(o.mean()), "predicted_mean_mm_per_day": float(s.mean()),
        "total_observed_km3": float(o.sum() * to_km3), "total_predicted_km3": float(s.sum() * to_km3),
        "total_discrepancy_km3": float(d.sum() * to_km3),
        "total_relative_discrepancy": float(d.sum() / o.sum()) if o.sum() > 0 else None,
        "daily_bias_mm_per_day": float(d.mean()), "daily_mae_mm_per_day": float(np.abs(d).mean()),
        "daily_rmse_mm_per_day": float(np.sqrt(np.mean(d ** 2))),
        "daily_abs_error_p95_mm_per_day": float(np.quantile(np.abs(d), 0.95)),
        "daily_mean_relative_error": float(rel.mean()), "daily_median_relative_error": float(np.median(rel)),
        "relative_error_eps_mm_per_day": eps,
        "daily_totals_corr": float(np.corrcoef(o, s)[0, 1]) if o.std() > 0 and s.std() > 0 else None,
        "days": int(len(o)),
    }


def evaluate(y_true, y_pred, train_totals, labels, n_intervals, areas_km2, eps):
    return {"regime_mse": regime_mse(y_true, y_pred, train_totals),
            "interval_mse": interval_mse(y_true, y_pred, labels, n_intervals),
            "spatial": spatial_scores(y_true, y_pred),
            "all_india_totals": total_scores(y_true, y_pred, areas_km2, eps)}
