import numpy as np


def time_to_years(times):
    """Calendar year of each time stamp (numpy datetime64 or cftime objects)."""
    times = np.asarray(times)
    if np.issubdtype(times.dtype, np.datetime64):
        return times.astype("datetime64[Y]").astype(int) + 1970
    return np.array([t.year for t in times])


def split_masks_by_years(times, train_years, val_years, test_years):
    """
    Boolean masks for inclusive (first_year, last_year) ranges.
    Raises if the ranges overlap or leave a requested year without data.
    """
    years = time_to_years(times)
    masks = {}
    for name, (y0, y1) in [("train", train_years), ("val", val_years), ("test", test_years)]:
        if y0 > y1:
            raise ValueError(f"{name} years are reversed: {(y0, y1)}")
        masks[name] = (years >= y0) & (years <= y1)
        missing = sorted(set(range(y0, y1 + 1)) - set(years[masks[name]].tolist()))
        if missing:
            raise ValueError(f"{name} split asks for years with no data: {missing}")

    if np.any(masks["train"] & masks["val"]) or np.any(masks["train"] & masks["test"]) \
            or np.any(masks["val"] & masks["test"]):
        raise ValueError("train/val/test year ranges overlap")
    return masks


def split_masks_by_dates(dates, split):
    """
    Boolean masks for inclusive [first_day, last_day] ranges, e.g.
    split = {"train": ["1990-01-01", "2010-12-31"], "val": [...], "test": [...]}.
    Raises if the ranges are reversed, overlap, or select no day.
    """
    days = np.asarray(dates).astype("datetime64[D]")
    masks = {}
    for name in ("train", "val", "test"):
        d0, d1 = (np.datetime64(d, "D") for d in split[name])
        if d0 > d1:
            raise ValueError(f"{name} dates are reversed: {split[name]}")
        masks[name] = (days >= d0) & (days <= d1)
        if not masks[name].any():
            raise ValueError(f"{name} split {split[name]} contains no day of the data")
    if np.any(masks["train"] & masks["val"]) or np.any(masks["train"] & masks["test"]) \
            or np.any(masks["val"] & masks["test"]):
        raise ValueError("train/val/test date ranges overlap")
    return masks


def channel_stats(x, names=None):
    """Per-channel mean/std/min/max of a (T, H, W, C) array, as a list of dicts."""
    names = names or [f"ch{i}" for i in range(x.shape[-1])]
    stats = []
    for i, name in enumerate(names):
        c = x[..., i].astype(np.float64)
        stats.append({"channel": name, "mean": float(c.mean()), "std": float(c.std()),
                      "min": float(c.min()), "max": float(c.max())})
    return stats


def looks_standardized(stats, tol_mean=0.1, tol_std=0.1):
    """True if every channel already has |mean| < tol_mean and |std - 1| < tol_std."""
    return all(abs(s["mean"]) < tol_mean and abs(s["std"] - 1.0) < tol_std for s in stats)


class ChannelStandardizer:
    """Per-channel (x - mean) / std with statistics fitted on the training split only."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x_train):
        self.mean = x_train.mean(axis=(0, 1, 2), dtype=np.float64)
        self.std = x_train.std(axis=(0, 1, 2), dtype=np.float64)
        if np.any(self.std < 1e-12):
            raise ValueError(f"constant predictor channel(s): std = {self.std}")
        return self

    def transform(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}
