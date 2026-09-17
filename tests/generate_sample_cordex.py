import os
import sys
import numpy as np
import xarray as xr

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
test_dir = os.path.join(WORKSPACE_ROOT, "data", "CORDEX_NZ_sample")

os.makedirs(os.path.join(test_dir, "train", "ESD_pseudo-reality", "predictors"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "train", "ESD_pseudo-reality", "target"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "test", "historical", "predictors", "perfect"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "test", "historical", "target"), exist_ok=True)

# 1961-1963: 1095 days (1961-01-01 to 1963-12-31)
times_tr = np.arange("1961-01-01", "1964-01-01", dtype="datetime64[D]")
# 1981: 365 days (1981-01-01 to 1981-12-31)
times_te = np.arange("1981-01-01", "1982-01-01", dtype="datetime64[D]")

T_tr = len(times_tr)  # 1095
T_te = len(times_te)  # 365

H_in, W_in = 16, 16
H_out, W_out = 128, 128
levels = [850, 700, 500]

lat_in = np.linspace(-51, -33, H_in)
lon_in = np.linspace(165, 184, W_in)
lat_out = np.linspace(-51, -33, H_out)
lon_out = np.linspace(165, 184, W_out)

print(f"Generating CORDEX sample data: Train+Val T={T_tr} days ({times_tr[0]} to {times_tr[-1]}), Test T={T_te} days ({times_te[0]} to {times_te[-1]})")

np.random.seed(42)

# Predictors (ACCESS-CM2)
ds_pred_tr = xr.Dataset(
    data_vars={
        "u": (["time", "level", "lat", "lon"], np.random.randn(T_tr, 3, H_in, W_in).astype(np.float32)),
        "v": (["time", "level", "lat", "lon"], np.random.randn(T_tr, 3, H_in, W_in).astype(np.float32)),
        "q": (["time", "level", "lat", "lon"], (np.random.rand(T_tr, 3, H_in, W_in) * 0.015).astype(np.float32)),
        "t": (["time", "level", "lat", "lon"], (280.0 + np.random.randn(T_tr, 3, H_in, W_in) * 5.0).astype(np.float32)),
        "z": (["time", "level", "lat", "lon"], (1400.0 + np.random.randn(T_tr, 3, H_in, W_in) * 50.0).astype(np.float32)),
    },
    coords={"time": times_tr, "level": levels, "lat": lat_in, "lon": lon_in}
)

ds_pred_te = xr.Dataset(
    data_vars={
        "u": (["time", "level", "lat", "lon"], np.random.randn(T_te, 3, H_in, W_in).astype(np.float32)),
        "v": (["time", "level", "lat", "lon"], np.random.randn(T_te, 3, H_in, W_in).astype(np.float32)),
        "q": (["time", "level", "lat", "lon"], (np.random.rand(T_te, 3, H_in, W_in) * 0.015).astype(np.float32)),
        "t": (["time", "level", "lat", "lon"], (280.0 + np.random.randn(T_te, 3, H_in, W_in) * 5.0).astype(np.float32)),
        "z": (["time", "level", "lat", "lon"], (1400.0 + np.random.randn(T_te, 3, H_in, W_in) * 50.0).astype(np.float32)),
    },
    coords={"time": times_te, "level": levels, "lat": lat_in, "lon": lon_in}
)

# Target (pr, tasmax)
raw_pr_tr = np.random.exponential(scale=5.0, size=(T_tr, H_out, W_out)).astype(np.float32)
raw_pr_te = np.random.exponential(scale=5.0, size=(T_te, H_out, W_out)).astype(np.float32)

# Realistic land mask
xx, yy = np.meshgrid(np.linspace(-1, 1, W_out), np.linspace(-1, 1, H_out))
island_shape = (np.abs(xx + 0.3 * yy) < 0.25) & (np.abs(yy) < 0.6)
raw_orog = np.where(island_shape, 500.0 + 800.0 * np.exp(-(xx**2 + yy**2) * 5), 0.0).astype(np.float32)

ds_target_tr = xr.Dataset(
    data_vars={
        "pr": (["time", "lat", "lon"], raw_pr_tr),
        "tasmax": (["time", "lat", "lon"], (290.0 + np.random.randn(T_tr, H_out, W_out) * 4.0).astype(np.float32))
    },
    coords={"time": times_tr, "lat": lat_out, "lon": lon_out}
)

ds_target_te = xr.Dataset(
    data_vars={
        "pr": (["time", "lat", "lon"], raw_pr_te),
        "tasmax": (["time", "lat", "lon"], (290.0 + np.random.randn(T_te, H_out, W_out) * 4.0).astype(np.float32))
    },
    coords={"time": times_te, "lat": lat_out, "lon": lon_out}
)

# Static
ds_static = xr.Dataset(
    data_vars={
        "orog": (["lat", "lon"], raw_orog),
        "sftlf": (["lat", "lon"], np.where(raw_orog > 0, 100.0, 0.0).astype(np.float32))
    },
    coords={"lat": lat_out, "lon": lon_out}
)

print("Writing NetCDF files...")
ds_pred_tr.to_netcdf(os.path.join(test_dir, "train", "ESD_pseudo-reality", "predictors", "ACCESS-CM2_1961-1980.nc"))
ds_target_tr.to_netcdf(os.path.join(test_dir, "train", "ESD_pseudo-reality", "target", "pr_tasmax_ACCESS-CM2_1961-1980.nc"))
ds_static.to_netcdf(os.path.join(test_dir, "train", "ESD_pseudo-reality", "static.nc"))
ds_pred_te.to_netcdf(os.path.join(test_dir, "test", "historical", "predictors", "perfect", "ACCESS-CM2_1981-2000.nc"))
ds_target_te.to_netcdf(os.path.join(test_dir, "test", "historical", "target", "pr_tasmax_ACCESS-CM2_1981-2000.nc"))

print("NetCDF generation complete!")
