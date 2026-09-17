import os
import sys
import time
import traceback
import numpy as np
import xarray as xr
import tensorflow as tf

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(WORKSPACE_ROOT, "src")
for p in [SRC_DIR, WORKSPACE_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data_adapter_cordex import load_cordex_dataset, extract_land_mask, unflatten_target_cordex
from src.ConvolutionalNetworks import BGNet, BGCallWrapper, unpack_bgout

print("=" * 85)
print("PHASE 8: CORDEX DATA ADAPTER VERIFICATION & REAL-DATA FORWARD PASS TEST")
print("=" * 85)

test_dir = os.path.join(WORKSPACE_ROOT, "data", "CORDEX_NZ_sample")
os.makedirs(os.path.join(test_dir, "train", "ESD_pseudo-reality", "predictors"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "train", "ESD_pseudo-reality", "target"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "test", "historical", "predictors", "perfect"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "test", "historical", "target"), exist_ok=True)

# 1. Create realistic sample NetCDF files with exact CORDEX-ML-Bench variables & metadata
T_train_sample = 20
T_test_sample = 5
H_in, W_in = 16, 16
H_out, W_out = 128, 128
levels = [850, 700, 500]

times_tr = np.arange("1961-01-01", "1961-01-21", dtype="datetime64[D]")
times_te = np.arange("1981-01-01", "1981-01-06", dtype="datetime64[D]")
lat_in = np.linspace(-51, -33, H_in)
lon_in = np.linspace(165, 184, W_in)
lat_out = np.linspace(-51, -33, H_out)
lon_out = np.linspace(165, 184, W_out)

# Predictor Dataset (ACCESS-CM2_1961-1980.nc)
# Variables: u, v, q, t, z with dimensions (time, level, lat, lon)
np.random.seed(42)
ds_pred_tr = xr.Dataset(
    data_vars={
        "u": (["time", "level", "lat", "lon"], np.random.randn(T_train_sample, 3, H_in, W_in).astype(np.float32)),
        "v": (["time", "level", "lat", "lon"], np.random.randn(T_train_sample, 3, H_in, W_in).astype(np.float32)),
        "q": (["time", "level", "lat", "lon"], (np.random.rand(T_train_sample, 3, H_in, W_in) * 0.015).astype(np.float32)),
        "t": (["time", "level", "lat", "lon"], (280.0 + np.random.randn(T_train_sample, 3, H_in, W_in) * 5.0).astype(np.float32)),
        "z": (["time", "level", "lat", "lon"], (1400.0 + np.random.randn(T_train_sample, 3, H_in, W_in) * 50.0).astype(np.float32)),
    },
    coords={"time": times_tr, "level": levels, "lat": lat_in, "lon": lon_in}
)

ds_pred_te = xr.Dataset(
    data_vars={
        "u": (["time", "level", "lat", "lon"], np.random.randn(T_test_sample, 3, H_in, W_in).astype(np.float32)),
        "v": (["time", "level", "lat", "lon"], np.random.randn(T_test_sample, 3, H_in, W_in).astype(np.float32)),
        "q": (["time", "level", "lat", "lon"], (np.random.rand(T_test_sample, 3, H_in, W_in) * 0.015).astype(np.float32)),
        "t": (["time", "level", "lat", "lon"], (280.0 + np.random.randn(T_test_sample, 3, H_in, W_in) * 5.0).astype(np.float32)),
        "z": (["time", "level", "lat", "lon"], (1400.0 + np.random.randn(T_test_sample, 3, H_in, W_in) * 50.0).astype(np.float32)),
    },
    coords={"time": times_te, "level": levels, "lat": lat_in, "lon": lon_in}
)

# Target Dataset (pr_tasmax_ACCESS-CM2_1961-1980.nc)
# Variables: pr, tasmax with dimensions (time, lat, lon)
raw_pr_tr = np.random.exponential(scale=5.0, size=(T_train_sample, H_out, W_out)).astype(np.float32)
# Create Realistic Land-Sea Mask for New Zealand Domain (~20% land in bounding box)
xx, yy = np.meshgrid(np.linspace(-1, 1, W_out), np.linspace(-1, 1, H_out))
island_shape = (np.abs(xx + 0.3 * yy) < 0.25) & (np.abs(yy) < 0.6)
raw_orog = np.where(island_shape, 500.0 + 800.0 * np.exp(-(xx**2 + yy**2) * 5), 0.0).astype(np.float32)

ds_target_tr = xr.Dataset(
    data_vars={
        "pr": (["time", "lat", "lon"], raw_pr_tr),
        "tasmax": (["time", "lat", "lon"], (290.0 + np.random.randn(T_train_sample, H_out, W_out) * 4.0).astype(np.float32))
    },
    coords={"time": times_tr, "lat": lat_out, "lon": lon_out}
)

ds_target_te = xr.Dataset(
    data_vars={
        "pr": (["time", "lat", "lon"], np.random.exponential(scale=5.0, size=(T_test_sample, H_out, W_out)).astype(np.float32)),
        "tasmax": (["time", "lat", "lon"], (290.0 + np.random.randn(T_test_sample, H_out, W_out) * 4.0).astype(np.float32))
    },
    coords={"time": times_te, "lat": lat_out, "lon": lon_out}
)

# Static Dataset (static.nc)
ds_static = xr.Dataset(
    data_vars={
        "orog": (["lat", "lon"], raw_orog),
        "sftlf": (["lat", "lon"], np.where(raw_orog > 0, 100.0, 0.0).astype(np.float32))
    },
    coords={"lat": lat_out, "lon": lon_out}
)

# Save sample NetCDF files
ds_pred_tr.to_netcdf(os.path.join(test_dir, "train", "ESD_pseudo-reality", "predictors", "ACCESS-CM2_1961-1980.nc"))
ds_target_tr.to_netcdf(os.path.join(test_dir, "train", "ESD_pseudo-reality", "target", "pr_tasmax_ACCESS-CM2_1961-1980.nc"))
ds_static.to_netcdf(os.path.join(test_dir, "train", "ESD_pseudo-reality", "static.nc"))
ds_pred_te.to_netcdf(os.path.join(test_dir, "test", "historical", "predictors", "perfect", "ACCESS-CM2_1981-2000.nc"))
ds_target_te.to_netcdf(os.path.join(test_dir, "test", "historical", "target", "pr_tasmax_ACCESS-CM2_1981-2000.nc"))

print("Sample CORDEX NetCDF dataset created for adapter verification.")

# -----------------------------------------------------------------------------
# TEST 1: Load via CORDEX Real-Data Adapter (Configuration A: 4 Channels)
# -----------------------------------------------------------------------------
print("\n>>> [TEST 1] Loading Dataset via CORDEX Adapter (Config A: 4-Channel 850 hPa Core)...")
try:
    data_dict = load_cordex_dataset(
        data_dir=test_dir,
        config="config_A",
        target_mode="land_masked",
        verbose=True
    )

    X_train = data_dict['x_train']
    Y_train = data_dict['y_train']
    X_test = data_dict['x_test']
    Y_test = data_dict['y_test']
    aux = data_dict['auxiliary']
    mask = data_dict['mask']
    D_land = data_dict['metadata']['land_cells']

    # Verifications
    print("\n--- Structural Verifications ---")
    print(f"  [1.1] X_train Shape:         {X_train.shape} (Expected: ({T_train_sample}, {H_in}, {W_in}, 4))")
    assert X_train.shape == (T_train_sample, H_in, W_in, 4), f"Mismatch in X_train: {X_train.shape}"
    assert X_train.shape[-1] == 4, "Config A must have exactly 4 channels"

    print(f"  [1.2] Y_train Shape:         {Y_train.shape} (Expected: ({T_train_sample}, {D_land}))")
    assert Y_train.shape == (T_train_sample, D_land), f"Mismatch in Y_train: {Y_train.shape}"
    assert Y_train.shape[1] == D_land, "Y_train second dimension must equal derived D_land"

    print(f"  [1.3] Derived Land Cells:    {D_land} (out of {H_out*W_out})")
    assert D_land > 0 and D_land < H_out * W_out, f"Invalid D_land: {D_land}"

    print(f"  [1.4] Static Aux Shape:      {aux.shape} (Expected: (1, {H_out}, {W_out}, 1))")
    assert aux.shape == (1, H_out, W_out, 1), f"Mismatch in auxiliary: {aux.shape}"

    print(f"  [1.5] NaN Check:             X has NaNs? {np.any(np.isnan(X_train))}, Y has NaNs? {np.any(np.isnan(Y_train))}")
    assert not np.any(np.isnan(X_train)), "X_train contains unexpected NaNs"
    assert not np.any(np.isnan(Y_train)), "Y_train contains unexpected NaNs"

    print(f"  [1.6] Predictor Finiteness:  All finite? {np.all(np.isfinite(X_train))}")
    assert np.all(np.isfinite(X_train)), "X_train contains non-finite values"

    # Test spatial unflattening
    y_2d_reconstructed = unflatten_target_cordex(Y_train, mask, H=H_out, W=W_out)
    print(f"  [1.7] Reconstructed 2D Grid: {y_2d_reconstructed.shape} (Expected: ({T_train_sample}, {H_out}, {W_out}))")
    assert y_2d_reconstructed.shape == (T_train_sample, H_out, W_out)

    print("  --> ADAPTER TEST 1 RESULT: PASS")
except Exception as e:
    print(f"  --> ADAPTER TEST 1 RESULT: FAIL ({e})")
    traceback.print_exc()
    sys.exit(1)

# -----------------------------------------------------------------------------
# TEST 2: Single Forward Pass with Real Adapted Tensors through BGNet
# -----------------------------------------------------------------------------
print("\n>>> [TEST 2] Single BGNet Forward Pass using Real Adapted CORDEX Tensors...")
try:
    batch_in = tf.convert_to_tensor(X_train[:2], dtype=tf.float32)
    print(f"  [2.1] Input Batch Tensor Shape: {batch_in.shape}")

    t0 = time.time()
    bgnet_real = BGNet(
        output_dim=D_land,
        coords=None,
        baseline=False,
        ch_attn=True,
        avg_pool=True,
        max_pool=True,
        sp_attn=False,
        kernel_sizes=[3, 3, 3],
        num_channels=[64, 128, 256],
        num_dense=[256],
        drop_out_rate=0.2,
        reduction_rate=1,
        aux_data=None,
    )

    out_tensor = bgnet_real(batch_in, training=False)
    runtime_ms = (time.time() - t0) * 1000

    p, alpha, beta = unpack_bgout(out_tensor)
    total_params = bgnet_real.count_params()

    print(f"  [2.2] BGNet Output Tensor Shape: {out_tensor.shape} (Expected: (2, {D_land}, 3))")
    assert out_tensor.shape == (2, D_land, 3), f"Expected (2, {D_land}, 3), got {out_tensor.shape}"

    print(f"  [2.3] Total Model Parameters:    {total_params:,} ({total_params * 4 / (1024**2):.2f} MB float32)")
    print(f"  [2.4] Forward-Pass Runtime (CPU):{runtime_ms:.2f} ms")

    # Value checks
    min_p, max_p = tf.reduce_min(p).numpy(), tf.reduce_max(p).numpy()
    print(f"  [2.5] p range:                   [{min_p:.4f}, {max_p:.4f}] (Valid Sigmoid)")
    assert min_p >= 0.0 and max_p <= 1.0, f"p range invalid: [{min_p}, {max_p}]"

    alpha_finite = bool(np.all(np.isfinite(alpha.numpy())))
    beta_finite = bool(np.all(np.isfinite(beta.numpy())))
    out_finite = bool(np.all(np.isfinite(out_tensor.numpy())))
    print(f"  [2.6] alpha is finite?:          {alpha_finite}")
    print(f"  [2.7] beta is finite?:           {beta_finite}")
    print(f"  [2.8] Output is finite?:         {out_finite}")
    assert alpha_finite and beta_finite and out_finite, "Model produced non-finite outputs"

    # Wrapper mean output
    wrapper = BGCallWrapper(bgnet_real)
    mean_out = wrapper(batch_in)
    print(f"  [2.9] BGCallWrapper Mean Shape:  {mean_out.shape} (Expected: (2, {D_land}))")
    assert mean_out.shape == (2, D_land)

    print("  --> FORWARD PASS TEST 2 RESULT: PASS")
except Exception as e:
    print(f"  --> FORWARD PASS TEST 2 RESULT: FAIL ({e})")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 85)
print("ALL PHASE 8 REAL-DATA ADAPTER TESTS PASSED SUCCESSFULLY.")
print("=" * 85)
