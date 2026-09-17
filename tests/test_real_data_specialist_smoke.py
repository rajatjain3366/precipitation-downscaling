import os
import sys
import time
import traceback
import numpy as np
import tensorflow as tf

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(WORKSPACE_ROOT, "src")
for p in [SRC_DIR, WORKSPACE_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data_adapter_cordex import load_cordex_dataset
from src.ConvolutionalNetworks import BGNet, BGCallWrapper, unpack_bgout, BG_mean
from src.GammaLoss import GammaLoss, EpochLogger
from src.useful_functions import data_between

print("=" * 85)
print("PHASE 9: REAL-DATA BGNET SPECIALIST TRAINING SMOKE TEST")
print("=" * 85)
print("NOTE: This is a software gradient/training verification test on real CORDEX data.")
print("It is NOT a research reproduction and must NOT be compared against paper metrics.")
print("=" * 85)

test_dir = os.path.join(WORKSPACE_ROOT, "data", "CORDEX_NZ_sample")

# 1. Load Real Data Subset through Adapter
print("\n>>> [1] Loading Real CORDEX-ML-Bench Data Subset via data_adapter_cordex.py...")
data_dict = load_cordex_dataset(
    data_dir=test_dir,
    config="config_A",         # 4-channel core (q850, t850, u850, v850)
    target_mode="land_masked", # D_land = 2418 land points
    sample_days=30,            # 30-day temporal subset
    verbose=False
)

X_train_full = data_dict['x_train'] # (30, 16, 16, 4)
Y_train_full = data_dict['y_train'] # (30, 2418)
D_land = data_dict['metadata']['land_cells']

print(f"  [1.1] Loaded X_train Shape: {X_train_full.shape} (dtype: {X_train_full.dtype})")
print(f"  [1.2] Loaded Y_train Shape: {Y_train_full.shape} (dtype: {Y_train_full.dtype})")
print(f"  [1.3] Target Land Dimension: D_land = {D_land}")

# 2. Target Precipitation Distribution Statistics
total_elements = Y_train_full.size
zero_count = int(np.sum(Y_train_full == 0.0))
zero_pct = (zero_count / total_elements) * 100.0
pos_mask = Y_train_full > 0.0
pos_values = Y_train_full[pos_mask]

min_val = float(np.min(Y_train_full))
max_val = float(np.max(Y_train_full))
mean_val = float(np.mean(Y_train_full))
std_val = float(np.std(Y_train_full))
pos_mean = float(np.mean(pos_values)) if len(pos_values) > 0 else 0.0
pos_max = float(np.max(pos_values)) if len(pos_values) > 0 else 0.0

print("\n>>> [2] Precipitation Target Statistics (Real CORDEX Data):")
print(f"  [2.1] Total Target Grid Points:   {total_elements:,}")
print(f"  [2.2] Dry / Zero Points:           {zero_count:,} ({zero_pct:.2f}%)")
print(f"  [2.3] Wet / Positive Points:       {len(pos_values):,} ({100.0 - zero_pct:.2f}%)")
print(f"  [2.4] Overall Precipitation Range: [{min_val:.2f}, {max_val:.2f}] mm/day")
print(f"  [2.5] Overall Mean +/- Std:        {mean_val:.2f} +/- {std_val:.2f} mm/day")
print(f"  [2.6] Positive Rain Mean:          {pos_mean:.2f} mm/day (Max: {pos_max:.2f} mm/day)")
print(f"  [2.7] Any NaNs in X?:              {np.any(np.isnan(X_train_full))}")
print(f"  [2.8] Any NaNs in Y?:              {np.any(np.isnan(Y_train_full))}")

# 3. Quantile / Intensity Partitioning
print("\n>>> [3] Creating Specialist Training Subset via data_between...")
q_low, q_high = 0.0, 0.5 # Specialist 1 covers lower 50% intensity regime
X_spec, Y_spec = data_between(Y_train_full, q_low, X_train_full, q_high)

N_spec = X_spec.shape[0]
print(f"  [3.1] Partition Range:             [{q_low:.2f}, {q_high:.2f}]")
print(f"  [3.2] Partitioned Samples Count:   {N_spec} (out of {X_train_full.shape[0]})")
print(f"  [3.3] Specialist X Shape:          {X_spec.shape}")
print(f"  [3.4] Specialist Y Shape:          {Y_spec.shape}")
assert X_spec.shape[0] == N_spec and Y_spec.shape[0] == N_spec
assert X_spec.shape[1:] == (16, 16, 4)
assert Y_spec.shape[1] == D_land
assert not np.any(np.isnan(X_spec))
assert not np.any(np.isnan(Y_spec))

# 4. BGNet Specialist Model Instantiation & Training
print("\n>>> [4] Instantiating and Training BGNet Specialist with GammaLoss on CPU...")
tf.random.set_seed(42)
spec_model = BGNet(
    output_dim=D_land,
    coords=None,
    baseline=False,
    ch_attn=True,
    avg_pool=True,
    max_pool=True,
    sp_attn=False,
    kernel_sizes=[3, 3, 3],
    num_channels=[32, 64, 128],
    num_dense=[128],
    drop_out_rate=0.1,
    reduction_rate=1,
    aux_data=None,
)

loss_fn = GammaLoss(epsilon=0.0001, y_thrs=0.5)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
spec_model.compile(optimizer=optimizer, loss=loss_fn)

# Capture initial weights to verify weight update
# Build model with dummy forward pass
_ = spec_model(X_spec[:2], training=False)
initial_weights = [w.numpy().copy() for w in spec_model.trainable_weights]

# Evaluate Initial Loss & Gradient Norm
with tf.GradientTape() as tape:
    pred_init = spec_model(X_spec, training=True)
    init_loss = loss_fn(Y_spec, pred_init)

grads = tape.gradient(init_loss, spec_model.trainable_weights)
none_grads = [w.name for g, w in zip(grads, spec_model.trainable_weights) if g is None]
assert len(none_grads) == 0, f"Variables with None gradients: {none_grads}"
grad_norm = float(np.sqrt(np.sum([tf.norm(g).numpy()**2 for g in grads])))

print(f"  [4.1] Initial GammaLoss:           {init_loss.numpy():.6f}")
print(f"  [4.2] Initial Gradient Norm:       {grad_norm:.6f} (Finite & Non-Zero)")
assert np.isfinite(init_loss.numpy()) and not np.isnan(init_loss.numpy())
assert np.isfinite(grad_norm) and not np.isnan(grad_norm)

# Train for 5 epochs
n_epochs = 5
batch_size = 4
epoch_losses = []

print(f"\n  --- Commencing {n_epochs} Epochs of Specialist Training (Batch Size = {batch_size}) ---")
t_start = time.time()
for epoch in range(n_epochs):
    # Mini-batch training loop
    indices = np.arange(N_spec)
    np.random.shuffle(indices)
    batch_loss_list = []
    
    for i in range(0, N_spec, batch_size):
        idx = indices[i:i+batch_size]
        xb, yb = X_spec[idx], Y_spec[idx]
        
        with tf.GradientTape() as tape:
            pb = spec_model(xb, training=True)
            loss_b = loss_fn(yb, pb)
            
        b_grads = tape.gradient(loss_b, spec_model.trainable_weights)
        optimizer.apply_gradients(zip(b_grads, spec_model.trainable_weights))
        batch_loss_list.append(loss_b.numpy())
        
    avg_ep_loss = float(np.mean(batch_loss_list))
    epoch_losses.append(avg_ep_loss)
    print(f"    Epoch {epoch+1}/{n_epochs} - GammaLoss: {avg_ep_loss:.6f}")

t_train = time.time() - t_start
final_eval_loss = float(loss_fn(Y_spec, spec_model(X_spec, training=False)).numpy())
print(f"  [4.3] Final Post-Training Loss:    {final_eval_loss:.6f} (Total Time: {t_train:.2f}s)")
assert np.isfinite(final_eval_loss) and not np.isnan(final_eval_loss)

# 5. Weight-Update Verification
final_weights = [w.numpy() for w in spec_model.trainable_weights]
weight_diffs = [float(np.max(np.abs(w_post - w_pre))) for w_pre, w_post in zip(initial_weights, final_weights)]
max_weight_delta = max(weight_diffs)
print(f"\n>>> [5] Weight Update Verification:")
print(f"  [5.1] Max Parameter Delta:         {max_weight_delta:.6e}")
print(f"  [5.2] Number of Updated Tensors:   {sum(d > 0.0 for d in weight_diffs)} / {len(weight_diffs)}")
assert max_weight_delta > 0.0, "Weights did not update during training!"

# 6. Post-Training Prediction & Output Integrity Checks
print("\n>>> [6] Post-Training Prediction & Shape Verification:")
pred_raw = spec_model(X_spec[:2], training=False)
p_out, alpha_out, beta_out = unpack_bgout(pred_raw)

print(f"  [6.1] Raw BGNet Output Shape:      {pred_raw.shape} (Expected: (2, {D_land}, 3))")
assert pred_raw.shape == (2, D_land, 3)

min_p, max_p = float(tf.reduce_min(p_out).numpy()), float(tf.reduce_max(p_out).numpy())
print(f"  [6.2] Probability p Range:         [{min_p:.4f}, {max_p:.4f}] (Strictly in [0, 1])")
assert min_p >= 0.0 and max_p <= 1.0

alpha_finite = bool(np.all(np.isfinite(alpha_out.numpy())))
beta_finite = bool(np.all(np.isfinite(beta_out.numpy())))
pred_finite = bool(np.all(np.isfinite(pred_raw.numpy())))
print(f"  [6.3] alpha Values Finite?:        {alpha_finite}")
print(f"  [6.4] beta Values Finite?:         {beta_finite}")
print(f"  [6.5] All Outputs Finite?:         {pred_finite}")
assert alpha_finite and beta_finite and pred_finite

# BGCallWrapper Mean Output
wrapped_model = BGCallWrapper(spec_model)
pred_mean = wrapped_model(X_spec[:2])
print(f"  [6.6] BGCallWrapper Output Shape:  {pred_mean.shape} (Expected: (2, {D_land}))")
assert pred_mean.shape == (2, D_land)
assert np.all(np.isfinite(pred_mean.numpy()))

# Software Sanity Check MSE
sanity_mse = float(tf.reduce_mean(tf.keras.metrics.mean_squared_error(Y_spec[:2], pred_mean)).numpy())
print(f"  [6.7] Software Sanity MSE (2 days):{sanity_mse:.4f} (Software sanity check only)")

print("\n" + "=" * 85)
print("ALL PHASE 9 SMOKE TEST VERIFICATIONS PASSED SUCCESSFULLY ON CPU.")
print("=" * 85)
