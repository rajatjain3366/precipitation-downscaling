"""
Phase 10: Real-Data Multi-Specialist QRE Integration Test
=========================================================
End-to-end integration test of the complete Quantile-Regression-Ensemble (QRE)
pipeline using real CORDEX-ML-Bench New Zealand data.

Components verified:
1. Real CORDEX-ML-Bench 4-channel predictor & land-masked precipitation ingestion
2. Quantile/intensity partitioning into N=3 ordered specialist regimes
3. Independent training of N=3 BGNet specialists with GammaLoss
4. Gradient flow and weight update verification across all specialists
5. Weight Network target generation (bin_y_var, one-hot (T, 3))
6. Omega Weight Network training with OmegaLoss (EMD)
7. Dynamic softmax weight verification (w >= 0, sum(w) = 1.0)
8. ynetwork assembly (BGCallWrapper specialists + Omega)
9. End-to-end prediction shape and value verification (batch, 2418)
10. Manual tensor aggregation check vs ynetwork (max absolute discrepancy)
11. Diagnostic software MSE check

IMPORTANT: Software integration test only. Not a scientific reproduction of paper results.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
import xarray as xr

# Add root directory to path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(WORKSPACE_ROOT, "src")
for p in [SRC_DIR, WORKSPACE_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data_adapter_cordex import load_cordex_dataset
from src.ConvolutionalNetworks import BGNet, BGCallWrapper
from src.GammaLoss import GammaLoss
from src.quantiles import Omega, OmegaLoss, ynetwork, bin_y_var
from src.useful_functions import data_between


def run_phase10_integration():
    print("=" * 85)
    print("PHASE 10: REAL-DATA MULTI-SPECIALIST QRE INTEGRATION TEST")
    print("=" * 85)
    print("NOTE: This is a software integration test on real CORDEX data.")
    print("It is NOT a research reproduction and must NOT be compared against paper metrics.")
    print("=" * 85)

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # -------------------------------------------------------------------------
    # Stage 1: Ingest Small Real CORDEX Temporal Slice
    # -------------------------------------------------------------------------
    print("\n>>> [1] Loading Real CORDEX-ML-Bench Data Subset...")
    test_dir = os.path.join(WORKSPACE_ROOT, "data", "CORDEX_NZ_sample")
    data_dict = load_cordex_dataset(
        data_dir=test_dir,
        config="config_A",         # 4-channel core (q850, t850, u850, v850)
        target_mode="land_masked", # D_land = 2418 land points
        verbose=False
    )

    X_train = data_dict['x_train']
    Y_train = data_dict['y_train']
    X_test = data_dict['x_test']
    Y_test = data_dict['y_test']

    T_train, H, W, C = X_train.shape
    T_test, D_land = Y_test.shape

    print(f"  [1.1] X_train Shape: {X_train.shape} (dtype: {X_train.dtype})")
    print(f"  [1.2] Y_train Shape: {Y_train.shape} (dtype: {Y_train.dtype})")
    print(f"  [1.3] X_test Shape:  {X_test.shape} (dtype: {X_test.dtype})")
    print(f"  [1.4] Y_test Shape:  {Y_test.shape} (dtype: {Y_test.dtype})")
    print(f"  [1.5] Land Points:   D_land = {D_land}")

    assert not np.isnan(X_train).any(), "NaNs in X_train"
    assert not np.isnan(Y_train).any(), "NaNs in Y_train"
    assert not np.isnan(X_test).any(), "NaNs in X_test"
    assert not np.isnan(Y_test).any(), "NaNs in Y_test"

    # -------------------------------------------------------------------------
    # Stage 2: Quantile Partitioning for N=3 Specialists
    # -------------------------------------------------------------------------
    print("\n>>> [2] Generating N=3 Specialist Regimes...")
    N_specialists = 3
    Q_ranges = [(0.0, 0.35), (0.35, 0.70), (0.70, 1.0)]

    specialist_data = []

    for n, (q_low, q_high) in enumerate(Q_ranges):
        Xq, yq = data_between(Y_train, q_low, X_train, q_high)
        n_samples = len(yq)
        specialist_data.append((Xq, yq))
        
        yq_mean = np.mean(yq) if n_samples > 0 else 0.0
        yq_max = np.max(yq) if n_samples > 0 else 0.0
        print(f"  [2.{n+1}] Specialist {n} ({q_low:.2f} - {q_high:.2f}): {n_samples} samples | "
              f"Mean Rain: {yq_mean:.3f} mm/day | Max Rain: {yq_max:.3f} mm/day | Xq: {Xq.shape}")
        
        assert n_samples > 0, f"Specialist {n} subset is empty!"
        assert Xq.shape[1:] == (H, W, C), f"Specialist {n} X shape mismatch: {Xq.shape}"
        assert yq.shape[1:] == (D_land,), f"Specialist {n} Y shape mismatch: {yq.shape}"
        assert not np.isnan(Xq).any(), f"NaNs in Specialist {n} X"
        assert not np.isnan(yq).any(), f"NaNs in Specialist {n} Y"

    # -------------------------------------------------------------------------
    # Stage 3: Train N=3 Independent BGNet Specialists
    # -------------------------------------------------------------------------
    print("\n>>> [3] Training N=3 Independent BGNet Specialists with GammaLoss (CPU)...")
    specialists = []
    specialist_wrappers = []
    specialist_loss_history = []
    specialist_weight_deltas = []

    loss_fn = GammaLoss(epsilon=0.0001, y_thrs=0.5)

    for n in range(N_specialists):
        Xq, yq = specialist_data[n]
        model = BGNet(
            output_dim=D_land,
            coords=None,
            baseline=False,
            ch_attn=True,
            avg_pool=True,
            max_pool=True,
            kernel_sizes=[3, 3, 3],
            num_channels=[32, 64, 128],
            num_dense=[128],
            drop_out_rate=0.1,
            reduction_rate=1,
            aux_data=None,
        )
        
        # Build model by running single dummy batch
        _ = model(Xq[:1])
        initial_weights = [w.numpy().copy() for w in model.trainable_weights]

        # Initial loss and gradients
        with tf.GradientTape() as tape:
            pred = model(Xq, training=True)
            loss_init = loss_fn(yq, pred)
        grads = tape.gradient(loss_init, model.trainable_weights)
        grad_norm = float(np.sqrt(np.sum([tf.norm(g).numpy()**2 for g in grads if g is not None])))
        
        assert np.isfinite(grad_norm) and grad_norm > 0, f"Specialist {n} gradient norm invalid: {grad_norm}"

        # Train for 3 epochs with Adam
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        epoch_losses = [float(loss_init.numpy())]

        for epoch in range(1, 4):
            with tf.GradientTape() as tape:
                pred = model(Xq, training=True)
                loss_val = loss_fn(yq, pred)
            g = tape.gradient(loss_val, model.trainable_weights)
            optimizer.apply_gradients(zip(g, model.trainable_weights))
            epoch_losses.append(float(loss_val.numpy()))

        # Check weight updates
        updated_weights = [w.numpy() for w in model.trainable_weights]
        weight_diffs = []
        for w_old, w_new in zip(initial_weights, updated_weights):
            diff = np.abs(w_new - w_old)
            val = float(np.max(diff)) if diff.size > 0 else 0.0
            weight_diffs.append(val)
        
        max_delta = max(weight_diffs)
        updated_count = sum(d > 0.0 for d in weight_diffs)
        total_non_empty = sum(w.size > 0 for w in initial_weights)

        specialists.append(model)
        specialist_wrappers.append(BGCallWrapper(model))
        specialist_loss_history.append(epoch_losses)
        specialist_weight_deltas.append((max_delta, updated_count, total_non_empty))

        print(f"  [3.{n+1}] Specialist {n}: Init Loss = {epoch_losses[0]:.4f} -> Final Loss = {epoch_losses[-1]:.4f} | "
              f"Grad Norm = {grad_norm:.4f} | Updated Tensors: {updated_count}/{total_non_empty} (max delta: {max_delta:.2e})")
        
        assert updated_count == total_non_empty, f"Specialist {n} did not update all non-empty weights!"

    # -------------------------------------------------------------------------
    # Stage 4: Generate Weight Network Targets (bin_y_var)
    # -------------------------------------------------------------------------
    print("\n>>> [4] Generating Weight Network Targets via bin_y_var...")
    quantiles_target = bin_y_var(Y_train, Q_ranges)
    Y_omega_train = np.array(quantiles_target, dtype=np.float32)

    print(f"  [4.1] Y_omega_train Shape: {Y_omega_train.shape} (Expected: ({T_train}, {N_specialists}))")
    print(f"  [4.2] Class Distribution:  {np.sum(Y_omega_train, axis=0)}")
    print(f"  [4.3] One-Hot Check:       Row Sums = {np.unique(np.sum(Y_omega_train, axis=1))} (Expected: [1.0])")

    assert Y_omega_train.shape == (T_train, N_specialists), f"Unexpected Y_omega shape: {Y_omega_train.shape}"
    assert np.allclose(np.sum(Y_omega_train, axis=1), 1.0), "Targets are not valid one-hot distributions!"

    # -------------------------------------------------------------------------
    # Stage 5: Train Omega Weight Network with OmegaLoss (EMD)
    # -------------------------------------------------------------------------
    print("\n>>> [5] Training Omega Weight Network with OmegaLoss (EMD)...")
    omega_net = Omega(output_dim=N_specialists, kernels=[3, 3, 3], channels=[32, 64, 128])
    omega_loss_fn = OmegaLoss(num_classes=N_specialists)

    # Forward pass to build
    _ = omega_net(X_train[:1])
    initial_omega_weights = [w.numpy().copy() for w in omega_net.trainable_weights]

    with tf.GradientTape() as tape:
        omega_pred = omega_net(X_train, training=True)
        loss_omega_init = tf.reduce_mean(omega_loss_fn(Y_omega_train, omega_pred))
    grads_omega = tape.gradient(loss_omega_init, omega_net.trainable_weights)
    grad_norm_omega = float(np.sqrt(np.sum([tf.norm(g).numpy()**2 for g in grads_omega if g is not None])))

    print(f"  [5.1] Initial OmegaLoss:   {float(loss_omega_init.numpy()):.6f}")
    print(f"  [5.2] Initial Grad Norm:   {grad_norm_omega:.6f}")

    optimizer_omega = tf.keras.optimizers.Adam(learning_rate=0.001)
    omega_losses = [float(loss_omega_init.numpy())]

    for epoch in range(1, 5):
        with tf.GradientTape() as tape:
            omega_pred = omega_net(X_train, training=True)
            loss_val = tf.reduce_mean(omega_loss_fn(Y_omega_train, omega_pred))
        g = tape.gradient(loss_val, omega_net.trainable_weights)
        optimizer_omega.apply_gradients(zip(g, omega_net.trainable_weights))
        omega_losses.append(float(loss_val.numpy()))

    updated_omega_weights = [w.numpy() for w in omega_net.trainable_weights]
    omega_diffs = []
    for w_old, w_new in zip(initial_omega_weights, updated_omega_weights):
        diff = np.abs(w_new - w_old)
        val = float(np.max(diff)) if diff.size > 0 else 0.0
        omega_diffs.append(val)

    max_delta_omega = max(omega_diffs)
    updated_count_omega = sum(d > 0.0 for d in omega_diffs)
    total_non_empty_omega = sum(w.size > 0 for w in initial_omega_weights)

    print(f"  [5.3] Final OmegaLoss:     {omega_losses[-1]:.6f}")
    print(f"  [5.4] Updated Tensors:     {updated_count_omega}/{total_non_empty_omega} (max delta: {max_delta_omega:.2e})")

    assert updated_count_omega == total_non_empty_omega, "Omega network did not update all non-empty weights!"

    # -------------------------------------------------------------------------
    # Stage 6: Dynamic Weight Verification on Test Data
    # -------------------------------------------------------------------------
    print("\n>>> [6] Dynamic Weight Verification on Test Set...")
    weights_test = omega_net(X_test, training=False).numpy()
    row_sums = np.sum(weights_test, axis=1)

    print(f"  [6.1] Weights Test Shape:  {weights_test.shape} (Expected: ({T_test}, {N_specialists}))")
    print(f"  [6.2] Min Weight Value:    {float(np.min(weights_test)):.4f} (>= 0.0)")
    print(f"  [6.3] Max Weight Value:    {float(np.max(weights_test)):.4f} (<= 1.0)")
    print(f"  [6.4] Row Sum Range:       [{float(np.min(row_sums)):.6f}, {float(np.max(row_sums)):.6f}] (Expected: ~1.0)")

    assert weights_test.shape == (T_test, N_specialists), f"Unexpected weights shape: {weights_test.shape}"
    assert np.all(weights_test >= 0.0), "Negative weights found!"
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Dynamic weights do not sum to 1.0!"

    # -------------------------------------------------------------------------
    # Stage 7: Assemble QRE ynetwork & Generate Predictions
    # -------------------------------------------------------------------------
    print("\n>>> [7] Assembling QRE ynetwork & Running Prediction...")
    CNN_dict = {f'CNN_{i}': [specialist_wrappers[i]] for i in range(N_specialists)}
    qre_model = ynetwork(CNN_dict, omega_net)

    # QRE Prediction
    qre_pred = qre_model(X_test)
    qre_pred_np = qre_pred.numpy() if isinstance(qre_pred, tf.Tensor) else np.array(qre_pred)

    print(f"  [7.1] QRE Prediction Shape:{qre_pred_np.shape} (Expected: ({T_test}, {D_land}))")
    print(f"  [7.2] Prediction Finiteness: All Finite = {np.all(np.isfinite(qre_pred_np))}")
    print(f"  [7.3] Prediction Range:    [{float(np.min(qre_pred_np)):.3f}, {float(np.max(qre_pred_np)):.3f}] mm/day")
    print(f"  [7.4] Prediction Mean:     {float(np.mean(qre_pred_np)):.3f} mm/day")

    assert qre_pred_np.shape == (T_test, D_land), f"Unexpected QRE pred shape: {qre_pred_np.shape}"
    assert not np.isnan(qre_pred_np).any(), "NaNs in QRE predictions!"
    assert not np.isinf(qre_pred_np).any(), "Infs in QRE predictions!"

    # -------------------------------------------------------------------------
    # Stage 8: Independent Manual Aggregation Verification
    # -------------------------------------------------------------------------
    print("\n>>> [8] Independent Manual Tensor Aggregation Verification...")
    spec_preds = []
    for i in range(N_specialists):
        f_i = specialist_wrappers[i](X_test).numpy()  # (T_test, D_land)
        spec_preds.append(f_i)
    spec_preds_stacked = np.stack(spec_preds, axis=1)  # (T_test, N_specialists, D_land)

    print(f"  [8.1] Stacked Specialist Preds Shape: {spec_preds_stacked.shape}")
    print(f"  [8.2] Test Dynamic Weights Shape:     {weights_test.shape}")

    # Manual aggregation: g(x) = sum_i omega_i(x) * f_i(x)
    weights_expanded = np.expand_dims(weights_test, axis=-1)  # (T_test, N_specialists, 1)
    manual_qre = np.sum(weights_expanded * spec_preds_stacked, axis=1)  # (T_test, D_land)

    max_abs_diff = float(np.max(np.abs(qre_pred_np - manual_qre)))
    print(f"  [8.3] Manual Aggregation Shape:       {manual_qre.shape}")
    print(f"  [8.4] Max Absolute Discrepancy:       {max_abs_diff:.2e} (Expected: < 1e-5)")

    assert max_abs_diff < 1e-5, f"Discrepancy between ynetwork and manual aggregation: {max_abs_diff}"

    # -------------------------------------------------------------------------
    # Stage 9: Diagnostic Software MSE Check
    # -------------------------------------------------------------------------
    print("\n>>> [9] Basic Test MSE Software Diagnostic...")
    test_mse = float(np.mean((qre_pred_np - Y_test) ** 2))
    print(f"  [9.1] Diagnostic Test MSE: {test_mse:.4f} mm^2/day^2")
    print(f"  [9.2] Diagnostic Note:     Strictly software sanity check; not for scientific comparison.")

    print("\n" + "=" * 85)
    print("ALL PHASE 10 MULTI-SPECIALIST QRE INTEGRATION CHECKS PASSED.")
    print("=" * 85)
    return True


if __name__ == "__main__":
    run_phase10_integration()
