import os
import sys
import traceback
import numpy as np
import tensorflow as tf
from sklearn.mixture import GaussianMixture

# Ensure C:/BTP is in python path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(WORKSPACE_ROOT, "src")
for p in [SRC_DIR, WORKSPACE_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.ConvolutionalNetworks import BGNet, BGCallWrapper, unpack_bgout, BG_mean
from src.GammaLoss import GammaLoss, EpochLogger
from src.quantiles import bin_y_var, OmegaLoss, Omega, ynetwork
from src.useful_functions import data_between

print("=" * 85)
print("PHASE 6: COMPLETE SYNTHETIC QRE PIPELINE INTEGRATION TEST")
print("=" * 85)
print("NOTE: This is a software pipeline verification test on CPU with synthetic data.")
print("It is NOT a research reproduction and must NOT be presented as such.")
print("=" * 85)

pipeline_results = {}

# -----------------------------------------------------------------------------
# STAGE 1: Synthetic Dataset Construction
# -----------------------------------------------------------------------------
print("\n>>> [STAGE 1] Creating CPU-Friendly Synthetic Dataset...")
try:
    np.random.seed(42)
    tf.random.set_seed(42)

    # Dimensions
    T_train = 40          # 40 daily timesteps
    T_test = 10           # 10 daily timesteps
    in_lat, in_lon = 36, 41
    in_channels = 5       # 5 ERA5 atmospheric variables
    toy_output_dim = 20   # 20 spatial grid points (simulating VCSN land cells)
    N_specialists = 3     # N = 3 intensity specialist models
    batch_size = 8
    epochs = 4

    # 1. Synthetic Atmospheric Predictors (standard normal Gaussian fields)
    x_train_synthetic = np.random.randn(T_train, in_lat, in_lon, in_channels).astype(np.float32)
    x_test_synthetic = np.random.randn(T_test, in_lat, in_lon, in_channels).astype(np.float32)

    # 2. Synthetic Zero-Inflated Heavy-Tailed Precipitation Target (exponential with dry days)
    # ~60% dry days (zeros), ~40% positive rain amounts (up to ~30 mm/day)
    raw_rain_train = np.random.exponential(scale=6.0, size=(T_train, toy_output_dim)).astype(np.float32)
    dry_mask_train = np.random.binomial(n=1, p=0.4, size=(T_train, toy_output_dim)).astype(np.float32)
    y_train_synthetic = raw_rain_train * dry_mask_train

    raw_rain_test = np.random.exponential(scale=6.0, size=(T_test, toy_output_dim)).astype(np.float32)
    dry_mask_test = np.random.binomial(n=1, p=0.4, size=(T_test, toy_output_dim)).astype(np.float32)
    y_test_synthetic = raw_rain_test * dry_mask_test

    print(f"  [1.1] X_train shape: {x_train_synthetic.shape}")
    print(f"  [1.2] Y_train shape: {y_train_synthetic.shape} (Zero-inflated, range: [{np.min(y_train_synthetic):.2f}, {np.max(y_train_synthetic):.2f}])")
    print(f"  [1.3] X_test shape:  {x_test_synthetic.shape}")
    print(f"  [1.4] Y_test shape:  {y_test_synthetic.shape}")

    pipeline_results["Stage 1 (Dataset Setup)"] = "PASS"
except Exception as e:
    print(f"  --> STAGE 1 FAILED: {e}")
    traceback.print_exc()
    pipeline_results["Stage 1 (Dataset Setup)"] = f"FAIL: {e}"

# -----------------------------------------------------------------------------
# STAGE 2: Quantile Intensity Range Generation (GMM Clustering on Daily Sums)
# -----------------------------------------------------------------------------
print("\n>>> [STAGE 2] Generating Quantile Intensity Regimes via GMM Clustering...")
try:
    # Compute daily spatial precipitation sum
    sum_y = np.expand_dims(np.sum(y_train_synthetic, axis=-1), axis=-1)
    print(f"  [2.1] Daily domain precipitation sums (first 5 days): {sum_y[:5].flatten().round(2)}")

    gmm = GaussianMixture(n_components=N_specialists, n_init=10, random_state=42).fit(sum_y)
    counts = np.sort([np.sum(np.where(sum_y <= y_upper, 1.0, 0.0)) for y_upper in gmm.means_])

    Q_ranges = [c / T_train for c in counts]
    Q_ranges.insert(0, 0.0)
    Q_ranges[-1] = 1.0
    Q_intervals = [(float(Q_ranges[i]), float(Q_ranges[i + 1])) for i in range(len(Q_ranges) - 1)]

    # Clean duplicates or degenerate edges if any
    clean_intervals = []
    prev_q = 0.0
    for q1, q2 in Q_intervals:
        q_start = prev_q
        q_end = max(q_start + 0.1, q2)
        if q_end > 1.0 or q1 == Q_intervals[-1][0]:
            q_end = 1.0
        clean_intervals.append((q_start, q_end))
        prev_q = q_end
        if q_end >= 1.0:
            break
    if clean_intervals[-1][1] < 1.0:
        clean_intervals[-1] = (clean_intervals[-1][0], 1.0)
    Q_intervals = clean_intervals

    print(f"  [2.2] Identified N={len(Q_intervals)} Intensity Regimes: {Q_intervals}")
    assert len(Q_intervals) >= 2, "Quantile intervals failed to generate properly"
    pipeline_results["Stage 2 (Quantile Generation)"] = "PASS"
except Exception as e:
    print(f"  --> STAGE 2 FAILED: {e}")
    traceback.print_exc()
    pipeline_results["Stage 2 (Quantile Generation)"] = f"FAIL: {e}"

# -----------------------------------------------------------------------------
# STAGE 3: Specialist BGNet Models Training on Quantile Subsets
# -----------------------------------------------------------------------------
print("\n>>> [STAGE 3] Training Specialist BGNet Models on Quantile Partitions...")
specialist_models = {}
specialist_losses_before = {}
specialist_losses_after = {}

try:
    for n, (q_low, q_high) in enumerate(Q_intervals):
        # 1. Partition data using data_between
        Xq, yq = data_between(y_train_synthetic, q_low, x_train_synthetic, q_high)
        num_q_samples = Xq.shape[0]
        print(f"\n  --- Specialist {n+1} (Regime [{q_low:.2f}, {q_high:.2f}]): {num_q_samples} training samples ---")

        if num_q_samples == 0:
            print(f"  [Warning] Fallback: using all samples for degenerate slice")
            Xq, yq = x_train_synthetic, y_train_synthetic
            num_q_samples = Xq.shape[0]

        # 2. Instantiate BGNet Specialist
        bg_model = BGNet(
            output_dim=toy_output_dim,
            coords=None,
            baseline=False,
            ch_attn=True,
            avg_pool=True,
            max_pool=True,
            kernel_sizes=[3, 3, 3],
            num_channels=[32, 64, 128],
            num_dense=[64],
            drop_out_rate=0.1,
            reduction_rate=1,
            aux_data=None,
        )

        loss_fn = GammaLoss(epsilon=0.0001, y_thrs=0.5)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        bg_model.compile(optimizer=optimizer, loss=loss_fn)

        # Initial loss evaluation
        init_loss = loss_fn(yq, bg_model(Xq, training=False)).numpy()
        specialist_losses_before[f"Specialist_{n+1}"] = float(init_loss)
        print(f"  [3.{n+1}.1] Initial GammaLoss:   {init_loss:.6f}")

        # Train for few epochs
        epoch_logger = EpochLogger()
        history = bg_model.fit(
            Xq, yq,
            batch_size=min(batch_size, num_q_samples),
            epochs=epochs,
            verbose=0,
            callbacks=[epoch_logger]
        )

        final_loss = loss_fn(yq, bg_model(Xq, training=False)).numpy()
        specialist_losses_after[f"Specialist_{n+1}"] = float(final_loss)
        print(f"  [3.{n+1}.2] Final GammaLoss:     {final_loss:.6f} (Loss delta: {final_loss - init_loss:.6f})")

        # Wrap specialist with BGCallWrapper
        wrapped_specialist = BGCallWrapper(bg_model)
        specialist_models[f"CNN_{n}"] = [wrapped_specialist]

    print(f"\n  [3.Final] Successfully trained and wrapped all {len(specialist_models)} specialists.")
    pipeline_results["Stage 3 (Specialist Training)"] = "PASS"
except Exception as e:
    print(f"  --> STAGE 3 FAILED: {e}")
    traceback.print_exc()
    pipeline_results["Stage 3 (Specialist Training)"] = f"FAIL: {e}"

# -----------------------------------------------------------------------------
# STAGE 4: Weight Network Target Discretization & Training via OmegaLoss
# -----------------------------------------------------------------------------
print("\n>>> [STAGE 4] Discretizing Targets and Training Weight Network (Omega)...")
try:
    num_bins = len(Q_intervals)

    # 1. Discretize training targets into one-hot class vectors
    quantiles_targets = bin_y_var(y_train_synthetic, Q_intervals)
    y_omega_train = np.array(quantiles_targets, dtype=np.float32)
    print(f"  [4.1] Discretized Target Shape:  {y_omega_train.shape}")
    assert y_omega_train.shape == (T_train, num_bins)
    assert np.all(np.sum(y_omega_train, axis=-1) == 1.0), "Targets are not valid one-hot vectors"

    # Class distribution
    class_counts = np.sum(y_omega_train, axis=0)
    print(f"  [4.2] Sample counts per regime: {class_counts}")

    # 2. Instantiate Weight Network (Omega)
    weight_network = Omega(
        output_dim=num_bins,
        kernels=[3, 3, 3],
        channels=[32, 64, 128]
    )

    omega_loss = OmegaLoss(num_classes=num_bins, p=1)
    weight_network.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=omega_loss
    )

    init_omega_loss = tf.reduce_mean(omega_loss(y_omega_train, weight_network(x_train_synthetic, training=False))).numpy()
    print(f"  [4.3] Initial OmegaLoss (EMD):   {init_omega_loss:.6f}")

    # Train Weight Network
    weight_network.fit(
        x_train_synthetic, y_omega_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0
    )

    final_omega_loss = tf.reduce_mean(omega_loss(y_omega_train, weight_network(x_train_synthetic, training=False))).numpy()
    print(f"  [4.4] Final OmegaLoss (EMD):     {final_omega_loss:.6f} (Loss delta: {final_omega_loss - init_omega_loss:.6f})")

    # Verify predicted weights properties
    pred_weights = weight_network(x_test_synthetic, training=False).numpy()
    print(f"  [4.5] Test Predicted Weights Shape: {pred_weights.shape}")
    row_sums = np.sum(pred_weights, axis=-1)
    print(f"  [4.6] Softmax Weight Row Sums (Test Set): {row_sums.round(5)}")
    assert np.allclose(row_sums, 1.0, atol=1e-4), "Softmax weights do not sum to 1.0"
    assert np.all(pred_weights >= 0.0), "Negative weights encountered"

    pipeline_results["Stage 4 (Weight Network Training)"] = "PASS"
except Exception as e:
    print(f"  --> STAGE 4 FAILED: {e}")
    traceback.print_exc()
    pipeline_results["Stage 4 (Weight Network Training)"] = f"FAIL: {e}"

# -----------------------------------------------------------------------------
# STAGE 5: Complete QRE Ensemble (ynetwork) Assembly & End-to-End Prediction
# -----------------------------------------------------------------------------
print("\n>>> [STAGE 5] Assembling QRE Ensemble and Performing End-to-End Prediction...")
try:
    # 1. Assemble QRE Ensemble
    qre_model = ynetwork(specialist_models, weight_model=weight_network)

    # 2. Forward pass on test set
    y_pred_qre = qre_model(x_test_synthetic)
    print(f"  [5.1] Final QRE Prediction Shape: {y_pred_qre.shape}")
    assert y_pred_qre.shape == (T_test, toy_output_dim), f"Expected {(T_test, toy_output_dim)}, got {y_pred_qre.shape}"

    # 3. Check finite and non-NaN
    assert not np.any(np.isnan(y_pred_qre.numpy())), "QRE output contains NaN!"
    assert not np.any(np.isinf(y_pred_qre.numpy())), "QRE output contains Inf!"
    print(f"  [5.2] QRE Prediction Range: [{np.min(y_pred_qre.numpy()):.4f}, {np.max(y_pred_qre.numpy()):.4f}] (Finite & Valid)")

    # 4. Mathematical Linearity Check: verify g(x) == sum(w_i * f_i)
    # Extract individual specialist predictions
    individual_preds = []
    for s_name, s_list in specialist_models.items():
        # s_list[0] is wrapped BGNet
        f_i_out = s_list[0](x_test_synthetic)
        individual_preds.append(f_i_out)
    individual_preds = np.stack(individual_preds, axis=1) # shape: (T_test, N, toy_output_dim)

    # Manual combination
    w_expanded = np.expand_dims(pred_weights, axis=-1)   # shape: (T_test, N, 1)
    manual_g = np.sum(w_expanded * individual_preds, axis=1) # shape: (T_test, toy_output_dim)

    max_diff = np.max(np.abs(y_pred_qre.numpy() - manual_g))
    print(f"  [5.3] Max Difference between ynetwork and Manual g(x): {max_diff:.8e}")
    assert np.isclose(max_diff, 0.0, atol=1e-5), f"Mathematical discrepancy in QRE aggregation: {max_diff}"

    # 5. Basic MSE Evaluation on Test Set
    test_mse = tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_test_synthetic, y_pred_qre)).numpy()
    print(f"  [5.4] Synthetic Test Set MSE:     {test_mse:.4f}")

    pipeline_results["Stage 5 (QRE Assembly & Prediction)"] = "PASS"
except Exception as e:
    print(f"  --> STAGE 5 FAILED: {e}")
    traceback.print_exc()
    pipeline_results["Stage 5 (QRE Assembly & Prediction)"] = f"FAIL: {e}"

# -----------------------------------------------------------------------------
# STAGE 6: Baseline Model Comparisons (BGNet Single Baseline vs QRE)
# -----------------------------------------------------------------------------
print("\n>>> [STAGE 6] Verifying Single BGNet Baseline Comparison Model...")
try:
    # Single BGNet trained on full training set without segmentation
    single_bgnet = BGNet(
        output_dim=toy_output_dim,
        coords=None,
        baseline=False,
        ch_attn=True,
        avg_pool=True,
        max_pool=True,
        kernel_sizes=[3, 3, 3],
        num_channels=[32, 64, 128],
        num_dense=[64],
        aux_data=None,
    )
    single_bgnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=GammaLoss())
    single_bgnet.fit(x_train_synthetic, y_train_synthetic, batch_size=batch_size, epochs=epochs, verbose=0)
    single_wrapped = BGCallWrapper(single_bgnet)

    y_pred_single = single_wrapped(x_test_synthetic)
    single_mse = tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_test_synthetic, y_pred_single)).numpy()
    print(f"  [6.1] Single BGNet Test MSE:      {single_mse:.4f}")
    print(f"  [6.2] QRE Test MSE:               {test_mse:.4f}")

    pipeline_results["Stage 6 (Baseline Comparison)"] = "PASS"
except Exception as e:
    print(f"  --> STAGE 6 FAILED: {e}")
    traceback.print_exc()
    pipeline_results["Stage 6 (Baseline Comparison)"] = f"FAIL: {e}"

# -----------------------------------------------------------------------------
# SUMMARY REPORT
# -----------------------------------------------------------------------------
print("\n" + "=" * 85)
print("PHASE 6 INTEGRATION TEST SUMMARY REPORT")
print("=" * 85)
all_passed = True
for stage_name, stage_status in pipeline_results.items():
    print(f"  {stage_name:<42}: {stage_status}")
    if stage_status != "PASS":
        all_passed = False
print("=" * 85)
if all_passed:
    print("ALL 6 PIPELINE INTEGRATION STAGES PASSED SUCCESSFULLY ON CPU.")
else:
    print("SOME STAGES FAILED. SEE DETAILED ERROR LOGS ABOVE.")
print("=" * 85)
