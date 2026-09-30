import os
import sys
import traceback
import numpy as np
import tensorflow as tf

# Ensure C:/BTP is in python path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(WORKSPACE_ROOT, "src")
for p in [SRC_DIR, WORKSPACE_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.ConvolutionalNetworks import BGNet, unpack_bgout, BG_mean, BG_variance, MSENet, BGCallWrapper
from src.GammaLoss import GammaLoss
from src.quantiles import bin_y_var, OmegaLoss, Omega, ynetwork
from src.useful_functions import data_between

print("=" * 80)
print("ISOLATED SYNTHETIC SMOKE TEST SUITE - QRE REPRODUCTION")
print("=" * 80)
print(f"TensorFlow Version: {tf.__version__}")
print(f"NumPy Version:      {np.__version__}")
print(f"Available Devices:  {tf.config.list_physical_devices()}")
print("=" * 80)

results = {}

# ----------------------------------------------------------------------
# TEST 1: BGNet Architecture and Forward Pass
# ----------------------------------------------------------------------
print("\n>>> [TEST 1] BGNet Forward Pass & Intermediate Tensor Tracing...")
try:
    batch_size = 2
    in_lat, in_lon, in_channels = 36, 41, 5
    toy_output_dim = 10

    # 1. Instantiate synthetic input
    tf.random.set_seed(42)
    x_sample = tf.random.normal((batch_size, in_lat, in_lon, in_channels))
    print(f"  [1.1] Input Tensor Shape:           {x_sample.shape}")

    # 2. Instantiate BGNet (baseline=False, ch_attn=True, small dense)
    model = BGNet(
        output_dim=toy_output_dim,
        coords=None,
        baseline=False,
        ch_attn=True,
        avg_pool=True,
        max_pool=True,
        num_dense=[64],
        kernel_sizes=[3, 3, 3],
        num_channels=[64, 128, 256],
        drop_out_rate=0.2,
        reduction_rate=1,
        aux_data=None,
    )

    # 3. Explicitly trace intermediate layers
    x_init = model.init_conv(x_sample)
    print(f"  [1.2] After init_conv (64 filters): {x_init.shape}")

    # Test the DownScaleModule with init_conv output
    x_ds = model.ds_module(x_init)
    print(f"  [1.3] After DownScaleModule:        {x_ds.shape}")

    x_dense = model.dense_module(x_ds)
    print(f"  [1.4] After Dense Module (Flatten+Dense): {x_dense.shape}")

    alpha_out = model.alpha(x_dense)
    beta_out = model.beta(x_dense)
    p_out = model.p(x_dense)
    print(f"  [1.5] alpha branch shape:           {alpha_out.shape}")
    print(f"  [1.6] beta branch shape:            {beta_out.shape}")
    print(f"  [1.7] p branch shape:               {p_out.shape}")

    # 4. Full forward pass through model.call()
    full_out = model(x_sample, training=False)
    print(f"  [1.8] Full BGNet Output Shape:      {full_out.shape}")

    # Assertions
    assert full_out.shape == (batch_size, toy_output_dim, 3), f"Expected shape {(batch_size, toy_output_dim, 3)}, got {full_out.shape}"
    p, alpha, beta = unpack_bgout(full_out)
    assert p.shape == (batch_size, toy_output_dim, 1)
    assert alpha.shape == (batch_size, toy_output_dim, 1)
    assert beta.shape == (batch_size, toy_output_dim, 1)

    # Check value ranges: p in [0, 1] due to sigmoid
    assert tf.reduce_all(p >= 0.0) and tf.reduce_all(p <= 1.0), "p values not within [0, 1]"
    print(f"  [1.9] p range: [{tf.reduce_min(p).numpy():.4f}, {tf.reduce_max(p).numpy():.4f}] (Valid Sigmoid)")
    print(f"  [1.10] alpha range: [{tf.reduce_min(alpha).numpy():.4f}, {tf.reduce_max(alpha).numpy():.4f}]")
    print(f"  [1.11] beta range:  [{tf.reduce_min(beta).numpy():.4f}, {tf.reduce_max(beta).numpy():.4f}]")

    # Mean calculation
    mean_pred = BG_mean(p, alpha, beta)
    print(f"  [1.12] BG_mean Output Shape:        {mean_pred.shape}")
    assert mean_pred.shape == (batch_size, toy_output_dim, 1)

    print("  --> TEST 1 RESULT: PASS")
    results["TEST 1 (BGNet Forward Pass)"] = "PASS"
except Exception as e:
    print(f"  --> TEST 1 RESULT: FAIL ({e})")
    traceback.print_exc()
    results["TEST 1 (BGNet Forward Pass)"] = f"FAIL: {e}"

# ----------------------------------------------------------------------
# TEST 2: GammaLoss Computation & Gradient Backpropagation
# ----------------------------------------------------------------------
print("\n>>> [TEST 2] GammaLoss Function & Gradient Backpropagation...")
try:
    loss_fn = GammaLoss(epsilon=0.0001, y_thrs=0.5)

    # Synthetic ground truth: batch of 2, 10 stations (mix of zero/low and high precipitation)
    y_true_sample = tf.constant([
        [0.0, 0.2, 0.0, 1.5, 5.0, 0.0, 12.0, 0.1, 0.0, 25.0],
        [0.0, 0.0, 0.0, 0.0, 2.2, 8.4, 0.0, 0.0, 15.0, 30.0]
    ], dtype=tf.float32)

    with tf.GradientTape() as tape:
        y_pred = model(x_sample, training=True)
        loss_val = loss_fn(y_true_sample, y_pred)

    print(f"  [2.1] GammaLoss Value:              {loss_val.numpy():.6f}")
    assert not np.isnan(loss_val.numpy()), "Loss is NaN!"
    assert not np.isinf(loss_val.numpy()), "Loss is Inf!"

    # Compute gradients wrt trainable variables
    grads = tape.gradient(loss_val, model.trainable_variables)
    assert len(grads) > 0, "No gradients computed!"
    none_grads = [v.name for g, v in zip(grads, model.trainable_variables) if g is None]
    assert len(none_grads) == 0, f"Some variables have None gradients: {none_grads}"

    grad_norms = [tf.norm(g).numpy() for g in grads]
    total_grad_norm = np.sqrt(np.sum([gn**2 for gn in grad_norms]))
    print(f"  [2.2] Total Gradient Norm:          {total_grad_norm:.6f} (Finite & Non-Zero)")
    assert not np.isnan(total_grad_norm) and not np.isinf(total_grad_norm)

    print("  --> TEST 2 RESULT: PASS")
    results["TEST 2 (GammaLoss & Gradients)"] = "PASS"
except Exception as e:
    print(f"  --> TEST 2 RESULT: FAIL ({e})")
    traceback.print_exc()
    results["TEST 2 (GammaLoss & Gradients)"] = f"FAIL: {e}"

# ----------------------------------------------------------------------
# TEST 3: Quantile Binning and OmegaLoss (Earth Mover's Distance)
# ----------------------------------------------------------------------
print("\n>>> [TEST 3] Quantile Discretization & OmegaLoss (EMD)...")
try:
    # 1. Test data_between
    # Create synthetic dataset with 20 days
    y_synthetic_days = np.random.exponential(scale=5.0, size=(20, 10))
    q_ranges = [(0.0, 0.5), (0.5, 1.0)]
    
    y_binned = bin_y_var(y_synthetic_days, q_ranges)
    y_binned = np.array(y_binned)
    print(f"  [3.1] bin_y_var Output Shape:       {y_binned.shape}")
    assert y_binned.shape == (20, 2), f"Expected (20, 2), got {y_binned.shape}"
    # Verify one-hot row sums
    assert np.all(np.sum(y_binned, axis=1) == 1.0), "Binned classes are not valid one-hot vectors"

    # 2. Test OmegaLoss with N=3 classes
    num_classes = 3
    omega_loss_fn = OmegaLoss(num_classes=num_classes, p=1)
    
    # Weight matrix should be symmetric absolute distance: [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
    expected_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.float32)
    actual_matrix = omega_loss_fn.weight_matrix.numpy()
    print(f"  [3.2] OmegaLoss Distance Matrix:\n{actual_matrix}")
    assert np.allclose(actual_matrix, expected_matrix), f"Matrix mismatch: {actual_matrix}"

    # Test loss on exact predictions vs misclassifications
    # y_true: class 0 (one hot: [1, 0, 0])
    y_true_cls = tf.constant([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=tf.float32)
    # Perfect prediction: class 0 and class 2
    y_pred_perfect = tf.constant([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=tf.float32)
    # Misclassified prediction: class 2 instead of class 0 (distance 2)
    y_pred_wrong = tf.constant([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=tf.float32)

    loss_perfect = tf.reduce_mean(omega_loss_fn(y_true_cls, y_pred_perfect)).numpy()
    loss_wrong = tf.reduce_mean(omega_loss_fn(y_true_cls, y_pred_wrong)).numpy()
    print(f"  [3.3] Perfect Prediction EMD Loss:  {loss_perfect:.4f} (Expected 0.0)")
    print(f"  [3.4] Distant Error EMD Loss:       {loss_wrong:.4f} (Expected > 0.0)")
    assert np.isclose(loss_perfect, 0.0)
    assert loss_wrong > loss_perfect

    print("  --> TEST 3 RESULT: PASS")
    results["TEST 3 (Quantile Binning & OmegaLoss)"] = "PASS"
except Exception as e:
    print(f"  --> TEST 3 RESULT: FAIL ({e})")
    traceback.print_exc()
    results["TEST 3 (Quantile Binning & OmegaLoss)"] = f"FAIL: {e}"

# ----------------------------------------------------------------------
# TEST 4: Weight Network (Omega) Forward Pass & Softmax
# ----------------------------------------------------------------------
print("\n>>> [TEST 4] Weight Network (Omega) Forward Pass & Softmax Output...")
try:
    num_specialists = 3
    weight_net = Omega(output_dim=num_specialists, kernels=[3, 3, 3], channels=[32, 64, 128])

    w_out = weight_net(x_sample, training=False)
    print(f"  [4.1] Weight Network Output Shape:  {w_out.shape}")
    assert w_out.shape == (batch_size, num_specialists)

    # Verify softmax properties: positive and sum to 1
    row_sums = tf.reduce_sum(w_out, axis=-1).numpy()
    print(f"  [4.2] Predicted Weights:\n{w_out.numpy()}")
    print(f"  [4.3] Softmax Row Sums:             {row_sums}")
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"Softmax weights do not sum to 1: {row_sums}"
    assert np.all(w_out.numpy() >= 0.0), "Weights contain negative numbers"

    print("  --> TEST 4 RESULT: PASS")
    results["TEST 4 (Weight Network Omega)"] = "PASS"
except Exception as e:
    print(f"  --> TEST 4 RESULT: FAIL ({e})")
    traceback.print_exc()
    results["TEST 4 (Weight Network Omega)"] = f"FAIL: {e}"

# ----------------------------------------------------------------------
# TEST 5: QRE Ensemble Aggregator (ynetwork) Mathematical Verification
# ----------------------------------------------------------------------
print("\n>>> [TEST 5] QRE Ensemble (ynetwork) Aggregation & Linearity...")
try:
    # Create N=2 mock specialist models f_1(x) and f_2(x) returning constant tensor outputs
    class MockSpecialist:
        def __init__(self, const_val):
            self.const_val = const_val
        def __call__(self, x):
            # return shape (batch, toy_output_dim)
            b = tf.shape(x)[0]
            return tf.fill((b, toy_output_dim), self.const_val)

    # Specialist 1 predicts 10.0 everywhere, Specialist 2 predicts 20.0 everywhere
    f1 = MockSpecialist(10.0)
    f2 = MockSpecialist(20.0)
    cnn_dict = {'CNN_0': [f1], 'CNN_1': [f2]}

    # Mock weight network returning fixed weights: [0.3, 0.7]
    class MockWeightNet:
        def __call__(self, x):
            b = tf.shape(x)[0]
            # Use tf.tile to broadcast fixed weights to batch size b
            return tf.tile(tf.constant([[0.3, 0.7]], dtype=tf.float32), [b, 1])

    w_model = MockWeightNet()
    qre_ensemble = ynetwork(cnn_dict, weight_model=w_model)

    # g(x) = 0.3 * 10.0 + 0.7 * 20.0 = 3.0 + 14.0 = 17.0
    expected_val = 0.3 * 10.0 + 0.7 * 20.0
    g_out = qre_ensemble(x_sample)
    print(f"  [5.1] ynetwork Output Shape:        {g_out.shape}")
    assert g_out.shape == (batch_size, toy_output_dim)

    actual_val = g_out.numpy()[0, 0]
    print(f"  [5.2] Aggregated Value:             {actual_val:.4f} (Expected: {expected_val:.4f})")
    assert np.isclose(actual_val, expected_val, atol=1e-5), f"Aggregation calculation mismatch: {actual_val} vs {expected_val}"

    # Also test ynetwork with real BGNet + BGCallWrapper + real Omega model
    bg_m1 = BGCallWrapper(BGNet(output_dim=toy_output_dim, coords=None, baseline=True, num_dense=[32]))
    bg_m2 = BGCallWrapper(BGNet(output_dim=toy_output_dim, coords=None, baseline=True, num_dense=[32]))
    real_cnn_dict = {'CNN_0': [bg_m1], 'CNN_1': [bg_m2]}
    real_omega = Omega(output_dim=2, kernels=[3, 3, 3], channels=[16, 32, 64])
    real_qre = ynetwork(real_cnn_dict, weight_model=real_omega)
    real_g_out = real_qre(x_sample)
    print(f"  [5.3] Real BGNet+Omega ynetwork Output Shape: {real_g_out.shape}")
    assert real_g_out.shape == (batch_size, toy_output_dim)

    print("  --> TEST 5 RESULT: PASS")
    results["TEST 5 (QRE ynetwork Aggregation)"] = "PASS"
except Exception as e:
    print(f"  --> TEST 5 RESULT: FAIL ({e})")
    traceback.print_exc()
    results["TEST 5 (QRE ynetwork Aggregation)"] = f"FAIL: {e}"

# ----------------------------------------------------------------------
# TEST 6: Single-Step Optimization Step (End-to-End Gradient Flow)
# ----------------------------------------------------------------------
print("\n>>> [TEST 6] End-to-End Single Optimization Step on CPU...")
try:
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    init_loss = loss_fn(y_true_sample, model(x_sample, training=True)).numpy()

    with tf.GradientTape() as tape:
        preds = model(x_sample, training=True)
        step_loss = loss_fn(y_true_sample, preds)
    grads = tape.gradient(step_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    post_loss = loss_fn(y_true_sample, model(x_sample, training=True)).numpy()
    print(f"  [6.1] Initial Loss:                 {init_loss:.6f}")
    print(f"  [6.2] Loss after 1 Gradient Step:   {post_loss:.6f}")
    assert not np.isnan(post_loss) and not np.isinf(post_loss)

    print("  --> TEST 6 RESULT: PASS")
    results["TEST 6 (End-to-End Gradient Step)"] = "PASS"
except Exception as e:
    print(f"  --> TEST 6 RESULT: FAIL ({e})")
    traceback.print_exc()
    results["TEST 6 (End-to-End Gradient Step)"] = f"FAIL: {e}"

print("\n" + "=" * 80)
print("TEST SUMMARY REPORT")
print("=" * 80)
all_pass = True
for tname, tstatus in results.items():
    print(f"  {tname:<40}: {tstatus}")
    if tstatus != "PASS":
        all_pass = False
print("=" * 80)
if all_pass:
    print("ALL 6 ISOLATED SMOKE TESTS PASSED SUCCESSFULLY.")
else:
    print("SOME TESTS FAILED. PLEASE REVIEW THE DETAILED LOGS ABOVE.")
print("=" * 80)
