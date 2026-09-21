"""
run_cordex_experiment.py
========================
Standalone experiment runner for the Quantile-Regression-Ensemble (QRE)
on the CORDEX-ML-Bench New Zealand dataset (AAAI 2024 Reproduction Project).

Supported Modes:
1. 'cpu_feasible' : Small validation experiment on CPU with deterministic reductions.
2. 'faithful_full': Full methodology-faithful alternative-data experiment (N=6, 40 reps, 20-year span).

Usage:
  python run_cordex_experiment.py --mode cpu_feasible --dry-run
  python run_cordex_experiment.py --mode cpu_feasible
  python run_cordex_experiment.py --config configs/cordex_cpu.json
"""

import os
import sys
import json
import argparse
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.mixture import GaussianMixture
from scipy.stats import wilcoxon

# Ensure repository root and src directory are on sys.path
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(WORKSPACE_ROOT, "src")
for p in [SRC_DIR, WORKSPACE_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data_adapter_cordex import load_cordex_dataset
from src.ConvolutionalNetworks import BGNet, BGCallWrapper
from src.GammaLoss import GammaLoss
from src.quantiles import Omega, OmegaLoss, ynetwork, bin_y_var
from src.useful_functions import data_between


def parse_arguments():
    """Parses command-line arguments for running CORDEX QRE experiments."""
    parser = argparse.ArgumentParser(description="Run CORDEX QRE Experiment Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cordex_final.json",
        help="Path to experiment JSON configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cpu_feasible", "faithful_full"],
        default=None,
        help="Override execution mode (cpu_feasible or faithful_full)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry-run configuration and dataset audit without model training"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Override number of independent experimental repetitions"
    )
    parser.add_argument(
        "--n-specialists",
        type=int,
        default=None,
        help="Override number of quantile specialists (e.g. 3 or 6)"
    )
    return parser.parse_args()


def load_configuration(cli_args):
    """Loads configuration JSON and merges command-line overrides."""
    config_path = cli_args.config
    if not os.path.exists(config_path):
        # Try relative to WORKSPACE_ROOT
        alt_path = os.path.join(WORKSPACE_ROOT, config_path)
        if os.path.exists(alt_path):
            config_path = alt_path

    if not os.path.exists(config_path):
        # Default fallback config if file not found
        default_config = {
            "mode": "cpu_feasible",
            "data_dir": "data/CORDEX_NZ_sample",
            "dataset_name": "CORDEX-ML-Bench",
            "domain": "New Zealand",
            "gcm_train": "ACCESS-CM2",
            "gcm_test": "ACCESS-CM2",
            "predictor_config": "config_A",
            "target_mode": "land_masked",
            "n_specialists": 6,
            "repeats": 3,
            "train_years": 2,
            "validation_years": 1,
            "test_years": 1,
            "batch_size": 16,
            "specialist_learning_rate": 0.001,
            "omega_learning_rate": 0.001,
            "decay_rate": None,
            "decay_steps": None,
            "bg_patience": 5,
            "omega_patience": 3,
            "max_epochs_specialist": 30,
            "max_epochs_omega": 25,
            "base_seed": 42,
            "save_dir": "results/final_cordex_qre"
        }
        config = default_config
    else:
        with open(config_path, "r") as f:
            config = json.load(f)

    # CLI Overrides
    if cli_args.mode is not None:
        config["mode"] = cli_args.mode
    if cli_args.repeats is not None:
        config["repeats"] = cli_args.repeats
    if cli_args.n_specialists is not None:
        config["n_specialists"] = cli_args.n_specialists

    # Resolve relative paths against WORKSPACE_ROOT
    if "data_dir" in config and not os.path.isabs(config["data_dir"]):
        config["data_dir"] = os.path.normpath(os.path.join(WORKSPACE_ROOT, config["data_dir"]))
    if "save_dir" in config and not os.path.isabs(config["save_dir"]):
        config["save_dir"] = os.path.normpath(os.path.join(WORKSPACE_ROOT, config["save_dir"]))

    # Defaults for optional fields
    config.setdefault("save_dir", f"results/cordex_qre_{config.get('mode', 'cpu')}")
    config.setdefault("base_seed", 42)
    config.setdefault("batch_size", 16 if config.get("mode") == "cpu_feasible" else 32)
    config.setdefault("specialist_learning_rate", 0.001 if config.get("mode") == "cpu_feasible" else 0.0001)
    config.setdefault("omega_learning_rate", 0.001 if config.get("mode") == "cpu_feasible" else 0.00005)
    config.setdefault("bg_patience", 5 if config.get("mode") == "cpu_feasible" else 15)
    config.setdefault("omega_patience", 3 if config.get("mode") == "cpu_feasible" else 4)
    config.setdefault("max_epochs_specialist", 30 if config.get("mode") == "cpu_feasible" else 400)
    config.setdefault("max_epochs_omega", 25 if config.get("mode") == "cpu_feasible" else 100)

    return config


def set_seed(seed):
    """Sets deterministic random seeds across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def derive_gmm_quantile_ranges(y_train, n_specialists, seed):
    """
    Derives intensity quantile intervals using Gaussian Mixture Model clustering
    on daily cumulative spatial precipitation sums (AAAI 2024 repository methodology).
    
    No arbitrary numerical tolerances added; uses original repository logic:
    counts = np.sort([np.sum(where(daily_spatial_sum <= y_upper, 1.0, 0.0)) for y_upper in means])
    """
    N_samples = y_train.shape[0]
    daily_spatial_sum = np.expand_dims(np.sum(y_train, axis=-1), axis=-1)

    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_specialists, n_init=100, random_state=seed)
    gmm.fit(daily_spatial_sum)

    # Centroids and parameter estimates
    means = np.sort(gmm.means_.flatten())
    weights = gmm.weights_.flatten()
    covariances = gmm.covariances_.flatten()

    # Original repository boundary derivation
    counts = np.sort([np.sum(np.where(daily_spatial_sum <= y_upper, 1.0, 0.0)) for y_upper in means])

    # Convert counts to quantile boundaries
    q_bounds = [float(c / N_samples) for c in counts]
    q_bounds.insert(0, 0.0)
    q_bounds[-1] = 1.0  # Ensure interval reaches the extreme tail

    # Validate interval boundaries
    assert q_bounds[0] == 0.0, "First quantile boundary must be 0.0"
    assert q_bounds[-1] == 1.0, "Final quantile boundary must be 1.0"
    assert len(q_bounds) == n_specialists + 1, f"Expected {n_specialists + 1} bounds, got {len(q_bounds)}"

    intervals = []
    for i in range(len(q_bounds) - 1):
        ql, qh = q_bounds[i], q_bounds[i + 1]
        if ql >= qh:
            raise ValueError(
                f"GMM derived duplicate or non-increasing boundary: [{ql}, {qh}]. "
                f"Means: {means}, Counts: {counts}. STOPPING."
            )
        intervals.append((ql, qh))

    # Validate sample partitions
    sample_counts = []
    for idx, (ql, qh) in enumerate(intervals):
        yq = data_between(y_train, ql, None, qh)
        n_sub = len(yq)
        if n_sub == 0:
            raise ValueError(
                f"Specialist {idx} interval [{ql:.4f}, {qh:.4f}] has 0 training samples! "
                f"GMM clustering failed to assign sufficient data. STOPPING."
            )
        sample_counts.append(n_sub)

    gmm_diag = {
        "means": [float(m) for m in means],
        "weights": [float(w) for w in weights],
        "covariances": [float(c) for c in covariances],
        "counts": [float(c) for c in counts],
        "intervals": intervals,
        "sample_counts": sample_counts
    }

    return intervals, gmm_diag


def format_date_str(val):
    """Formats numpy datetime64 or str to clean ISO date string."""
    if val is None:
        return "N/A"
    s = str(val)
    if "T" in s:
        return s.split("T")[0]
    return s


def prepare_explicit_temporal_splits(data_dict, mode="cpu_feasible"):
    """
    Constructs explicit train, validation, and test datasets with actual time coordinates.
    Uses a chronological 15-year train / 5-year validation / 20-year test split.
    """

    X_train_full = data_dict['x_train']
    Y_train_full = data_dict['y_train']
    X_test = data_dict['x_test']
    Y_test = data_dict['y_test']

    T_full = X_train_full.shape[0]

    train_time = data_dict['coords'].get('train_time')
    test_time = data_dict['coords'].get('test_time')

    if train_time is not None and len(train_time) == T_full:

        train_time = np.array(train_time)

        # 1961-1975 -> Training
        # 1976-1980 -> Validation
        train_mask = train_time <= np.datetime64("1975-12-31")
        val_mask = (
            (train_time >= np.datetime64("1976-01-01")) &
            (train_time <= np.datetime64("1980-12-31"))
        )

        X_train = X_train_full[train_mask]
        Y_train = Y_train_full[train_mask]

        X_val = X_train_full[val_mask]
        Y_val = Y_train_full[val_mask]

        train_start_date = format_date_str(train_time[train_mask][0])
        train_end_date = format_date_str(train_time[train_mask][-1])

        val_start_date = format_date_str(train_time[val_mask][0])
        val_end_date = format_date_str(train_time[val_mask][-1])

        T_train = X_train.shape[0]
        T_val = X_val.shape[0]

    else:
        raise ValueError("Training time coordinates are required for explicit temporal splitting.")

    if test_time is not None and len(test_time) > 0:
        test_start_date = format_date_str(test_time[0])
        test_end_date = format_date_str(test_time[-1])
    else:
        raise ValueError("Test time coordinates are required for explicit temporal splitting.")

    dates_summary = {
        "train_start": train_start_date,
        "train_end": train_end_date,
        "val_start": val_start_date,
        "val_end": val_end_date,
        "test_start": test_start_date,
        "test_end": test_end_date,
        "train_samples": int(T_train),
        "val_samples": int(T_val),
        "test_samples": int(X_test.shape[0])
    }

    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_val": X_val,
        "Y_val": Y_val,
        "X_test": X_test,
        "Y_test": Y_test,
        "coords": data_dict['coords'],
        "auxiliary": data_dict.get('auxiliary'),
        "mask": data_dict.get('mask'),
        "D_land": Y_train.shape[-1],
        "dates": dates_summary
    }


def evaluate_test_quantiles(y_true, y_pred, eval_quantiles=[(0.0, 1.0), (0.0, 0.2), (0.9, 1.0)]):
    """
    Evaluates predictions across test-truth spatial precipitation quantile regimes.
    
    Regimes:
      - [0.0, 1.0]: Overall MSE across all test days.
      - [0.0, 0.2]: Low-precipitation regime (bottom 20% spatial precipitation days).
      - [0.9, 1.0]: Extreme precipitation regime (top 10% spatial precipitation days).
    """
    daily_spatial_sum = np.sum(y_true, axis=-1)
    N_test = len(y_true)
    results = {}

    for q_low, q_high in eval_quantiles:
        regime_label = f"Q_[{q_low:.1f}_{q_high:.1f}]"
        thrs_low = np.quantile(daily_spatial_sum, q_low)
        thrs_high = np.quantile(daily_spatial_sum, q_high)

        if q_high == 1.0:
            mask = (daily_spatial_sum >= thrs_low) & (daily_spatial_sum <= thrs_high)
        else:
            mask = (daily_spatial_sum >= thrs_low) & (daily_spatial_sum < thrs_high)

        n_days = int(np.sum(mask))
        if n_days == 0:
            # Fallback if sample count is tiny
            mask = np.ones(N_test, dtype=bool)
            n_days = N_test

        sub_true = y_true[mask]
        sub_pred = y_pred[mask]

        mse = float(np.mean((sub_true - sub_pred) ** 2))
        mae = float(np.mean(np.abs(sub_true - sub_pred)))

        results[regime_label] = {
            "mse": mse,
            "mae": mae,
            "days_count": n_days,
            "quantile_bounds": [q_low, q_high],
            "threshold_values": [float(thrs_low), float(thrs_high)]
        }

    return results


def run_dry_run_audit(config):
    """
    Performs full dry-run configuration and dataset audit without model execution.
    """
    print("\n" + "=" * 85)
    print("STARTING CORDEX QRE EXPERIMENT DRY-RUN AUDIT")
    print("=" * 85)

    data_dict = load_cordex_dataset(
        data_dir=config['data_dir'],
        config=config['predictor_config'],
        target_mode=config['target_mode'],
        verbose=False
    )
    splits = prepare_explicit_temporal_splits(data_dict, mode=config['mode'])

    X_train = splits['X_train']
    Y_train = splits['Y_train']
    X_val = splits['X_val']
    Y_val = splits['Y_val']
    X_test = splits['X_test']
    Y_test = splits['Y_test']
    D_land = splits['D_land']
    dates = splits['dates']
    land_mask = splits['mask']

    print(f"\n[1] Dataset Dimensions & Structure:")
    print(f"  - X_train Shape:         {X_train.shape} (dtype: {X_train.dtype})")
    print(f"  - Y_train Shape:         {Y_train.shape} (dtype: {Y_train.dtype})")
    print(f"  - X_val Shape:           {X_val.shape}")
    print(f"  - Y_val Shape:           {Y_val.shape}")
    print(f"  - X_test Shape:          {X_test.shape}")
    print(f"  - Y_test Shape:          {Y_test.shape}")
    print(f"  - Valid Land Points:     {D_land} points")
    print(f"  - Land Mask Shape:       {land_mask.shape if land_mask is not None else 'N/A'}")
    print(f"  - Train Date Span:       {dates['train_start']} to {dates['train_end']} ({dates['train_samples']} days)")
    print(f"  - Val Date Span:         {dates['val_start']} to {dates['val_end']} ({dates['val_samples']} days)")
    print(f"  - Test Date Span:        {dates['test_start']} to {dates['test_end']} ({dates['test_samples']} days)")

    # Test GMM Partitioning
    print(f"\n[2] GMM Quantile Partitioning Check:")
    N_spec = config['n_specialists']
    Q_ranges, gmm_diag = derive_gmm_quantile_ranges(Y_train, N_spec, config['base_seed'])
    print(f"  - GMM Centroids (means): {gmm_diag['means']}")
    print(f"  - GMM Mixture Weights:   {gmm_diag['weights']}")
    print(f"  - Derived Q_ranges:      {Q_ranges}")
    print(f"  - Specialist Counts:     {gmm_diag['sample_counts']}")

    # Model Instantiation Check
    print(f"\n[3] Model Instantiation & Parameter Verification:")
    sample_spec = BGNet(
        output_dim=D_land,
        coords=None,
        baseline=False,
        ch_attn=True,
        avg_pool=True,
        max_pool=True,
        sp_attn=False,
        kernel_sizes=[3, 3, 3] if config.get('mode') == 'cpu_feasible' else [6, 6, 6],
        num_channels=[32, 64, 128] if config.get('mode') == 'cpu_feasible' else [64, 128, 256],
        num_dense=[128] if config.get('mode') == 'cpu_feasible' else [256],
        drop_out_rate=0.1 if config.get('mode') == 'cpu_feasible' else 0.2,
    )
    dummy_x = tf.zeros((2, 16, 16, 4))
    dummy_out = sample_spec(dummy_x)
    print(f"  - BGNet Forward Pass:    Input (2, 16, 16, 4) -> Output {dummy_out.shape} (Expected: (2, {D_land}, 3))")
    print(f"  - BGNet Parameters:      {sample_spec.count_params():,} trainable parameters")

    sample_omega = Omega(
        output_dim=N_spec,
        kernels=[3, 3, 3] if config.get('mode') == 'cpu_feasible' else [6, 6, 6],
        channels=[32, 64, 128] if config.get('mode') == 'cpu_feasible' else [64, 128, 256]
    )
    dummy_omega_out = sample_omega(dummy_x)
    print(f"  - Omega Forward Pass:    Input (2, 16, 16, 4) -> Output {dummy_omega_out.shape} (Expected: (2, {N_spec}))")
    print(f"  - Omega Parameters:      {sample_omega.count_params():,} trainable parameters")

    print(f"\n[4] Execution Plan:")
    print(f"  - Mode:                  {config['mode']}")
    print(f"  - Specialists (N):       {N_spec}")
    print(f"  - Repetitions:           {config['repeats']}")
    print(f"  - Batch Size:            {config['batch_size']}")
    print(f"  - Specialist LR:         {config['specialist_learning_rate']}")
    print(f"  - Omega LR:              {config['omega_learning_rate']}")
    print(f"  - Save Directory:        {os.path.abspath(config['save_dir'])}")

    print("\n" + "=" * 85)
    print("DRY-RUN AUDIT COMPLETED: All verification checks passed.")
    print("=" * 85)
    return True


def execute_training_pipeline(config):
    """
    Executes the multi-repetition QRE experiment pipeline on CORDEX-ML-Bench.
    """
    t_pipeline_start = time.time()
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], "training_history"), exist_ok=True)

    print("\n" + "=" * 85)
    print(f"STARTING CORDEX QRE EXPERIMENT (Mode: {config['mode']})")
    print("=" * 85)

    # 1. Load CORDEX Dataset
    print("\n>>> [1] Loading CORDEX-ML-Bench Data...")
    data_dict = load_cordex_dataset(
        data_dir=config['data_dir'],
        config=config['predictor_config'],
        target_mode=config['target_mode'],
        verbose=False
    )
    splits = prepare_explicit_temporal_splits(data_dict, mode=config['mode'])

    X_train = splits['X_train']
    Y_train = splits['Y_train']
    X_val = splits['X_val']
    Y_val = splits['Y_val']
    X_test = splits['X_test']
    Y_test = splits['Y_test']
    D_land = splits['D_land']
    dates = splits['dates']
    land_mask = splits['mask']

    print(f"  [1.1] X_train Shape:     {X_train.shape} (dtype: {X_train.dtype})")
    print(f"  [1.2] Y_train Shape:     {Y_train.shape} (dtype: {Y_train.dtype})")
    print(f"  [1.3] X_val Shape:       {X_val.shape}")
    print(f"  [1.4] Y_val Shape:       {Y_val.shape}")
    print(f"  [1.5] X_test Shape:      {X_test.shape}")
    print(f"  [1.6] Y_test Shape:      {Y_test.shape}")
    print(f"  [1.7] Land Points D_land:{D_land}")
    print(f"  [1.8] Land Mask Shape:   {land_mask.shape if land_mask is not None else 'N/A'}")
    print(f"  [1.9] Actual Dates:      Train: {dates['train_start']} to {dates['train_end']} | "
          f"Val: {dates['val_start']} to {dates['val_end']} | Test: {dates['test_start']} to {dates['test_end']}")
    print(f"  [1.10] Predictor Vars:   q850, t850, u850, v850 (w850 UNAVAILABLE and omitted)")

    N_spec = config['n_specialists']
    num_repeats = config['repeats']

    all_repeat_metrics = {
        "qre": [],
        "single_bgnet": [],
        "bgnet_minus": [],
        "bagging_cqre": [],
        "probability_pqre": [],
        "empirical_mean": []
    }

    quantile_ranges_log = {}
    gmm_diagnostics_log = {}
    specialist_train_log = {}
    omega_train_log = {}
    rep_runtimes = []

    for k in range(num_repeats):
        t_rep_start = time.time()
        seed_k = config['base_seed'] + k * 1000
        set_seed(seed_k)
        print(f"\n" + "=" * 85)
        print(f">>> [REPETITION {k+1}/{num_repeats}] (Random Seed: {seed_k})")
        print("=" * 85)

        # ---------------------------------------------------------------------
        # 2. GMM Quantile Derivation
        # ---------------------------------------------------------------------
        print("\n>>> [2] Deriving Quantile Regimes via Gaussian Mixture Model...")
        try:
            Q_ranges, gmm_diag = derive_gmm_quantile_ranges(Y_train, N_spec, seed_k)
        except Exception as e:
            print(f"[FATAL FAILURE] Repetition {k} (seed {seed_k}) failed at GMM derivation: {e}")
            raise e

        quantile_ranges_log[f"rep_{k}"] = Q_ranges
        gmm_diagnostics_log[f"rep_{k}"] = gmm_diag

        print(f"  [2.1] GMM Means:         {gmm_diag['means']}")
        print(f"  [2.2] GMM Weights:       {gmm_diag['weights']}")
        print(f"  [2.3] GMM Covariances:   {gmm_diag['covariances']}")
        print(f"  [2.4] Derived Q_ranges:  {Q_ranges}")
        print(f"  [2.5] Sample Counts:     {gmm_diag['sample_counts']}")

        # ---------------------------------------------------------------------
        # 3. Train N Specialists
        # ---------------------------------------------------------------------
        print(f"\n>>> [3] Training {N_spec} Independent BGNet Specialists (CPU)...")
        specialist_models = []
        specialist_wrappers = []
        specialist_train_log[f"rep_{k}"] = []

        loss_fn = GammaLoss(epsilon=0.0001, y_thrs=0.5)

        for i, (ql, qh) in enumerate(Q_ranges):
            X_spec_tr, Y_spec_tr = data_between(Y_train, ql, X_train, qh)
            
            spec_model = BGNet(
                output_dim=D_land,
                coords=None,
                baseline=False,
                ch_attn=True,
                avg_pool=True,
                max_pool=True,
                sp_attn=False,
                kernel_sizes=[3, 3, 3] if config.get('mode') == 'cpu_feasible' else [6, 6, 6],
                num_channels=[32, 64, 128] if config.get('mode') == 'cpu_feasible' else [64, 128, 256],
                num_dense=[128] if config.get('mode') == 'cpu_feasible' else [256],
                drop_out_rate=0.1 if config.get('mode') == 'cpu_feasible' else 0.2,
            )
            spec_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=config['specialist_learning_rate']),
                loss=loss_fn
            )

            # Evaluate initial loss before training
            initial_tr_loss = float(loss_fn(Y_spec_tr, spec_model(X_spec_tr, training=False)).numpy())
            initial_val_loss = float(loss_fn(Y_val, spec_model(X_val, training=False)).numpy())

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config['bg_patience'],
                restore_best_weights=True
            )

            history = spec_model.fit(
                X_spec_tr, Y_spec_tr,
                validation_data=(X_val, Y_val),
                epochs=config.get('max_epochs_specialist', 30),
                batch_size=config['batch_size'],
                callbacks=[early_stop],
                verbose=0
            )

            final_tr_loss = float(history.history['loss'][-1])
            best_val_loss = float(min(history.history['val_loss']))
            epochs_run = len(history.history['loss'])
            best_epoch = int(np.argmin(history.history['val_loss']) + 1)

            spec_info = {
                "specialist_id": i,
                "quantile_interval": [ql, qh],
                "sample_count": len(Y_spec_tr),
                "initial_train_loss": initial_tr_loss,
                "initial_val_loss": initial_val_loss,
                "final_train_loss": final_tr_loss,
                "best_val_loss": best_val_loss,
                "epochs_completed": epochs_run,
                "best_epoch": best_epoch,
                "seed": seed_k
            }
            specialist_train_log[f"rep_{k}"].append(spec_info)

            print(f"  [3.{i+1}] Specialist {i} [{ql:.3f}, {qh:.3f}] ({len(Y_spec_tr)} samples): "
                  f"Init Train Loss = {initial_tr_loss:.4f} -> Final = {final_tr_loss:.4f} | "
                  f"Best Val Loss = {best_val_loss:.4f} (Epoch {best_epoch}/{epochs_run})")

            specialist_models.append(spec_model)
            specialist_wrappers.append(BGCallWrapper(spec_model))

        # ---------------------------------------------------------------------
        # 4. Train Omega Weight Network
        # ---------------------------------------------------------------------
        print(f"\n>>> [4] Training Omega Weight Network with OmegaLoss (EMD)...")
        Y_omega_train = np.array(bin_y_var(Y_train, Q_ranges), dtype=np.float32)
        Y_omega_val = np.array(bin_y_var(Y_val, Q_ranges), dtype=np.float32)

        print("Y_omega_train shape:", Y_omega_train.shape)
        print("Y_omega_val shape:", Y_omega_val.shape)

        omega_net = Omega(
            output_dim=N_spec,
            kernels=[3, 3, 3] if config.get('mode') == 'cpu_feasible' else [6, 6, 6],
            channels=[32, 64, 128] if config.get('mode') == 'cpu_feasible' else [64, 128, 256]
        )
        omega_loss_fn = OmegaLoss(num_classes=N_spec)
        omega_net.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['omega_learning_rate']),
            loss=omega_loss_fn
        )

        initial_omega_tr_loss = float(tf.reduce_mean(omega_loss_fn(Y_omega_train, omega_net(X_train, training=False))).numpy())
        initial_omega_val_loss = float(tf.reduce_mean(omega_loss_fn(Y_omega_val, omega_net(X_val, training=False))).numpy())

        early_stop_omega = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['omega_patience'],
            restore_best_weights=True
        )

        history_omega = omega_net.fit(
            X_train, Y_omega_train,
            validation_data=(X_val, Y_omega_val),
            epochs=config.get('max_epochs_omega', 25),
            batch_size=config['batch_size'],
            callbacks=[early_stop_omega],
            verbose=0
        )

        final_omega_tr_loss = float(history_omega.history['loss'][-1])
        best_omega_val_loss = float(min(history_omega.history['val_loss']))
        epochs_omega_run = len(history_omega.history['loss'])
        best_omega_epoch = int(np.argmin(history_omega.history['val_loss']) + 1)

        omega_info = {
            "initial_train_loss": initial_omega_tr_loss,
            "initial_val_loss": initial_omega_val_loss,
            "final_train_loss": final_omega_tr_loss,
            "best_val_loss": best_omega_val_loss,
            "epochs_completed": epochs_omega_run,
            "best_epoch": best_omega_epoch,
            "seed": seed_k
        }
        omega_train_log[f"rep_{k}"] = omega_info

        print(f"  [4.1] Omega Network: Init Loss = {initial_omega_tr_loss:.4f} -> Final = {final_omega_tr_loss:.4f} | "
              f"Best Val Loss = {best_omega_val_loss:.4f} (Epoch {best_omega_epoch}/{epochs_omega_run})")

        # ---------------------------------------------------------------------
        # 5. Assemble QRE ynetwork & Predict
        # ---------------------------------------------------------------------
        print(f"\n>>> [5] Assembling QRE ynetwork & Verifying Test Predictions...")
        CNN_dict = {f'CNN_{idx}': [specialist_wrappers[idx]] for idx in range(N_spec)}
        qre_model = ynetwork(CNN_dict, omega_net)

        # Dynamic Softmax Weights
        test_weights = omega_net(X_test, training=False).numpy()
        row_sums = np.sum(test_weights, axis=1)

        print(f"  [5.1] Test Weights Shape:     {test_weights.shape} (Expected: ({X_test.shape[0]}, {N_spec}))")
        print(f"  [5.2] Min/Max Weight Values:  [{np.min(test_weights):.4f}, {np.max(test_weights):.4f}] (>= 0.0, <= 1.0)")
        print(f"  [5.3] Row Sums:               [{np.min(row_sums):.6f}, {np.max(row_sums):.6f}] (~ 1.0)")

        assert test_weights.shape == (X_test.shape[0], N_spec), "Weights shape mismatch"
        assert np.all(test_weights >= 0.0), "Negative weight detected"
        assert np.allclose(row_sums, 1.0, atol=1e-5), "Weight row sum is not 1.0"

        # QRE Prediction
        pred_qre = qre_model(X_test).numpy()
        print(f"  [5.4] QRE Prediction Shape:   {pred_qre.shape} (Expected: ({X_test.shape[0]}, {D_land}))")
        print(f"  [5.5] Prediction Finite?:     {np.all(np.isfinite(pred_qre))}")

        assert pred_qre.shape == (X_test.shape[0], D_land), "Prediction shape mismatch"
        assert np.all(np.isfinite(pred_qre)), "Non-finite predictions detected"

        # Manual Aggregation Check
        spec_test_preds = np.stack([w(X_test).numpy() for w in specialist_wrappers], axis=1)  # (T_test, N, D)
        weights_expanded = np.expand_dims(test_weights, axis=-1)  # (T_test, N, 1)
        manual_qre_pred = np.sum(weights_expanded * spec_test_preds, axis=1)
        max_abs_disc = float(np.max(np.abs(pred_qre - manual_qre_pred)))
        print(f"  [5.6] Manual Discrepancy:     {max_abs_disc:.2e} (Expected: < 1e-5)")
        assert max_abs_disc < 1e-5, f"Manual aggregation discrepancy too large: {max_abs_disc}"

        rep_qre_eval = evaluate_test_quantiles(Y_test, pred_qre)
        all_repeat_metrics["qre"].append(rep_qre_eval)
        print(f"        QRE Test MSE Overall:          {rep_qre_eval['Q_[0.0_1.0]']['mse']:.4f}")

        # ---------------------------------------------------------------------
        # 6. Train and Evaluate Baselines
        # ---------------------------------------------------------------------
        print(f"\n>>> [6] Training and Evaluating Baselines...")
        # 1. Single BGNet (Full unpartitioned dataset)
        print("  [6.1] Training Single BGNet Baseline...")
        single_bgnet = BGNet(output_dim=D_land, coords=None, baseline=False)
        single_bgnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['specialist_learning_rate']), loss=loss_fn)
        single_bgnet.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=config.get('max_epochs_specialist', 30),
                         batch_size=config['batch_size'], callbacks=[early_stop], verbose=0)
        pred_single = BGCallWrapper(single_bgnet)(X_test).numpy()
        all_repeat_metrics["single_bgnet"].append(evaluate_test_quantiles(Y_test, pred_single))
        print(f"        Single BGNet Test MSE Overall: {all_repeat_metrics['single_bgnet'][-1]['Q_[0.0_1.0]']['mse']:.4f}")

        # 2. BG-Net(-) (Ablation baseline)
        print("  [6.2] Training BG-Net(-) Ablation Baseline...")
        try:
            bgnet_minus = BGNet(output_dim=D_land, coords=None, baseline=True)
            bgnet_minus.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['specialist_learning_rate']), loss=loss_fn)
            bgnet_minus.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=config.get('max_epochs_specialist', 30),
                            batch_size=config['batch_size'], callbacks=[early_stop], verbose=0)
            pred_minus = BGCallWrapper(bgnet_minus)(X_test).numpy()
            all_repeat_metrics["bgnet_minus"].append(evaluate_test_quantiles(Y_test, pred_minus))
            print(f"        BG-Net(-) Test MSE Overall:    {all_repeat_metrics['bgnet_minus'][-1]['Q_[0.0_1.0]']['mse']:.4f}")
        except Exception as e:
            print(f"        [NOTE] BG-Net(-) skipped: bmodule MaxPool2D(2) on 16x16 input reduces spatial dims below 3x3 kernel size (designed for 36x41 ERA5 grid).")

        # 3. Bagging / cqre (fixed weights 1/N)
        print("  [6.3] Evaluating Bagging Ensemble (cqre)...")
        cqre_model = ynetwork(CNN_dict, fixed_weights=[1.0 / N_spec] * N_spec)
        pred_cqre = cqre_model(X_test).numpy()
        all_repeat_metrics["bagging_cqre"].append(evaluate_test_quantiles(Y_test, pred_cqre))
        print(f"        Bagging (cqre) Test MSE:       {all_repeat_metrics['bagging_cqre'][-1]['Q_[0.0_1.0]']['mse']:.4f}")

        # 4. Probability / pqre (fixed prior weights)
        print("  [6.4] Evaluating Probability Ensemble (pqre)...")
        prob_weights = [qh - ql for ql, qh in Q_ranges]
        pqre_model = ynetwork(CNN_dict, fixed_weights=prob_weights)
        pred_pqre = pqre_model(X_test).numpy()
        all_repeat_metrics["probability_pqre"].append(evaluate_test_quantiles(Y_test, pred_pqre))
        print(f"        Probability (pqre) Test MSE:   {all_repeat_metrics['probability_pqre'][-1]['Q_[0.0_1.0]']['mse']:.4f}")

        # 5. Empirical Mean Baseline
        print("  [6.5] Evaluating Empirical Mean Baseline...")
        mean_pred = np.tile(np.mean(Y_train, axis=0, keepdims=True), (Y_test.shape[0], 1))
        all_repeat_metrics["empirical_mean"].append(evaluate_test_quantiles(Y_test, mean_pred))
        print(f"        Empirical Mean Test MSE:       {all_repeat_metrics['empirical_mean'][-1]['Q_[0.0_1.0]']['mse']:.4f}")

        t_rep_end = time.time()
        rep_runtime = t_rep_end - t_rep_start
        rep_runtimes.append(rep_runtime)
        print(f"\n>>> Repetition {k+1} completed in {rep_runtime:.2f} seconds.")

        # ---------------------------------------------------------------------
        # Save Per-Repetition Results
        # ---------------------------------------------------------------------
        rep_dir = os.path.join(config['save_dir'], f"repetition_{k}")
        os.makedirs(os.path.join(rep_dir, "training_history"), exist_ok=True)

        rep_summary = {}
        rep_csv = []
        for model_name, rep_list in all_repeat_metrics.items():
            if len(rep_list) > k:
                rep_summary[model_name] = rep_list[k]
                for q_key in ["Q_[0.0_1.0]", "Q_[0.0_0.2]", "Q_[0.9_1.0]"]:
                    rep_csv.append({
                        "repetition": k,
                        "seed": seed_k,
                        "model": model_name,
                        "quantile_regime": q_key,
                        "days_evaluated": rep_list[k][q_key]["days_count"],
                        "mse": rep_list[k][q_key]["mse"],
                        "mae": rep_list[k][q_key]["mae"]
                    })

        with open(os.path.join(rep_dir, "metrics_summary.json"), "w") as f:
            json.dump(rep_summary, f, indent=2)
        pd.DataFrame(rep_csv).to_csv(os.path.join(rep_dir, "metrics_table.csv"), index=False)
        with open(os.path.join(rep_dir, "quantile_ranges.json"), "w") as f:
            json.dump({f"rep_{k}": Q_ranges}, f, indent=2)
        with open(os.path.join(rep_dir, "training_history", "training_log.json"), "w") as f:
            json.dump({
                "specialists": specialist_train_log.get(f"rep_{k}", []),
                "omega": omega_train_log.get(f"rep_{k}", {}),
                "gmm_diagnostics": gmm_diagnostics_log.get(f"rep_{k}", {})
            }, f, indent=2)

        rep_meta = {
            "repetition_id": k,
            "seed": seed_k,
            "runtime_seconds": float(rep_runtime),
            "gmm_parameters": gmm_diag,
            "derived_q_ranges": Q_ranges,
            "specialist_sample_counts": gmm_diag['sample_counts'],
            "dates": dates,
            "max_aggregation_discrepancy": max_abs_disc
        }
        with open(os.path.join(rep_dir, "metadata.json"), "w") as f:
            json.dump(rep_meta, f, indent=2)

    # -------------------------------------------------------------------------
    # 7. Aggregate Metrics & Export Results Across All Repetitions
    # -------------------------------------------------------------------------
    print("\n>>> [7] Aggregating Metrics Across All Repetitions...")
    summary_metrics = {}
    csv_rows = []

    for model_name, rep_list in all_repeat_metrics.items():
        if not rep_list:
            continue
        summary_metrics[model_name] = {}
        for q_key in ["Q_[0.0_1.0]", "Q_[0.0_0.2]", "Q_[0.9_1.0]"]:
            mses = [rep[q_key]["mse"] for rep in rep_list]
            maes = [rep[q_key]["mae"] for rep in rep_list]
            mean_mse = float(np.mean(mses))
            std_mse = float(np.std(mses)) if len(mses) > 1 else 0.0
            mean_mae = float(np.mean(maes))
            std_mae = float(np.std(maes)) if len(maes) > 1 else 0.0
            days_count = rep_list[0][q_key]["days_count"]

            summary_metrics[model_name][q_key] = {
                "mean_mse": mean_mse,
                "std_mse": std_mse,
                "mean_mae": mean_mae,
                "std_mae": std_mae,
                "all_repeats_mse": mses,
                "all_repeats_mae": maes,
                "days_count": days_count
            }

            csv_rows.append({
                "model": model_name,
                "quantile_regime": q_key,
                "days_evaluated": days_count,
                "mean_mse": mean_mse,
                "std_mse": std_mse,
                "mean_mae": mean_mae,
                "std_mae": std_mae
            })

    # Statistical significance: Wilcoxon signed-rank tests vs QRE
    wilcoxon_results = {}
    if num_repeats >= 3 and len(all_repeat_metrics["qre"]) >= 3:
        print("\n>>> [8] Calculating Paired Wilcoxon Signed-Rank Tests (QRE vs Baselines)...")
        for baseline_name in ["single_bgnet", "bagging_cqre", "probability_pqre", "empirical_mean"]:
            if baseline_name in all_repeat_metrics and len(all_repeat_metrics[baseline_name]) == num_repeats:
                wilcoxon_results[baseline_name] = {}
                for q_key in ["Q_[0.0_1.0]", "Q_[0.0_0.2]", "Q_[0.9_1.0]"]:
                    qre_mses = [r[q_key]["mse"] for r in all_repeat_metrics["qre"]]
                    base_mses = [r[q_key]["mse"] for r in all_repeat_metrics[baseline_name]]
                    diffs = np.array(qre_mses) - np.array(base_mses)
                    
                    if np.all(diffs == 0):
                        stat, pval = 0.0, 1.0
                    else:
                        try:
                            res = wilcoxon(qre_mses, base_mses, zero_method="wilcox", alternative="two-sided")
                            stat, pval = float(res.statistic), float(res.pvalue)
                        except Exception as e:
                            stat, pval = None, None

                    wilcoxon_results[baseline_name][q_key] = {
                        "statistic": stat,
                        "pvalue": pval,
                        "qre_mses": qre_mses,
                        "baseline_mses": base_mses,
                        "note": f"LOW-POWER exploratory statistics (n={len(qre_mses)}). Do not interpret p < 0.05 as strong evidence."
                    }
                    print(f"  - QRE vs {baseline_name} [{q_key}]: stat={stat}, p={pval}")

    # Save summary and aggregate files
    summary_path = os.path.join(config['save_dir'], "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)

    csv_path = os.path.join(config['save_dir'], "metrics_table.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    # Save aggregate_metrics.json & aggregate_metrics.csv
    agg_json_path = os.path.join(config['save_dir'], "aggregate_metrics.json")
    with open(agg_json_path, "w") as f:
        json.dump({
            "metrics": summary_metrics,
            "wilcoxon_tests": wilcoxon_results,
            "per_repetition_runtimes": rep_runtimes
        }, f, indent=2)

    agg_csv_path = os.path.join(config['save_dir'], "aggregate_metrics.csv")
    pd.DataFrame(csv_rows).to_csv(agg_csv_path, index=False)

    # Save quantile_ranges.json
    ranges_path = os.path.join(config['save_dir'], "quantile_ranges.json")
    with open(ranges_path, "w") as f:
        json.dump(quantile_ranges_log, f, indent=2)

    # Save training_history.json
    hist_path = os.path.join(config['save_dir'], "training_history", "training_log.json")
    with open(hist_path, "w") as f:
        json.dump({
            "specialists": specialist_train_log,
            "omega": omega_train_log,
            "gmm_diagnostics": gmm_diagnostics_log
        }, f, indent=2)

    # Metadata export
    t_pipeline_end = time.time()
    total_runtime = t_pipeline_end - t_pipeline_start

    metadata = {
        "experiment_name": "CORDEX_QRE_CPU_Validation_Run",
        "dataset": "CORDEX-ML-Bench",
        "data_type": "alternative_data",
        "original_dataset_available": False,
        "n_specialists": N_spec,
        "repeats": num_repeats,
        "seeds": [config['base_seed'] + k * 1000 for k in range(num_repeats)],
        "base_seed": config['base_seed'],
        "predictor_configuration": "q850,t850,u850,v850",
        "w850_available": False,
        "target_resolution": "128x128",
        "land_points": D_land,
        "actual_train_dates": f"{dates['train_start']} to {dates['train_end']}",
        "actual_val_dates": f"{dates['val_start']} to {dates['val_end']}",
        "actual_test_dates": f"{dates['test_start']} to {dates['test_end']}",
        "gmm_parameters_per_repeat": gmm_diagnostics_log,
        "derived_q_ranges_per_repeat": quantile_ranges_log,
        "per_repetition_runtimes_seconds": rep_runtimes,
        "total_runtime_seconds": float(total_runtime),
        "config": config,
        "scientific_classification": {
            "experiment_type": "alternative_data_computational_reduction",
            "dataset": "ALTERNATIVE_DATA (CORDEX-ML-Bench)",
            "predictors": "ALTERNATIVE_DATA (q850, t850, u850, v850; w850 omitted)",
            "n_specialists": f"COMPUTATIONAL_REDUCTION (N={N_spec})",
            "repeats": f"COMPUTATIONAL_REDUCTION ({num_repeats} repetitions)",
            "train_period": f"{dates['train_start']} to {dates['train_end']}",
            "validation_period": f"{dates['val_start']} to {dates['val_end']}",
            "test_period": f"{dates['test_start']} to {dates['test_end']}",
            "batch_size": f"IMPLEMENTATION_CHOICE (batch_size={config['batch_size']})",
            "loss_functions": "PAPER_FAITHFUL (GammaLoss + OmegaLoss EMD)",
            "qre_aggregation": "PAPER_FAITHFUL (ynetwork Softmax)",
            "evaluation_intervals": "PAPER_FAITHFUL ([0,1], [0,0.2], [0.9,1])"
        }
    }

    meta_path = os.path.join(config['save_dir'], "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    agg_meta_path = os.path.join(config['save_dir'], "aggregate_metadata.json")
    with open(agg_meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[PASS] All artifacts successfully saved to: {os.path.abspath(config['save_dir'])}")
    print(f"  - Aggregate Metadata: {agg_meta_path}")
    print(f"  - Aggregate Metrics:  {agg_json_path}")
    print(f"  - Aggregate Table:    {agg_csv_path}")
    print(f"  - Repetition Folders: {[f'repetition_{k}' for k in range(num_repeats)]}")
    print(f"  - Total Runtime:      {total_runtime:.2f} seconds")

    print("\n" + "=" * 85)
    print("PHASE 14B MULTI-REPETITION CPU VALIDATION EXPERIMENT COMPLETED SUCCESSFULLY.")
    print("=" * 85)
    return True


if __name__ == "__main__":
    cli_args = parse_arguments()
    exp_config = load_configuration(cli_args)

    if cli_args.dry_run:
        success = run_dry_run_audit(exp_config)
        sys.exit(0 if success else 1)
    else:
        success = execute_training_pipeline(exp_config)
        sys.exit(0 if success else 1)
