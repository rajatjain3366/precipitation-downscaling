# QRE Experiment Configuration Specification

This document provides the formal, machine-readable parameter configurations for the Quantile-Regression-Ensemble (QRE) experiments on the CORDEX-ML-Bench dataset.

---

## Configuration 1: Methodology-Faithful Alternative-Data Experiment
*Designed for full scientific fidelity on CORDEX-ML-Bench ($N=6$, 40 repetitions, multi-year temporal split, GPU cluster target).*

```json
{
  "experiment_name": "qre_cordex_nz_faithful_n6_r40",
  "dataset": {
    "source": "CORDEX-ML-Bench",
    "domain": "New Zealand",
    "gcm_train": "ACCESS-CM2",
    "gcm_test": "ACCESS-CM2",
    "predictor_config": "config_A",
    "predictors": ["q_850", "t_850", "u_850", "v_850"],
    "omitted_predictors": ["w_850"],
    "target_variable": "pr",
    "target_mode": "land_masked",
    "target_dim": 2418,
    "input_shape": [16, 16, 4],
    "auxiliary_shape": [1, 128, 128, 1],
    "train_period": {
      "start": "1961-01-01",
      "end": "1976-12-31",
      "days": 5844
    },
    "val_period": {
      "start": "1977-01-01",
      "end": "1980-12-31",
      "days": 1461
    },
    "test_period": {
      "start": "1981-01-01",
      "end": "2000-12-31",
      "days": 7305
    }
  },
  "ensemble": {
    "num_specialists": 6,
    "partition_method": "quantile_spatial_sum",
    "quantiles": [
      [0.0, 0.20],
      [0.20, 0.40],
      [0.40, 0.60],
      [0.60, 0.80],
      [0.80, 0.95],
      [0.95, 1.00]
    ],
    "num_repeats": 40,
    "random_seed_base": 42
  },
  "training": {
    "specialist": {
      "architecture": "BGNet",
      "baseline": false,
      "channel_attention": true,
      "spatial_attention": false,
      "kernel_sizes": [3, 3, 3],
      "num_channels": [64, 128, 256],
      "num_dense": [256],
      "dropout_rate": 0.2,
      "loss": "GammaLoss",
      "loss_params": {
        "epsilon": 0.0001,
        "y_thrs": 0.5
      },
      "optimizer": "Adam",
      "learning_rate_schedule": {
        "initial_lr": 0.0001,
        "decay_steps": 6000,
        "decay_rate": 0.7
      },
      "early_stopping_patience": 15,
      "batch_size": 32,
      "max_epochs": 100
    },
    "weight_network": {
      "architecture": "Omega",
      "kernels": [6, 6, 6],
      "channels": [64, 128, 256],
      "dense_units": [100, 100],
      "dropout_rate": 0.2,
      "activation": "softmax",
      "loss": "OmegaLoss",
      "loss_params": {
        "p": 1,
        "loss_weights": [0.0, 0.0, 1.0]
      },
      "optimizer": "Adam",
      "learning_rate_schedule": {
        "initial_lr": 0.00005,
        "decay_steps": 6000,
        "decay_rate": 0.8
      },
      "early_stopping_patience": 4,
      "batch_size": 32,
      "max_epochs": 50
    }
  },
  "baselines": [
    "single_bgnet",
    "constant_weight_ensemble_cqre",
    "probability_weight_ensemble_pqre",
    "empirical_mean"
  ],
  "evaluation": {
    "quantiles_evaluated": [
      [0.0, 1.0],
      [0.0, 0.2],
      [0.9, 1.0]
    ],
    "metrics": ["MSE", "Regional_MSE", "Wilcoxon_p_value"],
    "significance_alpha": 0.05
  },
  "hardware_target": "GPU_Cluster"
}
```

---

## Configuration 2: CPU-Feasible Validation Experiment
*Designed for complete pipeline verification, dynamic weighting evaluation, and extreme tail assessment on local CPU hardware.*

```json
{
  "experiment_name": "qre_cordex_nz_cpu_feasibility_n3_r3",
  "dataset": {
    "source": "CORDEX-ML-Bench",
    "domain": "New Zealand",
    "gcm_train": "ACCESS-CM2",
    "gcm_test": "ACCESS-CM2",
    "predictor_config": "config_A",
    "predictors": ["q_850", "t_850", "u_850", "v_850"],
    "omitted_predictors": ["w_850"],
    "target_variable": "pr",
    "target_mode": "land_masked",
    "target_dim": 2418,
    "input_shape": [16, 16, 4],
    "auxiliary_shape": [1, 128, 128, 1],
    "train_period": {
      "start": "1961-01-01",
      "end": "1962-12-31",
      "days": 730
    },
    "val_period": {
      "start": "1963-01-01",
      "end": "1963-12-31",
      "days": 365
    },
    "test_period": {
      "start": "1981-01-01",
      "end": "1981-12-31",
      "days": 365
    }
  },
  "ensemble": {
    "num_specialists": 3,
    "partition_method": "quantile_spatial_sum",
    "quantiles": [
      [0.0, 0.35],
      [0.35, 0.70],
      [0.70, 1.00]
    ],
    "num_repeats": 3,
    "random_seed_base": 42
  },
  "training": {
    "specialist": {
      "architecture": "BGNet",
      "baseline": false,
      "channel_attention": true,
      "spatial_attention": false,
      "kernel_sizes": [3, 3, 3],
      "num_channels": [32, 64, 128],
      "num_dense": [128],
      "dropout_rate": 0.1,
      "loss": "GammaLoss",
      "loss_params": {
        "epsilon": 0.0001,
        "y_thrs": 0.5
      },
      "optimizer": "Adam",
      "learning_rate": 0.001,
      "early_stopping_patience": 5,
      "batch_size": 16,
      "max_epochs": 20
    },
    "weight_network": {
      "architecture": "Omega",
      "kernels": [3, 3, 3],
      "channels": [32, 64, 128],
      "dense_units": [64, 64],
      "dropout_rate": 0.1,
      "activation": "softmax",
      "loss": "OmegaLoss",
      "loss_params": {
        "p": 1,
        "loss_weights": [0.0, 0.0, 1.0]
      },
      "optimizer": "Adam",
      "learning_rate": 0.001,
      "early_stopping_patience": 3,
      "batch_size": 16,
      "max_epochs": 15
    }
  },
  "baselines": [
    "single_bgnet",
    "constant_weight_ensemble_cqre",
    "probability_weight_ensemble_pqre"
  ],
  "evaluation": {
    "quantiles_evaluated": [
      [0.0, 1.0],
      [0.0, 0.2],
      [0.9, 1.0]
    ],
    "metrics": ["MSE", "Mean_Std_across_repeats"]
  },
  "hardware_target": "Local_CPU"
}
```
