import os
import numpy as np
import xarray as xr
import tensorflow as tf

"""
CORDEX-ML-Bench Real-Data Adapter for QRE Reproduction (AAAI 2024)

This module provides a standardized, decoupled interface for ingesting CORDEX-ML-Bench
NetCDF datasets into the Quantile-Regression-Ensemble (QRE) architecture.

Methodological Notes:
- Original Paper Predictors: (q850, t850, w850, u850, v850) from ERA5 reanalysis.
- CORDEX-ML-Bench Predictors: Vertical velocity w850 is UNAVAILABLE in the benchmark.
- Configuration A explicitly omits w850 and uses the 4-channel core: (q850, t850, u850, v850).
- Configuration B uses all 16 available multi-level channels (u, v, q, t, z at 850, 700, 500 hPa + static).
- Target Precipitation: Variable 'pr' on the 128x128 (~10 km) CCAM grid.
- Static Topography: Extracted from 'static.nc' (variable 'orog').
"""

def extract_land_mask(static_ds, threshold=0.0):
    """
    Derives a time-invariant boolean land mask from static orography dataset.
    
    Parameters:
        static_ds (xr.Dataset): Static dataset containing 'orog' (surface altitude) or 'sftlf'.
        threshold (float): Altitude threshold for land points (default > 0.0m).
        
    Returns:
        mask (np.ndarray): 2D boolean array of shape (H, W), True for land cells.
        D_land (int): Count of valid land cells.
    """
    if 'sftlf' in static_ds:
        # Land area fraction in [0, 100] or [0, 1]
        land_frac = static_ds['sftlf'].values.squeeze()
        mask = land_frac > 0.5
    elif 'orog' in static_ds:
        # Orography surface elevation (meters)
        orog_vals = static_ds['orog'].values.squeeze()
        mask = (orog_vals > threshold) & (~np.isnan(orog_vals))
    elif 'elevation' in static_ds:
        elev_vals = static_ds['elevation'].values.squeeze()
        mask = (elev_vals > threshold) & (~np.isnan(elev_vals))
    else:
        var_names = list(static_ds.data_vars)
        if len(var_names) > 0:
            first_var = static_ds[var_names[0]].values.squeeze()
            mask = ~np.isnan(first_var)
        else:
            raise ValueError(f"No recognizable topography variable found in static dataset. Available: {var_names}")

    D_land = int(np.sum(mask))
    return mask, D_land


def prep_aux_cordex(static_ds, mask=None):
    """
    Standardizes static topography over land points and formats shape as (1, H, W, 1).
    
    Parameters:
        static_ds (xr.Dataset): Static dataset containing topography.
        mask (np.ndarray): 2D boolean land mask.
        
    Returns:
        aux_tensor (np.ndarray): Processed auxiliary array of shape (1, H, W, 1).
    """
    if 'orog' in static_ds:
        raw_topo = static_ds['orog'].values.squeeze().astype(np.float32)
    elif 'elevation' in static_ds:
        raw_topo = static_ds['elevation'].values.squeeze().astype(np.float32)
    else:
        var_name = list(static_ds.data_vars)[0]
        raw_topo = static_ds[var_name].values.squeeze().astype(np.float32)

    H, W = raw_topo.shape
    if mask is not None:
        land_vals = raw_topo[mask]
        mean_val = float(np.mean(land_vals))
        std_val = float(np.std(land_vals)) if np.std(land_vals) > 1e-6 else 1.0
        
        # Standardize land cells, impute ocean with mean
        processed = np.where(mask, (raw_topo - mean_val) / std_val, 0.0)
        # Shift positive land points for embedding module
        processed = np.where(mask, processed + np.abs(np.min(processed)) + 1.0, 0.0)
    else:
        mean_val = float(np.nanmean(raw_topo))
        std_val = float(np.nanstd(raw_topo)) if np.nanstd(raw_topo) > 1e-6 else 1.0
        processed = np.nan_to_num((raw_topo - mean_val) / std_val, nan=0.0)

    aux_tensor = np.reshape(processed, (1, H, W, 1)).astype(np.float32)
    return aux_tensor


def unflatten_target_cordex(y_flat, mask, H=128, W=128, fill_value=0.0):
    """
    Reconstructs full 2D spatial grid (batch, H, W) from 1D land-masked vector (batch, D_land).
    
    Parameters:
        y_flat (np.ndarray): 2D array of shape (batch, D_land).
        mask (np.ndarray): 2D boolean array of shape (H, W).
        fill_value (float): Value for non-land cells.
        
    Returns:
        y_2d (np.ndarray): 3D array of shape (batch, H, W).
    """
    batch_size = y_flat.shape[0]
    y_2d = np.full((batch_size, H, W), fill_value, dtype=y_flat.dtype)
    y_2d[:, mask] = y_flat
    return y_2d


def load_cordex_dataset(
    data_dir,
    config="config_A",
    target_mode="land_masked",
    train_gcm="ACCESS-CM2",
    test_gcm="ACCESS-CM2",
    sample_days=None,
    verbose=True
):
    """
    Loads and structures the CORDEX-ML-Bench dataset for QRE training and evaluation.
    
    Parameters:
        data_dir (str): Root directory containing the extracted CORDEX NZ domain files.
        config (str): 
            - "config_A": 4-channel 850 hPa core physical subset (q850, t850, u850, v850).
            - "config_B": Full 16-channel multi-level atmospheric suite.
        target_mode (str): 
            - "land_masked": Flattens target to (batch, D_land) using time-invariant land mask.
            - "full_grid": Returns full flattened grid (batch, H*W).
        train_gcm (str): GCM identifier for training (default "ACCESS-CM2").
        test_gcm (str): GCM identifier for evaluation (default "ACCESS-CM2").
        sample_days (int): If specified, slices first N days for quick smoke testing.
        verbose (bool): If True, prints detailed tensor information.
        
    Returns:
        data_dict (dict):
            - 'x_train': np.ndarray (T_train, H_in, W_in, C)
            - 'x_test':  np.ndarray (T_test, H_in, W_in, C)
            - 'y_train': np.ndarray (T_train, D_out)
            - 'y_test':  np.ndarray (T_test, D_out)
            - 'auxiliary': np.ndarray (1, H_target, W_target, 1)
            - 'coords': dict with spatial/temporal coordinate metadata
            - 'mask': 2D boolean land mask (H_target, W_target)
            - 'metadata': dict of dataset characteristics
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"CORDEX data directory not found at: {os.path.abspath(data_dir)}")

    # 1. Resolve File Paths
    train_pred_path = os.path.join(data_dir, "train", "ESD_pseudo-reality", "predictors", f"{train_gcm}_1961-1980.nc")
    train_targ_path = os.path.join(data_dir, "train", "ESD_pseudo-reality", "target", f"pr_tasmax_{train_gcm}_1961-1980.nc")
    static_path = os.path.join(data_dir, "train", "ESD_pseudo-reality", "static.nc")
    test_pred_path = os.path.join(data_dir, "test", "historical", "predictors", "perfect", f"{test_gcm}_1981-2000.nc")
    test_targ_path = os.path.join(data_dir, "test", "historical", "target", f"pr_tasmax_{test_gcm}_1981-2000.nc")

    # Fallback paths if flat directory structure is used
    if not os.path.exists(train_pred_path):
        train_pred_path = os.path.join(data_dir, f"{train_gcm}_1961-1980_predictors.nc")
    if not os.path.exists(train_targ_path):
        train_targ_path = os.path.join(data_dir, f"pr_tasmax_{train_gcm}_1961-1980_target.nc")
    if not os.path.exists(static_path):
        static_path = os.path.join(data_dir, "static.nc")
    if not os.path.exists(test_pred_path):
        test_pred_path = os.path.join(data_dir, f"{test_gcm}_1981-2000_predictors.nc")
    if not os.path.exists(test_targ_path):
        test_targ_path = os.path.join(data_dir, f"pr_tasmax_{test_gcm}_1981-2000_target.nc")

    # Validate essential files exist
    for p_name, p_val in [("Train Predictor", train_pred_path), ("Train Target", train_targ_path), ("Static", static_path)]:
        if not os.path.exists(p_val):
            raise FileNotFoundError(f"Required {p_name} file missing: {os.path.abspath(p_val)}")

    # 2. Open Datasets with xarray
    ds_x_train = xr.open_dataset(train_pred_path)
    ds_y_train = xr.open_dataset(train_targ_path)
    ds_static = xr.open_dataset(static_path)

    has_test = os.path.exists(test_pred_path) and os.path.exists(test_targ_path)
    if has_test:
        ds_x_test = xr.open_dataset(test_pred_path)
        ds_y_test = xr.open_dataset(test_targ_path)
    else:
        ds_x_test = None
        ds_y_test = None

    # Slice temporal window if requested
    if sample_days is not None and sample_days > 0:
        ds_x_train = ds_x_train.isel(time=slice(0, sample_days))
        ds_y_train = ds_y_train.isel(time=slice(0, sample_days))
        if has_test:
            ds_x_test = ds_x_test.isel(time=slice(0, min(sample_days, ds_x_test.sizes['time'])))
            ds_y_test = ds_y_test.isel(time=slice(0, min(sample_days, ds_y_test.sizes['time'])))

    # 3. Process Static Orography & Land Mask
    land_mask, D_land = extract_land_mask(ds_static)
    aux_tensor = prep_aux_cordex(ds_static, mask=land_mask)

    # 4. Extract and Structure Predictors (X)
    # Available variables in CORDEX: u, v, q, t, z across levels [850, 700, 500]
    def extract_predictors(ds, cfg):
        # Determine variable layout
        if 'channel' in ds.dims or 'channel' in ds.coords:
            # Channel coordinate layout
            all_channels = list(ds.coords['channel'].values) if 'channel' in ds.coords else list(ds['channel'].values)
            if cfg == "config_A":
                # Select only the 4 core 850 hPa variables
                target_vars = ['q_850', 't_850', 'u_850', 'v_850']
                avail = [v for v in target_vars if v in all_channels]
                if len(avail) == 4:
                    sub = ds.sel(channel=avail)
                    x_arr = sub.to_array().values if isinstance(sub, xr.Dataset) else sub.values
                else:
                    # Try alternate channel naming (e.g. q850, t850, u850, v850)
                    alt_vars = ['q850', 't850', 'u850', 'v850']
                    avail_alt = [v for v in alt_vars if v in all_channels]
                    if len(avail_alt) == 4:
                        sub = ds.sel(channel=avail_alt)
                        x_arr = sub.to_array().values if isinstance(sub, xr.Dataset) else sub.values
                    else:
                        # Fallback to first 4 channels
                        x_arr = ds.values[..., :4]
            else:
                x_arr = ds.values
        else:
            # Multi-variable Dataset layout (e.g. ds['q'], ds['t'], ds['u'], ds['v'])
            if cfg == "config_A":
                # Check for 850 hPa slice or discrete variable names
                selected_arrays = []
                for vname in ['q', 't', 'u', 'v']:
                    if vname in ds:
                        var_da = ds[vname]
                        if 'level' in var_da.dims or 'plev' in var_da.dims:
                            lev_dim = 'level' if 'level' in var_da.dims else 'plev'
                            # Select 850 hPa level (850 or 85000 Pa)
                            lev_vals = var_da[lev_dim].values
                            if 850 in lev_vals:
                                var_slice = var_da.sel({lev_dim: 850})
                            elif 85000 in lev_vals:
                                var_slice = var_da.sel({lev_dim: 85000})
                            else:
                                var_slice = var_da.isel({lev_dim: 0})
                            selected_arrays.append(var_slice.values)
                        else:
                            selected_arrays.append(var_da.values)
                    elif f"{vname}_850" in ds:
                        selected_arrays.append(ds[f"{vname}_850"].values)
                    elif f"{vname}850" in ds:
                        selected_arrays.append(ds[f"{vname}850"].values)
                
                if len(selected_arrays) == 4:
                    # Stack along trailing channel dimension: (time, H, W, 4)
                    x_arr = np.stack(selected_arrays, axis=-1)
                else:
                    # Fallback to stacking all variables and taking first 4
                    all_arrs = [ds[v].values for v in ds.data_vars]
                    x_arr = np.stack(all_arrs, axis=-1)[..., :4]
            else:
                all_arrs = [ds[v].values for v in ds.data_vars]
                x_arr = np.stack(all_arrs, axis=-1)

        # Ensure shape is (batch, H, W, channels)
        if len(x_arr.shape) == 3:
            x_arr = np.expand_dims(x_arr, axis=-1)
        elif len(x_arr.shape) == 5:
            # (time, level, lat, lon, var) -> collapse to (time, lat, lon, channels)
            T, L, H, W, V = x_arr.shape
            x_arr = np.transpose(x_arr, (0, 2, 3, 1, 4)).reshape(T, H, W, L * V)

        return x_arr.astype(np.float32)

    x_train = extract_predictors(ds_x_train, config)
    x_test = extract_predictors(ds_x_test, config) if has_test else x_train[:min(10, x_train.shape[0])]

    # 5. Extract and Structure Precipitation Target (Y)
    if 'pr' in ds_y_train:
        y_train_da = ds_y_train['pr']
    elif 'Rain_bc' in ds_y_train:
        y_train_da = ds_y_train['Rain_bc']
    else:
        var_names = list(ds_y_train.data_vars)
        y_train_da = ds_y_train[var_names[0]]

    y_train_raw = y_train_da.values.astype(np.float32) # Shape: (time, H, W)
    if len(y_train_raw.shape) == 2:
        y_train_raw = np.expand_dims(y_train_raw, axis=0)

    if has_test:
        if 'pr' in ds_y_test:
            y_test_da = ds_y_test['pr']
        elif 'Rain_bc' in ds_y_test:
            y_test_da = ds_y_test['Rain_bc']
        else:
            y_test_da = ds_y_test[list(ds_y_test.data_vars)[0]]
        y_test_raw = y_test_da.values.astype(np.float32)
        if len(y_test_raw.shape) == 2:
            y_test_raw = np.expand_dims(y_test_raw, axis=0)
    else:
        y_test_raw = y_train_raw[:min(10, y_train_raw.shape[0])]

    # Format target output according to target_mode
    T_tr, H_tg, W_tg = y_train_raw.shape
    if target_mode == "land_masked":
        # Extract only valid land points: (T, D_land)
        y_train = y_train_raw[:, land_mask]
        y_test = y_test_raw[:, land_mask]
        out_dim = D_land
    else:
        # Full flattened 2D grid: (T, H*W)
        y_train = np.reshape(y_train_raw, (T_tr, H_tg * W_tg))
        y_test = np.reshape(y_test_raw, (y_test_raw.shape[0], H_tg * W_tg))
        out_dim = H_tg * W_tg

    # Check for NaNs and handle safely
    if np.any(np.isnan(x_train)):
        x_train = np.nan_to_num(x_train, nan=0.0)
    if np.any(np.isnan(y_train)):
        y_train = np.nan_to_num(y_train, nan=0.0)
    if np.any(np.isnan(x_test)):
        x_test = np.nan_to_num(x_test, nan=0.0)
    if np.any(np.isnan(y_test)):
        y_test = np.nan_to_num(y_test, nan=0.0)

    # 6. Build Metadata & Coordinates
    coords_dict = {
        'lat': ds_y_train['lat'].values if 'lat' in ds_y_train.coords else np.arange(H_tg),
        'lon': ds_y_train['lon'].values if 'lon' in ds_y_train.coords else np.arange(W_tg),
        'train_time': ds_y_train['time'].values if 'time' in ds_y_train.coords else None,
        'test_time': ds_y_test['time'].values if has_test and 'time' in ds_y_test.coords else None,
        'target_shape': (H_tg, W_tg),
        'D_land': D_land,
        'target_mode': target_mode
    }

    metadata = {
        'dataset_name': 'CORDEX-ML-Bench',
        'config': config,
        'target_mode': target_mode,
        'n_channels': x_train.shape[-1],
        'output_dim': out_dim,
        'T_train': x_train.shape[0],
        'T_test': x_test.shape[0],
        'H_in': x_train.shape[1],
        'W_in': x_train.shape[2],
        'H_target': H_tg,
        'W_target': W_tg,
        'land_cells': D_land,
        'w850_omitted': True
    }

    if verbose:
        print("=" * 80)
        print("CORDEX-ML-Bench Dataset Successfully Loaded via Adapter:")
        print(f"  Configuration:         {config} ({'4-channel core: q, t, u, v @ 850 hPa (w850 omitted)' if config == 'config_A' else 'Full multi-level suite'})")
        print(f"  Target Mode:           {target_mode} (Output Dim = {out_dim:,})")
        print(f"  X_train Tensor:        {x_train.shape} (dtype: {x_train.dtype})")
        print(f"  Y_train Tensor:        {y_train.shape} (dtype: {y_train.dtype}, range: [{np.min(y_train):.2f}, {np.max(y_train):.2f}] mm/day)")
        print(f"  X_test Tensor:         {x_test.shape}")
        print(f"  Y_test Tensor:         {y_test.shape}")
        print(f"  Auxiliary Topography:  {aux_tensor.shape}")
        print(f"  Land Mask:             {land_mask.shape} (Valid Land Cells: {D_land:,} / {H_tg*W_tg:,} = {D_land/(H_tg*W_tg)*100:.1f}%)")
        print("=" * 80)

    data_dict = {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'auxiliary': aux_tensor,
        'coords': coords_dict,
        'mask': land_mask,
        'metadata': metadata
    }

    return data_dict
