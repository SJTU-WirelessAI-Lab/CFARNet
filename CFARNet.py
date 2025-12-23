# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from train import IndexPredictionCNN

class IndexPredictionCNN(nn.Module):
    def __init__(self, M_plus_1, Ns, hidden_dim=512, dropout=0.2): # Ensure params match
        super().__init__()
        print("WARNING: Using Placeholder CNN Definition!")
        self.M_plus_1 = M_plus_1; self.Ns = Ns
        final_channels = 512; self.hidden_dim=hidden_dim; self.dropout=dropout
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.1), nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.1), nn.Dropout2d(0.1),
            nn.Conv2d(256, final_channels, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(final_channels), nn.LeakyReLU(0.1), nn.Dropout2d(0.1)
        )
        self.predictor = nn.Sequential(
            nn.Conv1d(final_channels, hidden_dim, kernel_size=3, padding=1, bias=False), nn.BatchNorm1d(hidden_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False), nn.BatchNorm1d(hidden_dim // 2), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1)
        )
        print("INFO: Using actual IndexPredictionCNN definition (copied).") # Inform if placeholder is overridden
    def forward(self, Y_complex):
        Y_fft=torch.fft.fft(Y_complex, dim=1); Y_fft_shift=torch.fft.fftshift(Y_fft, dim=1)
        Y_magnitude=torch.abs(Y_fft_shift); Y_magnitude_log=torch.log1p(Y_magnitude)
        Y_input=Y_magnitude_log.unsqueeze(1)
        features=self.feature_extractor(Y_input); features_pooled=torch.max(features, dim=2)[0]
        logits=self.predictor(features_pooled); logits=logits.squeeze(1)
        return logits, Y_magnitude_log

import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
import sys
import traceback
import math
import datetime
from itertools import islice
from functions import load_system_params
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment

# --- Constants ---
K_BOLTZMANN = 1.38e-23; T_NOISE_KELVIN = 290; c = 3e8

# --- Dataset Definition ---
class ChunkedEchoDataset(Dataset):
    """ Loads pre-computed echoes, m_peak, theta, r, vr from chunked files. """
    def __init__(self, data_root, start_idx, end_idx, expected_k=3):
        super().__init__(); self.data_root = data_root; self.start_idx = start_idx; self.end_idx = end_idx
        self.num_samples = end_idx - start_idx + 1; self.expected_k = expected_k
        print(f"   Dataset Init: Root='{data_root}', Range=[{start_idx}, {end_idx}], Num Samples={self.num_samples}") # English print
        self.echoes_dir = os.path.join(data_root, 'echoes'); assert os.path.isdir(self.echoes_dir), f"Dir not found: {self.echoes_dir}"
        params_path = os.path.join(data_root, 'system_params.npz'); assert os.path.isfile(params_path), f"File not found: {params_path}"
        try:
            params_data = np.load(params_path)
            self.chunk_size = int(params_data.get('samples_per_chunk', params_data.get('chunk_size', 0)))
            self.M_plus_1 = int(params_data['M']) + 1 if 'M' in params_data else None
            self.Ns = int(params_data['Ns']) if 'Ns' in params_data else None
        except Exception as e: raise IOError(f"Error loading system_params.npz: {e}")
        assert self.chunk_size > 0, "samples_per_chunk/chunk_size must be positive."
        traj_path = os.path.join(data_root, 'trajectory_data.npz'); assert os.path.isfile(traj_path), f"File not found: {traj_path}"
        try:
            traj_data = np.load(traj_path)
            self.m_peak_all = traj_data.get('m_peak_indices', traj_data.get('m_peak'))
            self.theta_all = traj_data['theta_traj']; self.r_all = traj_data['r_traj']; self.vr_all = traj_data['vr']
            assert self.m_peak_all is not None, "Missing 'm_peak_indices' or 'm_peak'"
            assert 'theta_traj' in traj_data and 'r_traj' in traj_data and 'vr' in traj_data, "Missing GT params"
            self.gt_needs_averaging = self.theta_all.ndim == 3
            total_samples_in_file = self.m_peak_all.shape[0]
            if self.end_idx >= total_samples_in_file: self.end_idx = total_samples_in_file - 1; self.num_samples = self.end_idx - self.start_idx + 1
            assert self.num_samples > 0, "Adjusted sample range invalid."
            self.m_peak_targets = self.m_peak_all[self.start_idx : self.end_idx + 1]
            self.theta_targets = self.theta_all[self.start_idx : self.end_idx + 1]
            self.r_targets = self.r_all[self.start_idx : self.end_idx + 1]
            self.vr_targets = self.vr_all[self.start_idx : self.end_idx + 1]
            for name, arr_ref in [('m_peak', 'm_peak_targets'), ('theta', 'theta_targets'), ('r', 'r_targets'), ('vr', 'vr_targets')]:
                arr = getattr(self, arr_ref); current_k = arr.shape[-1]
                if current_k != self.expected_k:
                    fill_value = -1 if name == 'm_peak' else np.nan; target_shape = list(arr.shape); target_shape[-1] = self.expected_k
                    new_arr = np.full(target_shape, fill_value, dtype=arr.dtype); k_copy = min(current_k, self.expected_k)
                    if arr.ndim == 2: new_arr[:, :k_copy] = arr[:, :k_copy]
                    elif arr.ndim == 3: new_arr[:, :, :k_copy] = arr[:, :, :k_copy]
                    setattr(self, arr_ref, new_arr)
            print(f"   Adjusted GT theta shape: {self.theta_targets.shape}") # English print
        except Exception as e: raise IOError(f"Error loading trajectory_data.npz: {e}")

    def __len__(self): return self.num_samples
    def __getitem__(self, index):
        if index < 0 or index >= self.num_samples: raise IndexError(f"Index {index} out of bounds.")
        try:
            absolute_idx=self.start_idx + index; chunk_idx=absolute_idx // self.chunk_size; index_in_chunk=absolute_idx % self.chunk_size
            echo_file_path=os.path.join(self.echoes_dir, f'echo_chunk_{chunk_idx}.npy'); assert os.path.isfile(echo_file_path), f"{echo_file_path}"
            echo_chunk=np.load(echo_file_path); assert index_in_chunk < echo_chunk.shape[0], f"Index {index_in_chunk} >= chunk size {echo_chunk.shape[0]}"
            clean_echo_signal=echo_chunk[index_in_chunk]; m_peak=self.m_peak_targets[index]; theta=self.theta_targets[index]; r=self.r_targets[index]; vr=self.vr_targets[index]
            if self.gt_needs_averaging: theta=np.nanmean(theta, axis=0) if theta.ndim==2 else theta; r=np.nanmean(r, axis=0) if r.ndim==2 else r; vr=np.nanmean(vr, axis=0) if vr.ndim==2 else vr
            return {'echo': torch.from_numpy(clean_echo_signal).to(torch.complex64), 'm_peak': torch.from_numpy(m_peak).to(torch.long), 'theta': torch.from_numpy(theta).to(torch.float32), 'r': torch.from_numpy(r).to(torch.float32), 'vr': torch.from_numpy(vr).to(torch.float32)}
        except Exception as e: print(f"Error loading index {index}: {e}"); traceback.print_exc(); raise

# --- Angle Calculation Helper ---
def calculate_angle_for_m(m_idx, f_scs, BW, f0,fc, phi_start_deg, phi_end_deg):
    """
    Calculates the beam angle corresponding to a given subcarrier index m_idx.
    Args:
        m_idx (int): The 0-based subcarrier index (0 to M).
        f_scs (float): Subcarrier spacing.
        BW (float): Bandwidth.
        fc (float): Center frequency.
        phi_start_deg (float): Start angle for rainbow beam (degrees).
        phi_end_deg (float): End angle for rainbow beam (degrees).
    Returns:
        float: Calculated angle in degrees. Returns NaN if calculation fails.
    """
    fm_base = m_idx * f_scs + f0 # Make sure fm includes f0 offset if needed by formula
    # Check for invalid fm (including zero or negative based on original logic's continue)
    if fm_base <= 1e-9: return np.nan # Use a small epsilon instead of direct <= 0
    # Check for denominator close to zero
    denom = BW * (fm_base) # Simplified based on typical rainbow beam formulas
    if abs(denom) < 1e-9: return np.nan

    # Revisit the formula - standard linear mapping is simpler usually
    # Linear interpolation based on frequency index relative to total bandwidth
    fraction = (fm_base - f0) / BW if BW > 1e-9 else 0.5 # Handle BW=0 case
    fraction = np.clip(fraction, 0.0, 1.0)
    angle_rad = np.deg2rad(phi_start_deg) * (1 - fraction) + np.deg2rad(phi_end_deg) * fraction

    fm_for_calc = m_idx * f_scs # Use frequency relative to start f0 for BW calculations?
    
    fm_abs = f0 + m_idx * f_scs
    if fm_abs <= 1e-9: return np.nan

    denom_orig = BW * fm_abs # Original calculation used fm, let's stick to that
    if abs(denom_orig) < 1e-9: return np.nan

    # term1_num = (BW) * f0 
    
    fm_calc = m_idx * f_scs # Use the frequency offset from f0


    denom = BW * (fm_calc + f0)
                              
    if abs(denom) < 1e-9: return np.nan

    term1_num = (BW - fm_calc) * f0 # Use fc as in code
    term1 = (term1_num / denom) * np.sin(np.deg2rad(phi_start_deg))

    term2_num = (BW + f0) * fm_calc # Use fc and fm_calc (offset)
    term2 = (term2_num / denom) * np.sin(np.deg2rad(phi_end_deg))

    arcsin_arg = term1 + term2
   
    arcsin_arg = np.clip(arcsin_arg, -1.0, 1.0)

    try:
        angle_rad = np.arcsin(arcsin_arg)
        angle_value = np.rad2deg(angle_rad)
        if np.isnan(angle_value): return np.nan # Should not happen after clip, but safe check
    except ValueError: return np.nan # Should not happen after clip
    return angle_value

# --- Main Test Function ---
def main_test():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='CNN+MUSIC Parameter Estimation with Matching')
    parser.add_argument('--pt_dbm', type=float, default=40.0, help='Transmit Power (dBm)')
    parser.add_argument('--cuda_device', type=str, default='0', help='CUDA device ID ("0", "1", ...) or "cpu"')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--model_dir', type=str, required=True, help='CNN model directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (MUST BE 1)')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--num_test_samples', type=int, default=None, help='Max test samples')
    parser.add_argument('--top_k_cnn', type=int, default=3, help='Top-K peaks from CNN')
    args = parser.parse_args()

    # --- Device Setup ---
    if args.cuda_device.lower() == 'cpu': device = torch.device('cpu')
    else:
        try: device_id = int(args.cuda_device); assert 0 <= device_id < torch.cuda.device_count(); device = torch.device(f'cuda:{device_id}')
        except: print(f"Warning: CUDA device '{args.cuda_device}' invalid. Using CPU."); device = torch.device('cpu')
    print(f"Using device: {device}") # English print
    if args.batch_size != 1: print("Warning: Batch size forced to 1."); args.batch_size = 1

    # --- Load System Parameters ---
    data_root = args.data_dir; params_file = os.path.join(data_root, 'system_params.npz'); assert os.path.exists(params_file), f"{params_file}"
    try:
        # <<< Unpack the tuple returned by load_system_params >>>
        Nt, Nr, M, Ns, fc, f_scs, Delta_T, D_rayleigh, K_data_max, d, lambda_c, fm_list_np = load_system_params(params_file)
        M_plus_1 = M + 1
        fm_list = torch.from_numpy(fm_list_np.astype(np.float32)).to(device)
        BW = M * f_scs
        f0 = fm_list_np[0] # Start frequency
        Ts = 1 / f_scs
        phi_start_deg = -60 # Should ideally load from params if saved
        phi_end_deg = 60   # Should ideally load from params if saved
        c = 3e8
    except ImportError: 
        print("Error: Could not import load_system_params from functions.py. Cannot proceed without system parameters.")
        exit(1)
    except Exception as e: raise IOError(f"Error loading or unpacking system_params.npz: {e}") # General error handling

    # M_plus_1 = M + 1; phi_start_deg = -60; phi_end_deg = 60 # Defined above
    # <<< Use unpacked variables directly >>>
    print(f"System params: M={M}, Ns={Ns}, K_data(Max True)={K_data_max}, fc={fc/1e9:.2f}GHz, BW={BW:.2e}Hz, Ts={Ts:.2e}s, f0={f0/1e9:.3f}GHz") # English print
    print(f"CNN Top-K (Predictions Processed): {args.top_k_cnn}") # English print
    K_eval = K_data_max # Evaluation dimension = Max True Targets
    K_pred = args.top_k_cnn # Number of predictions to process
    assert K_pred > 0, "top_k_cnn must be > 0"
    print(f"Evaluation Dim K_eval: {K_eval}, Prediction Processing K_pred: {K_pred}") # English print

    # --- Calculate Noise ---
    pt_linear_mw = 10**(args.pt_dbm / 10.0); pt_scaling_factor = math.sqrt(pt_linear_mw)
    noise_power = K_BOLTZMANN * T_NOISE_KELVIN * BW* 1000.0; noise_std_dev = math.sqrt(noise_power / 2.0)
    noise_std_dev_tensor = torch.tensor(noise_std_dev, dtype=torch.float32).to(device)
    print(f"Pt: {args.pt_dbm} dBm, Noise Std Dev: {noise_std_dev:.2e}") # English print

    # --- Load Trained Model ---
    print(f"Loading CNN model from: {args.model_dir}") # English print
    model_path = None; potential_models = []
    if os.path.isdir(args.model_dir):
        potential_models = sorted([f for f in os.listdir(args.model_dir) if f.startswith('best_model') and f.endswith('.pt')], key=lambda f: os.path.getmtime(os.path.join(args.model_dir, f)), reverse=True)
    if potential_models: model_path = os.path.join(args.model_dir, potential_models[0])
    if not model_path or not os.path.exists(model_path): fallback_path = os.path.join(args.model_dir, 'best_model.pt'); model_path = fallback_path if os.path.exists(fallback_path) else None
    assert model_path, f"Model file not found in {args.model_dir}"
    cnn_hidden_dim = 512; cnn_dropout = 0.1 # Example hyperparameters - Ensure these match the *trained* model
    # <<< Make sure M_plus_1 and Ns passed to model are correct >>>
    model = IndexPredictionCNN(M_plus_1, Ns, hidden_dim=cnn_hidden_dim, dropout=cnn_dropout).to(device)
    try: model.load_state_dict(torch.load(model_path, map_location=device)); print(f"Loaded Model weights: {model_path}") # English print
    except Exception as e: raise RuntimeError(f"Error loading state_dict: {e}")
    model.eval()

    # --- Prepare Data Loader ---
    print("Initializing test dataset...") # English print
    # Estimate total samples to determine test set range (Adjust if needed)
    # This assumes the full dataset has 50k samples and 15% are for testing.
    # You might need to adjust 'total_samples_assumed' or load it from somewhere.
    total_samples_assumed = 50000
    test_fraction = 0.15
    num_test_data_in_set = int(total_samples_assumed * test_fraction)
    test_start_idx = 0 # Or adjust based on your training/validation split
    test_end_idx = num_test_data_in_set - 1

    test_dataset = ChunkedEchoDataset(data_root, test_start_idx, test_end_idx, expected_k=K_eval)
    num_samples_in_dataset = len(test_dataset)
    num_samples_to_process = min(num_samples_in_dataset, args.num_test_samples) if args.num_test_samples is not None else num_samples_in_dataset
    assert num_samples_to_process > 0, "No samples to process!"
    print(f"Processing {num_samples_to_process} samples from test set.") # English print
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device.type=='cuda')

    # --- Result Storage Initialization ---
    est_theta_raw = np.full((num_samples_to_process, K_pred), np.nan)
    est_r_raw = np.full((num_samples_to_process, K_pred), np.nan)
    est_v_raw = np.full((num_samples_to_process, K_pred), np.nan)
    est_indices_raw = np.full((num_samples_to_process, K_pred), -1, dtype=int)
    true_theta = np.full((num_samples_to_process, K_eval), np.nan)
    true_r = np.full((num_samples_to_process, K_eval), np.nan)
    true_v = np.full((num_samples_to_process, K_eval), np.nan)
    matched_est_theta = np.full((num_samples_to_process, K_eval), np.nan)
    matched_est_r = np.full((num_samples_to_process, K_eval), np.nan)
    matched_est_v = np.full((num_samples_to_process, K_eval), np.nan)
    raw_predicted_indices_log = []

    # --- MUSIC Parameters ---
    sidelobe_window = 10
    v_search_range = np.linspace(-10.5, 10.5, 2001) # Adjust as needed
    r_search_range = np.arange(9.5, 100.5, 0.01)   # Adjust as needed

    # --- Main Processing Loop ---
    print(f"\nStarting processing for {num_samples_to_process} test samples (CNN+MUSIC)...") # English print
    sample_idx_counter = 0
    # <<< Use islice correctly >>>
    limited_loader = islice(test_loader, num_samples_to_process)
    test_pbar = tqdm(limited_loader, total=num_samples_to_process, desc="Testing CNN+MUSIC", file=sys.stdout) # English desc

    # ==============================================================
    # Pass 1: CNN -> Angle -> MUSIC R/V -> Store Raw
    # ==============================================================
    with torch.no_grad():
        for batch in test_pbar:
            if sample_idx_counter >= num_samples_to_process: break

            # 1. Get Data & Store GT
            clean_echo = batch['echo'].to(device)
            gt_theta = batch['theta'].cpu().numpy().squeeze(0); gt_r = batch['r'].cpu().numpy().squeeze(0); gt_vr = batch['vr'].cpu().numpy().squeeze(0)
            true_theta[sample_idx_counter, :] = gt_theta; true_r[sample_idx_counter, :] = gt_r; true_v[sample_idx_counter, :] = gt_vr

            # 2. Add Noise
            scaled_echo = clean_echo * pt_scaling_factor
            noise = (torch.randn_like(scaled_echo.real) + 1j * torch.randn_like(scaled_echo.imag)) * noise_std_dev_tensor
            y_echo_noisy = scaled_echo + noise

            # 3. CNN Prediction & Get Top-K Indices
            pred_logits, _ = model(y_echo_noisy)
            pred_probs = torch.sigmoid(pred_logits.squeeze(0)) # Assuming sigmoid output from model
            k_actual_cnn = min(args.top_k_cnn, M_plus_1)
            if k_actual_cnn <= 0: sample_idx_counter += 1; test_pbar.update(1); continue # Skip if no peaks requested/possible
            _, top_indices = torch.topk(pred_probs, k=k_actual_cnn)
            predicted_indices_np = top_indices.cpu().numpy()

            # Log raw indices for debugging first few samples
            if sample_idx_counter < 10: raw_predicted_indices_log.append(predicted_indices_np.copy())
            est_indices_raw[sample_idx_counter, :k_actual_cnn] = predicted_indices_np # Store raw indices

            # Prepare echo for MUSIC (move to CPU and NumPy)
            Y_dynamic = y_echo_noisy.squeeze(0).cpu().numpy()

            # 4. Estimate Angle, R, V for each Top-K index
            for k_idx in range(k_actual_cnn):
                m_peak_pred = predicted_indices_np[k_idx]

                # 4.1 Angle Estimation
                angle_value = calculate_angle_for_m(m_peak_pred, f_scs, BW, f0,fc, phi_start_deg, phi_end_deg)
                est_theta_raw[sample_idx_counter, k_idx] = angle_value

                # 4.2 R/V Estimation (MUSIC) - Use rank=1 & normalization
                r_est, v_est = np.nan, np.nan # Initialize estimates for this peak
                m_min_local = max(0, m_peak_pred - sidelobe_window); m_max_local = min(M, m_peak_pred + sidelobe_window)
                sub_index = np.arange(m_min_local, m_max_local + 1)

                if len(sub_index) > 0 and np.all(sub_index <= M): # Ensure indices are valid
                    Y_RD_raw = Y_dynamic[:, sub_index]
                    if Y_RD_raw.size > 0: # Check if submatrix is not empty
                        # Normalize for MUSIC stability (optional but often helpful)
                        Y_RD = Y_RD_raw / (np.abs(Y_RD_raw) + 1e-10)
                        current_win_size = Y_RD.shape[1]; current_Ns = Y_RD.shape[0]
                        rank_sig = 1 # Rank 1 assumption for MUSIC

                        # Velocity MUSIC
                        try:
                            if current_Ns > rank_sig: # Need more samples than rank
                                R_D = (Y_RD @ np.conjugate(Y_RD.T)) / current_win_size
                                eigvals_D, U_D = np.linalg.eigh(R_D); idx_sort_D = np.argsort(np.abs(eigvals_D))[::-1]; U_D = U_D[:, idx_sort_D]
                                U_nD = U_D[:, rank_sig:] # Noise subspace
                                F_v = np.zeros_like(v_search_range, dtype=float)
                                v_time_indices = np.arange(current_Ns)
                                for ii, v_cand in enumerate(v_search_range):
                                    a_v = np.exp(1j * (4 * np.pi * f0 * v_cand * Ts / c) * v_time_indices) # Steering vector
                                    projection = np.conjugate(a_v) @ U_nD
                                    F_v[ii] = 1 / (np.linalg.norm(projection)**2 + 1e-10) # MUSIC spectrum
                                if np.any(F_v > 0): v_est = v_search_range[np.argmax(F_v)] # Find peak
                        except np.linalg.LinAlgError: pass # Handle potential numerical issues
                        est_v_raw[sample_idx_counter, k_idx] = v_est

                        # Distance MUSIC
                        try:
                            if current_win_size > rank_sig: # Need more window size than rank
                                R_r = (Y_RD.T @ np.conjugate(Y_RD)) / Ns
                                eigvals_R, U_R = np.linalg.eigh(R_r); idx_sort_R = np.argsort(np.abs(eigvals_R))[::-1]; U_R = U_R[:, idx_sort_R]
                                U_nR = U_R[:, rank_sig:] # Noise subspace
                                F_r = np.zeros_like(r_search_range, dtype=float)
                                r_freq_indices = np.arange(current_win_size)
                                for ii, rr_cand in enumerate(r_search_range):
                                    a_r = np.exp(-1j * (4 * np.pi * rr_cand * f_scs / c) * r_freq_indices) # Steering vector
                                    projection_r = np.conjugate(a_r) @ U_nR
                                    F_r[ii] = 1 / (np.linalg.norm(projection_r)**2 + 1e-10) # MUSIC spectrum
                                if np.any(F_r > 0): r_est = r_search_range[np.argmax(F_r)] # Find peak
                        except np.linalg.LinAlgError: pass # Handle potential numerical issues
                        est_r_raw[sample_idx_counter, k_idx] = r_est
                # If sub_index was invalid or Y_RD_raw was empty, r_est/v_est remain NaN

            sample_idx_counter += 1
            test_pbar.update(1)

    test_pbar.close()
    print("\nRaw parameter estimation complete. Starting matching and evaluation...") # English print

    # ==============================================================
    # Pass 2: Hungarian Matching (Angle-based) -> Align R/V -> Evaluate
    # ==============================================================
    print("Applying Hungarian matching based on angle estimates...") # English print
    num_total_matched_pairs = 0
    for i in tqdm(range(num_samples_to_process), desc="Matching Samples", file=sys.stdout): # English desc
        sample_est_theta = est_theta_raw[i, :]; sample_est_r = est_r_raw[i, :]; sample_est_v = est_v_raw[i, :]
        sample_true_theta = true_theta[i, :]; sample_true_r = true_r[i, :]; sample_true_v = true_v[i, :]

        # Find indices of valid (non-NaN) estimates and true values
        valid_est_indices = np.where(~np.isnan(sample_est_theta))[0]
        valid_true_indices = np.where(~np.isnan(sample_true_theta))[0]
        num_valid_est = len(valid_est_indices); num_valid_true = len(valid_true_indices)

        if num_valid_est == 0 or num_valid_true == 0: continue # Skip if no valid estimates or targets

        # Select the valid angles for cost matrix calculation
        valid_est_angles = sample_est_theta[valid_est_indices]
        valid_true_angles = sample_true_theta[valid_true_indices]

        # Create cost matrix based on absolute angle difference
        cost_matrix = np.abs(valid_est_angles[:, np.newaxis] - valid_true_angles[np.newaxis, :])

        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        num_total_matched_pairs += len(row_ind) # Count pairs matched in this sample

        # Store matched estimates aligned with true target indices
        for r, c in zip(row_ind, col_ind):
            original_est_idx = valid_est_indices[r] # Index in the original est_..._raw array
            original_true_idx = valid_true_indices[c] # Index in the original true_... array

            # Ensure the true index is within bounds of the matched arrays
            if original_true_idx < matched_est_theta.shape[1]:
                # Store the matched estimate at the position corresponding to the true target
                matched_est_theta[i, original_true_idx] = sample_est_theta[original_est_idx]
                matched_est_r[i, original_true_idx] = sample_est_r[original_est_idx]
                matched_est_v[i, original_true_idx] = sample_est_v[original_est_idx]

    print(f"Matching complete. Total matched pairs: {num_total_matched_pairs}") # English print

    # --- Calculate Errors on Matched Pairs ---
    print("Calculating errors and RMSE based on matched pairs...") # English print
    abs_error_theta = np.abs(matched_est_theta - true_theta)
    abs_error_r = np.abs(matched_est_r - true_r)
    abs_error_v = np.abs(matched_est_v - true_v)

    # Filter out NaNs (where no match occurred or true value was NaN)
    # These arrays contain *all* the individual absolute errors for matched pairs
    valid_errors_theta = abs_error_theta[~np.isnan(abs_error_theta)]
    valid_errors_r = abs_error_r[~np.isnan(abs_error_r)]
    valid_errors_v = abs_error_v[~np.isnan(abs_error_v)]

    num_valid_pairs_theta = len(valid_errors_theta)
    num_valid_pairs_r = len(valid_errors_r)
    num_valid_pairs_v = len(valid_errors_v)

    # Calculate overall RMSE 
    rmse_theta = np.sqrt(np.mean(valid_errors_theta**2)) if num_valid_pairs_theta > 0 else np.nan
    rmse_r = np.sqrt(np.mean(valid_errors_r**2)) if num_valid_pairs_r > 0 else np.nan
    rmse_v = np.sqrt(np.mean(valid_errors_v**2)) if num_valid_pairs_v > 0 else np.nan

    # Calculate 95th percentile
    print("Calculating 95th percentile errors...") # English print
    percentile_95_theta = np.percentile(valid_errors_theta, 95) if num_valid_pairs_theta > 0 else np.nan
    percentile_95_r = np.percentile(valid_errors_r, 95) if num_valid_pairs_r > 0 else np.nan
    percentile_95_v = np.percentile(valid_errors_v, 95) if num_valid_pairs_v > 0 else np.nan

    print("\nSaving all matched absolute errors for post-processing...")
    try:
        # Determine script directory (handle interactive vs script execution)
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError: # __file__ not defined, e.g., in interactive session
            script_dir = os.getcwd()

        # Create a timestamped filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        errors_filename = f'matched_abs_errors_cnn{args.top_k_cnn}_pt{args.pt_dbm}dBm_{timestamp}.npz'
        errors_save_path = os.path.join(script_dir, errors_filename)

        # Save the arrays into an NPZ file
        np.savez(errors_save_path,
                 abs_error_theta=valid_errors_theta, # Key 'abs_error_theta' will store valid_errors_theta array
                 abs_error_r=valid_errors_r,       # Key 'abs_error_r' will store valid_errors_r array
                 abs_error_v=valid_errors_v)       # Key 'abs_error_v' will store valid_errors_v array

        print(f"All matched absolute errors saved successfully to: {errors_save_path}") # English print
        print(f"  You can load this data later using: data = np.load('{errors_save_path}')") # English print
        print(f"  Access arrays via: theta_errors = data['abs_error_theta'], etc.") # English print

    except Exception as e:
        print(f"Error saving matched absolute errors: {e}")
        traceback.print_exc() # Print detailed traceback for debugging


    # --- Print Performance Summary ---
    print("\n========== Performance Evaluation Results (CNN+MUSIC, Rank=1, Hungarian Matching) ==========") # English Title
    print(f"Model Used: {model_path}"); print(f"Noise Setting: Pt = {args.pt_dbm} dBm"); print(f"CNN Top-K: {args.top_k_cnn}") # English prints
    print(f"Total Samples Processed: {num_samples_to_process}"); print(f"Total Valid Matched Pairs (Angle/Dist/Vel): {num_valid_pairs_theta}/{num_valid_pairs_r}/{num_valid_pairs_v}") # English print
    total_possible_true_targets = np.sum(~np.isnan(true_theta)); print(f"   (Total possible true targets: {total_possible_true_targets})") # English print
    print("-" * 70)
    print(f"Angle RMSE: {rmse_theta:.4f}°     | 95th Percentile Error: {percentile_95_theta:.4f}°") # English print
    print(f"Distance RMSE: {rmse_r:.4f} m    | 95th Percentile Error: {percentile_95_r:.4f} m") # English print
    print(f"Velocity RMSE: {rmse_v:.4f} m/s  | 95th Percentile Error: {percentile_95_v:.4f} m/s") # English print
    print("======================================================================")

    # --- Plot CDF ---
    print("\nPlotting error CDF curves (based on matched errors)...") # English print
    fig_cdf, axs_cdf = plt.subplots(1, 3, figsize=(18, 5))
    # --- plot_cdf function with English labels ---
    def plot_cdf(ax, data, label, unit):
        if data is None or len(data)==0:
            ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label} Error CDF')
            ax.set_xlabel(f'Absolute Error ({unit})'); ax.set_ylabel('CDF'); ax.grid(True); return
        sorted_data = np.sort(data); cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf, marker='.', linestyle='-', markersize=2)
        p95_val = np.percentile(sorted_data, 95)
        ax.axvline(p95_val, color='r', linestyle='--', label=f'95th Percentile ({p95_val:.2f} {unit})') # English Legend
        ax.axhline(0.95, color='r', linestyle=':', alpha=0.7)
        ax.set_title(f'{label} Error CDF (Matched)'); ax.set_xlabel(f'Absolute Error ({unit})'); ax.set_ylabel('CDF') # English Title/Labels
        ax.legend(); ax.grid(True); ax.set_ylim(0, 1.05); ax.set_xlim(left=0)

    plot_cdf(axs_cdf[0], valid_errors_theta, 'Angle', 'degrees'); plot_cdf(axs_cdf[1], valid_errors_r, 'Distance', 'm'); plot_cdf(axs_cdf[2], valid_errors_v, 'Velocity', 'm/s')
    plt.suptitle(f'CNN(Top-{args.top_k_cnn})+MUSIC (Rank=1) Error CDF - Pt={args.pt_dbm}dBm (Hungarian Matching)') # English Title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Use the same script_dir logic for saving plot
    try: script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: script_dir = os.getcwd()
    timestamp_plot = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') # Use separate timestamp if needed
    cdf_filename = f'cnn{args.top_k_cnn}_music_rank1_matched_error_cdf_pt{args.pt_dbm}dBm_{timestamp_plot}.png'
    cdf_save_path = os.path.join(script_dir, cdf_filename)
    try: plt.savefig(cdf_save_path); print(f"CDF plot saved to: {cdf_save_path}") # English print
    except Exception as e: print(f"Error saving CDF plot: {e}")
    # plt.show() # Uncomment to display plot interactively

    # --- Print Sample Comparison ---
    print("\nFirst 10 Samples Comparison (CNN+MUSIC with Hungarian Matching):") # English print
    print("-" * 95)
    num_to_print = min(10, num_samples_to_process)
    for i in range(num_to_print):
        print(f"Sample {i+1}:") # English print
        if i < len(raw_predicted_indices_log): print(f"   Raw Predicted M-Indices (Top-{args.top_k_cnn}): [{', '.join(map(str, raw_predicted_indices_log[i]))}]") # English print
        # --- English Headers ---
        print(f"   {'True Idx':<8} {'Matched Est Ang':<15} {'True Ang':<10} | {'Matched Est Dist':<16} {'True Dist':<11} | {'Matched Est Vel':<15} {'True Vel':<10}")
        num_printed = 0
        for k in range(K_eval):
            true_a_k = true_theta[i, k]; true_d_k = true_r[i, k]; true_vel_k = true_v[i, k]
            # Only print rows for which there is a true target
            if not np.isnan(true_a_k):
                matched_a=matched_est_theta[i, k]; matched_d=matched_est_r[i, k]; matched_v=matched_est_v[i, k]
                # Format strings, handling potential NaNs in matched estimates
                est_a_str = f"{matched_a:.2f}°" if ~np.isnan(matched_a) else "NaN (Unmatched)" # English print
                est_d_str = f"{matched_d:.2f}m" if ~np.isnan(matched_d) else "NaN (Unmatched)" # English print
                est_ve_str= f"{matched_v:.2f}m/s" if ~np.isnan(matched_v) else "NaN (Unmatched)" # English print
                true_a_str=f"{true_a_k:.2f}°"; true_d_str=f"{true_d_k:.2f}m"; true_ve_str=f"{true_vel_k:.2f}m/s"
                print(f"   {k:<8} {est_a_str:<15} {true_a_str:<10} | {est_d_str:<16} {true_d_str:<11} | {est_ve_str:<15} {true_ve_str:<10}")
                num_printed += 1
        if num_printed == 0: print("   (No valid true targets for this sample)") # English print
        print("-" * 95)

    print("\nScript finished.") # English print

if __name__ == "__main__":
    main_test()