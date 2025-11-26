# -*- coding: utf-8 -*-
import numpy as np
from trajectory import generate_trajectory
import math
import os
import datetime
import argparse
import torch # Import torch globally now
from tqdm import tqdm
import traceback 
import gc 
f0 = 28e9

def initial_rainbow_beam_ULA_YOLO(N, d, BW, f_scs, fm_list, phi_1, phi_M):
    """
    Rainbow beam parameter calculation implemented according to MATLAB code

    Args:
        N: Number of antennas
        d: Antenna spacing
        BW: Bandwidth
        M: Number of subcarriers (Derived from fm_list length)
        fm_list: Subcarrier frequency list [M+1]
        phi_1: Phi angle corresponding to the lowest frequency (unit: degrees)
        phi_M: Phi angle corresponding to the highest frequency (unit: degrees)

    Returns:
        TTD: Time delay compensation [N] (ns)
        PS: Phase compensation [N] (rad)
    """
    c = 3e8

    # Calculation logic according to generate.py
    antenna_idx = (np.arange(N) - (N - 1) / 2)

    # Calculate initial PS and TTD
    PS_val = -fm_list[0] * antenna_idx * d * np.sin(np.deg2rad(phi_1)) / c
    TTD_val = -PS_val / BW - ((fm_list[0] + BW) * antenna_idx * d * np.sin(np.deg2rad(phi_M))) / (BW * c)

    PS_val  = 2.0 * np.pi * PS_val           # [rad]
    TTD_val = 1e9 * TTD_val                  # [ns]

    PS_val = np.mod(PS_val , 2*np.pi)       # [0, 2π] might be better than original comment's [-π, π] based on mod
    TTD_val = np.mod(TTD_val, 1e9/f_scs)

    # Return TTD first, then PS to match potential usage patterns, but clarify in docstring
    return TTD_val, PS_val # NOTE: Returning TTD then PS

def compute_echo_from_factors_optimized(
    chan_factor,  # [B, Ns, M, K], complex
    a_vectors,    # [B, Ns, M, K, Nt], complex
    PS_T, TTD_T,  # both [B, Nt], real
    PS_R, TTD_R,  # both [B, Nt], real
    fm_list       # [M], real
):
   
    # Ensure input tensors are PyTorch tensors
    if not isinstance(chan_factor, torch.Tensor): chan_factor = torch.from_numpy(chan_factor)
    if not isinstance(a_vectors, torch.Tensor): a_vectors = torch.from_numpy(a_vectors)
    if not isinstance(PS_T, torch.Tensor): PS_T = torch.from_numpy(PS_T).float() # Ensure float
    if not isinstance(TTD_T, torch.Tensor): TTD_T = torch.from_numpy(TTD_T).float()
    if not isinstance(PS_R, torch.Tensor): PS_R = torch.from_numpy(PS_R).float()
    if not isinstance(TTD_R, torch.Tensor): TTD_R = torch.from_numpy(TTD_R).float()
    if not isinstance(fm_list, torch.Tensor): fm_list = torch.from_numpy(fm_list).float()


    B, Ns, M, K = chan_factor.shape
    Nt = a_vectors.shape[-1]
    # Try to infer device from chan_factor, default to CPU if not a tensor yet
    device = chan_factor.device if isinstance(chan_factor, torch.Tensor) else torch.device('cpu')
    dtype = torch.complex64

    # Ensure all tensors are on the same device
    chan_factor = chan_factor.to(device=device, dtype=dtype)
    a_vectors = a_vectors.to(device=device, dtype=dtype)
    fm = fm_list.to(device=device, dtype=torch.float32)          # [M]
    PS_T = PS_T.to(device=device, dtype=torch.float32)           # [B, Nt]
    TTD_T = TTD_T.to(device=device, dtype=torch.float32)         # [B, Nt]
    PS_R = PS_R.to(device=device, dtype=torch.float32)
    TTD_R = TTD_R.to(device=device, dtype=torch.float32)


    f0 = fm[0]                       # Reference frequency
    freq_diff = fm - f0              # [M]
    scale = 1e9

    # 1) First construct phase_t and phase_r: shape [B, M, Nt]
    #    phase_t[b,m,n] = -PS_T[b,n] - 2π * TTD_T[b,n] * freq_diff[m]/scale
    # Unsqueeze PS/TTD for broadcasting with freq_diff
    PS_T_exp = PS_T.unsqueeze(1) # [B, 1, Nt]
    TTD_T_exp = TTD_T.unsqueeze(1) # [B, 1, Nt]
    PS_R_exp = PS_R.unsqueeze(1) # [B, 1, Nt]
    TTD_R_exp = TTD_R.unsqueeze(1) # [B, 1, Nt]
    freq_diff_exp = freq_diff.view(1, M, 1) # [1, M, 1]

    phase_t = - PS_T_exp - 2 * np.pi * (freq_diff_exp * TTD_T_exp) / scale # [B, M, Nt]
    phase_r = - PS_R_exp - 2 * np.pi * (freq_diff_exp * TTD_R_exp) / scale # [B, M, Nt]


    # 2) Calculate BF_t_all, BF_r_all: shape [B, M, Nt]
    BF_t_all = torch.exp(1j * phase_t).to(dtype)/math.sqrt(Nt)  # [B,M,Nt]
    BF_r_all = torch.exp(1j * phase_r).to(dtype)/math.sqrt(Nt)  # [B,M,Nt]

    # 3) Initialize echo
    echo = torch.zeros((B, Ns, M), dtype=dtype, device=device)

    # 4) Loop over each subcarrier
    # --- Vectorized echo calculation across subcarriers ---
    # Reshape BF vectors for broadcasting: [B, 1, M, 1, Nt]
    bf_t_exp = BF_t_all.unsqueeze(1).unsqueeze(3)
    bf_r_exp = BF_r_all.unsqueeze(1).unsqueeze(3)

    # Reshape factors for broadcasting:
    # chan_factor: [B, Ns, M, K] -> [B, Ns, M, K, 1]
    h_exp = chan_factor.unsqueeze(-1)
    # a_vectors: [B, Ns, M, K, Nt]

    # Transmit contribution: conj(a) * bf_t -> sum over Nt -> [B, Ns, M, K]
    # Element-wise product: a_vectors.conj() * bf_t_exp results in [B, Ns, M, K, Nt]
    tx = torch.sum(a_vectors.conj() * bf_t_exp, dim=-1) # Sum over Nt -> [B, Ns, M, K]

    # Receive contribution: conj(bf_r) * a -> sum over Nt -> [B, Ns, M, K]
    # Element-wise product: bf_r_exp.conj() * a_vectors results in [B, Ns, M, K, Nt]
    rx = torch.sum(bf_r_exp.conj() * a_vectors, dim=-1) # Sum over Nt -> [B, Ns, M, K]

    # Overall echo: sum over K -> [B, Ns, M]
    # Element-wise product: h_exp.squeeze(-1) * tx * rx results in [B, Ns, M, K]
    echo = torch.sum(h_exp.squeeze(-1) * tx * rx, dim=-1) # Sum over K -> [B, Ns, M]

    return echo

def calculate_angle_for_m(m_idx, f_scs, BW, fc, phi_start_deg, phi_end_deg):
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

    term1_num = (BW) * f0 
    
    fm_calc = m_idx * f_scs # Use the frequency offset from f0


    denom = BW * (fm_calc + fc)
                              
    if abs(denom) < 1e-9: return np.nan

    term1_num = (BW - fm_calc) * fc # Use fc as in code
    term1 = (term1_num / denom) * np.sin(np.deg2rad(phi_start_deg))

    term2_num = (BW + fc) * fm_calc # Use fc and fm_calc (offset)
    term2 = (term2_num / denom) * np.sin(np.deg2rad(phi_end_deg))

    arcsin_arg = term1 + term2
   
    arcsin_arg = np.clip(arcsin_arg, -1.0, 1.0)

    try:
        angle_rad = np.arcsin(arcsin_arg)
        angle_value = np.rad2deg(angle_rad)
        if np.isnan(angle_value): return np.nan # Should not happen after clip, but safe check
    except ValueError: return np.nan # Should not happen after clip
    return angle_value


def find_closest_m_idx(target_angle_deg, angles_for_all_m):
    """
    Finds the index m that produces the angle closest to the target angle.
    Args:
        target_angle_deg (float): The target angle in degrees.
        angles_for_all_m (np.ndarray): Pre-calculated angles for each m_idx (size M+1).
    Returns:
        int: The 0-based index m_idx whose corresponding angle is closest.
             Returns -1 if no valid angles were found.
    """
    valid_indices = np.where(~np.isnan(angles_for_all_m))[0]
    if len(valid_indices) == 0:
        print("Warning: No valid pre-calculated angles found.")
        return -1

    valid_angles = angles_for_all_m[valid_indices]
    diffs = np.abs(valid_angles - target_angle_deg)
    closest_valid_idx_in_subset = np.argmin(diffs)
    best_m_idx = valid_indices[closest_valid_idx_in_subset]
    return int(best_m_idx)


def main(sample_num=1, chunk_size=500, experiment_name='moving_target_v2logic_chunked', random_trajectory_flag=0):
    """
    Generates moving target dataset (based on V2 script's logic/params)
    with chunked saving, m_peak indices, and calculated echo signals. VECTORIZED VERSION.

    Args:
        sample_num (int): Total number of samples to generate.
        chunk_size (int): Number of samples to save per chunk file.
        experiment_name (str): Name for the experiment directory.
        random_trajectory_flag (int): Trajectory generation mode.
    """
    # --- Directory Setup ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    base_dir = os.path.join(script_dir, "data", f"{experiment_name}_{timestamp}")
    folders = {
        'root': base_dir,
        'channel_factors': os.path.join(base_dir, 'channel_factors'),
        'array_vectors': os.path.join(base_dir, 'array_vectors'),
        'echoes': os.path.join(base_dir, 'echoes') # <<< Added echoes directory
    }
    for folder in folders.values(): os.makedirs(folder, exist_ok=True)
    print(f"Data will be saved to: {base_dir}")
    channel_dir, array_vectors_dir = folders['channel_factors'], folders['array_vectors']
    echoes_dir = folders['echoes'] # <<< Get echoes directory path

    # ===== System Parameters (Matching V2 Script Logic) =====
    Nt = 128; Nr = 128; c = 3e8
    f0 = 28e9        # Start frequency
    BW = 1e9          # Bandwidth Δf
    M = 2048 - 1      # Highest subcarrier index (0 to M) -> M+1 total subcarriers
    f_scs = BW / M    # Subcarrier spacing
    fc = f0 + BW / 2  # Center frequency
    print(f"System Params (V2 Logic): M={M}, Nt={Nt}, f0={f0/1e9}GHz, BW={BW/1e9}GHz, f_scs={f_scs/1e3:.2f}kHz, fc={fc/1e9}GHz")

    lambda_c = c / fc; d = lambda_c / 2; D = (Nt - 1) * d
    D_rayleigh = 2 * D**2 / lambda_c
    print(f"Rayleigh distance: {D_rayleigh:.2f} m")

    # Target & Time Params
    K = 3; L = 40; Ns = 32
    Delta_T = 1 / f_scs # Symbol duration
    total_time = Delta_T * L
    print(f"Targets K={K}, Symbols Ns={Ns}, Symbol time Delta_T={Delta_T*1e6:.2f} us")

    # Trajectory Params
    theta_min_deg = -60; theta_max_deg = 60
    r_min = 5; r_max = 100
    min_speed = 3.6; max_speed = 36 # Assuming m/s

    # Subcarrier Frequencies (V2 logic: M+1 frequencies)
    fm_list = f0 + f_scs * np.arange(M + 1) # Shape [M+1,] -> Indices 0 to M
    print(f"Frequency list shape: {fm_list.shape}.")

    # Rainbow Beam Angles for m_peak calculation and Initial PS/TTD
    phi_start_deg = theta_min_deg
    phi_end_deg = theta_max_deg

    # --- Pre-calculate angles for m_peak ---
    print("Pre-calculating angles for m_peak (m=0 to M)...")
    angles_for_all_m = np.array([
        calculate_angle_for_m(m_idx, f_scs, BW, fc, phi_start_deg, phi_end_deg)
        for m_idx in range(M + 1) # 0 to M
    ])
    num_valid = np.sum(~np.isnan(angles_for_all_m))
    print(f"Pre-calculated {num_valid}/{M+1} valid angles.")

    # --- Calculate Initial PS/TTD ---
    print("Calculating initial PS and TTD...")
    # Ensure N matches Nt/Nr for the function call
    initial_TTD, initial_PS = initial_rainbow_beam_ULA_YOLO(
        N=Nt, # Assuming Nt=Nr
        d=d,
        BW=BW,
        f_scs=f_scs,
        fm_list=fm_list,
        phi_1=phi_start_deg,
        phi_M=phi_end_deg
    )
    print(f"Calculated initial PS (shape {initial_PS.shape}) and TTD (shape {initial_TTD.shape})")
    # Convert to float32 for consistency
    initial_PS = initial_PS.astype(np.float32)
    initial_TTD = initial_TTD.astype(np.float32)


    # ===== Trajectory Generation =====
    t_samples = np.linspace(0, total_time, L)
    t_ofdm = t_samples[:Ns] # Shape [Ns,]

    # Pre-allocate full data arrays
    x_traj_all = np.zeros((sample_num, Ns, K)); y_traj_all = np.zeros((sample_num, Ns, K))
    vx_all = np.zeros((sample_num, Ns, K)); vy_all = np.zeros((sample_num, Ns, K))
    vr_all = np.zeros((sample_num, Ns, K)); vt_all = np.zeros((sample_num, Ns, K))
    r_traj_all = np.zeros((sample_num, Ns, K)); theta_traj_all = np.zeros((sample_num, Ns, K))
    m_peak_indices_all = np.full((sample_num, K), -1, dtype=np.int64)

    print("Generating trajectories and calculating m_peak...")
    traj_pbar = tqdm(range(sample_num), desc="Generating Trajectories")
    for sample_idx in traj_pbar:
        # --- (Trajectory generation loop remains the same) ---
        if hasattr(generate_trajectory, 'chosen_angles'):
            try: delattr(generate_trajectory, 'chosen_angles')
            except AttributeError: pass
        for k_idx in range(K):
            x, y, vx_vals, vy_vals, vr_vals, vt_vals, r_vals, theta_vals = generate_trajectory(
                total_time, Delta_T, theta_min_deg, theta_max_deg,
                r_min, r_max, min_speed, max_speed, sector_idx=None, total_sectors=K,
                random_flag=random_trajectory_flag, circle_mode=False )
            x_traj_all[sample_idx, :, k_idx]=x[:Ns]; y_traj_all[sample_idx, :, k_idx]=y[:Ns]
            vx_all[sample_idx, :, k_idx]=vx_vals[:Ns]; vy_all[sample_idx, :, k_idx]=vy_vals[:Ns]
            vr_all[sample_idx, :, k_idx]=vr_vals[:Ns]; vt_all[sample_idx, :, k_idx]=vt_vals[:Ns]
            r_traj_all[sample_idx, :, k_idx]=r_vals[:Ns]; theta_traj_all[sample_idx, :, k_idx]=theta_vals[:Ns]

            initial_target_angle = theta_traj_all[sample_idx, 0, k_idx]
            best_m_idx = find_closest_m_idx(initial_target_angle, angles_for_all_m)
            m_peak_indices_all[sample_idx, k_idx] = best_m_idx
    # --- End Trajectory Generation ---


    # ===== Channel Factor, Array Vector, and Echo Calculation & Chunked Saving =====
    num_chunks = math.ceil(sample_num / chunk_size)
    print(f"\nCalculating factors, echo (VECTORIZED) & saving in {num_chunks} chunks (Size: {chunk_size})...")

    # Pre-calculate constants and arrays needed for vectorization
    idx_antenna = np.arange(Nt) # Shape [Nt,]
    phase_const_array = (1j * 2 * np.pi * d / c)
    phase_const_doppler = (1j * 2 * np.pi * f0 * 2 / c)
    phase_const_distance = (-1j * 2 * np.pi * 2 / c)

    # Reshape common arrays for broadcasting
    fm_list_M1 = fm_list[:, np.newaxis] # Shape [M+1, 1]
    fm_list_M1_1 = fm_list[np.newaxis, :, np.newaxis] # Shape [1, M+1, 1]
    fm_list_M1_K = fm_list[np.newaxis, :, np.newaxis] # Shape [1, M+1, 1] (For chan factor)
    t_ofdm_Ns_1 = t_ofdm[np.newaxis, :, np.newaxis] # Shape [1, Ns, 1]
    idx_antenna_Nt = idx_antenna[np.newaxis, np.newaxis, :] # Shape [1, 1, Nt]

    # Mask for invalid frequencies (fm <= 0)
    invalid_m_mask = (fm_list <= 1e-9) # Use epsilon for safety
    fm_list_torch = torch.from_numpy(fm_list.astype(np.float32)) # Convert fm_list to tensor once

    try:
        process_pbar = tqdm(range(num_chunks), desc="Processing Chunks (Vectorized)")
        for chunk_idx in process_pbar:
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, sample_num)
            current_chunk_sample_count = end_idx - start_idx
            # B = current_chunk_sample_count (for brevity in comments)

            # --- Get data for the current chunk (factors) ---
            # Initial states (at t=0) for alpha and array vector
            r_initial_chunk = r_traj_all[start_idx:end_idx, 0, :]     # Shape [B, K]
            theta_initial_chunk = theta_traj_all[start_idx:end_idx, 0, :] # Shape [B, K]
            # Constant r and vr used in the original factor loop (at t=0)
            r_const_chunk = r_traj_all[start_idx:end_idx, 0, :]     # Shape [B, K]
            vr_const_chunk = vr_all[start_idx:end_idx, 0, :]        # Shape [B, K]

            # --- Create Masks for Validity (factors) ---
            invalid_sample_mask_r = np.any(r_initial_chunk <= 1e-9, axis=1) # Shape [B,] Use epsilon
            invalid_sample_mask_r_const = np.any(r_const_chunk <= 1e-9, axis=1) # Shape [B,] Use epsilon
            invalid_sample_mask = np.logical_or(invalid_sample_mask_r, invalid_sample_mask_r_const) # Shape [B,]
            valid_sample_mask = ~invalid_sample_mask # Shape [B,]

            # --- Calculate alphaVal_k for the chunk (factors) ---
            r_initial_safe = r_initial_chunk + 1e-12
            rcs=0.1
            alphaVal_k_chunk = np.sqrt((lambda_c**2 *rcs/ (4*np.pi)**3)) / (r_initial_safe**2) # Shape [B, K]
            
            alphaVal_k_chunk[invalid_sample_mask, :] = 0

            # --- Vectorized Array Vector Calculation (factors) ---
            theta_initial_rad_chunk = np.deg2rad(theta_initial_chunk) # Shape [B, K]
            sin_theta_chunk = np.sin(theta_initial_rad_chunk)         # Shape [B, K]
            # term_array = (phase_const_array *
            #               fm_list[np.newaxis, :, np.newaxis, np.newaxis] *
            #               sin_theta_chunk[:, np.newaxis, :, np.newaxis] *
            #               idx_antenna[np.newaxis, np.newaxis, np.newaxis, :])
            linear_term = (phase_const_array *
               fm_list[np.newaxis, :, np.newaxis, np.newaxis] *
               sin_theta_chunk[:, np.newaxis, :, np.newaxis] * # 对应 $d \cos \theta_k$ (根据您的约定)
               idx_antenna[np.newaxis, np.newaxis, np.newaxis, :])

            # --- 2. 非线性项 (Spherical Wave Correction) ---
            r_k_array = r_const_chunk # Shape [B, K]
            cos_sq_theta_chunk = np.square(np.cos(theta_initial_rad_chunk)) # Shape [B, K]
            nonlinear_term = (phase_const_array *
                            fm_list[np.newaxis, :, np.newaxis, np.newaxis] *
                            (d / 2) * 
                            np.square(idx_antenna)[np.newaxis, np.newaxis, np.newaxis, :] * # x_n^2
                            cos_sq_theta_chunk[:, np.newaxis, :, np.newaxis] / # $\sin^2\theta_k$
                            r_k_array[:, np.newaxis, :, np.newaxis] # $1/r_k$
                            )

            # --- 3. 完整相位项 (带二阶修正) ---
            term_array = linear_term - nonlinear_term
            array_vector_chunk = np.exp(term_array, dtype=np.complex64)
            array_vector_chunk[:, invalid_m_mask, :, :] = 0
            array_vector_chunk[invalid_sample_mask, :, :, :] = 0

            # --- Vectorized Channel Factor Calculation ---
            phase_doppler = phase_const_doppler * vr_const_chunk[:, np.newaxis, :] * t_ofdm_Ns_1
            dopFactor_chunk = np.exp(phase_doppler) # Shape [B, Ns, K]
            phase_distance = phase_const_distance * r_const_chunk[:, np.newaxis, :] * fm_list_M1_K
            distPhase_chunk = np.exp(phase_distance) # Shape [B, M+1, K]
            chan_factor_chunk = (alphaVal_k_chunk[:, np.newaxis, np.newaxis, :] *
                                 dopFactor_chunk[:, :, np.newaxis, :] *
                                 distPhase_chunk[:, np.newaxis, :, :]).astype(np.complex64)
            chan_factor_chunk[:, :, invalid_m_mask, :] = 0
            chan_factor_chunk[invalid_sample_mask, :, :, :] = 0
            # --- End Factor Calculation ---

            # <<< --- Start Echo Calculation --- >>>
            # Prepare PS/TTD for the chunk size B
            PS_T_chunk = np.tile(initial_PS, (current_chunk_sample_count, 1))   # [B, Nt]
            TTD_T_chunk = np.tile(initial_TTD, (current_chunk_sample_count, 1)) # [B, Nt]
            PS_R_chunk = PS_T_chunk  # Assuming Tx=Rx for initial values
            TTD_R_chunk = TTD_T_chunk # Assuming Tx=Rx for initial values

            # Convert inputs to PyTorch tensors (on CPU initially)
            # Use .copy() to avoid potential issues with views if factors are loaded later
            chan_factor_tensor = torch.from_numpy(chan_factor_chunk.copy())     # [B, Ns, M+1, K]
            array_vector_tensor = torch.from_numpy(array_vector_chunk.copy())  # [B, M+1, K, Nt]

            # Expand array_vector to include Ns dimension [B, 1, M+1, K, Nt] -> [B, Ns, M+1, K, Nt]
            array_vector_tensor_expanded = array_vector_tensor.unsqueeze(1).expand(
                current_chunk_sample_count, Ns, M + 1, K, Nt
            )

            PS_T_tensor = torch.from_numpy(PS_T_chunk)
            TTD_T_tensor = torch.from_numpy(TTD_T_chunk)
            PS_R_tensor = torch.from_numpy(PS_R_chunk)
            TTD_R_tensor = torch.from_numpy(TTD_R_chunk)

            # Compute echo
            yecho_chunk_tensor = compute_echo_from_factors_optimized(
                chan_factor=chan_factor_tensor,
                a_vectors=array_vector_tensor_expanded, # Use expanded version
                PS_T=PS_T_tensor, TTD_T=TTD_T_tensor,
                PS_R=PS_R_tensor, TTD_R=TTD_R_tensor,
                fm_list=fm_list_torch # Use the pre-converted tensor
            )

            # Convert echo back to NumPy array for saving
            yecho_chunk_numpy = yecho_chunk_tensor.cpu().numpy()
            # <<< --- End Echo Calculation --- >>>


            # --- Save the chunk data (Factors and Echo) ---
            chan_save_path = os.path.join(channel_dir, f'chan_factors_chunk_{chunk_idx}.npy')
            array_save_path = os.path.join(array_vectors_dir, f'array_vectors_chunk_{chunk_idx}.npy')
            echo_save_path = os.path.join(echoes_dir, f'echo_chunk_{chunk_idx}.npy') # <<< Echo save path

            # np.save(chan_save_path, chan_factor_chunk)
            # np.save(array_save_path, array_vector_chunk)
            np.save(echo_save_path, yecho_chunk_numpy) # <<< Save echo

            # Explicitly delete and collect intermediate data for the chunk
            del chan_factor_chunk, array_vector_chunk, yecho_chunk_numpy
            del chan_factor_tensor, array_vector_tensor, array_vector_tensor_expanded, yecho_chunk_tensor
            del PS_T_tensor, TTD_T_tensor, PS_R_tensor, TTD_R_tensor
            gc.collect()

    except Exception as e:
        print(f"\nError during chunk processing (Chunk {chunk_idx}): {e}"); traceback.print_exc(); raise

    # --- Save Trajectory and System Parameters ---
    print("\nSaving trajectory data (including m_peak_indices)...")
    traj_filename = os.path.join(base_dir, 'trajectory_data.npz')
    np.savez(traj_filename,
             x_traj=x_traj_all, y_traj=y_traj_all, vx=vx_all, vy=vy_all,
             vr=vr_all, vt=vt_all, r_traj=r_traj_all, theta_traj=theta_traj_all,
             m_peak_indices=m_peak_indices_all)
    print(f"Saved trajectory data: {traj_filename}")

    print("\nSaving system parameters...")
    params_filename = os.path.join(base_dir, 'system_params.npz')
    np.savez(params_filename,
             Nt=Nt, Nr=Nr, M=M, Ns=Ns, fc=fc, f_scs=f_scs, Delta_T=Delta_T,
             D_rayleigh=D_rayleigh, K=K, d=d, lambda_c=lambda_c, D=D,
             fm_list=fm_list, f0=f0, BW=BW, L=L,
             theta_min_deg=theta_min_deg, theta_max_deg=theta_max_deg,
             r_min=r_min, r_max=r_max, min_speed=min_speed, max_speed=max_speed,
             random_trajectory_flag=random_trajectory_flag,
             sample_num=sample_num, samples_per_chunk=chunk_size,
             phi_start_deg=phi_start_deg, phi_end_deg=phi_end_deg,
             # Optionally save initial PS/TTD if needed later
             initial_PS=initial_PS, initial_TTD=initial_TTD
            )
    print(f"Saved system parameters: {params_filename}")

    print(f"\nAll data saved to: {folders['root']}")
    print(f"Total samples: {sample_num}, Samples/chunk: {chunk_size}")
    print(f"Dims per sample: Ns={Ns}, M+1={M+1}, K={K}, Nt={Nt}")

    try:
        latest_exp_file = os.path.join(script_dir, 'latest_experiment.txt')
        with open(latest_exp_file, 'w') as f: f.write(folders['root'])
        print(f"Latest experiment path saved to: {latest_exp_file}")
    except Exception as write_err: print(f"Warning: Could not save latest experiment path: {write_err}")

    bytes_per_complex = np.complex64(0).nbytes
    chan_size_mb = Ns * (M + 1) * K * bytes_per_complex / (1024*1024)
    array_size_mb = (M + 1) * K * Nt * bytes_per_complex / (1024*1024) # Note: Saved array vector doesn't have Ns dim
    echo_size_mb = Ns * (M + 1) * bytes_per_complex / (1024*1024)
    total_chunk_data_gb = (chan_size_mb + array_size_mb + echo_size_mb) * sample_num / 1024 # Add echo size estimate
    print(f"\nStorage Statistics (V2 Logic):")
    print(f"  Single Sample Chan Factor Size: {chan_size_mb:.2f} MB")
    print(f"  Single Sample Array Vector Size (saved): {array_size_mb:.2f} MB") # Clarify saved size
    print(f"  Single Sample Echo Size: {echo_size_mb:.2f} MB")
    print(f"  Total Estimated Chunk Data Size: {total_chunk_data_gb:.2f} GB")

    return folders['root']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate moving target data (V2 Logic, VECTORIZED Chunked Saving + m_peak + Echo)') # Updated description
    parser.add_argument('--samples', type=int, default=1, help='Number of samples')
    parser.add_argument('--chunk', type=int, default=500, help='Number of samples saved per chunk file (Reduced default for testing echo gen)') # Adjusted default
    parser.add_argument('--name', type=str, default='moving_target_v2logic_chunked_vectorized_echo', help='Experiment name') # Updated default name
    parser.add_argument('--random', type=int, default=0, choices=[0, 1], help='Trajectory generation mode (0: ensure angle difference, 1: completely random)')
    args = parser.parse_args()

    print(f"--- Running with Samples={args.samples}, Chunk Size={args.chunk} ---")

    experiment_path = main(sample_num=args.samples,
                           chunk_size=args.chunk,
                           experiment_name=args.name,
                           random_trajectory_flag=args.random)
    print(f"\nDataset generation completed, path: {experiment_path}")
