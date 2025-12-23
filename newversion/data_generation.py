# -*- coding: utf-8 -*-
import numpy as np
from trajectory import generate_trajectory
import math
import os
import datetime
import argparse
import torch
from tqdm import tqdm
import traceback
import gc
f0 = 77e9

def initial_rainbow_beam_ULA_YOLO(N, d, BW, f_scs, fm_list, phi_1, phi_M):
    c = 3e8
    antenna_idx = (np.arange(N) - (N - 1) / 2)
    PS_val = -fm_list[0] * antenna_idx * d * np.sin(np.deg2rad(phi_1)) / c
    TTD_val = -PS_val / BW - ((fm_list[0] + BW) * antenna_idx * d * np.sin(np.deg2rad(phi_M))) / (BW * c)
    PS_val  = 2.0 * np.pi * PS_val
    TTD_val = 1e9 * TTD_val
    PS_val = np.mod(PS_val , 2*np.pi)
    TTD_val = np.mod(TTD_val, 1e9/f_scs)
    return TTD_val, PS_val

def compute_echo_from_factors_optimized(chan_factor, a_vectors, PS_T, TTD_T, PS_R, TTD_R, fm_list):
    if not isinstance(chan_factor, torch.Tensor): chan_factor = torch.from_numpy(chan_factor)
    if not isinstance(a_vectors, torch.Tensor): a_vectors = torch.from_numpy(a_vectors)
    if not isinstance(PS_T, torch.Tensor): PS_T = torch.from_numpy(PS_T).float()
    if not isinstance(TTD_T, torch.Tensor): TTD_T = torch.from_numpy(TTD_T).float()
    if not isinstance(PS_R, torch.Tensor): PS_R = torch.from_numpy(PS_R).float()
    if not isinstance(TTD_R, torch.Tensor): TTD_R = torch.from_numpy(TTD_R).float()
    if not isinstance(fm_list, torch.Tensor): fm_list = torch.from_numpy(fm_list).float()

    B, Ns, M, K = chan_factor.shape
    Nt = a_vectors.shape[-1]
    device = chan_factor.device if isinstance(chan_factor, torch.Tensor) else torch.device('cpu')
    dtype = torch.complex64

    chan_factor = chan_factor.to(device=device, dtype=dtype)
    a_vectors = a_vectors.to(device=device, dtype=dtype)
    fm = fm_list.to(device=device, dtype=torch.float32)
    PS_T = PS_T.to(device=device, dtype=torch.float32)
    TTD_T = TTD_T.to(device=device, dtype=torch.float32)
    PS_R = PS_R.to(device=device, dtype=torch.float32)
    TTD_R = TTD_R.to(device=device, dtype=torch.float32)

    f0 = fm[0]
    freq_diff = fm - f0
    scale = 1e9

    PS_T_exp = PS_T.unsqueeze(1)
    TTD_T_exp = TTD_T.unsqueeze(1)
    PS_R_exp = PS_R.unsqueeze(1)
    TTD_R_exp = TTD_R.unsqueeze(1)
    freq_diff_exp = freq_diff.view(1, M, 1)

    phase_t = - PS_T_exp - 2 * np.pi * (freq_diff_exp * TTD_T_exp) / scale
    phase_r = - PS_R_exp - 2 * np.pi * (freq_diff_exp * TTD_R_exp) / scale

    BF_t_all = torch.exp(1j * phase_t).to(dtype)/math.sqrt(Nt)
    BF_r_all = torch.exp(1j * phase_r).to(dtype)/math.sqrt(Nt)

    echo = torch.zeros((B, Ns, M), dtype=dtype, device=device)
    bf_t_exp = BF_t_all.unsqueeze(1).unsqueeze(3)
    bf_r_exp = BF_r_all.unsqueeze(1).unsqueeze(3)
    h_exp = chan_factor.unsqueeze(-1)
    tx = torch.sum(a_vectors.conj() * bf_t_exp, dim=-1)
    rx = torch.sum(bf_r_exp.conj() * a_vectors, dim=-1)
    echo = torch.sum(h_exp.squeeze(-1) * tx * rx, dim=-1)
    return echo

def calculate_angle_for_m(m_idx, f_scs, BW, fc, phi_start_deg, phi_end_deg):
    fm_base = m_idx * f_scs + f0
    if fm_base <= 1e-9: return np.nan
    denom = BW * (fm_base)
    if abs(denom) < 1e-9: return np.nan
    
    fm_calc = m_idx * f_scs
    denom = BW * (fm_calc + f0)
    if abs(denom) < 1e-9: return np.nan

    term1_num = (BW - fm_calc) * f0
    term1 = (term1_num / denom) * np.sin(np.deg2rad(phi_start_deg))

    term2_num = (BW + f0) * fm_calc
    term2 = (term2_num / denom) * np.sin(np.deg2rad(phi_end_deg))

    arcsin_arg = term1 + term2
    arcsin_arg = np.clip(arcsin_arg, -1.0, 1.0)

    try:
        angle_rad = np.arcsin(arcsin_arg)
        angle_value = np.rad2deg(angle_rad)
        if np.isnan(angle_value): return np.nan
    except ValueError: return np.nan
    return angle_value

def find_closest_m_idx(target_angle_deg, angles_for_all_m):
    valid_indices = np.where(~np.isnan(angles_for_all_m))[0]
    if len(valid_indices) == 0: return -1
    valid_angles = angles_for_all_m[valid_indices]
    diffs = np.abs(valid_angles - target_angle_deg)
    closest_valid_idx_in_subset = np.argmin(diffs)
    best_m_idx = valid_indices[closest_valid_idx_in_subset]
    return int(best_m_idx)

def main(sample_num=5000, chunk_size=500, experiment_name='moving_target', 
         random_trajectory_flag=0, min_angle_diff=10.0, num_targets=3):
    """
    Main function with K and D parameters exposed
    """
    # --- Directory Setup with K{}_d{} Format ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    # <--- 修改: 强制格式化文件夹名称 --->
    folder_suffix = f"_k{num_targets}_d{int(min_angle_diff)}"
    # 将 experiment_name 拼接参数，或者直接用参数构造
    full_exp_name = f"{experiment_name}{folder_suffix}_{timestamp}"
    
    base_dir = os.path.join(script_dir, "data", full_exp_name)
    
    folders = {
        'root': base_dir,
        'channel_factors': os.path.join(base_dir, 'channel_factors'),
        'array_vectors': os.path.join(base_dir, 'array_vectors'),
        'echoes': os.path.join(base_dir, 'echoes')
    }
    for folder in folders.values(): os.makedirs(folder, exist_ok=True)
    print(f"Data will be saved to: {base_dir}")
    print(f"Config: K={num_targets}, MinDelta={min_angle_diff}")

    channel_dir, array_vectors_dir = folders['channel_factors'], folders['array_vectors']
    echoes_dir = folders['echoes']

    # ===== System Parameters =====
    Nt = 128; Nr = 128; c = 3e8
    BW = 1e9
    M = 2048 - 1
    f_scs = BW / M
    fc = f0 + BW / 2
    
    lambda_c = c / fc; d = lambda_c / 2; D = (Nt - 1) * d
    D_rayleigh = 2 * D**2 / lambda_c

    # Target & Time Params
    K = num_targets # <--- 使用传入参数
    L = 40; Ns = 32
    Delta_T = 1 / f_scs
    total_time = Delta_T * L

    # Trajectory Params
    theta_min_deg = -60; theta_max_deg = 60
    r_min = 35; r_max = 100
    min_speed = 3.6; max_speed = 36

    fm_list = f0 + f_scs * np.arange(M + 1)
    phi_start_deg = theta_min_deg
    phi_end_deg = theta_max_deg

    print("Pre-calculating angles for m_peak...")
    angles_for_all_m = np.array([
        calculate_angle_for_m(m_idx, f_scs, BW, fc, phi_start_deg, phi_end_deg)
        for m_idx in range(M + 1)
    ])

    print("Calculating initial PS and TTD...")
    initial_TTD, initial_PS = initial_rainbow_beam_ULA_YOLO(
        N=Nt, d=d, BW=BW, f_scs=f_scs, fm_list=fm_list, phi_1=phi_start_deg, phi_M=phi_end_deg
    )
    initial_PS = initial_PS.astype(np.float32)
    initial_TTD = initial_TTD.astype(np.float32)

    # ===== Trajectory Generation =====
    t_samples = np.linspace(0, total_time, L)
    t_ofdm = t_samples[:Ns]

    x_traj_all = np.zeros((sample_num, Ns, K)); y_traj_all = np.zeros((sample_num, Ns, K))
    vx_all = np.zeros((sample_num, Ns, K)); vy_all = np.zeros((sample_num, Ns, K))
    vr_all = np.zeros((sample_num, Ns, K)); vt_all = np.zeros((sample_num, Ns, K))
    r_traj_all = np.zeros((sample_num, Ns, K)); theta_traj_all = np.zeros((sample_num, Ns, K))
    m_peak_indices_all = np.full((sample_num, K), -1, dtype=np.int64)

    print(f"Generating trajectories (Min Delta={min_angle_diff})...")
    traj_pbar = tqdm(range(sample_num), desc="Generating Trajectories")
    for sample_idx in traj_pbar:
        if hasattr(generate_trajectory, 'chosen_angles'):
            try: delattr(generate_trajectory, 'chosen_angles')
            except AttributeError: pass
        for k_idx in range(K):
            # <--- 传递 min_angle_diff --->
            x, y, vx_vals, vy_vals, vr_vals, vt_vals, r_vals, theta_vals = generate_trajectory(
                total_time, Delta_T, theta_min_deg, theta_max_deg,
                r_min, r_max, min_speed, max_speed, sector_idx=None, total_sectors=K,
                random_flag=random_trajectory_flag, circle_mode=False,
                min_angle_diff=min_angle_diff) 
            
            x_traj_all[sample_idx, :, k_idx]=x[:Ns]; y_traj_all[sample_idx, :, k_idx]=y[:Ns]
            vx_all[sample_idx, :, k_idx]=vx_vals[:Ns]; vy_all[sample_idx, :, k_idx]=vy_vals[:Ns]
            vr_all[sample_idx, :, k_idx]=vr_vals[:Ns]; vt_all[sample_idx, :, k_idx]=vt_vals[:Ns]
            r_traj_all[sample_idx, :, k_idx]=r_vals[:Ns]; theta_traj_all[sample_idx, :, k_idx]=theta_vals[:Ns]

            initial_target_angle = theta_traj_all[sample_idx, 0, k_idx]
            best_m_idx = find_closest_m_idx(initial_target_angle, angles_for_all_m)
            m_peak_indices_all[sample_idx, k_idx] = best_m_idx

    # ===== Channel Factor & Echo Calculation =====
    num_chunks = math.ceil(sample_num / chunk_size)
    print(f"\nCalculating factors & echo...")

    idx_antenna = np.arange(Nt)
    phase_const_array = (1j * 2 * np.pi * d / c)
    phase_const_doppler = (1j * 2 * np.pi * f0 * 2 / c)
    phase_const_distance = (-1j * 2 * np.pi * 2 / c)

    fm_list_M1 = fm_list[:, np.newaxis]
    fm_list_M1_1 = fm_list[np.newaxis, :, np.newaxis]
    fm_list_M1_K = fm_list[np.newaxis, :, np.newaxis]
    t_ofdm_Ns_1 = t_ofdm[np.newaxis, :, np.newaxis]
    idx_antenna_Nt = idx_antenna[np.newaxis, np.newaxis, :]
    
    invalid_m_mask = (fm_list <= 1e-9)
    fm_list_torch = torch.from_numpy(fm_list.astype(np.float32))

    try:
        process_pbar = tqdm(range(num_chunks), desc="Processing Chunks")
        for chunk_idx in process_pbar:
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, sample_num)
            current_chunk_sample_count = end_idx - start_idx
            
            r_initial_chunk = r_traj_all[start_idx:end_idx, 0, :]
            theta_initial_chunk = theta_traj_all[start_idx:end_idx, 0, :]
            r_const_chunk = r_traj_all[start_idx:end_idx, 0, :]
            vr_const_chunk = vr_all[start_idx:end_idx, 0, :]

            invalid_sample_mask_r = np.any(r_initial_chunk <= 1e-9, axis=1)
            invalid_sample_mask_r_const = np.any(r_const_chunk <= 1e-9, axis=1)
            invalid_sample_mask = np.logical_or(invalid_sample_mask_r, invalid_sample_mask_r_const)
            valid_sample_mask = ~invalid_sample_mask

            r_initial_safe = r_initial_chunk + 1e-12
            rcs = 0.1
            alphaVal_k_chunk = np.sqrt((lambda_c**2 *rcs/ (4*np.pi)**3)) / (r_initial_safe**2)
            alphaVal_k_chunk[invalid_sample_mask, :] = 0

            theta_initial_rad_chunk = np.deg2rad(theta_initial_chunk)
            sin_theta_chunk = np.sin(theta_initial_rad_chunk)
            term_array = (phase_const_array * fm_list[np.newaxis, :, np.newaxis, np.newaxis] *
                          sin_theta_chunk[:, np.newaxis, :, np.newaxis] *
                          idx_antenna[np.newaxis, np.newaxis, np.newaxis, :])
            array_vector_chunk = np.exp(term_array, dtype=np.complex64)
            array_vector_chunk[:, invalid_m_mask, :, :] = 0
            array_vector_chunk[invalid_sample_mask, :, :, :] = 0

            phase_doppler = phase_const_doppler * vr_const_chunk[:, np.newaxis, :] * t_ofdm_Ns_1
            dopFactor_chunk = np.exp(phase_doppler)
            phase_distance = phase_const_distance * r_const_chunk[:, np.newaxis, :] * fm_list_M1_K
            distPhase_chunk = np.exp(phase_distance)
            chan_factor_chunk = (alphaVal_k_chunk[:, np.newaxis, np.newaxis, :] *
                                 dopFactor_chunk[:, :, np.newaxis, :] *
                                 distPhase_chunk[:, np.newaxis, :, :]).astype(np.complex64)
            chan_factor_chunk[:, :, invalid_m_mask, :] = 0
            chan_factor_chunk[invalid_sample_mask, :, :, :] = 0

            PS_T_chunk = np.tile(initial_PS, (current_chunk_sample_count, 1))
            TTD_T_chunk = np.tile(initial_TTD, (current_chunk_sample_count, 1))
            PS_R_chunk = PS_T_chunk
            TTD_R_chunk = TTD_T_chunk

            chan_factor_tensor = torch.from_numpy(chan_factor_chunk.copy())
            array_vector_tensor = torch.from_numpy(array_vector_chunk.copy())
            array_vector_tensor_expanded = array_vector_tensor.unsqueeze(1).expand(
                current_chunk_sample_count, Ns, M + 1, K, Nt
            )
            PS_T_tensor = torch.from_numpy(PS_T_chunk)
            TTD_T_tensor = torch.from_numpy(TTD_T_chunk)
            PS_R_tensor = torch.from_numpy(PS_R_chunk)
            TTD_R_tensor = torch.from_numpy(TTD_R_chunk)

            yecho_chunk_tensor = compute_echo_from_factors_optimized(
                chan_factor=chan_factor_tensor, a_vectors=array_vector_tensor_expanded,
                PS_T=PS_T_tensor, TTD_T=TTD_T_tensor, PS_R=PS_R_tensor, TTD_R=TTD_R_tensor,
                fm_list=fm_list_torch
            )
            yecho_chunk_numpy = yecho_chunk_tensor.cpu().numpy()

            echo_save_path = os.path.join(echoes_dir, f'echo_chunk_{chunk_idx}.npy')
            np.save(echo_save_path, yecho_chunk_numpy)

            del chan_factor_chunk, array_vector_chunk, yecho_chunk_numpy
            del chan_factor_tensor, array_vector_tensor, array_vector_tensor_expanded, yecho_chunk_tensor
            gc.collect()

    except Exception as e: print(f"Error: {e}"); traceback.print_exc(); raise

    print("Saving trajectory data...")
    traj_filename = os.path.join(base_dir, 'trajectory_data.npz')
    np.savez(traj_filename,
             x_traj=x_traj_all, y_traj=y_traj_all, vx=vx_all, vy=vy_all,
             vr=vr_all, vt=vt_all, r_traj=r_traj_all, theta_traj=theta_traj_all,
             m_peak_indices=m_peak_indices_all, m_peak=m_peak_indices_all) # Backwards compat

    print("Saving system parameters...")
    params_filename = os.path.join(base_dir, 'system_params.npz')
    np.savez(params_filename,
             Nt=Nt, Nr=Nr, M=M, Ns=Ns, fc=fc, f_scs=f_scs, Delta_T=Delta_T,
             D_rayleigh=D_rayleigh, K=K, d=d, lambda_c=lambda_c, D=D,
             fm_list=fm_list, f0=f0, BW=BW, L=L,
             theta_min_deg=theta_min_deg, theta_max_deg=theta_max_deg,
             r_min=r_min, r_max=r_max, min_speed=min_speed, max_speed=max_speed,
             random_trajectory_flag=random_trajectory_flag,
             sample_num=sample_num, samples_per_chunk=chunk_size,
             min_angle_diff=min_angle_diff # Save this for reference
            )

    print(f"Latest experiment path: {folders['root']}")
    return folders['root']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate moving target data (Auto Pipeline)')
    parser.add_argument('--samples', type=int, default=5000, help='Number of samples')
    parser.add_argument('--chunk', type=int, default=500, help='Chunk size')
    parser.add_argument('--name', type=str, default='moving_target', help='Base experiment name')
    parser.add_argument('--random', type=int, default=0, choices=[0, 1], help='Random mode')
    # <--- 新增参数 --->
    parser.add_argument('--min_angle_diff', type=float, default=10.0, help='Minimum angle difference')
    parser.add_argument('--num_targets', type=int, default=3, help='Number of targets (K)')
    
    args = parser.parse_args()

    main(sample_num=args.samples,
         chunk_size=args.chunk,
         experiment_name=args.name,
         random_trajectory_flag=args.random,
         min_angle_diff=args.min_angle_diff,
         num_targets=args.num_targets)