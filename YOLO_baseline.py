# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import argparse
from torch.utils.data import Dataset, DataLoader
import time
import math
import datetime
import traceback
import sys
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from itertools import islice
import pandas as pd 

# ==============================================================================
# 0. Initialization and Configuration
# ==============================================================================
print("--- Initialization: Baseline (CFAR+MUSIC) Full Test (Pt + Thermal Noise) ---")

class ChunkedEchoDataset(Dataset):
    def __init__(self, data_root, start_idx, end_idx, expected_k):
        super().__init__()
        self.data_root = data_root
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_samples = end_idx - start_idx + 1
        self.expected_k = expected_k
        
        self.echoes_dir = os.path.join(data_root, 'echoes')
        params_path = os.path.join(data_root, 'system_params.npz')
        params_data = np.load(params_path)
        self.chunk_size = int(params_data['samples_per_chunk'] if 'samples_per_chunk' in params_data else params_data['chunk_size'])
        
        traj_path = os.path.join(data_root, 'trajectory_data.npz')
        traj_data = np.load(traj_path)
        
        # Load full arrays then slice
        m_peak_all = traj_data['m_peak_indices'] if 'm_peak_indices' in traj_data else traj_data['m_peak']
        theta_all = traj_data['theta_traj']
        r_all = traj_data['r_traj']
        vr_all = traj_data['vr']

        # Handle Averaging
        self.gt_needs_averaging = (theta_all.ndim == 3)

        # Safety Truncate
        total_samples = m_peak_all.shape[0]
        if self.end_idx >= total_samples:
            self.end_idx = total_samples - 1
            self.num_samples = self.end_idx - self.start_idx + 1

        # Slicing
        self.m_peak_targets = m_peak_all[self.start_idx : self.end_idx + 1]
        self.theta_targets = theta_all[self.start_idx : self.end_idx + 1]
        self.r_targets = r_all[self.start_idx : self.end_idx + 1]
        self.vr_targets = vr_all[self.start_idx : self.end_idx + 1]

    def __len__(self): return self.num_samples
    
    def __getitem__(self, index):
        absolute_idx = self.start_idx + index
        chunk_idx = absolute_idx // self.chunk_size
        idx_in_chunk = absolute_idx % self.chunk_size
        
        echo_path = os.path.join(self.echoes_dir, f'echo_chunk_{chunk_idx}.npy')
        # Note: In production, cache loaded chunks to avoid opening file every time
        echo_chunk = np.load(echo_path) 
        clean_echo = echo_chunk[idx_in_chunk]
        
        sample = {
            'echo': torch.from_numpy(clean_echo).to(torch.complex64),
            'theta': torch.from_numpy(self.theta_targets[index]).to(torch.float32),
            'r': torch.from_numpy(self.r_targets[index]).to(torch.float32),
            'vr': torch.from_numpy(self.vr_targets[index]).to(torch.float32)
        }
        if self.gt_needs_averaging:
            for k in ['theta', 'r', 'vr']: sample[k] = torch.nanmean(sample[k], dim=0)
        return sample

# --- Arguments & Config ---
parser = argparse.ArgumentParser()
parser.add_argument('--cuda_device', type=str, default='cpu')
parser.add_argument('--data_dir', type=str, default=None) 
args = parser.parse_args()

# >>> CONFIGURATION (MODIFIED: Use Pt list instead of SNR list) <<<
pt_dbm_test_list = [-10, 0, 10, 20, 30] 
print(f"Test Transmit Powers (Pt): {pt_dbm_test_list} dBm")

# --- Constants for Noise Calculation ---
K_BOLTZMANN = 1.38e-23
T_NOISE_KELVIN = 290

BATCH_SIZE = 1
GUARD_DOPPLER, REF_DOPPLER = 2, 4
GUARD_ANGLE, REF_ANGLE = 1, 4
CFAR_ALPHA = 1.5
MUSIC_WIN, MUSIC_EXCL = 10, 10
V_SEARCH = np.linspace(-10.5, 10.5, 2001)
R_SEARCH = np.arange(34.5, 200.5, 0.01)
PHI_START, PHI_END = -60, 60

# --- Data Path Setup ---
def get_exp_path():
    return "data/k=3_delta=10" # Placeholder or use args.data_dir

DATA_ROOT = args.data_dir if args.data_dir else get_exp_path()
PARAMS_FILE = os.path.join(DATA_ROOT, 'system_params.npz')

# Load Params
p = np.load(PARAMS_FILE)
M, Ns = int(p['M']), int(p['Ns'])
fc, f_scs = float(p['fc']), float(p['f_scs'])
BW = M * f_scs
_f0_raw = p.get('fm_list', [fc - BW/2])[0]
f0 = float(_f0_raw) if np.ndim(_f0_raw)==0 else float(_f0_raw[0]) if np.ndim(_f0_raw)>0 else float(_f0_raw)
Ts = float(p.get('Delta_T', 1/f_scs))
K_eff = int(p.get('K', 3))
c = 3e8
device = torch.device('cuda' if args.cuda_device != 'cpu' and torch.cuda.is_available() else 'cpu')

# --- Data Loading ---
total_samples_assumed = 50000
test_fraction = 0.15
num_test_data_in_set = int(total_samples_assumed * test_fraction)
test_start_idx = 0
test_end_idx = num_test_data_in_set - 1
num_samples_to_run = num_test_data_in_set

test_dataset = ChunkedEchoDataset(DATA_ROOT, test_start_idx, test_end_idx, expected_k=K_eff)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ==============================================================================
# 2. Main Processing Loop
# ==============================================================================
overall_results = [] 

# >>> MODIFIED LOOP: Iterate over Pt (dBm) <<<
for pt_dbm in pt_dbm_test_list:
    print(f"\n{'='*40}")
    print(f" STARTING TEST FOR Pt = {pt_dbm} dBm")
    print(f"{'='*40}")
    
    # Re-initialize storage for EACH Pt level
    est_th = np.full((num_samples_to_run, K_eff), np.nan)
    est_r = np.full((num_samples_to_run, K_eff), np.nan)
    est_v = np.full((num_samples_to_run, K_eff), np.nan)
    
    true_th = np.full((num_samples_to_run, K_eff), np.nan)
    true_r = np.full((num_samples_to_run, K_eff), np.nan)
    true_v = np.full((num_samples_to_run, K_eff), np.nan)
    
    matched_th = np.full((num_samples_to_run, K_eff), np.nan)
    matched_r = np.full((num_samples_to_run, K_eff), np.nan)
    matched_v = np.full((num_samples_to_run, K_eff), np.nan)

    # >>> Calculate Thermal Noise Standard Deviation (Fixed for current system BW) <<<
    # Noise Power (Watts) = k * T * B
    # Convert to mW by multiplying by 1000.0
    noise_pwr_mw = K_BOLTZMANN * T_NOISE_KELVIN * BW * 1000.0
    # For complex noise, variance is split between real and imag parts: sigma^2 = P_noise / 2
    noise_std = math.sqrt(noise_pwr_mw / 2.0)
    
    # >>> Calculate Signal Scaling Factor based on Pt <<<
    pt_mw = 10**(pt_dbm / 10.0)
    pt_scale_factor = math.sqrt(pt_mw)

    iter_loader = islice(test_loader, num_samples_to_run)
    pbar = tqdm(enumerate(iter_loader), total=num_samples_to_run, desc=f"Pt={pt_dbm}dBm", leave=True)
    
    for idx, batch in pbar:
        try:
            clean_echo = batch['echo'].to(device) 
            gt_th = batch['theta'].numpy(); gt_r = batch['r'].numpy(); gt_v = batch['vr'].numpy()
        except Exception as e:
            print(f"Batch error: {e}")
            continue
        
        true_th[idx] = gt_th[0]; true_r[idx] = gt_r[0]; true_v[idx] = gt_v[0]

        # --- MODIFIED: Add Noise using Pt and Thermal Noise ---
        # 1. Scale clean echo by sqrt(Pt)
        scaled_echo = clean_echo * pt_scale_factor
        
        # 2. Generate thermal noise (fixed intensity based on BW/Temp)
        noise_tensor = (torch.randn_like(scaled_echo.real) + 1j * torch.randn_like(scaled_echo.imag)) * noise_std
        
        # 3. Add
        y_noisy = scaled_echo + noise_tensor
        
        # ======================================================================
        # ALGORITHM: CFAR + MUSIC
        # ======================================================================
        Y_np = y_noisy[0].cpu().numpy() 
        G_AD = np.abs(np.fft.fftshift(np.fft.fft(Y_np, axis=0), axes=0))
        Ns_dim, M_dim = G_AD.shape
        
        # 1. CFAR
        mask = np.zeros_like(G_AD, dtype=bool)
        d_rng = range(REF_DOPPLER+GUARD_DOPPLER, Ns_dim-(REF_DOPPLER+GUARD_DOPPLER))
        m_rng = range(REF_ANGLE+GUARD_ANGLE, M_dim-(REF_ANGLE+GUARD_ANGLE))
        
        for d in d_rng:
            for m in m_rng:
                d_start, d_end = d-(REF_DOPPLER+GUARD_DOPPLER), d+(REF_DOPPLER+GUARD_DOPPLER)+1
                m_start, m_end = m-(REF_ANGLE+GUARD_ANGLE), m+(REF_ANGLE+GUARD_ANGLE)+1
                win = G_AD[d_start:d_end, m_start:m_end]
                
                grd_d_s, grd_d_e = d-GUARD_DOPPLER, d+GUARD_DOPPLER+1
                grd_m_s, grd_m_e = m-GUARD_ANGLE, m+GUARD_ANGLE+1
                grd = G_AD[grd_d_s:grd_d_e, grd_m_s:grd_m_e]
                
                noise_lvl = (np.sum(win) - np.sum(grd)) / (win.size - grd.size + 1e-10)
                if G_AD[d,m] > CFAR_ALPHA * noise_lvl: mask[d,m] = True

        cands = np.argwhere(mask)
        
        if len(cands) > 0:
            # NMS
            d_c, m_c = cands[:,0], cands[:,1]
            pwr = G_AD[d_c, m_c]
            keep = np.zeros(len(d_c), bool)
            valid_nms = (d_c >= 1) & (d_c < Ns_dim-1) & (m_c >= 1) & (m_c < M_dim-1)
            
            for i in np.where(valid_nms)[0]:
                sub = G_AD[d_c[i]-1:d_c[i]+2, m_c[i]-1:m_c[i]+2]
                if pwr[i] >= np.max(sub): keep[i] = True
            
            d_final, m_final = d_c[keep], m_c[keep]
            pwr_final = pwr[keep]
            
            # Pick Top K
            sort_idx = np.argsort(-pwr_final)
            picks = []
            for i in sort_idx:
                curr = m_final[i]
                if not any(abs(curr - m_final[p]) < MUSIC_EXCL for p in picks):
                    picks.append(i)
                if len(picks) >= K_eff: break
            
            # Param Estimation
            for k_i, p_idx in enumerate(picks):
                m_k = m_final[p_idx]
                
                # Angle
                fm = m_k * f_scs
                den = BW * (fm + fc)
                ang = np.nan
                if abs(den) > 1e-9:
                    t1 = ((BW - fm)*fc/den) * np.sin(np.deg2rad(PHI_START))
                    t2 = ((BW + fc)*fm/den) * np.sin(np.deg2rad(PHI_END))
                    val = np.clip(t1+t2, -1, 1)
                    ang = np.rad2deg(np.arcsin(val))
                est_th[idx, k_i] = ang
                
                # MUSIC Prep
                m_lo, m_hi = max(0, m_k-MUSIC_WIN), min(M, m_k+MUSIC_WIN)
                Y_sub_raw = Y_np[:, m_lo:m_hi+1]
                
                if Y_sub_raw.shape[1] >= 2:
                    Y_sub = Y_sub_raw / (np.abs(Y_sub_raw) + 1e-10)
                    current_win = Y_sub.shape[1]
                    current_Ns = Y_sub.shape[0]
                    rank_sig = 1
                    
                    # Velocity MUSIC
                    try:
                        R_D = (Y_sub @ Y_sub.conj().T) / current_win
                        eig, U = np.linalg.eigh(R_D)
                        U_noise = U[:, :max(0, current_Ns - rank_sig)]
                        
                        av_all = np.exp(1j * (4 * np.pi * f0 * Ts / 3e8) * np.outer(V_SEARCH, np.arange(current_Ns)))
                        spec = 1.0 / (np.sum(np.abs(av_all.conj() @ U_noise)**2, axis=1) + 1e-10)
                        est_v[idx, k_i] = V_SEARCH[np.argmax(spec)]
                    except: pass
                    
                    # Range MUSIC
                    try:
                        R_R = (Y_sub.T @ Y_sub.conj()) / current_Ns
                        eig, U = np.linalg.eigh(R_R)
                        U_noise = U[:, :max(0, current_win - rank_sig)]
                        
                        ar_all = np.exp(-1j * (4 * np.pi * f_scs / 3e8) * np.outer(R_SEARCH, np.arange(current_win)))
                        spec = 1.0 / (np.sum(np.abs(ar_all.conj() @ U_noise)**2, axis=1) + 1e-10)
                        est_r[idx, k_i] = R_SEARCH[np.argmax(spec)]
                    except: pass

        # --- Hungarian Matching ---
        v_est = np.where(~np.isnan(est_th[idx]))[0]
        v_tru = np.where(~np.isnan(true_th[idx]))[0]
        
        if len(v_est) > 0 and len(v_tru) > 0:
            cost = np.abs(est_th[idx][v_est, None] - true_th[idx][None, v_tru])
            r_ind, c_ind = linear_sum_assignment(cost)
            for r, col_idx in zip(r_ind, c_ind):
                t_idx = v_tru[col_idx]
                e_idx = v_est[r]
                matched_th[idx, t_idx] = est_th[idx, e_idx]
                matched_r[idx, t_idx] = est_r[idx, e_idx]
                matched_v[idx, t_idx] = est_v[idx, e_idx]

    # --- Metrics Calculation (RMSE & Percentiles) ---
    err_th = np.abs(matched_th - true_th).flatten()
    err_r = np.abs(matched_r - true_r).flatten()
    err_v = np.abs(matched_v - true_v).flatten()
    
    valid_th = err_th[~np.isnan(err_th)]
    valid_r = err_r[~np.isnan(err_r)]
    valid_v = err_v[~np.isnan(err_v)]
    
    # Helper for metrics
    def get_stats(errors):
        if len(errors) == 0: return np.nan, np.nan, np.nan
        rmse = np.sqrt(np.mean(errors**2))
        p90 = np.percentile(errors, 90)
        p95 = np.percentile(errors, 95)
        return rmse, p90, p95

    rmse_th, p90_th, p95_th = get_stats(valid_th)
    rmse_r, p90_r, p95_r = get_stats(valid_r)
    rmse_v, p90_v, p95_v = get_stats(valid_v)
    
    res = {
        "Pt(dBm)": pt_dbm,  # Modified Key
        "RMSE_Angle": rmse_th, "P90_Angle": p90_th, "P95_Angle": p95_th,
        "RMSE_Range": rmse_r,  "P90_Range": p90_r,  "P95_Range": p95_r,
        "RMSE_Vel":   rmse_v,  "P90_Vel":   p90_v,  "P95_Vel":   p95_v
    }
    overall_results.append(res)
    
    print(f"Finished Pt={pt_dbm}dBm:")
    print(f"  > Angle: RMSE={rmse_th:.4f}, P95={p95_th:.4f}")
    print(f"  > Range: RMSE={rmse_r:.4f}, P95={p95_r:.4f}")

# ==============================================================================
# 3. Final Grand Summary
# ==============================================================================
print("\n" + "="*100)
print("FINAL PERFORMANCE SUMMARY (ALL Pt Levels)")
print("="*100)
df_res = pd.DataFrame(overall_results)
# Columns in specific order
cols = [
    "Pt(dBm)", 
    "RMSE_Angle", "P90_Angle", "P95_Angle",
    "RMSE_Range", "P90_Range", "P95_Range",
    "RMSE_Vel",   "P90_Vel",   "P95_Vel"
]
print(df_res[cols].to_string(index=False, float_format="%.4f"))
print("="*100)