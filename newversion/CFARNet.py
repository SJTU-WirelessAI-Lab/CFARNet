# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import sys
import math
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# --- 1. Dataset & Model Definition (需与训练保持一致) ---

class ChunkedEchoDataset(Dataset):
    def __init__(self, data_root, start_idx, end_idx, expected_k):
        super().__init__()
        self.data_root = data_root; self.start_idx = start_idx; self.end_idx = end_idx
        self.num_samples = end_idx - start_idx + 1; self.expected_k = expected_k
        self.echoes_dir = os.path.join(data_root, 'echoes')
        
        try:
            params = np.load(os.path.join(data_root, 'system_params.npz'))
            self.chunk_size = int(params.get('samples_per_chunk', 500))
            self.M_plus_1 = int(params['M']) + 1
            self.Ns = int(params['Ns'])
            
            traj = np.load(os.path.join(data_root, 'trajectory_data.npz'))
            m_peak_all = traj.get('m_peak_indices', traj.get('m_peak'))
            if self.end_idx >= m_peak_all.shape[0]: self.end_idx = m_peak_all.shape[0] - 1
            self.m_peak_targets = m_peak_all[self.start_idx : self.end_idx + 1]
            
            # Pad or Truncate K
            k_in = self.m_peak_targets.shape[1] if self.m_peak_targets.ndim > 1 else 1
            if k_in < self.expected_k:
                self.m_peak_targets = np.pad(self.m_peak_targets, ((0,0), (0, self.expected_k - k_in)), constant_values=-1)
            elif k_in > self.expected_k:
                self.m_peak_targets = self.m_peak_targets[:, :self.expected_k]
                
        except Exception as e: raise IOError(f"Dataset Init Failed: {e}")

    def __len__(self): return self.num_samples
    def __getitem__(self, index):
        abs_idx = self.start_idx + index
        chunk_idx = abs_idx // self.chunk_size
        idx_in_chunk = abs_idx % self.chunk_size
        try:
            echo_chunk = np.load(os.path.join(self.echoes_dir, f'echo_chunk_{chunk_idx}.npy'))
            return {
                'echo': torch.from_numpy(echo_chunk[idx_in_chunk]).to(torch.complex64),
                'm_peak': torch.from_numpy(self.m_peak_targets[index]).to(torch.long)
            }
        except Exception as e: print(f"Load Error {index}: {e}"); raise

class IndexPredictionCNN(nn.Module):
    def __init__(self, M_plus_1, Ns, hidden_dim=512, dropout=0.2):
        super().__init__()
        C = 512
        self.feat = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 3, (2,1), 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.1), nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 3, (2,1), 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.1), nn.Dropout2d(0.1),
            nn.Conv2d(256, C, 3, (2,1), 1, bias=False), nn.BatchNorm2d(C), nn.LeakyReLU(0.1), nn.Dropout2d(0.1),
        )
        self.pred = nn.Sequential(
            nn.Conv1d(C, hidden_dim, 3, 1, 1, bias=False), nn.BatchNorm1d(hidden_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim//2, 3, 1, 1, bias=False), nn.BatchNorm1d(hidden_dim//2), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim//2, 1, 1)
        )
    def forward(self, x):
        x = torch.log1p(torch.abs(torch.fft.fftshift(torch.fft.fft(x, dim=1), dim=1))).unsqueeze(1)
        x = self.feat(x)
        x = torch.max(x, dim=2)[0]
        return self.pred(x).squeeze(1)

# --- 2. Physics & Algorithm Helpers ---

def calculate_angle_from_m(m_idx, f_scs, BW, f0, phi_start=-60, phi_end=60):
    """Reverses the Rainbow Beam mapping to estimate Angle from Frequency Index."""
    # Logic matches generate_data.py
    fm = f0 + m_idx * f_scs
    
    # Approx logic based on simplified linear relation or the specific formula used in gen
    # This formula needs to match the one in generate_data.py exactly to be accurate.
    # Assuming standard mapping:
    # We solve for theta in: f_m = f0 + ... (Rainbow beam formula)
    # Re-using the logic from the generator's `calculate_angle_for_m`
    
    # Reconstruct terms
    fm_calc = m_idx * f_scs
    denom = BW * (fm_calc + f0)
    if abs(denom) < 1e-9: return 0.0
    
    term1 = ((BW - fm_calc) * f0 / denom) * np.sin(np.deg2rad(phi_start))
    term2 = ((BW + f0) * fm_calc / denom) * np.sin(np.deg2rad(phi_end))
    
    arcsin_arg = np.clip(term1 + term2, -1.0, 1.0)
    return np.rad2deg(np.arcsin(arcsin_arg))

def run_music(y_sample, m_peak, M, f_scs, f0):
    """
    Runs simplistic 1D MUSIC for Range and Velocity on the sub-carriers around m_peak.
    """
    win = 10 # Window size around peak
    m_start = max(0, int(m_peak) - win)
    m_end = min(M, int(m_peak) + win)
    y_sub = y_sample[:, m_start:m_end+1] # [Ns, Subcarriers]
    
    Ns, K_sub = y_sub.shape
    if K_sub < 2: return np.nan, np.nan

    # --- Velocity Estimation (Temporal MUSIC) ---
    # R_t: [Ns, Ns]
    R_t = y_sub @ y_sub.T.conj() / K_sub
    try:
        _, V_t = np.linalg.eigh(R_t)
        Un_t = V_t[:, :-1] # Noise subspace (Rank 1 signal assumption)
        
        # Grid search for Velocity
        v_grid = np.linspace(-15, 15, 400) # m/s
        P_v = []
        t_vec = np.arange(Ns) * (1/f_scs) # Symbol duration is 1/f_scs roughly
        for v in v_grid:
            # Doppler steering vector: exp(j * 4pi * f0 * v * t / c)
            steer = np.exp(1j * 4 * np.pi * f0 * v / 3e8 * t_vec)
            # Projection
            denom = (np.abs(steer.conj() @ Un_t)**2).sum()
            P_v.append(1.0 / (denom + 1e-9))
        v_est = v_grid[np.argmax(P_v)]
    except:
        v_est = np.nan

    # --- Range Estimation (Frequency MUSIC on sub-band) ---
    # R_f: [K_sub, K_sub]
    R_f = y_sub.T @ y_sub.conj() / Ns
    try:
        _, V_f = np.linalg.eigh(R_f)
        Un_f = V_f[:, :-1]
        
        # Grid search for Range
        r_grid = np.linspace(10, 100, 400) # m
        P_r = []
        f_vec = np.arange(K_sub) * f_scs # Relative freq in window
        for r in r_grid:
            # Range steering: exp(-j * 4pi * f * r / c)
            steer = np.exp(-1j * 4 * np.pi * r / 3e8 * f_vec)
            denom = (np.abs(steer.conj() @ Un_f)**2).sum()
            P_r.append(1.0 / (denom + 1e-9))
        r_est = r_grid[np.argmax(P_r)]
    except:
        r_est = np.nan

    return r_est, v_est

# --- 3. Main Testing Logic ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True, help="Path to .pt weights")
    parser.add_argument('--save_dir', type=str, required=True, help="Where to save log/npz")
    parser.add_argument('--pt_dbm', type=float, required=True)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--max_targets', type=int, required=True) # K
    parser.add_argument('--num_test_samples', type=int, default=1000)
    parser.add_argument('--test_set_mode', type=str, default='last', choices=['first', 'last'])
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    
    # 1. Load System Parameters
    try:
        sys_params = np.load(os.path.join(args.data_dir, 'system_params.npz'))
        M = int(sys_params['M']); Ns = int(sys_params['Ns'])
        BW = float(sys_params['BW']); f_scs = float(sys_params['f_scs'])
        f0 = float(sys_params['f0'])
    except Exception as e:
        print(f"Error loading params: {e}")
        return

    # 2. Load Model
    model = IndexPredictionCNN(M+1, Ns).to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"[CFARNet] Loaded weights: {args.model_path}")
    else:
        print(f"[Error] Model not found: {args.model_path}")
        return
    model.eval()

    # 3. Load Test Data
    traj = np.load(os.path.join(args.data_dir, 'trajectory_data.npz'))
    total_samples = traj['m_peak_indices'].shape[0]
    
    if args.test_set_mode == 'first':
        start_idx = 0
        end_idx = args.num_test_samples - 1
        print(f"[CFARNet] Test Mode: FIRST {args.num_test_samples} samples.")
    else:
        start_idx = max(0, total_samples - args.num_test_samples)
        end_idx = total_samples - 1
        print(f"[CFARNet] Test Mode: LAST {args.num_test_samples} samples.")
    
    test_ds = ChunkedEchoDataset(args.data_dir, start_idx, end_idx, args.max_targets)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    print(f"[CFARNet] Testing {len(test_ds)} samples at Pt={args.pt_dbm}dBm...")

    # GT Data for matching
    gt_theta = traj['theta_traj'][start_idx:]
    gt_r = traj['r_traj'][start_idx:]
    gt_v = traj['vr'][start_idx:]

    # Noise Config
    noise_pow = 1.38e-23 * 290 * BW * 1000
    noise_std = math.sqrt(noise_pow / 2)
    scale_factor = math.sqrt(10**(args.pt_dbm/10))

    # Metric Containers
    err_angle_list = []
    err_range_list = []
    err_velo_list = []
    err_2d_list = []

    # 4. Testing Loop
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=min(len(test_ds), args.num_test_samples)):
            if i >= args.num_test_samples: break
            
            # Prepare Input
            echo = batch['echo'].to(device) # [1, Ns, M]
            
            # Add Noise
            noise = (torch.randn_like(echo.real) + 1j*torch.randn_like(echo.imag)) * noise_std
            input_sig = echo * scale_factor + noise
            
            # Inference
            logits = model(input_sig)
            probs = torch.sigmoid(logits)
            
            # Get Top-K Indices
            _, topk = torch.topk(probs, args.max_targets)
            pred_indices = topk.cpu().numpy()[0] # [K]
            
            # Estimate Params for each predicted peak
            est_targets = [] # List of [ang, r, v]
            input_np = input_sig.cpu().numpy()[0] # [Ns, M]
            
            for m_idx in pred_indices:
                # Angle from Formula
                ang = calculate_angle_from_m(m_idx, f_scs, BW, f0)
                # Range/Vel from MUSIC
                r, v = run_music(input_np, m_idx, int(sys_params['M']), f_scs, f0)
                est_targets.append([ang, r, v])
            
            est_targets = np.array(est_targets) # [K_pred, 3]
            
            # Get GT for this sample (at t=0)
            # gt_theta shape: [Samples, Ns, K] -> [Ns, K]
            # We compare with the first timestamp or average? Usually t=0 for estimation.
            # Using t=0
            curr_gt_theta = gt_theta[i, 0, :] # [K]
            curr_gt_r = gt_r[i, 0, :]
            curr_gt_v = gt_v[i, 0, :]
            
            true_targets = np.column_stack((curr_gt_theta, curr_gt_r, curr_gt_v)) # [K_true, 3]
            
            # Hungarian Matching based on ANGLE
            # Cost matrix: abs diff of angles
            # est: [K, 1], true: [1, K] -> [K, K]
            cost_matrix = np.abs(est_targets[:, 0:1] - true_targets[:, 0:1].T)
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Collect Errors
            for r, c in zip(row_ind, col_ind):
                # Angle Error
                err_a = np.abs(est_targets[r, 0] - true_targets[c, 0])
                err_angle_list.append(err_a)
                
                # Range Error (only if valid)
                if not np.isnan(est_targets[r, 1]):
                    err_r = np.abs(est_targets[r, 1] - true_targets[c, 1])
                    err_range_list.append(err_r)
                    
                # Velocity Error (only if valid)
                if not np.isnan(est_targets[r, 2]):
                    err_v = np.abs(est_targets[r, 2] - true_targets[c, 2])
                    err_velo_list.append(err_v)
                
                # 2D Position Error
                if not np.isnan(est_targets[r, 1]):
                    ang_pred = est_targets[r, 0]
                    r_pred = est_targets[r, 1]
                    ang_true = true_targets[c, 0]
                    r_true = true_targets[c, 1]
                    
                    rad_p = np.deg2rad(ang_pred); rad_t = np.deg2rad(ang_true)
                    x_p, y_p = r_pred * np.cos(rad_p), r_pred * np.sin(rad_p)
                    x_t, y_t = r_true * np.cos(rad_t), r_true * np.sin(rad_t)
                    err_2d = np.sqrt((x_p - x_t)**2 + (y_p - y_t)**2)
                    err_2d_list.append(err_2d)
    # 5. Statistics & Logging
    def calc_stats(errors):
        if not errors: return 0.0, 0.0, 0.0
        arr = np.array(errors)
        rmse = np.sqrt(np.mean(arr**2))
        p90 = np.percentile(arr, 90)
        p95 = np.percentile(arr, 95)
        return rmse, p90, p95

    rmse_a, p90_a, p95_a = calc_stats(err_angle_list)
    rmse_r, p90_r, p95_r = calc_stats(err_range_list)
    rmse_v, p90_v, p95_v = calc_stats(err_velo_list)
    rmse_2d, p90_2d, p95_2d = calc_stats(err_2d_list)

    log_file = os.path.join(args.save_dir, f"log_test_cfarnet_pt{int(args.pt_dbm)}.txt")
    
    log_content = [
        f"=== CFARNet Test Results (Pt={args.pt_dbm}dBm) ===",
        f"Samples: {len(test_ds)} | K: {args.max_targets}",
        f"Angle (deg) : RMSE={rmse_a:.4f} | 90%={p90_a:.4f} | 95%={p95_a:.4f}",
        f"Range (m)   : RMSE={rmse_r:.4f} | 90%={p90_r:.4f} | 95%={p95_r:.4f}",
        f"Velocity(m/s): RMSE={rmse_v:.4f} | 90%={p90_v:.4f} | 95%={p95_v:.4f}",
        f"2D Pos (m)  : RMSE={rmse_2d:.4f} | 90%={p90_2d:.4f} | 95%={p95_2d:.4f}",
        f"=================================================="
    ]
    
    print("\n".join(log_content))
    
    with open(log_file, 'w') as f:
        f.write("\n".join(log_content))

    # 6. Save Errors to NPZ
    npz_file = os.path.join(args.save_dir, f"errors_cfarnet_pt{int(args.pt_dbm)}.npz")
    np.savez(npz_file, 
             err_angle=err_angle_list,
             err_2d=err_2d_list,
             err_range=err_range_list, 
             err_velo=err_velo_list,
             pt_dbm=args.pt_dbm)
    
    print(f"[CFARNet] Results saved to {args.save_dir}")

if __name__ == "__main__":
    main()