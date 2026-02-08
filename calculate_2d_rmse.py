# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import os
import glob
import math
import sys
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================= CONFIG =================
NUM_TEST_SAMPLES = 7500
BATCH_SIZE = 32  # Optimized Batch Size
K_TARGETS_LIST = [3]
PT_LIST = [45, 50,55,60]
# Note: The folder auto_pipeline_k3_d1 contains min_angle_diff=1.5
TRAIN_D_TARGET = 1.0                                                                                                                                                                                 
TEST_D_LIST = [1.0, 1.5,3, 5,10]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Center frequency for MUSIC velocity steering (use latest requested value)
FC_HZ = 220.5e9
# Toggle to enable/disable additive noise in evaluation
NOISE_ENABLED = True
# ==========================================

# --- 1. Dataset & Model Classes (Copied from CFARNet.py) ---

class ChunkedEchoDataset(Dataset):
    def __init__(self, data_root, start_idx, end_idx, expected_k):
        super().__init__()
        self.data_root = data_root; self.start_idx = start_idx; self.end_idx = end_idx
        self.num_samples = end_idx - start_idx + 1; self.expected_k = expected_k
        self.echoes_dir = os.path.join(data_root, 'echoes')
        
        try:
            params = np.load(os.path.join(data_root, 'system_params.npz'))
            self.chunk_size = int(params.get('samples_per_chunk', 500))
            
            traj = np.load(os.path.join(data_root, 'trajectory_data.npz'))
            # Support both naming conventions
            m_peak_all = traj['m_peak_indices'] if 'm_peak_indices' in traj else traj['m_peak']
            theta_all = traj['theta_traj']
            r_all = traj['r_traj']
            vr_all = traj['vr']

            if self.end_idx >= m_peak_all.shape[0]: self.end_idx = m_peak_all.shape[0] - 1
            
            # Slice Data
            self.m_peak_targets = m_peak_all[self.start_idx : self.end_idx + 1]
            self.theta_targets = theta_all[self.start_idx : self.end_idx + 1]
            self.r_targets = r_all[self.start_idx : self.end_idx + 1]
            self.vr_targets = vr_all[self.start_idx : self.end_idx + 1]
            
            # Handle GT Averaging (if stored as [Ns, K])
            if self.theta_targets.ndim == 3:
                self.theta_targets = np.mean(self.theta_targets, axis=1)
                self.r_targets = np.mean(self.r_targets, axis=1)
                self.vr_targets = np.mean(self.vr_targets, axis=1)

        except Exception as e: raise IOError(f"Dataset Init Failed: {e}")

    def __len__(self): return self.num_samples
    def __getitem__(self, index):
        abs_idx = self.start_idx + index
        chunk_idx = abs_idx // self.chunk_size
        idx_in_chunk = abs_idx % self.chunk_size
        try:
            # Added mmap_mode='r' and debug print
            # print(f"[DEBUG] Loading chunk {chunk_idx}", flush=True) 
            echo_chunk = np.load(os.path.join(self.echoes_dir, f'echo_chunk_{chunk_idx}.npy'), mmap_mode='r')
            arr = np.ascontiguousarray(echo_chunk[idx_in_chunk].astype(np.complex64))
            return {
                'echo': torch.from_numpy(arr),
                'theta': self.theta_targets[index],
                'r': self.r_targets[index],
                'vr': self.vr_targets[index]
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

# --- 2. Helpers ---

def calculate_angle_from_m(m_idx, f_scs, BW, f0, phi_start=-60, phi_end=60):
    fm_calc = m_idx * f_scs
    denom = BW * (fm_calc + f0)
    if abs(denom) < 1e-9: return 0.0
    term1 = ((BW - fm_calc) * f0 / denom) * np.sin(np.deg2rad(phi_start))
    term2 = ((BW + f0) * fm_calc / denom) * np.sin(np.deg2rad(phi_end))
    arcsin_arg = np.clip(term1 + term2, -1.0, 1.0)
    return np.rad2deg(np.arcsin(arcsin_arg))


def noise_std_and_scale(pt_dbm, BW):
    """Return noise_std and scale_factor used by both YOLO and CFARNet."""
    noise_pow = 1.38e-23 * 290 * BW * 1000
    noise_std = math.sqrt(noise_pow / 2)
    if not NOISE_ENABLED:
        noise_std = 0.0
    scale_factor = math.sqrt(10**(pt_dbm/10))
    return noise_std, scale_factor

def run_music(y_sample, m_peak, M, f_scs, f0):
    """
    Vectorized MUSIC implementation for speed.
    """
    win = 10
    m_start = max(0, int(m_peak) - win)
    m_end = min(M, int(m_peak) + win)
    y_sub = y_sample[:, m_start:m_end+1]
    Ns, K_sub = y_sub.shape
    if K_sub < 2: return np.nan, np.nan

    # --- Velocity Estimation ---
    # R_t shape: (Ns, Ns)
    R_t = y_sub @ y_sub.conj().T / K_sub
    try:
        # eigh returns eigenvalues in ascending order
        _, V_t = np.linalg.eigh(R_t)
        # Noise subspace (all but last eigenvector) - Assuming 1 target per peak
        Un_t = V_t[:, :-1] 

        # High-precision vectorized search (latest parameters)
        v_grid = np.linspace(-10.5, 10.5, 1001)
        t_vec = np.arange(Ns) * (1.0 / f_scs)

        # Use configured center frequency for velocity steering
        fm = f0 + (int(m_peak) * f_scs)
        scale_v = 4 * np.pi * fm / 3e8
        phase_mat = np.outer(t_vec, scale_v * v_grid)
        steer_mat = np.exp(1j * phase_mat)

        proj = Un_t.conj().T @ steer_mat
        denom = np.sum(np.abs(proj)**2, axis=0)
        v_est = v_grid[np.argmin(denom)]
    except: v_est = np.nan

    # --- Range Estimation ---
    # R_f shape: (K_sub, K_sub)
    R_f = y_sub.T @ y_sub.conj() / Ns
    try:
        _, V_f = np.linalg.eigh(R_f)
        Un_f = V_f[:, :-1]

        # High-precision range grid (latest parameters)
        r_grid = np.arange(34.5, 100.5, 0.01)
        f_vec = np.arange(K_sub) * f_scs

        scale_r = -4 * np.pi / 3e8
        phase_mat = np.outer(f_vec, scale_r * r_grid)
        steer_mat = np.exp(1j * phase_mat)

        proj = Un_f.conj().T @ steer_mat
        denom = np.sum(np.abs(proj)**2, axis=0)
        r_est = r_grid[np.argmin(denom)]
    except: r_est = np.nan

    return r_est, v_est

def compute_metrics(est_targets, true_targets):
    """
    est_targets: [K_est, 3] (angle, range, velocity)
    true_targets: [K_true, 3]
    Returns list of 2D errors and Angle errors for matched pairs.
    """
    if len(est_targets) == 0 or len(true_targets) == 0:
        return [], []

    # Match based on Angle (same as original)
    cost_matrix = np.abs(est_targets[:, 0:1] - true_targets[:, 0:1].T)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    errors_2d = []
    errors_angle = []
    errors_range = []
    errors_velocity = []
    
    for r, c in zip(row_ind, col_ind):
        # Pred
        ang_p_deg = est_targets[r, 0]
        ang_p = np.deg2rad(ang_p_deg)
        rng_p = est_targets[r, 1]
        vel_p = est_targets[r, 2]
        
        # GT
        ang_t_deg = true_targets[c, 0]
        ang_t = np.deg2rad(ang_t_deg)
        rng_t = true_targets[c, 1]
        vel_t = true_targets[c, 2]
        
        # Angle Error (deg)
        errors_angle.append(np.abs(ang_p_deg - ang_t_deg))
        
        # Range Error
        errors_range.append(np.abs(rng_p - rng_t))
        
        # Velocity Error
        errors_velocity.append(np.abs(vel_p - vel_t))

        # 2D Error
        if np.isnan(rng_p): continue
        x_p = rng_p * np.cos(ang_p)
        y_p = rng_p * np.sin(ang_p)
        
        x_t = rng_t * np.cos(ang_t)
        y_t = rng_t * np.sin(ang_t)
        
        dist = np.sqrt((x_p - x_t)**2 + (y_p - y_t)**2)
        errors_2d.append(dist)
        
    return errors_2d, errors_angle, errors_range, errors_velocity

# --- 3. Evaluation Functions ---

def evaluate_yolo(data_dir, pt_dbm, sys_params, k_targets):
    M = int(sys_params['M']); Ns = int(sys_params['Ns'])
    BW = float(sys_params['BW']); f_scs = float(sys_params['f_scs'])
    f0 = float(sys_params['f0'])
    
    # YOLO Config (use latest parameters)
    GUARD_DOPPLER, REF_DOPPLER = 2, 4
    GUARD_ANGLE, REF_ANGLE = 1, 4
    CFAR_ALPHA = 0.8
    MUSIC_EXCL = 80
    LOCAL_MAX_WINDOW = 1

    ds = ChunkedEchoDataset(data_dir, 0, NUM_TEST_SAMPLES-1, k_targets)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    noise_std, scale_factor = noise_std_and_scale(pt_dbm, BW)

    all_2d_errors = []
    all_angle_errors = []
    all_range_errors = []
    all_velocity_errors = []

    # Pre-calculate CFAR kernel with inner guard zeros
    outer_d = 2 * (REF_DOPPLER + GUARD_DOPPLER) + 1
    outer_a = 2 * (REF_ANGLE + GUARD_ANGLE) + 1
    kernel_cfar = torch.ones((1, 1, outer_d, outer_a), device=DEVICE)
    inner_d = 2 * GUARD_DOPPLER + 1
    inner_a = 2 * GUARD_ANGLE + 1
    start_d = (outer_d - inner_d) // 2
    start_a = (outer_a - inner_a) // 2
    kernel_cfar[:, :, start_d:start_d+inner_d, start_a:start_a+inner_a] = 0
    n_norm = kernel_cfar.sum()

    print(f"[DEBUG] Starting YOLO Loop. Loader Length: {len(loader)}", flush=True)

    for i, batch in enumerate(tqdm(loader, desc=f"YOLO K={k_targets} Pt={pt_dbm}", leave=False)):
        echo = batch['echo'].to(DEVICE)
        gt_th_batch = batch['theta'].numpy()
        gt_r_batch = batch['r'].numpy()
        gt_v_batch = batch['vr'].numpy()

        B_size = echo.shape[0]

        # Add Noise & FFT
        noise = (torch.randn_like(echo.real) + 1j*torch.randn_like(echo.imag)) * noise_std
        y_noisy_gpu = echo * scale_factor + noise

        y_fft = torch.fft.fft(y_noisy_gpu, dim=1)
        G_AD_batch = torch.abs(torch.fft.fftshift(y_fft, dim=1))

        img = G_AD_batch.unsqueeze(1)
        pad_d = (outer_d - 1) // 2
        pad_a = (outer_a - 1) // 2
        img_padded = F.pad(img, (pad_a, pad_a, pad_d, pad_d), mode='replicate')

        noise_sum = F.conv2d(img_padded, kernel_cfar)
        threshold = (noise_sum / n_norm) * CFAR_ALPHA
        mask_batch = (img > threshold)

        # Edge suppression
        cut_d = REF_DOPPLER + GUARD_DOPPLER
        cut_a = REF_ANGLE + GUARD_ANGLE
        mask_batch[:, :, :cut_d, :] = False
        mask_batch[:, :, -cut_d:, :] = False
        mask_batch[:, :, :, :cut_a] = False
        mask_batch[:, :, :, -cut_a:] = False

        # NMS (local max)
        nms_win = 2 * LOCAL_MAX_WINDOW + 1
        pad_nms = LOCAL_MAX_WINDOW
        G_max = F.max_pool2d(img, kernel_size=nms_win, stride=1, padding=pad_nms)
        is_peak = (img == G_max) & mask_batch

        peak_coords = torch.nonzero(is_peak, as_tuple=False)

        y_noisy_np = y_noisy_gpu.cpu().numpy()
        G_AD_batch_cpu = G_AD_batch.cpu()

        for b in range(B_size):
            b_mask = (peak_coords[:, 0] == b)
            sample_peaks = peak_coords[b_mask]

            if sample_peaks.shape[0] == 0:
                picks = []
            else:
                d_indices = sample_peaks[:, 2]
                m_indices = sample_peaks[:, 3]
                powers = G_AD_batch[b, d_indices, m_indices]
                sorted_vals, sorted_idx = torch.sort(powers, descending=True)
                sorted_d = d_indices[sorted_idx]
                sorted_m = m_indices[sorted_idx]

                picks = []
                is_suppressed = torch.zeros(len(sorted_vals), dtype=torch.bool, device=DEVICE)

                for k in range(len(sorted_vals)):
                    if is_suppressed[k]:
                        continue
                    picks.append((sorted_d[k].item(), sorted_m[k].item()))
                    if len(picks) >= k_targets:
                        break
                    diff_m = torch.abs(sorted_m - sorted_m[k])
                    suppress_mask = (diff_m < MUSIC_EXCL)
                    is_suppressed = is_suppressed | suppress_mask

            # MUSIC & Metrics
            y_sample = y_noisy_np[b]
            est_targets = []
            for (d_idx, m_idx) in picks:
                ang = calculate_angle_from_m(m_idx, f_scs, BW, f0)
                r, v = run_music(y_sample, m_idx, M, f_scs, f0)
                est_targets.append([ang, r, v])

            est_targets = np.array(est_targets)

            gt_th = gt_th_batch[b]
            gt_r = gt_r_batch[b]
            gt_v = gt_v_batch[b]
            true_targets = np.column_stack((gt_th, gt_r, gt_v))

            errs_2d, errs_ang, errs_rng, errs_vel = compute_metrics(est_targets, true_targets)
            all_2d_errors.extend(errs_2d)
            all_angle_errors.extend(errs_ang)
            all_range_errors.extend(errs_rng)
            all_velocity_errors.extend(errs_vel)

    return np.array(all_2d_errors), np.array(all_angle_errors), np.array(all_range_errors), np.array(all_velocity_errors)

def evaluate_cfarnet(data_dir, model_path, pt_dbm, sys_params, k_targets):
    M = int(sys_params['M']); Ns = int(sys_params['Ns'])
    BW = float(sys_params['BW']); f_scs = float(sys_params['f_scs'])
    f0 = float(sys_params['f0'])
    
    model = IndexPredictionCNN(M+1, Ns).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    ds = ChunkedEchoDataset(data_dir, 0, NUM_TEST_SAMPLES-1, k_targets)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # Use same noise model and scaling as YOLO
    noise_std, scale_factor = noise_std_and_scale(pt_dbm, BW)
    
    all_range_errors = []
    all_velocity_errors = []
    all_2d_errors = []
    all_angle_errors = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"CFARNet K={k_targets} Pt={pt_dbm}", leave=False)):
            echo = batch['echo'].to(DEVICE)
            # GT Data
            gt_th_batch = batch['theta'].numpy()
            gt_r_batch = batch['r'].numpy()
            gt_v_batch = batch['vr'].numpy()
            
            noise = (torch.randn_like(echo.real) + 1j*torch.randn_like(echo.imag)) * noise_std
            input_sig = echo * scale_factor + noise
            
            logits = model(input_sig)
            probs = torch.sigmoid(logits)
            _, topk = torch.topk(probs, k_targets)
            pred_indices_batch = topk.cpu().numpy() # [B, K]
            
            # --- Per Sample Processing ---
            input_np_batch = input_sig.cpu().numpy()
            B_size = echo.shape[0]

            for b in range(B_size):
                input_np = input_np_batch[b]
                pred_indices = pred_indices_batch[b]
                
                est_targets = []
                for m_idx in pred_indices:
                    ang = calculate_angle_from_m(m_idx, f_scs, BW, f0)
                    r, v = run_music(input_np, m_idx, M, f_scs, f0)
                    est_targets.append([ang, r, v])
                
                est_targets = np.array(est_targets)
                gt_th = gt_th_batch[b]
                gt_r = gt_r_batch[b]
                gt_v = gt_v_batch[b]
                true_targets = np.column_stack((gt_th, gt_r, gt_v))
                
                errs_2d, errs_ang, errs_rng, errs_vel = compute_metrics(est_targets, true_targets)
                all_2d_errors.extend(errs_2d)
                all_angle_errors.extend(errs_ang)
                all_range_errors.extend(errs_rng)
                all_velocity_errors.extend(errs_vel)
            
    return np.array(all_2d_errors), np.array(all_angle_errors), np.array(all_range_errors), np.array(all_velocity_errors)

# --- 4. Main ---

def main():
    print("Searching for datasets...")
    # Use script directory as base to handle CWD differences
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, "data")
    output_root = os.path.join(script_dir, "bce0122") # Updated to correct output folder
    
    results_file = os.path.join(script_dir, "results_2d_rmse.txt")
    with open(results_file, "w") as f:
        f.write("K | Dataset_D | Pt(dBm) | Model | RMSE_2D | 90%_2D | 95%_2D | RMSE_Ang | 90%_Ang | 95%_Ang | RMSE_Rng | 90%_Rng | 95%_Rng | RMSE_Vel | 90%_Vel | 95%_Vel\n")
        f.write("-" * 150 + "\n")

    for k_val in K_TARGETS_LIST:
        print(f"\n{'='*40}")
        print(f"Processing K={k_val}")
        print(f"{'='*40}")

        # Find Train Dataset (D=1.5) for this K
        train_dir = None
        candidates = glob.glob(os.path.join(data_root, f"auto_pipeline_k{k_val}_d*"))
        for d in candidates:
            try:
                p = np.load(os.path.join(d, "system_params.npz"))
                if int(p['K']) == k_val and abs(float(p['min_angle_diff']) - TRAIN_D_TARGET) < 0.1:
                    train_dir = d
                    break
            except: continue
            
        if not train_dir:
            print(f"[Error] Could not find training dataset for K={k_val}, D={TRAIN_D_TARGET}")
            continue
        print(f"Found Train Dir: {train_dir}")
        
        # Find Test Datasets for this K
        test_dirs = []
        for d in candidates:
            try:
                p = np.load(os.path.join(d, "system_params.npz"))
                diff = float(p['min_angle_diff'])
                if int(p['K']) == k_val and any(abs(diff - td) < 0.1 for td in TEST_D_LIST):
                    test_dirs.append((d, diff))
            except: continue
            
        print(f"Found {len(test_dirs)} Test Dirs: {[t[1] for t in test_dirs]}")

        for test_dir, d_val in test_dirs:
            print(f"\n  Processing D={d_val} ({test_dir})")
            
            # Load Params
            sys_params = np.load(os.path.join(test_dir, "system_params.npz"))
            
            for pt in PT_LIST:
                # 1. YOLO
                print(f"    > Running YOLO Pt={pt}...")
                errs_yolo_2d, errs_yolo_ang, errs_yolo_rng, errs_yolo_vel = evaluate_yolo(test_dir, pt, sys_params, k_val)
                if len(errs_yolo_2d) > 0:
                    rmse_2d = np.sqrt(np.mean(errs_yolo_2d**2))
                    p90_2d = np.percentile(errs_yolo_2d, 90)
                    p95_2d = np.percentile(errs_yolo_2d, 95)
                    
                    rmse_ang = np.sqrt(np.mean(errs_yolo_ang**2))
                    p90_ang = np.percentile(errs_yolo_ang, 90)
                    p95_ang = np.percentile(errs_yolo_ang, 95)

                    rmse_rng = np.sqrt(np.mean(errs_yolo_rng**2))
                    p90_rng = np.percentile(errs_yolo_rng, 90)
                    p95_rng = np.percentile(errs_yolo_rng, 95)

                    rmse_vel = np.sqrt(np.mean(errs_yolo_vel**2))
                    p90_vel = np.percentile(errs_yolo_vel, 90)
                    p95_vel = np.percentile(errs_yolo_vel, 95)
                    
                    line = f"{k_val:<2} | {d_val:<4} | {pt:<7} | YOLO  | {rmse_2d:.4f}  | {p90_2d:.4f} | {p95_2d:.4f} | {rmse_ang:.4f}   | {p90_ang:.4f}  | {p95_ang:.4f} | {rmse_rng:.4f}  | {p90_rng:.4f} | {p95_rng:.4f} | {rmse_vel:.4f}  | {p90_vel:.4f} | {p95_vel:.4f}"
                    print(line)
                    with open(results_file, "a") as f: f.write(line + "\n")
                
                # 2. CFARNet
                # model_path = os.path.join(train_dir, f"experiment_pt{int(pt)}", f"model_pt{int(pt)}_best.pt")
                train_dataset_name = os.path.basename(train_dir)
                model_path = os.path.join(output_root, train_dataset_name, f"pt{int(pt)}", f"model_pt{int(pt)}_best.pt")
                
                if os.path.exists(model_path):
                    print(f"    > Running CFARNet Pt={pt}...")
                    errs_cfar_2d, errs_cfar_ang, errs_cfar_rng, errs_cfar_vel = evaluate_cfarnet(test_dir, model_path, pt, sys_params, k_val)
                    if len(errs_cfar_2d) > 0:
                        rmse_2d = np.sqrt(np.mean(errs_cfar_2d**2))
                        p90_2d = np.percentile(errs_cfar_2d, 90)
                        p95_2d = np.percentile(errs_cfar_2d, 95)
                        
                        rmse_ang = np.sqrt(np.mean(errs_cfar_ang**2))
                        p90_ang = np.percentile(errs_cfar_ang, 90)
                        p95_ang = np.percentile(errs_cfar_ang, 95)

                        rmse_rng = np.sqrt(np.mean(errs_cfar_rng**2))
                        p90_rng = np.percentile(errs_cfar_rng, 90)
                        p95_rng = np.percentile(errs_cfar_rng, 95)

                        rmse_vel = np.sqrt(np.mean(errs_cfar_vel**2))
                        p90_vel = np.percentile(errs_cfar_vel, 90)
                        p95_vel = np.percentile(errs_cfar_vel, 95)
                        
                        line = f"{k_val:<2} | {d_val:<4} | {pt:<7} | CFAR  | {rmse_2d:.4f}  | {p90_2d:.4f} | {p95_2d:.4f} | {rmse_ang:.4f}   | {p90_ang:.4f}  | {p95_ang:.4f} | {rmse_rng:.4f}  | {p90_rng:.4f} | {p95_rng:.4f} | {rmse_vel:.4f}  | {p90_vel:.4f} | {p95_vel:.4f}"
                        print(line)
                        with open(results_file, "a") as f: f.write(line + "\n")
                else:
                    print(f"    [Warning] Model not found: {model_path}")

    # After all evaluations, plot 90th-percentile metrics per Pt for each delta and model
    try:
        plot_dir = os.path.join(script_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        # Use dynamic PT list from config and auto-assign line styles per delta
        pt_plot = PT_LIST
        deltas = TEST_D_LIST
        models = ['YOLO', 'CFAR']
        style_list = ['-', '--', ':', '-.']
        linestyles = {float(d): style_list[i % len(style_list)] for i, d in enumerate(deltas)}
        colors = {'YOLO': 'tab:blue', 'CFAR': 'tab:red'}

        # Parse results file
        data = {m: {float(d): {pt: np.nan for pt in pt_plot} for d in deltas} for m in models}
        with open(results_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('K |') or line.startswith('-'):
                    continue
                parts = [p.strip() for p in line.split('|')]
                try:
                    k = int(parts[0]); d = float(parts[1]); pt = int(parts[2]); model = parts[3]
                    vals = [float(v) for v in parts[4:]]
                    # vals indices: 0:RMSE_2D,1:90%_2D,2:95%_2D,3:RMSE_Ang,4:90%_Ang,5:95%_Ang,6:RMSE_Rng,7:90%_Rng,8:95%_Rng,9:RMSE_Vel,10:90%_Vel,11:95%_Vel
                    p90_ang = vals[4]
                    p90_rng = vals[7]
                    p90_vel = vals[10]
                    if model in models and d in deltas and pt in pt_plot:
                        data[model][d][pt] = {'90%_Ang': p90_ang, '90%_Rng': p90_rng, '90%_Vel': p90_vel}
                except Exception:
                    continue

        def build_series(model, delta, metric):
            ys = []
            for pt in pt_plot:
                entry = data.get(model, {}).get(delta, {}).get(pt, np.nan)
                if isinstance(entry, dict): ys.append(entry.get(metric, np.nan))
                else: ys.append(np.nan)
            return np.array(ys)

        # Plot each metric
        for metric, fname, ylabel in [ ('90%_Ang', '90th_angle.png', '90th percentile angle error (°)'),
                                       ('90%_Rng', '90th_range.png', '90th percentile range error (m)'),
                                       ('90%_Vel', '90th_velocity.png', '90th percentile velocity error (m/s)')]:
            plt.figure(figsize=(6,4))
            for d in deltas:
                for model in models:
                    ys = build_series(model, float(d), metric)
                    label = f"{model} (Δφ = {d:.1f}°)"
                    plt.plot(pt_plot, ys, label=label, linestyle=linestyles.get(float(d), '-'), color=colors[model], marker='o')
            plt.xlabel('Transmit Power Pt (dBm)')
            plt.ylabel(ylabel)
            plt.xticks(pt_plot)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(fontsize=8)
            outpath = os.path.join(plot_dir, fname)
            plt.tight_layout()
            plt.savefig(outpath, dpi=150)
            plt.close()
            print(f"Wrote {outpath}")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
