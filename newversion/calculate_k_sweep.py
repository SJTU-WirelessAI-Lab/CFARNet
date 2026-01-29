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

# ================= CONFIG =================
NUM_TEST_SAMPLES = 7500
BATCH_SIZE = 1 # Keep user config
K_TARGETS_LIST = [1, 2, 3, 4, 5]
PT_LIST = [50]
# We want to test on the "d1" dataset for each K.
RESULTS_FILENAME = "results_k_sweep.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                # Match CFARNet.py: use t=0
                self.theta_targets = self.theta_targets[:, 0, :]
                self.r_targets = self.r_targets[:, 0, :]
                self.vr_targets = self.vr_targets[:, 0, :]

        except Exception as e: raise IOError(f"Dataset Init Failed: {e}")

    def __len__(self): return self.num_samples
    def __getitem__(self, index):
        abs_idx = self.start_idx + index
        chunk_idx = abs_idx // self.chunk_size
        idx_in_chunk = abs_idx % self.chunk_size
        try:
            echo_chunk = np.load(os.path.join(self.echoes_dir, f'echo_chunk_{chunk_idx}.npy'), mmap_mode='r')
            return {
                'echo': torch.from_numpy(echo_chunk[idx_in_chunk]).to(torch.complex64),
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

        # Vectorized Search
        v_grid = np.linspace(-15, 15, 400)
        t_vec = np.arange(Ns) * (1/f_scs)
        
        # Steering Matrix: (Ns, 400)
        # Phase = 4*pi*f0/c * t * v
        scale_v = 4 * np.pi * f0 / 3e8
        phase_mat = np.outer(t_vec, scale_v * v_grid)
        steer_mat = np.exp(1j * phase_mat) # (Ns, 400)
        
        # Project: |U^H @ a(v)|^2
        proj = Un_t.conj().T @ steer_mat
        denom = np.sum(np.abs(proj)**2, axis=0) # (400,)
        
        v_est = v_grid[np.argmin(denom)]
    except: v_est = np.nan

    # --- Range Estimation ---
    # R_f shape: (K_sub, K_sub)
    R_f = y_sub.T @ y_sub.conj() / Ns
    try:
        _, V_f = np.linalg.eigh(R_f)
        Un_f = V_f[:, :-1]
        
        r_grid = np.linspace(10, 100, 400)
        f_vec = np.arange(K_sub) * f_scs
        
        # Steering Matrix: (K_sub, 400)
        # Phase = -4*pi/c * f * r
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
    
    # YOLO Config
    GUARD_DOPPLER, REF_DOPPLER = 2, 4
    GUARD_ANGLE, REF_ANGLE = 1, 4
    CFAR_ALPHA = 1.5
    MUSIC_EXCL = 12
    
    ds = ChunkedEchoDataset(data_dir, 0, NUM_TEST_SAMPLES-1, k_targets)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    noise_pow = 1.38e-23 * 290 * BW * 1000
    noise_std = math.sqrt(noise_pow / 2)
    scale_factor = math.sqrt(10**(pt_dbm/10))
    
    all_2d_errors = []
    all_angle_errors = []
    all_range_errors = []
    all_velocity_errors = []

    # Pre-calculate Kernels for CFAR Conv2d
    outer_d = 2 * (REF_DOPPLER + GUARD_DOPPLER) + 1
    outer_a = 2 * (REF_ANGLE + GUARD_ANGLE) + 1
    kernel_outer = torch.ones((1, 1, outer_d, outer_a), device=DEVICE)
    
    inner_d = 2 * GUARD_DOPPLER + 1
    inner_a = 2 * GUARD_ANGLE + 1
    kernel_inner = torch.ones((1, 1, inner_d, inner_a), device=DEVICE)
    
    n_norm = (outer_d * outer_a) - (inner_d * inner_a)
    
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
        
        # CFAR
        img = G_AD_batch.unsqueeze(1)
        pad_d, pad_a = (REF_DOPPLER + GUARD_DOPPLER), (REF_ANGLE + GUARD_ANGLE)
        sum_outer = F.conv2d(img, kernel_outer, padding=(pad_d, pad_a)) 
        sum_inner = F.conv2d(img, kernel_inner, padding=(GUARD_DOPPLER, GUARD_ANGLE))
        threshold = CFAR_ALPHA * (sum_outer - sum_inner) / n_norm
        mask_batch = (img > threshold) 
        
        # Edge Cutting
        cut_d, cut_a = pad_d, pad_a
        mask_batch[:, :, :cut_d, :] = False
        mask_batch[:, :, -cut_d:, :] = False
        mask_batch[:, :, :, :cut_a] = False
        mask_batch[:, :, :, -cut_a:] = False

        # Process Peaks (CPU)
        mask_np = mask_batch.squeeze(1).cpu().numpy()
        G_AD_np = G_AD_batch.cpu().numpy()
        y_noisy_np = y_noisy_gpu.cpu().numpy()
        
        for b in range(B_size):
            mask = mask_np[b]
            G_AD = G_AD_np[b]
            y_sample = y_noisy_np[b]
            
            # === [Updated Logic] Forced K Selection with NMS ===
            picks = [] # [(d, m), ...]
            
            def is_conflict(new_m, current_picks):
                for (_, pm) in current_picks:
                    if abs(new_m - pm) < MUSIC_EXCL:
                        return True
                return False
            
            # 1. Select from CFAR Candidates
            cands = np.argwhere(mask)
            if len(cands) > 0:
                d_c, m_c = cands[:,0], cands[:,1]
                pwr = G_AD[d_c, m_c]
                sort_idx = np.argsort(-pwr)
                
                for pid in sort_idx:
                    curr_d, curr_m = d_c[pid], m_c[pid]
                    if not is_conflict(curr_m, picks):
                        picks.append((curr_d, curr_m))
                    if len(picks) >= k_targets: break
            
            # 2. Fallback (Forced Completion)
            if len(picks) < k_targets:
                flat_indices = np.argsort(-G_AD.ravel())
                for flat_idx in flat_indices:
                    if len(picks) >= k_targets: break
                    
                    d_idx, m_idx = np.unravel_index(flat_idx, G_AD.shape)
                    if not is_conflict(m_idx, picks):
                        picks.append((d_idx, m_idx))
            
            # Run MUSIC
            est_targets = []
            for (d_idx, m_idx) in picks:
                ang = calculate_angle_from_m(m_idx, f_scs, BW, f0)
                r, v = run_music(y_sample, m_idx, M, f_scs, f0)
                est_targets.append([ang, r, v])
                
            est_targets = np.array(est_targets)
            
            # GT & Metrics
            gt_th = gt_th_batch[b]
            gt_r = gt_r_batch[b]
            gt_v = gt_v_batch[b]
            # Handle variable K in GT (if batching > 1, this needs care, but here B=1 or manual loop handles it)
            # If K varies per sample, gt_* might be padded. 
            # But here we assume dataset is homogeneous for k_targets (ChunkedEchoDataset expects expected_k)
            # Or gt_* is [K] shape.
            
            # Safely stacking GT
            # Ensure dims match
            if np.ndim(gt_th) == 0: gt_th = np.array([gt_th])
            if np.ndim(gt_r) == 0: gt_r = np.array([gt_r])
            if np.ndim(gt_v) == 0: gt_v = np.array([gt_v])
            
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
    
    noise_pow = 1.38e-23 * 290 * BW * 1000
    noise_std = math.sqrt(noise_pow / 2)
    scale_factor = math.sqrt(10**(pt_dbm/10))
    
    all_range_errors = []
    all_velocity_errors = []
    all_2d_errors = []
    all_angle_errors = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"CFARNet K={k_targets} Pt={pt_dbm}", leave=False)):
            echo = batch['echo'].to(DEVICE)
            gt_th_batch = batch['theta'].numpy()
            gt_r_batch = batch['r'].numpy()
            gt_v_batch = batch['vr'].numpy()
            
            noise = (torch.randn_like(echo.real) + 1j*torch.randn_like(echo.imag)) * noise_std
            input_sig = echo * scale_factor + noise
            
            logits = model(input_sig)
            probs = torch.sigmoid(logits)
            
            _, topk = torch.topk(probs, k_targets)
            pred_indices_batch = topk.cpu().numpy() # [B, K]

            input_np_batch = input_sig.cpu().numpy()
            B_size = echo.shape[0]

            for b in range(B_size):
                input_np = input_np_batch[b]
                pred_indices = pred_indices_batch[b]
                
                est_targets = []
                for m_idx in pred_indices:
                    ang = calculate_angle_from_m(m_idx, f_scs, BW, f0)
                    r, v = run_music(input_np, m_idx, int(sys_params['M']), f_scs, f0)
                    est_targets.append([ang, r, v])

                est_targets = np.array(est_targets)
                
                # Handle GT shapes for K=1 case where numpy might squeeze
                gt_th = gt_th_batch[b]
                gt_r = gt_r_batch[b]
                gt_v = gt_v_batch[b]
                
                if np.ndim(gt_th) == 0: gt_th = np.array([gt_th])
                if np.ndim(gt_r) == 0: gt_r = np.array([gt_r])
                if np.ndim(gt_v) == 0: gt_v = np.array([gt_v])

                true_targets = np.stack([gt_th, gt_r, gt_v], axis=1)
                
                errs_2d, errs_ang, errs_rng, errs_vel = compute_metrics(est_targets, true_targets)
                all_2d_errors.extend(errs_2d)
                all_angle_errors.extend(errs_ang)
                all_range_errors.extend(errs_rng)
                all_velocity_errors.extend(errs_vel)
            
    return np.array(all_2d_errors), np.array(all_angle_errors), np.array(all_range_errors), np.array(all_velocity_errors)

# --- 4. Main ---

def main():
    print("Starting K-Sweep Evaluation...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, "data")
    output_root = os.path.join(script_dir, "bce0112")
    results_file = os.path.join(script_dir, RESULTS_FILENAME)
    
    with open(results_file, "w") as f:
        f.write("K | Pt(dBm) | Model | RMSE_2D | 90%_2D | 95%_2D | RMSE_Ang | 90%_Ang | 95%_Ang | RMSE_Rng | 90%_Rng | 95%_Rng | RMSE_Vel | 90%_Vel | 95%_Vel\n")
        f.write("-" * 150 + "\n")

    for k_val in K_TARGETS_LIST:
        print(f"\n{'='*40}")
        print(f"Processing K={k_val}")
        print(f"{'='*40}")

        # Find Dataset matching "d1" and K
        candidates = glob.glob(os.path.join(data_root, f"auto_pipeline_k{k_val}_d*"))
        
        target_dir = None
        for d in candidates:
            folder_name = os.path.basename(d)
            if f"_d1_" in folder_name:
                target_dir = d
                break
        
        if not target_dir:
            for d in candidates:
                try:
                    p = np.load(os.path.join(d, "system_params.npz"))
                    if int(p['K']) == k_val and float(p['min_angle_diff']) <= 1.5:
                         target_dir = d
                         break
                except: continue

        if not target_dir:
            print(f"[Error] No D1 dataset found for K={k_val}")
            continue
            
        print(f"Using Dataset: {target_dir}")
        sys_params = np.load(os.path.join(target_dir, "system_params.npz"))
        
        for pt in PT_LIST:
            # 1. YOLO
            errs_yolo_2d, errs_yolo_ang, errs_yolo_rng, errs_yolo_vel = evaluate_yolo(target_dir, pt, sys_params, k_val)
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
                
                line = f"{k_val:<2} | {pt:<7} | YOLO  | {rmse_2d:.4f}  | {p90_2d:.4f} | {p95_2d:.4f} | {rmse_ang:.4f}   | {p90_ang:.4f}  | {p95_ang:.4f} | {rmse_rng:.4f}  | {p90_rng:.4f} | {p95_rng:.4f} | {rmse_vel:.4f}  | {p90_vel:.4f} | {p95_vel:.4f}"
                print(line)
                with open(results_file, "a") as f: f.write(line + "\n")
            
            # 2. CFARNet
            train_dataset_name = os.path.basename(target_dir)
            model_path = os.path.join(output_root, train_dataset_name, f"pt{int(pt)}", f"model_pt{int(pt)}_best.pt")
            
            if os.path.exists(model_path):
                print(f"    > Running CFARNet Pt={pt}...")
                errs_cfar_2d, errs_cfar_ang, errs_cfar_rng, errs_cfar_vel = evaluate_cfarnet(target_dir, model_path, pt, sys_params, k_val)
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
                    
                    line = f"{k_val:<2} | {pt:<7} | CFAR  | {rmse_2d:.4f}  | {p90_2d:.4f} | {p95_2d:.4f} | {rmse_ang:.4f}   | {p90_ang:.4f}  | {p95_ang:.4f} | {rmse_rng:.4f}  | {p90_rng:.4f} | {p95_rng:.4f} | {rmse_vel:.4f}  | {p90_vel:.4f} | {p95_vel:.4f}"
                    print(line)
                    with open(results_file, "a") as f: f.write(line + "\n")
            else:
                print(f"    [Warning] Model not found: {model_path}")

if __name__ == "__main__":
    main()