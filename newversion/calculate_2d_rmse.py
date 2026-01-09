# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import os
import glob
import math
import sys
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# ================= CONFIG =================
NUM_TEST_SAMPLES = 7500
BATCH_SIZE = 1
K_TARGETS_LIST = [1, 2, 3, 4, 5] # 支持列表
PT_LIST = [45, 50, 55, 60]
TRAIN_D_TARGET = 1.5
TEST_D_LIST = [1.5, 3, 5, 10]
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
            echo_chunk = np.load(os.path.join(self.echoes_dir, f'echo_chunk_{chunk_idx}.npy'))
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
    win = 10
    m_start = max(0, int(m_peak) - win)
    m_end = min(M, int(m_peak) + win)
    y_sub = y_sample[:, m_start:m_end+1]
    Ns, K_sub = y_sub.shape
    if K_sub < 2: return np.nan, np.nan

    # Velocity
    R_t = y_sub @ y_sub.T.conj() / K_sub
    try:
        _, V_t = np.linalg.eigh(R_t)
        Un_t = V_t[:, :-1]
        v_grid = np.linspace(-15, 15, 400)
        t_vec = np.arange(Ns) * (1/f_scs)
        P_v = []
        for v in v_grid:
            steer = np.exp(1j * 4 * np.pi * f0 * v / 3e8 * t_vec)
            denom = (np.abs(steer.conj() @ Un_t)**2).sum()
            P_v.append(1.0 / (denom + 1e-9))
        v_est = v_grid[np.argmax(P_v)]
    except: v_est = np.nan

    # Range
    R_f = y_sub.T @ y_sub.conj() / Ns
    try:
        _, V_f = np.linalg.eigh(R_f)
        Un_f = V_f[:, :-1]
        r_grid = np.linspace(10, 100, 400)
        f_vec = np.arange(K_sub) * f_scs
        P_r = []
        for r in r_grid:
            steer = np.exp(-1j * 4 * np.pi * r / 3e8 * f_vec)
            denom = (np.abs(steer.conj() @ Un_f)**2).sum()
            P_r.append(1.0 / (denom + 1e-9))
        r_est = r_grid[np.argmax(P_r)]
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
    
    for r, c in zip(row_ind, col_ind):
        # Pred
        ang_p_deg = est_targets[r, 0]
        ang_p = np.deg2rad(ang_p_deg)
        rng_p = est_targets[r, 1]
        
        # GT
        ang_t_deg = true_targets[c, 0]
        ang_t = np.deg2rad(ang_t_deg)
        rng_t = true_targets[c, 1]
        
        # Angle Error (deg)
        errors_angle.append(np.abs(ang_p_deg - ang_t_deg))

        # 2D Error
        if np.isnan(rng_p): continue
        x_p = rng_p * np.cos(ang_p)
        y_p = rng_p * np.sin(ang_p)
        
        x_t = rng_t * np.cos(ang_t)
        y_t = rng_t * np.sin(ang_t)
        
        dist = np.sqrt((x_p - x_t)**2 + (y_p - y_t)**2)
        errors_2d.append(dist)
        
    return errors_2d, errors_angle

# --- 3. Evaluation Functions ---

def evaluate_yolo(data_dir, pt_dbm, sys_params, k_targets):
    M = int(sys_params['M']); Ns = int(sys_params['Ns'])
    BW = float(sys_params['BW']); f_scs = float(sys_params['f_scs'])
    f0 = float(sys_params['f0'])
    
    # YOLO Config
    GUARD_DOPPLER, REF_DOPPLER = 2, 4
    GUARD_ANGLE, REF_ANGLE = 1, 4
    CFAR_ALPHA = 1.5
    MUSIC_EXCL = 20
    
    ds = ChunkedEchoDataset(data_dir, 0, NUM_TEST_SAMPLES-1, k_targets)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    noise_pow = 1.38e-23 * 290 * BW * 1000
    noise_std = math.sqrt(noise_pow / 2)
    scale_factor = math.sqrt(10**(pt_dbm/10))
    
    all_2d_errors = []
    all_angle_errors = []
    
    for i, batch in enumerate(tqdm(loader, desc=f"YOLO K={k_targets} Pt={pt_dbm}", leave=False)):
        echo = batch['echo'].to(DEVICE)
        gt_th = batch['theta'].numpy()[0]
        gt_r = batch['r'].numpy()[0]
        gt_v = batch['vr'].numpy()[0]
        
        # Add Noise
        noise = (torch.randn_like(echo.real) + 1j*torch.randn_like(echo.imag)) * noise_std
        y_noisy = (echo * scale_factor + noise).cpu().numpy()[0] # [Ns, M]
        
        # Range-Doppler
        G_AD = np.abs(np.fft.fftshift(np.fft.fft(y_noisy, axis=0), axes=0))
        Ns_dim, M_dim = G_AD.shape
        
        # CFAR
        mask = np.zeros_like(G_AD, dtype=bool)
        for d in range(REF_DOPPLER+GUARD_DOPPLER, Ns_dim-(REF_DOPPLER+GUARD_DOPPLER)):
            for m in range(REF_ANGLE+GUARD_ANGLE, M_dim-(REF_ANGLE+GUARD_ANGLE)):
                if G_AD[d,m] < 1e-6: continue
                win = G_AD[d-(REF_DOPPLER+GUARD_DOPPLER):d+(REF_DOPPLER+GUARD_DOPPLER)+1, 
                           m-(REF_ANGLE+GUARD_ANGLE):m+(REF_ANGLE+GUARD_ANGLE)+1]
                grd = G_AD[d-GUARD_DOPPLER:d+GUARD_DOPPLER+1, 
                           m-GUARD_ANGLE:m+GUARD_ANGLE+1]
                noise_lvl = (np.sum(win) - np.sum(grd)) / (win.size - grd.size + 1e-10)
                if G_AD[d,m] > CFAR_ALPHA * noise_lvl: mask[d,m] = True
                
        cands = np.argwhere(mask)
        est_targets = []
        
        if len(cands) > 0:
            d_c, m_c = cands[:,0], cands[:,1]
            pwr = G_AD[d_c, m_c]
            keep = np.ones(len(d_c), dtype=bool)
            sort_idx = np.argsort(-pwr)
            picks = []
            for pid in sort_idx:
                if not keep[pid]: continue
                picks.append(pid)
                curr_m = m_c[pid]
                for other_pid in sort_idx:
                    if keep[other_pid] and abs(m_c[other_pid] - curr_m) < MUSIC_EXCL and other_pid != pid:
                        keep[other_pid] = False
                if len(picks) >= k_targets: break
            
            for pid in picks:
                m_k = m_c[pid]
                ang = calculate_angle_from_m(m_k, f_scs, BW, f0)
                r, v = run_music(y_noisy, m_k, M, f_scs, f0)
                est_targets.append([ang, r, v])
                
        est_targets = np.array(est_targets)
        true_targets = np.column_stack((gt_th, gt_r, gt_v))
        
        errs_2d, errs_ang = compute_metrics(est_targets, true_targets)
        all_2d_errors.extend(errs_2d)
        all_angle_errors.extend(errs_ang)
        
    return np.array(all_2d_errors), np.array(all_angle_errors)

def evaluate_cfarnet(data_dir, model_path, pt_dbm, sys_params, k_targets):
    M = int(sys_params['M']); Ns = int(sys_params['Ns'])
    BW = float(sys_params['BW']); f_scs = float(sys_params['f_scs'])
    f0 = float(sys_params['f0'])
    
    model = IndexPredictionCNN(M+1, Ns).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    ds = ChunkedEchoDataset(data_dir, 0, NUM_TEST_SAMPLES-1, k_targets)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    noise_pow = 1.38e-23 * 290 * BW * 1000
    noise_std = math.sqrt(noise_pow / 2)
    scale_factor = math.sqrt(10**(pt_dbm/10))
    
    all_2d_errors = []
    all_angle_errors = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"CFARNet K={k_targets} Pt={pt_dbm}", leave=False)):
            echo = batch['echo'].to(DEVICE)
            gt_th = batch['theta'].numpy()[0]
            gt_r = batch['r'].numpy()[0]
            gt_v = batch['vr'].numpy()[0]
            
            noise = (torch.randn_like(echo.real) + 1j*torch.randn_like(echo.imag)) * noise_std
            input_sig = echo * scale_factor + noise
            
            logits = model(input_sig)
            probs = torch.sigmoid(logits)
            _, topk = torch.topk(probs, k_targets)
            pred_indices = topk.cpu().numpy()[0]
            
            est_targets = []
            input_np = input_sig.cpu().numpy()[0]
            for m_idx in pred_indices:
                ang = calculate_angle_from_m(m_idx, f_scs, BW, f0)
                r, v = run_music(input_np, m_idx, M, f_scs, f0)
                est_targets.append([ang, r, v])
            
            est_targets = np.array(est_targets)
            true_targets = np.column_stack((gt_th, gt_r, gt_v))
            
            errs_2d, errs_ang = compute_metrics(est_targets, true_targets)
            all_2d_errors.extend(errs_2d)
            all_angle_errors.extend(errs_ang)
            
    return np.array(all_2d_errors), np.array(all_angle_errors)

# --- 4. Main ---

def main():
    print("Searching for datasets...")
    data_root = "data"
    
    results_file = "results_2d_rmse.txt"
    with open(results_file, "w") as f:
        f.write("K | Dataset_D | Pt(dBm) | Model | RMSE_2D | 90%_2D | 95%_2D | RMSE_Ang | 90%_Ang | 95%_Ang\n")
        f.write("-" * 100 + "\n")

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
                errs_yolo_2d, errs_yolo_ang = evaluate_yolo(test_dir, pt, sys_params, k_val)
                if len(errs_yolo_2d) > 0:
                    rmse_2d = np.sqrt(np.mean(errs_yolo_2d**2))
                    p90_2d = np.percentile(errs_yolo_2d, 90)
                    p95_2d = np.percentile(errs_yolo_2d, 95)
                    
                    rmse_ang = np.sqrt(np.mean(errs_yolo_ang**2))
                    p90_ang = np.percentile(errs_yolo_ang, 90)
                    p95_ang = np.percentile(errs_yolo_ang, 95)
                    
                    line = f"{k_val:<2} | {d_val:<4} | {pt:<7} | YOLO  | {rmse_2d:.4f}  | {p90_2d:.4f} | {p95_2d:.4f} | {rmse_ang:.4f}   | {p90_ang:.4f}  | {p95_ang:.4f}"
                    print(line)
                    with open(results_file, "a") as f: f.write(line + "\n")
                
                # 2. CFARNet
                model_path = os.path.join(train_dir, f"experiment_pt{int(pt)}", f"model_pt{int(pt)}_best.pt")
                if os.path.exists(model_path):
                    print(f"    > Running CFARNet Pt={pt}...")
                    errs_cfar_2d, errs_cfar_ang = evaluate_cfarnet(test_dir, model_path, pt, sys_params, k_val)
                    if len(errs_cfar_2d) > 0:
                        rmse_2d = np.sqrt(np.mean(errs_cfar_2d**2))
                        p90_2d = np.percentile(errs_cfar_2d, 90)
                        p95_2d = np.percentile(errs_cfar_2d, 95)
                        
                        rmse_ang = np.sqrt(np.mean(errs_cfar_ang**2))
                        p90_ang = np.percentile(errs_cfar_ang, 90)
                        p95_ang = np.percentile(errs_cfar_ang, 95)
                        
                        line = f"{k_val:<2} | {d_val:<4} | {pt:<7} | CFAR  | {rmse_2d:.4f}  | {p90_2d:.4f} | {p95_2d:.4f} | {rmse_ang:.4f}   | {p90_ang:.4f}  | {p95_ang:.4f}"
                        print(line)
                        with open(results_file, "a") as f: f.write(line + "\n")
                else:
                    print(f"    [Warning] Model not found: {model_path}")

if __name__ == "__main__":
    main()
