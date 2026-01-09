# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
import argparse
import math
import sys
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from scipy.signal import convolve2d
import traceback
from tqdm import tqdm

# --- 1. Dataset Definition (保持一致) ---
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
            vr_all = traj['vr'] # Handle potential naming diffs if needed

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

# --- 2. Main Processing ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="newversion/data/auto_pipeline_k3_d1_20251214_174958")
    parser.add_argument('--save_dir', type=str, default="newversion/results/auto_pipeline_k3_d1_20251214_174958")
    parser.add_argument('--pt_dbm', type=float, default=60.0)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--max_targets', type=int, default=3) # K
    parser.add_argument('--num_test_samples', type=int, default=50)
    parser.add_argument('--test_set_mode', type=str, default='last', choices=['first', 'last'])
    parser.add_argument('--progress_file', type=str, default=None, help="Path to write progress stats")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")

    # Load System Params
    try:
        p = np.load(os.path.join(args.data_dir, 'system_params.npz'))
        M = int(p['M']); Ns = int(p['Ns'])
        fc = float(p['fc']); f_scs = float(p['f_scs'])
        BW = float(p['BW'])
        # Handle f0 loading robustly
        _f0_raw = p.get('fm_list', [fc - BW/2])[0]
        f0 = float(_f0_raw) if np.ndim(_f0_raw)==0 else float(_f0_raw[0]) if np.ndim(_f0_raw)>0 else float(_f0_raw)
        Ts = float(p.get('Delta_T', 1/f_scs))
        
        # Rainbow Beam Angle Config (Usually fixed)
        PHI_START, PHI_END = -60, 60 
    except Exception as e:
        print(f"Error loading params: {e}")
        return

    # Data Loader
    traj = np.load(os.path.join(args.data_dir, 'trajectory_data.npz'))
    total_samples = traj['m_peak_indices'].shape[0] if 'm_peak_indices' in traj else traj['m_peak'].shape[0]
    
    if args.test_set_mode == 'first':
        start_idx = 0
        end_idx = args.num_test_samples - 1
        print(f"[YOLO-Baseline] Test Mode: FIRST {args.num_test_samples} samples.")
    else:
        start_idx = max(0, total_samples - args.num_test_samples)
        end_idx = total_samples - 1
        print(f"[YOLO-Baseline] Test Mode: LAST {args.num_test_samples} samples.")

    test_ds = ChunkedEchoDataset(args.data_dir, start_idx, end_idx, args.max_targets)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    print(f"[YOLO-Baseline] Testing {len(test_ds)} samples at Pt={args.pt_dbm}dBm...")

    # Noise & Config
    K_BOLTZMANN = 1.38e-23; T_NOISE_KELVIN = 290
    noise_pwr = K_BOLTZMANN * T_NOISE_KELVIN * BW * 1000.0
    noise_std = math.sqrt(noise_pwr / 2.0)
    scale_factor = math.sqrt(10**(args.pt_dbm/10))

    # CFAR Config
    GUARD_DOPPLER, REF_DOPPLER = 2, 4
    GUARD_ANGLE, REF_ANGLE = 1, 4
    CFAR_ALPHA = 1.5
    MUSIC_WIN, MUSIC_EXCL = 10, 15
    V_SEARCH = np.linspace(-15, 15, 400) # Reduced grid for speed
    R_SEARCH = np.linspace(10, 100, 400)

    # Metric Storage
    err_angle_list = []
    err_range_list = []
    err_velo_list = []
    err_2d_list = []
    
    stats_fa = 0
    stats_miss = 0
    stats_gt_total = 0

    # Processing Loop
    total_to_process = min(len(test_ds), args.num_test_samples)
    i = 0
    try:
        for i_loop, batch in tqdm(enumerate(test_loader), total=total_to_process):
            i = i_loop
            if i >= args.num_test_samples: break
            
            # Monitor Progress
            if args.progress_file and i % 10 == 0:
                with open(args.progress_file, 'w') as f:
                    f.write(f"{i}/{total_to_process}")
            
            # 1. Prepare Data
            clean_echo = batch['echo'].to(device)
            # GT (batch size is 1)
            gt_th = batch['theta'].numpy()[0]
            gt_r = batch['r'].numpy()[0]
            gt_v = batch['vr'].numpy()[0]
            
            # Add Noise
            scaled_echo = clean_echo * scale_factor
            noise = (torch.randn_like(scaled_echo.real) + 1j * torch.randn_like(scaled_echo.imag)) * noise_std
            y_noisy = scaled_echo + noise
            
            # 2. Algorithm: Range-Doppler Map Generation
            Y_np = y_noisy[0].cpu().numpy() # [Ns, M]
            # FFT along Doppler dim (axis 0)
            # Use Magnitude of FFT (No Log)
            G_AD = np.fft.fftshift(np.fft.fft(Y_np, axis=0), axes=0)
            G_AD = np.abs(G_AD)
            Ns_dim, M_dim = G_AD.shape
            
            # 3. CFAR Detection (Vectorized)
            
            # 1. Total Window Sum
            r_tot_d = REF_DOPPLER + GUARD_DOPPLER
            r_tot_a = REF_ANGLE + GUARD_ANGLE
            kernel_win = np.ones((2*r_tot_d+1, 2*r_tot_a+1))
            
            # 2. Guard Window Sum
            r_grd_d = GUARD_DOPPLER
            r_grd_a = GUARD_ANGLE
            kernel_grd = np.ones((2*r_grd_d+1, 2*r_grd_a+1))
            
            # Pad Guard Kernel to match Win Kernel size for subtraction (center aligned)
            kernel_grd_padded = np.zeros_like(kernel_win)
            start_d = (kernel_win.shape[0] - kernel_grd.shape[0]) // 2
            start_a = (kernel_win.shape[1] - kernel_grd.shape[1]) // 2
            kernel_grd_padded[start_d:start_d+kernel_grd.shape[0], start_a:start_a+kernel_grd.shape[1]] = 1
            
            # Effective Noise Kernel (Win - Guard)
            kernel_noise = kernel_win - kernel_grd_padded
            num_cells = np.sum(kernel_noise)
            
            # Convolve to get noise level
            # Mode='same' keeps output size same as input
            noise_sum_map = convolve2d(G_AD, kernel_noise, mode='same', boundary='symm')
            noise_lvl_map = noise_sum_map / (num_cells + 1e-10)
            
            # Thresholding
            mask = (G_AD > (CFAR_ALPHA * noise_lvl_map)) & (G_AD > 1e-6)
            
            # Remove edges (invalid CFAR zones)
            mask[:r_tot_d, :] = False
            mask[-r_tot_d:, :] = False
            mask[:, :r_tot_a] = False
            mask[:, -r_tot_a:] = False


            cands = np.argwhere(mask)
            est_targets = [] # List of [ang, r, v]

            if len(cands) > 0:
                # 4. NMS & Parameter Estimation
                d_c, m_c = cands[:,0], cands[:,1]
                pwr = G_AD[d_c, m_c]
                keep = np.ones(len(d_c), dtype=bool)
                
                # Simple NMS sort
                sort_idx = np.argsort(-pwr)
                picks = []
                for pid in sort_idx:
                    if not keep[pid]: continue
                    picks.append(pid)
                    # Suppress neighbors
                    curr_m = m_c[pid]
                    # Invalidate indices with close m
                    # (Simple NMS on m-dimension as per original code logic)
                    for other_pid in sort_idx:
                        if keep[other_pid] and abs(m_c[other_pid] - curr_m) < MUSIC_EXCL and other_pid != pid:
                            keep[other_pid] = False
                    if len(picks) >= args.max_targets: break
                
                # 5. Estimation for picked peaks
                for pid in picks:
                    m_k = m_c[pid]
                    
                    # A. Angle (Rainbow Beam Formula)
                    fm_offset = m_k * f_scs
                    den = BW * (f0 + fm_offset)
                    ang_est = 0.0
                    if abs(den) > 1e-9:
                        t1 = ((BW - fm_offset)*f0/den) * np.sin(np.deg2rad(PHI_START))
                        t2 = ((BW + f0)*fm_offset/den) * np.sin(np.deg2rad(PHI_END))
                        ang_est = np.rad2deg(np.arcsin(np.clip(t1+t2, -1, 1)))
                    
                    # B. MUSIC for V/R
                    m_lo, m_hi = max(0, m_k-MUSIC_WIN), min(M, m_k+MUSIC_WIN)
                    Y_sub = Y_np[:, m_lo:m_hi+1]
                    
                    # Normalize
                    # if Y_sub.shape[1] >= 2:
                    #     Y_sub = Y_sub / (np.abs(Y_sub) + 1e-10)
                    
                    # Velocity
                    v_est = np.nan
                    try:
                        R_D = (Y_sub @ Y_sub.conj().T) / Y_sub.shape[1]
                        _, U = np.linalg.eigh(R_D); U_noise = U[:, :-1]
                        # Steering vector
                        t_vec = np.arange(Ns)
                        av_all = np.exp(1j * (4 * np.pi * f0 * Ts / 3e8) * np.outer(V_SEARCH, t_vec))
                        spec = 1.0 / (np.sum(np.abs(av_all.conj() @ U_noise)**2, axis=1) + 1e-10)
                        v_est = V_SEARCH[np.argmax(spec)]
                    except: pass
                    
                    # Range
                    r_est = np.nan
                    try:
                        R_R = (Y_sub.T @ Y_sub.conj()) / Ns
                        _, U = np.linalg.eigh(R_R); U_noise = U[:, :-1]
                        f_vec = np.arange(Y_sub.shape[1])
                        ar_all = np.exp(-1j * (4 * np.pi * f_scs / 3e8) * np.outer(R_SEARCH, f_vec))
                        spec = 1.0 / (np.sum(np.abs(ar_all.conj() @ U_noise)**2, axis=1) + 1e-10)
                        r_est = R_SEARCH[np.argmax(spec)]
                    except: pass
                    
                    est_targets.append([ang_est, r_est, v_est])

            # 6. Matching & Error Calculation
            # Clean GT NaNs
            valid_gt_indices = ~np.isnan(gt_th)
            true_targets = np.column_stack((gt_th[valid_gt_indices], gt_r[valid_gt_indices], gt_v[valid_gt_indices]))
            
            if i < 10:
                print(f"\n[DEBUG Sample {i}]")
                print("GT (Deg, m, m/s):")
                print(true_targets)
                print("EST (Deg, m, m/s):")
                print(np.array(est_targets) if len(est_targets) > 0 else "[]")

            num_est = len(est_targets)
            num_gt = len(true_targets)
            stats_gt_total += num_gt
            
            if num_est > 0 and num_gt > 0:
                est_arr = np.array(est_targets)
                # Cost based on Angle
                cost = np.abs(est_arr[:, 0:1] - true_targets[:, 0:1].T)
                row, col = linear_sum_assignment(cost)
                
                for r, c in zip(row, col):
                    # Angle
                    ang_pred, ang_true = est_arr[r,0], true_targets[c,0]
                    err_angle_list.append(np.abs(ang_pred - ang_true))
                    
                    # Range
                    r_pred, r_true = est_arr[r,1], true_targets[c,1]
                    if not np.isnan(r_pred): 
                        err_range_list.append(np.abs(r_pred - r_true))

                    # Velocity
                    if not np.isnan(est_arr[r,2]): 
                        err_velo_list.append(np.abs(est_arr[r,2] - true_targets[c,2]))
                    
                    # 2D Position Error
                    if not np.isnan(r_pred):
                        rad_p = np.deg2rad(ang_pred); rad_t = np.deg2rad(ang_true)
                        x_p, y_p = r_pred * np.cos(rad_p), r_pred * np.sin(rad_p)
                        x_t, y_t = r_true * np.cos(rad_t), r_true * np.sin(rad_t)
                        err_2d_list.append(np.sqrt((x_p - x_t)**2 + (y_p - y_t)**2))
                
                stats_fa += (num_est - len(row))
                stats_miss += (num_gt - len(row))
            else:
                stats_fa += num_est
                stats_miss += num_gt

    except Exception as e:
        print(f"\n[YOLO FATAL ERROR] Loop crashed at sample {i}: {e}")
        traceback.print_exc()
    except KeyboardInterrupt:
        print(f"\n[YOLO] Interrupted by user at sample {i}.")
    
    finally:
        # 7. Statistics & Logging
        print(f"\n[YOLO] Saving partial/full results for {len(err_angle_list)} detected targets...")
        
        def calc_stats(errors):
            if not errors: return 0.0, 0.0, 0.0
            arr = np.array(errors)
            return np.sqrt(np.mean(arr**2)), np.percentile(arr, 90), np.percentile(arr, 95)

        rmse_a, p90_a, p95_a = calc_stats(err_angle_list)
        rmse_r, p90_r, p95_r = calc_stats(err_range_list)
        rmse_v, p90_v, p95_v = calc_stats(err_velo_list)
        rmse_2d, p90_2d, p95_2d = calc_stats(err_2d_list)
        
        # Avoid division by zero
        samples_processed = i + 1 if i > 0 else 1
        far = stats_fa / samples_processed
        mdr = stats_miss / stats_gt_total if stats_gt_total > 0 else 0

        log_file = os.path.join(args.save_dir, f"log_test_yolo_pt{int(args.pt_dbm)}.txt")
        
        log_content = [
            f"=== YOLO Baseline Test (Pt={args.pt_dbm}dBm) ===",
            f"Samples Processed: {samples_processed} | K: {args.max_targets}",
            f"Angle (deg) : RMSE={rmse_a:.4f} | 90%={p90_a:.4f} | 95%={p95_a:.4f}",
            f"Range (m)   : RMSE={rmse_r:.4f} | 90%={p90_r:.4f} | 95%={p95_r:.4f}",
            f"Velocity(m/s): RMSE={rmse_v:.4f} | 90%={p90_v:.4f} | 95%={p95_v:.4f}",
            f"2D Pos (m)  : RMSE={rmse_2d:.4f} | 90%={p90_2d:.4f} | 95%={p95_2d:.4f}",
            f"Stats       : Total FA={stats_fa} (Avg {far:.2f}) | Total Miss={stats_miss} (MDR {mdr:.2%})",
            f"=================================================="
        ]
        
        print("\n".join(log_content))
        
        try:
            with open(log_file, 'w') as f:
                f.write("\n".join(log_content))

            # 8. Save Raw Errors
            npz_file = os.path.join(args.save_dir, f"errors_yolo_pt{int(args.pt_dbm)}.npz")
            np.savez(npz_file, 
                     err_angle=err_angle_list,
                     err_range=err_range_list, 
                     err_velo=err_velo_list,
                     err_2d=err_2d_list,
                     pt_dbm=args.pt_dbm)
            
            print(f"[YOLO] Results saved successfully to {args.save_dir}")
        except Exception as e:
            print(f"[YOLO] Failed to save results to disk: {e}")

if __name__ == "__main__":
    main()