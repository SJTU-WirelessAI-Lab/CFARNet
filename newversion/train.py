# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import datetime
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys

# --- Constants ---
K_BOLTZMANN = 1.38e-23
T_NOISE_KELVIN = 290 

class ChunkedEchoDataset(Dataset):
    """Simple dataset loader for chunked .npy files."""
    def __init__(self, data_root, start_idx, end_idx, expected_k):
        super().__init__()
        self.data_root = data_root
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_samples = end_idx - start_idx + 1
        self.expected_k = expected_k
        self.echoes_dir = os.path.join(data_root, 'echoes')
        
        try:
            params = np.load(os.path.join(data_root, 'system_params.npz'))
            self.chunk_size = int(params.get('samples_per_chunk', 500))
            self.M_plus_1 = int(params['M']) + 1
            self.Ns = int(params['Ns'])
            
            traj = np.load(os.path.join(data_root, 'trajectory_data.npz'))
            m_peak_all = traj.get('m_peak_indices', traj.get('m_peak'))
            
            # Bound check
            if self.end_idx >= m_peak_all.shape[0]: 
                self.end_idx = m_peak_all.shape[0] - 1
                self.num_samples = self.end_idx - self.start_idx + 1
            
            self.m_peak_targets = m_peak_all[self.start_idx : self.end_idx + 1]
            
            # Handle K mismatch (Pad or Truncate)
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
        # Input: [B, Ns, M] Complex -> Log Magnitude [B, 1, Ns, M]
        x_fft = torch.fft.fftshift(torch.fft.fft(x, dim=1), dim=1)
        x = torch.log1p(torch.abs(x_fft)).unsqueeze(1)
        x = self.feat(x)
        x = torch.max(x, dim=2)[0] # Max pool over Doppler dim
        return self.pred(x).squeeze(1)

def create_gaussian_target(peak_indices, M_plus_1, sigma, device):
    """Creates smooth target distribution."""
    valid = peak_indices[(peak_indices >= 0) & (peak_indices < M_plus_1)]
    if valid.numel() == 0: return torch.zeros(M_plus_1, device=device)
    pos = torch.arange(M_plus_1, device=device).unsqueeze(1)
    gauss = torch.sum(torch.exp(-0.5 * ((pos - valid.unsqueeze(0)) / sigma) ** 2), dim=1)
    return gauss / gauss.sum() if gauss.sum() > 0 else gauss

def calculate_tophit(pred_logits, true_indices, k_hit=3, tolerance=3):
    """Calculates batch Top-K Hit Rate."""
    batch_size = pred_logits.shape[0]
    pred_probs = torch.sigmoid(pred_logits)
    _, topk_indices = torch.topk(pred_probs, k=k_hit, dim=1)
    
    total_hits = 0
    total_targets = 0
    
    for b in range(batch_size):
        true_b = true_indices[b]
        valid_true = true_b[true_b >= 0]
        if valid_true.numel() == 0: continue
        total_targets += valid_true.numel()
        
        preds_b = topk_indices[b]
        dist_matrix = torch.abs(valid_true.unsqueeze(1) - preds_b.unsqueeze(0))
        min_dists, _ = torch.min(dist_matrix, dim=1)
        hits_b = torch.sum(min_dists <= tolerance).item()
        total_hits += hits_b
        
    return total_hits, total_targets

def main():
    parser = argparse.ArgumentParser()
    # Required for Pipeline
    parser.add_argument('--data_dir', type=str, required=True, help="Input data directory")
    parser.add_argument('--save_dir', type=str, required=True, help="Output directory for model and logs")
    parser.add_argument('--pt_dbm', type=float, required=True, help="Fixed Training Power (dBm)")
    parser.add_argument('--max_targets', type=int, required=True, help="K value")
    
    # Optional / Hyperparams
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_test_samples', type=int, default=0, help="Number of samples reserved for testing")
    parser.add_argument('--test_set_mode', type=str, default='last', choices=['first', 'last'], help="Where test set is located")
    args = parser.parse_args()

    # Device Setup
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device} | Pt: {args.pt_dbm} dBm | K: {args.max_targets}")

    # Load System Params
    try:
        sys_params = np.load(os.path.join(args.data_dir, 'system_params.npz'))
        M = int(sys_params['M'])
        Ns = int(sys_params['Ns'])
        BW = float(sys_params['BW'])
        M_plus_1 = M + 1
    except Exception as e:
        print(f"Error loading system_params: {e}")
        return

    # Data Loaders
    traj = np.load(os.path.join(args.data_dir, 'trajectory_data.npz'))
    total_samples = traj.get('m_peak_indices', traj.get('m_peak')).shape[0]
    
    if args.test_set_mode == 'first' and args.num_test_samples > 0:
        # Mode: First N samples are TEST. Train on the rest.
        print(f"[Train] Test Mode: FIRST {args.num_test_samples} samples reserved for testing.")
        start_train_idx = args.num_test_samples
        available_samples = total_samples - start_train_idx
        
        if available_samples <= 0:
            raise ValueError("No samples left for training after reserving test set!")
            
        # Split remaining into Train (90%) / Val (10%)
        train_len = int(available_samples * 0.9)
        val_len = available_samples - train_len
        
        train_ds = ChunkedEchoDataset(args.data_dir, start_train_idx, start_train_idx + train_len - 1, args.max_targets)
        val_ds = ChunkedEchoDataset(args.data_dir, start_train_idx + train_len, total_samples - 1, args.max_targets)
        
    else:
        # Mode: Last 20% is Val (Original Logic, or if test set is at the end)
        # If test_set_mode is 'last', usually we assume the test set is separate or part of Val.
        # But to keep original behavior:
        print(f"[Train] Test Mode: Standard 80/20 Split (Last 20% Val).")
        train_size = int(total_samples * 0.8)
        train_ds = ChunkedEchoDataset(args.data_dir, 0, train_size-1, args.max_targets)
        val_ds = ChunkedEchoDataset(args.data_dir, train_size, total_samples-1, args.max_targets)
    
    print(f"[Train] Train Samples: {len(train_ds)} | Val Samples: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model & Opt
    model = IndexPredictionCNN(M_plus_1, Ns).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Power Scaling Factors
    noise_pow = K_BOLTZMANN * T_NOISE_KELVIN * BW * 1000
    noise_std = math.sqrt(noise_pow / 2)
    scale_factor = math.sqrt(10**(args.pt_dbm/10))

    # History Logging
    history = {
        'train_loss': [], 'val_loss': [], 
        'val_tophit': [], 'epochs': []
    }
    
    best_val_tophit = -1.0
    model_save_path = os.path.join(args.save_dir, f"model_pt{int(args.pt_dbm)}_best.pt")

    print(f"[Train] Starting {args.epochs} epochs...")

    for epoch in range(args.epochs):
        # --- Training ---
        model.train()
        train_loss_accum = 0.0
        
        for batch in train_loader:
            echo = batch['echo'].to(device)
            target = batch['m_peak'].to(device)
            
            # Apply Noise & Scale
            noise = (torch.randn_like(echo.real) + 1j*torch.randn_like(echo.imag)) * noise_std
            input_sig = echo * scale_factor + noise
            
            logits = model(input_sig)
            
            # Smooth Target
            smooth_target = torch.zeros_like(logits)
            for b in range(logits.shape[0]):
                smooth_target[b] = create_gaussian_target(target[b], M_plus_1, 1.0, device)
            
            loss = criterion(logits, smooth_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            
        avg_train_loss = train_loss_accum / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss_accum = 0.0
        val_hits = 0
        val_total_targets = 0
        
        with torch.no_grad():
            for batch in val_loader:
                echo = batch['echo'].to(device)
                target = batch['m_peak'].to(device)
                
                noise = (torch.randn_like(echo.real) + 1j*torch.randn_like(echo.imag)) * noise_std
                input_sig = echo * scale_factor + noise
                
                logits = model(input_sig)
                
                # Val Loss
                smooth_target = torch.zeros_like(logits)
                for b in range(logits.shape[0]):
                    smooth_target[b] = create_gaussian_target(target[b], M_plus_1, 1.0, device)
                val_loss_accum += criterion(logits, smooth_target).item()
                
                # Val TopHit
                h, t = calculate_tophit(logits, target, k_hit=args.max_targets, tolerance=3)
                val_hits += h
                val_total_targets += t

        avg_val_loss = val_loss_accum / len(val_loader)
        avg_val_tophit = val_hits / val_total_targets if val_total_targets > 0 else 0.0
        
        scheduler.step()
        
        # --- Logging & Saving ---
        print(f"Ep {epoch+1:02d}: TrLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f} | ValTopHit={avg_val_tophit:.4f}")
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_tophit'].append(avg_val_tophit)
        history['epochs'].append(epoch+1)
        
        if avg_val_tophit > best_val_tophit:
            best_val_tophit = avg_val_tophit
            torch.save(model.state_dict(), model_save_path)
            # print(f"  --> Best model saved: {avg_val_tophit:.4f}")

    # Save History for plotting later
    np.savez(
        os.path.join(args.save_dir, f"history_pt{int(args.pt_dbm)}.npz"), 
        **history
    )
    print("[Train] Finished.")

if __name__ == "__main__":
    main()