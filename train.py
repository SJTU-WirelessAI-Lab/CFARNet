# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import datetime
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
from tqdm import tqdm
import math
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc 
import sys 
import traceback 
from typing import List, Dict, Tuple, Any 
# from functions import load_system_params # Assuming this is local, if not available, code handles it.

# --- Constants ---
K_BOLTZMANN = 1.38e-23
T_NOISE_KELVIN = 290 # Standard noise temperature

# --- Define custom dataset class (read echo and m_peak) ---
class ChunkedEchoDataset(Dataset):
    """
    Load pre-computed *noiseless* echo signals (yecho) from chunked .npy files
    and target peaks (m_peak).
    """
    def __init__(self, data_root, start_idx, end_idx, expected_k):
        super().__init__()
        self.data_root = data_root
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_samples = end_idx - start_idx + 1
        self.expected_k = expected_k 

        print(f"  Dataset initialization: root='{data_root}', range=[{start_idx}, {end_idx}], count={self.num_samples}, expected_K={self.expected_k}")

        self.echoes_dir = os.path.join(data_root, 'echoes')
        if not os.path.isdir(self.echoes_dir):
            raise FileNotFoundError(f"Echoes directory not found: {self.echoes_dir}")

        params_path = os.path.join(data_root, 'system_params.npz')
        if not os.path.isfile(params_path):
            raise FileNotFoundError(f"System parameters file not found: {params_path}")
        try:
            params_data = np.load(params_path)
            if 'samples_per_chunk' not in params_data:
                if 'chunk_size' in params_data: self.chunk_size = int(params_data['chunk_size'])
                else: raise KeyError("'samples_per_chunk' or 'chunk_size' not found in system_params.npz.")
            else: self.chunk_size = int(params_data['samples_per_chunk'])
            self.M_plus_1 = int(params_data['M']) + 1 if 'M' in params_data else None
            self.Ns = int(params_data['Ns']) if 'Ns' in params_data else None
            print(f"  Loaded chunk_size from params: {self.chunk_size}")
            if self.M_plus_1: print(f"  Loaded M+1 from params: {self.M_plus_1}")
            if self.Ns: print(f"  Loaded Ns from params: {self.Ns}")
        except Exception as e: raise IOError(f"Error loading or parsing system_params.npz: {e}")
        if self.chunk_size <= 0: raise ValueError("samples_per_chunk must be positive.")

        traj_path = os.path.join(data_root, 'trajectory_data.npz')
        if not os.path.isfile(traj_path): raise FileNotFoundError(f"Trajectory data file not found: {traj_path}")
        try:
            traj_data = np.load(traj_path)
            if 'm_peak_indices' not in traj_data:
                if 'm_peak' in traj_data: m_peak_all = traj_data['m_peak']
                else: raise KeyError("'m_peak_indices' or 'm_peak' not found in trajectory_data.npz")
            else: m_peak_all = traj_data['m_peak_indices']
            total_samples_in_file = m_peak_all.shape[0]
            if self.end_idx >= total_samples_in_file:
                print(f"Warning: Requested end_idx ({self.end_idx}) exceeds available samples in trajectory_data.npz ({total_samples_in_file}).")
                self.end_idx = total_samples_in_file - 1; self.num_samples = self.end_idx - self.start_idx + 1
                if self.num_samples <= 0: raise ValueError(f"Invalid adjusted sample range [{self.start_idx}, {self.end_idx}]")
                print(f"  Adjusted dataset range: [{self.start_idx}, {self.end_idx}], count={self.num_samples}")
            self.m_peak_targets = m_peak_all[self.start_idx : self.end_idx + 1]
            print(f"  Loaded m_peak_targets, original shape: {self.m_peak_targets.shape}")

            # --- Adjust loaded targets based on expected_k ---
            actual_k_in_data = self.m_peak_targets.shape[1] if self.m_peak_targets.ndim > 1 else 1
            if actual_k_in_data < self.expected_k:
                print(f"  Info: m_peak_targets K dimension ({actual_k_in_data}) is less than expected_k ({self.expected_k}). Will pad.")
                pad_width = self.expected_k - actual_k_in_data
                self.m_peak_targets = np.pad(self.m_peak_targets, ((0, 0), (0, pad_width)), 'constant', constant_values=-1)
            elif actual_k_in_data > self.expected_k:
                 print(f"  Warning: m_peak_targets K dimension ({actual_k_in_data}) is greater than expected_k ({self.expected_k}). Will truncate.")
                 self.m_peak_targets = self.m_peak_targets[:, :self.expected_k]
            # ---

        except Exception as e: raise IOError(f"Error loading or processing trajectory_data.npz: {e}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if index < 0 or index >= self.num_samples: raise IndexError(f"Index {index} out of range [0, {self.num_samples - 1}]")
        try:
            absolute_idx = self.start_idx + index
            chunk_idx = absolute_idx // self.chunk_size
            index_in_chunk = absolute_idx % self.chunk_size
            echo_file_path = os.path.join(self.echoes_dir, f'echo_chunk_{chunk_idx}.npy')
            if not os.path.isfile(echo_file_path): raise FileNotFoundError(f"Echo data file not found: {echo_file_path} (requested chunk {chunk_idx})")
            echo_chunk = np.load(echo_file_path)
            
            if self.Ns and self.M_plus_1 and echo_chunk.ndim >= 3 and echo_chunk.shape[1:] != (self.Ns, self.M_plus_1):
                print(f"Warning (idx={absolute_idx}, chunk={chunk_idx}): echo_chunk shape {echo_chunk.shape} does not match expected ({(-1, self.Ns, self.M_plus_1)})")
            if index_in_chunk >= echo_chunk.shape[0]: raise IndexError(f"Index {index_in_chunk} exceeds loaded chunk size ({echo_chunk.shape[0]}) for file echo_chunk_{chunk_idx}.npy (absolute index {absolute_idx})")

            clean_echo_signal = echo_chunk[index_in_chunk]
            m_peak = self.m_peak_targets[index] 

            echo_tensor = torch.from_numpy(clean_echo_signal).to(torch.complex64)
            m_peak_tensor = torch.from_numpy(m_peak).to(torch.long)

            sample = {'echo': echo_tensor, 'm_peak': m_peak_tensor}
            return sample
        except Exception as e: print(f"Error loading index {index}: {e}", flush=True); traceback.print_exc(); raise

# --- Model definition ---

class IndexPredictionCNN(nn.Module):
    def __init__(self, M_plus_1, Ns, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.M_plus_1 = M_plus_1
        self.Ns = Ns
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        final_channels = 512

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.1), nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.1), nn.Dropout2d(0.1),
            nn.Conv2d(256, final_channels, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(final_channels), nn.LeakyReLU(0.1), nn.Dropout2d(0.1),
        )

        self.predictor = nn.Sequential(
            nn.Conv1d(final_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim // 2), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1) 
        )

        self.apply(self._init_weights) 

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        if isinstance(module, nn.Conv1d) and module.out_channels == 1:
             if hasattr(module, 'weight') and module.weight is not None:
                 nn.init.normal_(module.weight, mean=0.0, std=0.01) 
             if hasattr(module, 'bias') and module.bias is not None:
                 nn.init.constant_(module.bias, 0)

    def forward(self, Y_complex):
        B, Ns_actual, M_plus_1_actual = Y_complex.shape
        Y_fft = torch.fft.fft(Y_complex, dim=1)
        Y_fft_shift = torch.fft.fftshift(Y_fft, dim=1) 
        Y_magnitude = torch.abs(Y_fft_shift)
        Y_magnitude_log = torch.log1p(Y_magnitude) 

        Y_input = Y_magnitude_log.unsqueeze(1)
        features = self.feature_extractor(Y_input) 
        features_pooled = torch.max(features, dim=2)[0] 
        logits = self.predictor(features_pooled) 
        logits = logits.squeeze(1) 

        return logits, Y_magnitude_log 

# --- Helper functions ---

def create_gaussian_target(peak_indices, M_plus_1, sigma, device):
    valid_peaks = peak_indices[(peak_indices >= 0) & (peak_indices < M_plus_1)]
    if valid_peaks.numel() == 0:
        return torch.zeros(M_plus_1, dtype=torch.float32, device=device)
    
    positions = torch.arange(M_plus_1, dtype=torch.float32, device=device).unsqueeze(1)
    valid_peaks_expanded = valid_peaks.unsqueeze(0).float()
    gaussian_sum = torch.sum(torch.exp(-0.5 * ((positions - valid_peaks_expanded) / sigma) ** 2), dim=1)
    
    if gaussian_sum.sum() > 0:
        gaussian_sum = gaussian_sum / gaussian_sum.sum()
    return gaussian_sum


class CombinedLoss(nn.Module):
    def __init__(self, main_loss_type='bce', loss_sigma=1.0, device='cpu'):
        super().__init__()
        self.main_loss_type = main_loss_type
        self.device = device
        self.loss_sigma = loss_sigma 

        if main_loss_type == 'bce':
            self.main_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        elif main_loss_type == 'kldiv':
            self.main_criterion = nn.KLDivLoss(reduction='batchmean', log_target=False)
        else:
            raise ValueError(f"Unknown main_loss_type: {main_loss_type}")
        self.main_criterion.to(device)

    def forward(self, pred_logits, target_smooth):
        pred_logits = pred_logits.to(self.device)
        target_smooth = target_smooth.to(self.device)

        if self.main_loss_type == 'bce':
            main_loss = self.main_criterion(pred_logits, target_smooth)
        elif self.main_loss_type == 'kldiv':
            target_dist = target_smooth / (target_smooth.sum(dim=-1, keepdim=True) + 1e-9) 
            pred_logprob = F.log_softmax(pred_logits, dim=-1)
            main_loss = self.main_criterion(pred_logprob, target_dist)

        return main_loss

def get_latest_experiment_path():
    try:
        try: script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError: script_dir = os.getcwd() 
        paths_to_check = [
            '/mnt/sda/liangqiushi/CFARnet/latest_experiment.txt', 
            os.path.join(script_dir, 'latest_experiment.txt'),         
            os.path.join(os.getcwd(), 'latest_experiment.txt')         
        ]
        file_path = next((p for p in paths_to_check if os.path.exists(p)), None)
        if file_path is None: raise FileNotFoundError("latest_experiment.txt not found.")
        with open(file_path, 'r') as f: return f.read().strip()
    except Exception as e: print(f"Error reading latest_experiment.txt: {e}", flush=True); raise

def create_timestamp_folders(base_data_root=None):
    if base_data_root is None:
        try: data_root = get_latest_experiment_path()
        except Exception: data_root = './output/default_experiment'; os.makedirs(data_root, exist_ok=True)
    else: data_root = base_data_root
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S'); norm_data_root = os.path.normpath(data_root)
    experiment_name = os.path.basename(norm_data_root) if norm_data_root not in ['.', '/'] else 'default_experiment'
    
    output_base_template = os.path.join('.', 'output', f"{experiment_name}_{timestamp}_PtFixed")
    folders = { 'root': data_root, 'output_base_template': output_base_template, 'output_base': None,
                'figures': None, 'models': None, 'outputs': None }
    return folders, timestamp

def set_matplotlib_english():
    try:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 
        plt.rcParams['axes.unicode_minus'] = False 
        plt.rcParams['font.size'] = 12
    except Exception: pass

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# <<< Top-K Accuracy Calculation Function >>> 
def calculate_accuracy_topk(pred_probs, true_peak_indices_batch, k, tolerance):
    batch_size = pred_probs.shape[0]
    M_plus_1 = pred_probs.shape[1]
    if batch_size == 0: return 0.0

    total_true_peaks_count = 0
    total_hits = 0
    k = min(k, M_plus_1) 
    if k <= 0: return 0.0 

    with torch.no_grad():
        _, topk_indices_batch = torch.topk(pred_probs, k=k, dim=1) 

        for b in range(batch_size):
            true_indices_b = true_peak_indices_batch[b] 
            valid_true_mask = (true_indices_b >= 0) & (true_indices_b < M_plus_1)
            valid_true_peaks = true_indices_b[valid_true_mask] 

            num_true = valid_true_peaks.numel()
            if num_true == 0: continue
            total_true_peaks_count += num_true

            pred_indices_b_topk = topk_indices_batch[b] 
            
            dist_matrix = torch.abs(valid_true_peaks.unsqueeze(1) - pred_indices_b_topk.unsqueeze(0))
            min_dists_to_topk_preds, _ = torch.min(dist_matrix, dim=1) 
            hits_b = torch.sum(min_dists_to_topk_preds <= tolerance).item()
            total_hits += hits_b

    if total_true_peaks_count == 0: accuracy = 1.0
    else: accuracy = total_hits / total_true_peaks_count
    return accuracy

def visualize_predictions(pred_probs_list, target_list, folders, timestamp, M_plus_1,
                          acc_threshold=0.5, acc_tolerance=3, 
                          is_target_distribution=False, num_samples=4):
    if not pred_probs_list or not target_list: return
    if not folders['figures']: return
    num_total_samples = len(pred_probs_list); num_samples = min(num_samples, num_total_samples)
    if num_samples == 0: return
    indices = np.random.choice(num_total_samples, num_samples, replace=False)
    fig, axs = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples), sharex=True, squeeze=False)
    subcarrier_indices = np.arange(M_plus_1)
    target_label = "Target Distribution" if is_target_distribution else "Smoothed Target"
    for i, idx in enumerate(indices):
        pred_probs = pred_probs_list[idx].cpu().numpy(); target_values = target_list[idx].cpu().numpy()
        ax = axs[i, 0]
        ax.plot(subcarrier_indices, pred_probs, label='Predicted', alpha=0.7, color='blue')
        ax.plot(subcarrier_indices, target_values, label=target_label, alpha=0.7, color='red', linestyle='--')
        true_peaks_indices = np.where(target_values > 0.5)[0]
        if len(true_peaks_indices) > 0: ax.plot(subcarrier_indices[true_peaks_indices], target_values[true_peaks_indices], 'ro', markersize=6)
        ax.set_title(f'Sample {idx}'); ax.legend(fontsize='small'); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(folders['figures'], f'peak_predictions_{timestamp}.png')
    try: plt.savefig(plot_path); print(f"Viz saved: {plot_path}", flush=True)
    except Exception: pass
    plt.close(fig)

# --- Simplified Test Function (Single Pt) ---
def test_model(model: nn.Module,
               test_loader: DataLoader,
               device: torch.device,
               args: argparse.Namespace,
               M_plus_1: int,
               pt_dbm: float, # Single Pt
               noise_std_dev_tensor: torch.Tensor
               ) -> Tuple[Dict[str, float], List[torch.Tensor], List[torch.Tensor]]:
    """
    Test model at a SINGLE specified transmit power level.
    """
    print(f"\nTesting at Pt = {pt_dbm:.1f} dBm (loss: {args.loss_type})...", flush=True)
    model.eval()

    all_pred_probs_list_viz = []
    all_target_smooth_list_viz = []
    
    loss_fn = CombinedLoss(main_loss_type=args.loss_type, loss_sigma=args.loss_sigma, device=device)
    
    pt_linear_mw = 10**(pt_dbm / 10.0)
    pt_scaling_factor = math.sqrt(pt_linear_mw)
    pt_scaling_factor_tensor = torch.tensor(pt_scaling_factor, dtype=torch.float32, device=device)

    test_pbar = tqdm(test_loader, desc=f"Testing Pt={pt_dbm:.1f}dBm", leave=False, file=sys.stdout)
    batch_loss_accum = 0.0
    batch_acc_accum = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_pbar):
            clean_echo = batch['echo'].to(device)
            m_peak_targets_original = batch['m_peak'].to(device)
            batch_size = clean_echo.shape[0]

            # Fixed power scaling
            scaled_echo = clean_echo * pt_scaling_factor_tensor 
            noise = (torch.randn_like(scaled_echo.real) + 1j * torch.randn_like(scaled_echo.imag)) * noise_std_dev_tensor.to(device)
            yecho_input = scaled_echo + noise

            pred_logits, _ = model(yecho_input)

            target_smooth_batch = torch.zeros_like(pred_logits, dtype=torch.float32)
            for b in range(batch_size):
                peak_indices_b = m_peak_targets_original[b]
                target_smooth_batch[b, :] = create_gaussian_target(peak_indices_b, M_plus_1, args.loss_sigma, device)

            loss = loss_fn(pred_logits, target_smooth_batch)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                batch_loss_accum += loss.item()
                pred_probs = torch.sigmoid(pred_logits)

                accuracy = calculate_accuracy_topk(
                    pred_probs, m_peak_targets_original, k=args.top_k, tolerance=args.accuracy_tolerance
                )
                batch_acc_accum += accuracy
                batch_count += 1

                # Collect viz data
                if len(all_pred_probs_list_viz) < 50: # Limit memory
                    all_pred_probs_list_viz.extend(list(pred_probs.cpu()))
                    all_target_smooth_list_viz.extend(list(target_smooth_batch.cpu()))
            
            if batch_idx % 50 == 0:
                 test_pbar.set_postfix({'L': f"{loss.item():.4f}", f'Top{args.top_k}': f"{accuracy:.3f}"})

    results = {'loss': float('inf'), 'accuracy': 0.0, 'count': 0}
    if batch_count > 0:
        results['loss'] = batch_loss_accum / batch_count
        results['accuracy'] = batch_acc_accum / batch_count
        results['count'] = batch_count
        print(f"  Result: Avg Loss: {results['loss']:.4f}, Avg Top-{args.top_k} Hit: {results['accuracy']:.4f}", flush=True)
    else:
        print(f"  Result: No valid batches.", flush=True)

    gc.collect();
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return results, all_pred_probs_list_viz, all_target_smooth_list_viz


# --- Main execution function ---
def main():
    parser = argparse.ArgumentParser(description='Train peak index prediction model (CNN - Fixed Power)')
    # --- Training/Test Power Parameter (Single Value) ---
    parser.add_argument('--pt_dbm', type=float, default=20.0, help='Transmit power (dBm) used for BOTH Training and Testing')
    
    # --- Standard Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=60, help="Total number of training epochs")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='Gradient clipping norm')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    
    # --- System/Data Parameters ---
    parser.add_argument('--data_dir', type=str, default=None, help='Data root directory')
    parser.add_argument('--max_targets', type=int, default=3, help='Expected maximum number of targets')
    
    # --- Model Hyperparameters ---
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # --- Loss/Accuracy Parameters ---
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'kldiv'], help='Main loss function type')
    parser.add_argument('--loss_sigma', type=float, default=1.0, help='Sigma for Gaussian smooth target')
    parser.add_argument('--top_k', type=int, default=4, help='Top-K for accuracy')
    parser.add_argument('--accuracy_threshold', type=float, default=0.5, help='Viz threshold')
    parser.add_argument('--accuracy_tolerance', type=int, default=3, help='Hit tolerance')
    
    # --- Execution Control ---
    parser.add_argument('--num_workers', type=int, default=4, help='Workers')
    parser.add_argument('--cuda_device', type=int, default=0, help="CUDA device ID")
    parser.add_argument('--test_only', action='store_true', help='Only run testing')
    parser.add_argument('--load_model', action='store_true', help='Load best model')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to .pt file')
    parser.add_argument('--model_dir', type=str, default=None, help='Directory with best_model')

    args = parser.parse_args()

    print(f"Fixed Transmit Power for ALL phases: {args.pt_dbm} dBm")
    
    # --- Setup Device, Paths, Folders ---
    if torch.cuda.is_available() and args.cuda_device >= 0 :
        device = torch.device(f"cuda:{args.cuda_device}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)
    set_matplotlib_english()

    try: data_root = args.data_dir if args.data_dir else get_latest_experiment_path()
    except Exception: print("Error: No data directory.", flush=True); return
    
    folders, timestamp = create_timestamp_folders(data_root) 
    folders['output_base'] = folders['output_base_template'] 
    folders['figures'] = os.path.join(folders['output_base'], 'figures')
    folders['models'] = os.path.join(folders['output_base'], 'models')
    folders['outputs'] = os.path.join(folders['output_base'], 'outputs')
    for folder_key in ['figures', 'models', 'outputs']: os.makedirs(folders[folder_key], exist_ok=True)
    print(f"Output files: {folders['output_base']}", flush=True)

    # --- Load System Params ---
    params_file = os.path.join(data_root, 'system_params.npz')
    try:
        params_data = np.load(params_file)
        M = int(params_data['M']); Ns = int(params_data['Ns'])
        if 'BW' in params_data: BW = float(params_data['BW'])
        else: BW = float(params_data['f_scs']) * int(params_data['M'])
    except Exception as e: print(f"Error loading system_params.npz: {e}", flush=True); return
    M_plus_1 = M + 1

    # --- Calculate noise standard deviation ---
    noise_power_total_linear = K_BOLTZMANN * T_NOISE_KELVIN * BW* 1000.0
    noise_variance_per_component = noise_power_total_linear / 2.0; noise_std_dev = math.sqrt(noise_variance_per_component)
    noise_std_dev_tensor = torch.tensor(noise_std_dev, dtype=torch.float32)

    # --- Calculate Fixed Power Scaling Factor ---
    fixed_pt_linear_mw = 10**(args.pt_dbm / 10.0)
    fixed_pt_scaling_factor = math.sqrt(fixed_pt_linear_mw)
    fixed_pt_scaling_factor_tensor = torch.tensor(fixed_pt_scaling_factor, dtype=torch.float32, device=device)
    print(f"Pt={args.pt_dbm}dBm -> Scaling Factor: {fixed_pt_scaling_factor:.4f}")

    # --- Datasets ---
    print("Setting up datasets...", flush=True)
    try:
        traj_path_check = os.path.join(data_root, 'trajectory_data.npz')
        traj_data_check = np.load(traj_path_check)
        key_to_check = 'm_peak_indices' if 'm_peak_indices' in traj_data_check else 'm_peak'
        num_total = traj_data_check[key_to_check].shape[0]
    except: num_total = 50000 

    test_frac = 0.15; val_frac = 0.15
    test_size = int(num_total * test_frac); val_size = int(num_total * val_frac); train_size = num_total - test_size - val_size
    test_start=0; test_end=test_size-1; val_start=test_end+1; val_end=val_start+val_size-1; train_start=val_end+1; train_end=train_start+train_size-1
    
    test_dataset = ChunkedEchoDataset(data_root, test_start, test_end, expected_k=args.max_targets)
    val_dataset = ChunkedEchoDataset(data_root, val_start, val_end, expected_k=args.max_targets)
    train_dataset = ChunkedEchoDataset(data_root, train_start, train_end, expected_k=args.max_targets)

    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    # --- Build Model ---
    model = IndexPredictionCNN(M_plus_1, Ns, args.hidden_dim, args.dropout).to(device)

    # --- Load Pretrained Model Logic ---
    load_path = args.load_model_path
    if not load_path and args.model_dir:
         if os.path.exists(os.path.join(args.model_dir, 'best_model.pt')): load_path = os.path.join(args.model_dir, 'best_model.pt')
    
    if (args.load_model or args.test_only) and load_path and os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path, map_location=device)); print(f"Loaded: {load_path}")

    # --- Test Only Mode ---
    if args.test_only:
        test_results, all_probs, all_targets = test_model(model, test_loader, device, args, M_plus_1, args.pt_dbm, noise_std_dev_tensor)
        visualize_predictions(all_probs, all_targets, folders, timestamp + "_test_only", M_plus_1)
        return

    # --- Training Setup ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=max(1e-8, args.lr * 0.001))
    criterion = CombinedLoss(main_loss_type=args.loss_type, loss_sigma=args.loss_sigma, device=device)

    epochs = args.epochs; best_val_loss = float('inf'); early_stop_counter = 0
    train_losses, val_losses, val_accs = [], [], []

    print(f"\nStarting training (Pt={args.pt_dbm}dBm)...", flush=True)
    
    for epoch in range(epochs):
        # === Training ===
        model.train()
        epoch_loss = 0.0; epoch_acc = 0.0; count = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs}", leave=False, file=sys.stdout)

        for batch in pbar:
            clean_echo = batch['echo'].to(device)
            targets = batch['m_peak'].to(device)
            optimizer.zero_grad()

            # --- Fixed Power Scaling ---
            scaled_echo = clean_echo * fixed_pt_scaling_factor_tensor
            noise = (torch.randn_like(scaled_echo.real) + 1j * torch.randn_like(scaled_echo.imag)) * noise_std_dev_tensor.to(device)
            yecho_input = scaled_echo + noise

            pred_logits, _ = model(yecho_input)
            
            target_smooth = torch.zeros_like(pred_logits)
            for b in range(clean_echo.shape[0]):
                target_smooth[b, :] = create_gaussian_target(targets[b], M_plus_1, args.loss_sigma, device)

            loss = criterion(pred_logits, target_smooth)
            if torch.isnan(loss): continue

            loss.backward()
            if args.clip_grad_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            acc = calculate_accuracy_topk(torch.sigmoid(pred_logits), targets, k=args.top_k, tolerance=args.accuracy_tolerance)
            epoch_loss += loss.item(); epoch_acc += acc; count += 1
            if count % 50 == 0: pbar.set_postfix({'L': f"{loss.item():.3f}", 'Acc': f"{acc:.2f}"})

        avg_train_loss = epoch_loss / count if count > 0 else float('inf')
        avg_train_acc = epoch_acc / count if count > 0 else 0.0
        train_losses.append(avg_train_loss)

        # === Validation (Single Pt) ===
        val_results, _, _ = test_model(model, val_loader, device, args, M_plus_1, args.pt_dbm, noise_std_dev_tensor)
        val_loss = val_results['loss']
        val_acc = val_results['accuracy']
        val_losses.append(val_loss); val_accs.append(val_acc)

        print(f"Ep {epoch+1}: TrL={avg_train_loss:.4f}, ValL={val_loss:.4f}, ValAcc={val_acc:.3f}")
        scheduler.step()

        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss; early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(folders['models'], f'best_model_{timestamp}.pt'))
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience: print("Early stopping."); break

    # --- Final Test ---
    print("\nFinal Test with Best Model...")
    best_model_path = os.path.join(folders['models'], f'best_model_{timestamp}.pt')
    if os.path.exists(best_model_path): model.load_state_dict(torch.load(best_model_path))
    
    test_results, all_probs, all_targets = test_model(model, test_loader, device, args, M_plus_1, args.pt_dbm, noise_std_dev_tensor)

    # --- Plotting ---
    if len(train_losses) > 0:
        plt.figure(figsize=(10, 8))
        plt.subplot(2,1,1); plt.plot(train_losses, label='Train Loss'); plt.plot(val_losses, label='Val Loss'); plt.legend(); plt.title('Loss')
        plt.subplot(2,1,2); plt.plot(val_accs, label='Val Acc'); plt.legend(); plt.title('Accuracy')
        plt.savefig(os.path.join(folders['figures'], f'curves_{timestamp}.png'))
    
    visualize_predictions(all_probs, all_targets, folders, timestamp + "_final", M_plus_1)

    # --- Summary ---
    with open(os.path.join(folders['outputs'], f'summary_{timestamp}.txt'), 'w') as f:
        f.write(f"Pt: {args.pt_dbm} dBm\nTest Acc: {test_results['accuracy']:.4f}\n")

if __name__ == "__main__":
    main()