# -*- coding: utf-8 -*- # 添加编码声明
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
from functions import load_system_params

# --- 常量 ---
K_BOLTZMANN = 1.38e-23
T_NOISE_KELVIN = 290 # Standard noise temperature




# --- 定义自定义数据集类 (读取 echo 和 m_peak) ---
class ChunkedEchoDataset(Dataset):
    """
    从分块保存的 .npy 文件加载预先计算的 *无噪声* 回波信号 (yecho)
    和目标峰值 (m_peak)。
    """
    # expected_k now directly controlled by args.max_targets
    def __init__(self, data_root, start_idx, end_idx, expected_k):
        """
        初始化数据集。

        Args:
            data_root (str): 包含 echoes 子目录和 trajectory_data.npz,
                             system_params.npz 的根目录。
            start_idx (int): 此数据集分片的起始绝对索引。
            end_idx (int): 此数据集分片的结束绝对索引。
            expected_k (int): 期望的最大目标数 K (用于 m_peak 填充/截断)。
        """
        super().__init__()
        self.data_root = data_root
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_samples = end_idx - start_idx + 1
        self.expected_k = expected_k # Now set by args.max_targets

        print(f"  数据集初始化: 根目录='{data_root}', 范围=[{start_idx}, {end_idx}], 数量={self.num_samples}, 预期K={self.expected_k}") # Added expected_k here

        self.echoes_dir = os.path.join(data_root, 'echoes')
        if not os.path.isdir(self.echoes_dir):
            raise FileNotFoundError(f"Echoes 目录未找到: {self.echoes_dir}")

        params_path = os.path.join(data_root, 'system_params.npz')
        if not os.path.isfile(params_path):
            raise FileNotFoundError(f"系统参数文件未找到: {params_path}")
        try:
            params_data = np.load(params_path)
            if 'samples_per_chunk' not in params_data:
                if 'chunk_size' in params_data: self.chunk_size = int(params_data['chunk_size'])
                else: raise KeyError("在 system_params.npz 中未找到 'samples_per_chunk' 或 'chunk_size'。")
            else: self.chunk_size = int(params_data['samples_per_chunk'])
            self.M_plus_1 = int(params_data['M']) + 1 if 'M' in params_data else None
            self.Ns = int(params_data['Ns']) if 'Ns' in params_data else None
            print(f"  从参数加载 chunk_size: {self.chunk_size}")
            if self.M_plus_1: print(f"  从参数加载 M+1: {self.M_plus_1}")
            if self.Ns: print(f"  从参数加载 Ns: {self.Ns}")
        except Exception as e: raise IOError(f"加载或解析 system_params.npz 时出错: {e}")
        if self.chunk_size <= 0: raise ValueError("samples_per_chunk 必须为正数。")

        traj_path = os.path.join(data_root, 'trajectory_data.npz')
        if not os.path.isfile(traj_path): raise FileNotFoundError(f"轨迹数据文件未找到: {traj_path}")
        try:
            traj_data = np.load(traj_path)
            if 'm_peak_indices' not in traj_data:
                if 'm_peak' in traj_data: m_peak_all = traj_data['m_peak']
                else: raise KeyError("在 trajectory_data.npz 中未找到 'm_peak_indices' 或 'm_peak'")
            else: m_peak_all = traj_data['m_peak_indices']
            total_samples_in_file = m_peak_all.shape[0]
            if self.end_idx >= total_samples_in_file:
                print(f"警告：请求的 end_idx ({self.end_idx}) 超出 trajectory_data.npz 中的可用样本 ({total_samples_in_file})。")
                self.end_idx = total_samples_in_file - 1; self.num_samples = self.end_idx - self.start_idx + 1
                if self.num_samples <= 0: raise ValueError(f"调整后的样本范围无效 [{self.start_idx}, {self.end_idx}]")
                print(f"  调整后的数据集范围: [{self.start_idx}, {self.end_idx}], 数量={self.num_samples}")
            self.m_peak_targets = m_peak_all[self.start_idx : self.end_idx + 1]
            print(f"  已加载 m_peak_targets，原始形状: {self.m_peak_targets.shape}")

            # --- Adjust loaded targets based on expected_k ---
            actual_k_in_data = self.m_peak_targets.shape[1] if self.m_peak_targets.ndim > 1 else 1
            if actual_k_in_data < self.expected_k:
                print(f"  信息: m_peak_targets K维度 ({actual_k_in_data}) 小于 expected_k ({self.expected_k})。将进行填充。")
                pad_width = self.expected_k - actual_k_in_data
                # Pad with -1 (invalid index)
                self.m_peak_targets = np.pad(self.m_peak_targets, ((0, 0), (0, pad_width)), 'constant', constant_values=-1)
                print(f"  填充后的 m_peak_targets 形状: {self.m_peak_targets.shape}")
            elif actual_k_in_data > self.expected_k:
                 print(f"  警告: m_peak_targets K维度 ({actual_k_in_data}) 大于 expected_k ({self.expected_k})。将截断。")
                 self.m_peak_targets = self.m_peak_targets[:, :self.expected_k]
                 print(f"  截断后的 m_peak_targets 形状: {self.m_peak_targets.shape}")
            # ---

        except Exception as e: raise IOError(f"加载或处理 trajectory_data.npz 时出错: {e}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if index < 0 or index >= self.num_samples: raise IndexError(f"索引 {index} 超出范围 [0, {self.num_samples - 1}]")
        try:
            absolute_idx = self.start_idx + index
            chunk_idx = absolute_idx // self.chunk_size
            index_in_chunk = absolute_idx % self.chunk_size
            echo_file_path = os.path.join(self.echoes_dir, f'echo_chunk_{chunk_idx}.npy')
            if not os.path.isfile(echo_file_path): raise FileNotFoundError(f"Echo 数据文件未找到: {echo_file_path} (请求 chunk {chunk_idx})")
            echo_chunk = np.load(echo_file_path)
            # Check dimensions if M_plus_1 and Ns are known
            if self.Ns and self.M_plus_1 and echo_chunk.ndim >= 3 and echo_chunk.shape[1:] != (self.Ns, self.M_plus_1):
                print(f"警告 (idx={absolute_idx}, chunk={chunk_idx}): echo_chunk 形状 {echo_chunk.shape} 与预期 ({(-1, self.Ns, self.M_plus_1)}) 不符")
            if index_in_chunk >= echo_chunk.shape[0]: raise IndexError(f"索引 {index_in_chunk} 超出加载的块大小 ({echo_chunk.shape[0]}) 对于文件 echo_chunk_{chunk_idx}.npy (绝对索引 {absolute_idx})")

            clean_echo_signal = echo_chunk[index_in_chunk]
            m_peak = self.m_peak_targets[index] # Already adjusted to expected_k

            # Convert to tensors
            echo_tensor = torch.from_numpy(clean_echo_signal).to(torch.complex64)
            # Ensure m_peak is LONG type for indexing/embedding later if needed
            m_peak_tensor = torch.from_numpy(m_peak).to(torch.long)

            sample = {'echo': echo_tensor, 'm_peak': m_peak_tensor}
            return sample
        except FileNotFoundError as e: print(f"错误：加载索引 {index} (绝对 {self.start_idx + index}) 时文件未找到: {e}", flush=True); raise
        except IndexError as e: print(f"错误：加载索引 {index} (绝对 {self.start_idx + index}) 时索引超出范围: {e}", flush=True); raise
        except Exception as e: print(f"错误：加载索引 {index} (绝对 {self.start_idx + index}) 时发生意外错误: {e}", flush=True); traceback.print_exc(); raise

# --- 模型定义 ---

# --- CNN Model (IndexPredictionCNN) ---
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
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1) # Output logits per M+1 position
        )

        self.apply(self._init_weights) # Apply weight initialization

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        # Initialize the final prediction layer differently if needed
        if isinstance(module, nn.Conv1d) and module.out_channels == 1:
             if hasattr(module, 'weight') and module.weight is not None:
                 nn.init.normal_(module.weight, mean=0.0, std=0.01) # Small init for final layer
             if hasattr(module, 'bias') and module.bias is not None:
                 nn.init.constant_(module.bias, 0)

    def forward(self, Y_complex):
        B, Ns_actual, M_plus_1_actual = Y_complex.shape
        # --- START: FFT Processing ---
        # FFT along Ns dimension (Doppler)
        Y_fft = torch.fft.fft(Y_complex, dim=1)
        Y_fft_shift = torch.fft.fftshift(Y_fft, dim=1) # Shift zero-frequency component to center
        # --- END: FFT Processing ---

        # --- START: Feature Extraction Input Preparation ---
        # Use magnitude as input features
        Y_magnitude = torch.abs(Y_fft_shift)
        # Log magnitude to compress dynamic range
        Y_magnitude_log = torch.log1p(Y_magnitude) # log(1+x) for stability

        # Standardization (Optional - uncomment to try)
        # mean = torch.mean(Y_magnitude_log, dim=(1, 2), keepdim=True)
        # std = torch.std(Y_magnitude_log, dim=(1, 2), keepdim=True)
        # Y_magnitude_log = (Y_magnitude_log - mean) / (std + 1e-6) # Add epsilon for stability

        # Add channel dimension: (B, Ns, M+1) -> (B, 1, Ns, M+1)
        Y_input = Y_magnitude_log.unsqueeze(1)
        # --- END: Feature Extraction Input Preparation ---

        # --- START: Feature Extraction (CNN) ---
        features = self.feature_extractor(Y_input) # Output: (B, C_final, Ns_out, M+1_out)
        # Max pooling over the Doppler (Ns) dimension
        # Note: kernel_size, stride, padding affect Ns_out, M+1_out
        # Here, M+1 dimension remains the same due to padding/stride
        features_pooled = torch.max(features, dim=2)[0] # Output: (B, C_final, M+1)
        # --- END: Feature Extraction (CNN) ---

        # --- START: Prediction Head (1D CNN) ---
        logits = self.predictor(features_pooled) # Output: (B, 1, M+1)
        logits = logits.squeeze(1) # Output: (B, M+1) - Logits for each subcarrier index
        # --- END: Prediction Head (1D CNN) ---

        return logits, Y_magnitude_log # Return logits and the processed input for visualization

# --- 辅助函数定义 ---

# --- Gaussian Target Generation ---
def create_gaussian_target(peak_indices, M_plus_1, sigma, device):
    if isinstance(peak_indices, (list, np.ndarray)):
        peak_indices = torch.tensor(peak_indices, device=device, dtype=torch.long)
    elif isinstance(peak_indices, torch.Tensor):
        peak_indices = peak_indices.to(device=device, dtype=torch.long)
    else:
        raise TypeError(f"peak_indices type {type(peak_indices)} not supported.")

    # Filter out invalid indices (like -1 padding) before processing
    valid_mask = (peak_indices >= 0) & (peak_indices < M_plus_1)
    valid_peaks = peak_indices[valid_mask]

    # If no valid peaks remain after filtering, return zeros
    if valid_peaks.numel() == 0:
        return torch.zeros(M_plus_1, device=device, dtype=torch.float32)

    # Proceed with Gaussian generation only for valid peaks
    valid_peaks = valid_peaks.float() # Ensure float for calculations
    idx = torch.arange(M_plus_1, device=device, dtype=torch.float32)
    idx_expanded = idx.unsqueeze(1) # (M+1, 1)
    peaks_expanded = valid_peaks.unsqueeze(0) # (1, num_valid_peaks)
    sigma_safe = max(float(sigma), 1e-6) # Prevent division by zero

    # Calculate Gaussian bumps for all valid peaks simultaneously
    gaussians = torch.exp(-0.5 * ((idx_expanded - peaks_expanded) / sigma_safe)**2) # (M+1, num_valid_peaks)

    # Sum the Gaussians and clamp
    target_smooth = torch.sum(gaussians, dim=1) # (M+1,)
    target_smooth = torch.clamp(target_smooth, max=1.0) # Ensure max value is 1.0

    return target_smooth


# --- Simplified Combined Loss (Only Main Loss) --- 
class CombinedLoss(nn.Module):
    def __init__(self, main_loss_type='bce', loss_sigma=1.0, device='cpu'):
        """
        Initializes the loss function (only main component).
        Args:
            main_loss_type (str): Type of the main loss ('bce' or 'kldiv').
            loss_sigma (float): Standard deviation for Gaussian target smoothing (used if main_loss_type is 'bce').
                                Not directly used by KLDivLoss itself but kept for consistency if target generation needs it.
            device (str): Device to perform calculations on.
        """
        super().__init__()
        self.main_loss_type = main_loss_type
        self.device = device
        self.loss_sigma = loss_sigma # Keep sigma if needed for target generation outside

        if main_loss_type == 'bce':
            # Use reduction='mean' to get average loss per batch element
            self.main_criterion = nn.BCEWithLogitsLoss(reduction='mean')
            print(f"  Loss Function: BCEWithLogitsLoss (reduction='mean')")
        elif main_loss_type == 'kldiv':
            # Use reduction='batchmean' which averages over the batch dimension
            self.main_criterion = nn.KLDivLoss(reduction='batchmean', log_target=False)
            print(f"  Loss Function: KLDivLoss (reduction='batchmean', log_target=False)")
        else:
            raise ValueError(f"Unknown main_loss_type: {main_loss_type}")

        self.main_criterion.to(device)

    def forward(self, pred_logits, target_smooth):
        """
        Calculates the main loss.
        Args:
            pred_logits (torch.Tensor): Predicted logits from the model (B, M+1).
            target_smooth (torch.Tensor): Smoothed target tensor (B, M+1).
                                          For BCE, values are 0-1 probabilities.
                                          For KLDiv, should represent a probability distribution (sums to 1 per sample).
        Returns:
            torch.Tensor: The calculated main loss (scalar).
        """
        pred_logits = pred_logits.to(self.device)
        target_smooth = target_smooth.to(self.device)

        if self.main_loss_type == 'bce':
            main_loss = self.main_criterion(pred_logits, target_smooth)
        elif self.main_loss_type == 'kldiv':
            # Ensure target is a valid probability distribution for KLDiv
            target_dist = target_smooth / (target_smooth.sum(dim=-1, keepdim=True) + 1e-9) # Normalize target
            # KLDivLoss expects log-probabilities as input
            pred_logprob = F.log_softmax(pred_logits, dim=-1)
            main_loss = self.main_criterion(pred_logprob, target_dist)

        # Only return the main loss
        return main_loss


# --- 其他辅助函数 ---
def get_latest_experiment_path():
    """Tries to find the latest experiment path from standard locations."""
    try:
        try: script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError: script_dir = os.getcwd() # Fallback for interactive environments
        paths_to_check = [
            '/mnt/sda/liangqiushi/CFARnet/latest_experiment.txt', # Specific absolute path
            os.path.join(script_dir, 'latest_experiment.txt'),         # Path relative to script
            os.path.join(os.getcwd(), 'latest_experiment.txt')         # Path relative to current working dir
        ]
        file_path = next((p for p in paths_to_check if os.path.exists(p)), None)
        if file_path is None: raise FileNotFoundError("未在标准位置找到 latest_experiment.txt。")
        with open(file_path, 'r') as f: return f.read().strip()
    except FileNotFoundError: print("错误：未找到 'latest_experiment.txt'。", flush=True); raise
    except Exception as e: print(f"读取 latest_experiment.txt 时发生错误: {e}", flush=True); raise


def create_timestamp_folders(base_data_root=None):
    """Creates timestamped output folders based on the data root directory."""
    if base_data_root is None:
        try: data_root = get_latest_experiment_path()
        except Exception: print("警告：未找到 latest_experiment.txt 或读取错误... 将使用默认输出路径。", flush=True); data_root = './output/default_experiment'; os.makedirs(data_root, exist_ok=True)
    else: data_root = base_data_root
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S'); norm_data_root = os.path.normpath(data_root)
    # Use experiment name from data root, or a default if data_root is trivial
    experiment_name = os.path.basename(norm_data_root) if norm_data_root not in ['.', '/'] else 'default_experiment'
    # --- Modified Folder Name ---
    # (MODIFIED: Added sampling type info to folder name later based on args)
    output_base_template = os.path.join('.', 'output', f"{experiment_name}_{timestamp}_Train{{SamplingType}}") # Placeholder
    # --- Create Structure ---
    folders = { 'root': data_root, 'output_base_template': output_base_template, 'output_base': None, # Will be finalized later
                'figures': None, 'models': None, 'outputs': None }
    # Directories will be created after sampling type is known
    return folders, timestamp

def set_matplotlib_english():
    """Sets Matplotlib parameters for English labels and consistent font sizes."""
    try:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # A common sans-serif font
        plt.rcParams['axes.unicode_minus'] = False # Display minus signs correctly
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    except Exception as e:
        print(f"设置 Matplotlib 字体时出错: {e}", flush=True)

def count_parameters(model):
    """Counts the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- 指标计算函数 ---

# <<< Top-K Accuracy Calculation Function >>> (Handles padding in true_peak_indices_batch)
def calculate_accuracy_topk(pred_probs, true_peak_indices_batch, k, tolerance):
    """
    计算批次内基于 Top-K 预测的目标检测准确率（召回率@TopK, 容忍度）。
    Args:
        pred_probs (torch.Tensor): 模型输出的预测概率 (B, M+1)。
        true_peak_indices_batch (torch.Tensor): 真实的峰值索引 (B, K_true)，可能包含填充值 (如 -1)。
        k (int): Top-K 中的 K 值。
        tolerance (int): 命中容忍度（子载波数量）。
    Returns:
        float: 该批次的平均 Top-K 命中率/召回率。
    """
    batch_size = pred_probs.shape[0]
    M_plus_1 = pred_probs.shape[1]
    if batch_size == 0:
        return 0.0 # Or NaN? 0.0 seems reasonable for empty batch

    device = pred_probs.device
    total_true_peaks_count = 0
    total_hits = 0

    k = min(k, M_plus_1) # Ensure k is not larger than the prediction dimension
    if k <= 0: return 0.0 # Cannot have non-positive k

    with torch.no_grad():
        # Get the indices of the top k predictions for each sample in the batch
        _, topk_indices_batch = torch.topk(pred_probs, k=k, dim=1) # (B, k)

        for b in range(batch_size):
            true_indices_b = true_peak_indices_batch[b] # (K_true_padded,)
            # Filter out invalid true indices (e.g., padding -1)
            valid_true_mask = (true_indices_b >= 0) & (true_indices_b < M_plus_1)
            valid_true_peaks = true_indices_b[valid_true_mask] # (num_valid_true,)

            num_true = valid_true_peaks.numel()
            if num_true == 0:
                # If a sample has no true peaks, it doesn't contribute to hits or misses
                continue
            total_true_peaks_count += num_true

            pred_indices_b_topk = topk_indices_batch[b] # (k,)
            if pred_indices_b_topk.numel() == 0: # Should not happen if k > 0
                continue

            # Calculate distance between each true peak and all top-k predicted peaks
            # Expand dims for broadcasting: (num_valid_true, 1) vs (1, k) -> (num_valid_true, k)
            dist_matrix = torch.abs(valid_true_peaks.unsqueeze(1) - pred_indices_b_topk.unsqueeze(0))

            # Find the minimum distance from each true peak to *any* of the top-k predictions
            min_dists_to_topk_preds, _ = torch.min(dist_matrix, dim=1) # (num_valid_true,)

            # A true peak is considered "hit" if its nearest top-k prediction is within the tolerance
            hits_b = torch.sum(min_dists_to_topk_preds <= tolerance).item()
            total_hits += hits_b

    # Accuracy (Recall@TopK) is defined as the ratio of hits to the total number of VALID true peaks across the batch
    if total_true_peaks_count == 0: # Avoid division by zero if no valid true peaks in the entire batch
        # If no true peaks, arguably perfect recall (detected all 0 peaks), or undefined. Let's return 1.0.
        accuracy = 1.0
    else:
        accuracy = total_hits / total_true_peaks_count
    return accuracy


# --- 可视化函数 ---
def visualize_predictions(pred_probs_list, target_list, folders, timestamp, M_plus_1,
                          acc_threshold=0.5, acc_tolerance=3, # Tolerance not used here
                          is_target_distribution=False, num_samples=4):
    """Visualizes a few samples of predictions vs targets."""
    if not pred_probs_list or not target_list: print("No data to visualize.", flush=True); return
    if not folders['figures']: print("Figures directory not set for visualization.", flush=True); return
    num_total_samples = len(pred_probs_list); num_samples = min(num_samples, num_total_samples)
    if num_samples == 0: print("Zero samples requested or available for visualization.", flush=True); return
    indices = np.random.choice(num_total_samples, num_samples, replace=False)
    fig, axs = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples), sharex=True, squeeze=False)
    subcarrier_indices = np.arange(M_plus_1)
    target_label = "Target Distribution (Sum=1)" if is_target_distribution else "Smoothed Target (0-1)"
    for i, idx in enumerate(indices):
        pred_probs = pred_probs_list[idx].cpu().numpy(); target_values = target_list[idx].cpu().numpy()
        ax = axs[i, 0]
        ax.plot(subcarrier_indices, pred_probs, label='Predicted Probability', alpha=0.7, color='blue', linewidth=1.5)
        ax.plot(subcarrier_indices, target_values, label=target_label, alpha=0.7, color='red', linestyle='--', linewidth=1.5)
        # Use target values > 0.5 to find approximate true peak locations for visualization
        true_peaks_indices = np.where(target_values > 0.5)[0]
        if len(true_peaks_indices) > 0: ax.plot(subcarrier_indices[true_peaks_indices], target_values[true_peaks_indices], 'ro', markersize=6, label='True Peak Location (Approx)', alpha=0.7)
        # Use accuracy_threshold for marking predicted peaks in the plot (visual aid only)
        pred_peaks_indices_viz = np.where(pred_probs > acc_threshold)[0]
        if len(pred_peaks_indices_viz) > 0: ax.plot(subcarrier_indices[pred_peaks_indices_viz], pred_probs[pred_peaks_indices_viz], 'bx', markersize=6, label=f'Predicted Peak (>{acc_threshold:.2f}, for Viz)', alpha=0.7)
        ax.set_ylabel('Probability / Target Value'); ax.set_title(f'Sample Index in Batch: {idx}'); ax.legend(fontsize='small'); ax.grid(True, alpha=0.3)
        if is_target_distribution: ax.set_ylim(bottom=-0.01)
        else: ax.set_ylim(-0.05, 1.05)
    axs[-1, 0].set_xlabel('Subcarrier Index (M+1)')
    plot_title = f'Predicted Probability vs {target_label} ({timestamp}) - {num_samples} Samples'; plt.suptitle(plot_title); plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plot_filename = f'peak_predictions_{"dist" if is_target_distribution else "smooth"}_{timestamp}.png'
    plot_path = os.path.join(folders['figures'], plot_filename)
    try: plt.savefig(plot_path); print(f"Prediction visualization saved to {plot_path}", flush=True)
    except Exception as e: print(f"Error saving visualization plot: {e}", flush=True)
    plt.close(fig)


# --- 测试函数 ---
def test_model(model: nn.Module,
               test_loader: DataLoader,
               device: torch.device,
               args: argparse.Namespace,
               M_plus_1: int,
               pt_dbm_list_test: List[float], # MODIFIED: Accepts list of dBm values
               noise_std_dev_tensor: torch.Tensor
               ) -> Tuple[Dict[float, Dict[str, float]], List[torch.Tensor], List[torch.Tensor]]:
    """
    在多个指定的发射功率水平上测试模型。

    Args:
        model: 要测试的模型。
        test_loader: 测试数据加载器。
        device: 计算设备。
        args: 包含参数 (loss_type, max_targets, top_k, accuracy_tolerance 等) 的命名空间。
        M_plus_1: 子载波数量 + 1。
        pt_dbm_list_test: 用于测试的发射功率 (dBm) 列表。
        noise_std_dev_tensor: 噪声标准差张量。

    Returns:
        Tuple containing:
        - results_per_pt (Dict[float, Dict[str, float]]): 包含每个 Pt_dBm 的平均损失和准确率的字典。
            Example: {10.0: {'loss': 0.1, 'accuracy': 0.9}, -10.0: {'loss': 0.5, 'accuracy': 0.6}}
        - all_pred_probs_list (List[torch.Tensor]): (来自第一个 Pt) 预测概率列表，用于可视化。
        - all_target_smooth_list (List[torch.Tensor]): (来自第一个 Pt) 平滑目标列表，用于可视化。
    """
    print(f"\n开始多点模型测试 (损失: {args.loss_type}, Pts_test={pt_dbm_list_test} dBm, MaxTargets={args.max_targets}, TopK={args.top_k})...", flush=True)
    model.eval()

    # Initialize results structure
    results_per_pt = {pt: {'loss': 0.0, 'accuracy': 0.0, 'count': 0} for pt in pt_dbm_list_test}

    # Lists for visualization (collect from the first power level only for consistency)
    all_pred_probs_list_viz = []
    all_target_smooth_list_viz = []
    collect_viz_data = True # Flag to collect only during the first power level iteration

    loss_fn = CombinedLoss(main_loss_type=args.loss_type, loss_sigma=args.loss_sigma, device=device)
    print(f"  准确率指标: Top-{args.top_k} 命中率 @ 容差={args.accuracy_tolerance}", flush=True)

    with torch.no_grad():
        # Outer loop for power levels
        for pt_idx, current_pt_dbm in enumerate(pt_dbm_list_test):
            print(f"  --- Testing at Pt = {current_pt_dbm:.1f} dBm ---", flush=True)
            current_pt_linear_mw = 10**(current_pt_dbm / 10.0)
            current_pt_scaling_factor = math.sqrt(current_pt_linear_mw)
            current_pt_scaling_factor_tensor = torch.tensor(current_pt_scaling_factor, dtype=torch.float32, device=device)

            test_pbar = tqdm(test_loader, desc=f"测试 Pt={current_pt_dbm:.1f}dBm", leave=False, file=sys.stdout)
            batch_loss_accum = 0.0
            batch_acc_accum = 0.0
            batch_count = 0

            for batch_idx, batch in enumerate(test_pbar):
                clean_echo = batch['echo'].to(device)
                # m_peak_targets_original shape is (B, args.max_targets) due to Dataset loading
                m_peak_targets_original = batch['m_peak'].to(device)
                batch_size = clean_echo.shape[0]

                # Apply *current fixed* power scaling and noise
                scaled_echo = clean_echo * current_pt_scaling_factor_tensor # Use current factor
                noise = (torch.randn_like(scaled_echo.real) + 1j * torch.randn_like(scaled_echo.imag)) * noise_std_dev_tensor.to(device)
                yecho_input = scaled_echo + noise

                pred_logits, _ = model(yecho_input)

                # Prepare target for loss calculation
                target_smooth_batch = torch.zeros_like(pred_logits, dtype=torch.float32)
                for b in range(batch_size):
                    # peak_indices_b already has shape (args.max_targets,) possibly with -1 padding
                    peak_indices_b = m_peak_targets_original[b]
                    # create_gaussian_target handles filtering invalid indices internally
                    target_smooth_batch[b, :] = create_gaussian_target(peak_indices_b, M_plus_1, args.loss_sigma, device)

                # Calculate loss
                loss = loss_fn(pred_logits, target_smooth_batch)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    batch_loss_accum += loss.item()
                    # Convert logits to probabilities (e.g., using sigmoid for BCE-like interpretation)
                    pred_probs = torch.sigmoid(pred_logits)

                    # Calculate Top-K accuracy metric using args.top_k
                    accuracy = calculate_accuracy_topk(
                        pred_probs,
                        m_peak_targets_original, # Pass the original labels (B, max_targets)
                        k=args.top_k,
                        tolerance=args.accuracy_tolerance
                    )
                    batch_acc_accum += accuracy
                    batch_count += 1

                    # Store results for visualization (only for the first power level)
                    if collect_viz_data:
                        all_pred_probs_list_viz.extend(list(pred_probs.cpu()))
                        all_target_smooth_list_viz.extend(list(target_smooth_batch.cpu()))

                    # Update TQDM postfix
                    if batch_idx % 50 == 0 or batch_idx == len(test_loader) - 1:
                         postfix_dict = {'L': f"{loss.item():.4f}", f'Top{args.top_k}Hit': f"{accuracy:.3f}"}
                         test_pbar.set_postfix(postfix_dict)
                else:
                    print(f"警告：测试批次 {batch_idx} (Pt={current_pt_dbm:.1f}dBm) 中遇到 NaN/Inf 损失。", flush=True)
                    if batch_idx % 50 == 0 or batch_idx == len(test_loader) - 1:
                        test_pbar.set_postfix({'loss': "NaN"})

            # Calculate average results for the current power level
            if batch_count > 0:
                results_per_pt[current_pt_dbm]['loss'] = batch_loss_accum / batch_count
                results_per_pt[current_pt_dbm]['accuracy'] = batch_acc_accum / batch_count
                results_per_pt[current_pt_dbm]['count'] = batch_count
                print(f"    Pt={current_pt_dbm:.1f}dBm - Avg Loss: {results_per_pt[current_pt_dbm]['loss']:.4f}, Avg Top-{args.top_k} Hit: {results_per_pt[current_pt_dbm]['accuracy']:.4f}", flush=True)
            else:
                results_per_pt[current_pt_dbm]['loss'] = float('inf')
                results_per_pt[current_pt_dbm]['accuracy'] = 0.0
                results_per_pt[current_pt_dbm]['count'] = 0
                print(f"    Pt={current_pt_dbm:.1f}dBm - 无有效批次进行评估。", flush=True)

            # Stop collecting visualization data after the first power level
            collect_viz_data = False


    print("\n多点测试结果汇总:", flush=True)
    for pt, results in results_per_pt.items():
        print(f"  Pt = {pt:.1f} dBm:")
        print(f"    平均损失 ({args.loss_type.upper()}): {results['loss']:.4f}")
        print(f"    平均 Top-{args.top_k} 命中率/召回率 (Recall@Top{args.top_k}, Tol={args.accuracy_tolerance}): {results['accuracy']:.4f}")
        print(f"    有效批次数: {results['count']}")

    gc.collect();
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return results_per_pt, all_pred_probs_list_viz, all_target_smooth_list_viz


# --- 主执行函数 ---
def main():
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description='训练峰值索引预测模型 (CNN - 使用随机发射功率训练)')
    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=60, help="训练的总轮数")
    parser.add_argument('--batch_size', type=int, default=100, help="批处理大小")
    parser.add_argument('--lr', type=float, default=1e-4, help='模型 (CNN) 的学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='模型 (CNN) 的权重衰减')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='模型 (CNN) 的梯度裁剪范数上限')
    parser.add_argument('--patience', type=int, default=7, help='早停轮数 (基于最低验证功率点的损失)')
    # --- System/Data Parameters ---
    parser.add_argument('--data_dir', type=str, default=None, help='包含 echoes/ 和 *.npz 文件的数据根目录路径')
    parser.add_argument('--min_pt_dbm', type=float, default=-10.0, help='训练期间使用的最小发射功率 (dBm)')
    parser.add_argument('--max_pt_dbm', type=float, default=30.0, help='训练期间使用的最大发射功率 (dBm)')

    parser.add_argument('--power_sampling', type=str, default='linear', choices=['linear', 'dbm'], help='训练时发射功率的采样方式 (linear: 在mW上均匀采样, dbm: 在dBm上均匀采样)')

    parser.add_argument('--val_pt_dbm_list', type=str, default="-10,0,10", help='用于验证和测试的固定发射功率 (dBm) 列表，以逗号分隔')
    # ---

    parser.add_argument('--max_targets', type=int, default=4, help='数据集中预期的最大目标/用户数量 (K_max)')
    # ---
    # --- Model Hyperparameters ---
    parser.add_argument('--hidden_dim', type=int, default=512, help='CNN 预测头中的隐藏维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='CNN 中的 Dropout 比率')
    # --- Loss/Accuracy Parameters ---
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'kldiv'], help='主损失函数类型')
    parser.add_argument('--loss_sigma', type=float, default=1.0, help='高斯平滑目标的标准差 (BCE时用于目标生成)')

    parser.add_argument('--top_k', type=int, default=4, help='计算准确率时使用的Top-K预测数量 (建议设置为 --max_targets 的值)')
    # ---
    parser.add_argument('--accuracy_threshold', type=float, default=0.5, help='[仅用于可视化] 标记预测峰值的概率阈值')
    parser.add_argument('--accuracy_tolerance', type=int, default=3, help='计算Top-K准确率(命中率)时的容忍度（子载波数量）')
    # --- Execution Control ---
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器使用的工作进程数')
    parser.add_argument('--cuda_device', type=int, default=0, help="要使用的 CUDA 设备 ID (若可用)")
    parser.add_argument('--test_only', action='store_true', help='仅在加载的模型上运行测试')
    parser.add_argument('--load_model', action='store_true', help='尝试加载最佳模型 (若存在于 model_dir 或指定路径)')
    parser.add_argument('--load_model_path', type=str, default=None, help='指定加载 CNN 模型权重的 .pt 文件路径')
    parser.add_argument('--model_dir', type=str, default=None, help='包含 best_model*.pt 的目录路径 (用于加载/查找模型)')


    args = parser.parse_args()

    # --- Parse val_pt_dbm_list ---
    try:
        args.val_pt_dbm_list = sorted([float(p.strip()) for p in args.val_pt_dbm_list.split(',')])
        if not args.val_pt_dbm_list: raise ValueError("验证功率列表不能为空")
    except Exception as e:
        print(f"错误：解析 --val_pt_dbm_list '{args.val_pt_dbm_list}' 失败: {e}", flush=True)
        return
    print(f"用于验证/测试的功率点 (dBm): {args.val_pt_dbm_list}")
    # Determine the critical power level for early stopping (lowest one)
    critical_val_pt_dbm = min(args.val_pt_dbm_list)
    print(f"早停将基于最低验证功率点的损失: {critical_val_pt_dbm:.1f} dBm")

    # --- Sanity check/recommendation for top_k ---
    if args.top_k != args.max_targets:
        print(f"警告: --top_k ({args.top_k}) 与 --max_targets ({args.max_targets}) 不同。评估指标将基于 Top-{args.top_k}。", flush=True)


    # --- Setup Device, Paths, Folders ---
    if torch.cuda.is_available() and args.cuda_device >= 0 :
        try:
            if args.cuda_device < torch.cuda.device_count():
                 device = torch.device(f"cuda:{args.cuda_device}")
                 torch.cuda.set_device(device)
            else:
                 print(f"警告: CUDA 设备 {args.cuda_device} 无效 (只有 {torch.cuda.device_count()} 个设备)。使用 CPU。", flush=True)
                 device = torch.device("cpu")
        except Exception as e:
            print(f"设置 CUDA 设备时出错: {e}。使用 CPU。", flush=True)
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("CUDA 不可用或未请求。使用 CPU。", flush=True)
    print(f"使用设备: {device}", flush=True)
    set_matplotlib_english()

    try: data_root = args.data_dir if args.data_dir else get_latest_experiment_path()
    except Exception: print("错误：未指定数据目录 (--data_dir) 且无法读取 latest_experiment.txt。请提供数据源。", flush=True); return
    print(f"使用数据根目录: {data_root}", flush=True)
    # --- Modified folder creation ---
    folders, timestamp = create_timestamp_folders(data_root) # Template name created
    # --- Finalize output folder name based on sampling type ---
    sampling_type_str = "LinearPt" if args.power_sampling == 'linear' else "DbPt"
    folders['output_base'] = folders['output_base_template'].format(SamplingType=sampling_type_str)
    folders['figures'] = os.path.join(folders['output_base'], 'figures')
    folders['models'] = os.path.join(folders['output_base'], 'models')
    folders['outputs'] = os.path.join(folders['output_base'], 'outputs')
    for folder_key in ['figures', 'models', 'outputs']: os.makedirs(folders[folder_key], exist_ok=True)
    print(f"输出文件位于: {folders['output_base']}", flush=True)
    print(f"训练功率采样方式: {args.power_sampling}")


    # --- Load System Params, Calculate Noise/Scaling ---
    params_file = os.path.join(data_root, 'system_params.npz')
    K_data_param = None # Max targets specified in the data generation params file
    try:
        params_data = np.load(params_file)
        M = int(params_data['M']); Ns = int(params_data['Ns'])
        # --- MODIFIED: Store K from params, compare with args.max_targets ---
        if 'K' in params_data and params_data['K'] is not None:
             K_data_param = int(params_data['K'])
        # ---
        if 'BW' in params_data:
             BW = float(params_data['BW'])
        elif 'f_scs' in params_data and 'M' in params_data:
             BW = float(params_data['f_scs']) * int(params_data['M'])
        else:
             raise KeyError("在 system_params.npz 中未找到 'BW' 或 'f_scs'/'M' 以计算带宽。")
        if M is None or Ns is None or BW is None: raise KeyError("Missing M, Ns, or BW")
    except Exception as e: print(f"加载 system_params.npz 时出错: {e}", flush=True); return
    M_plus_1 = M + 1

    print(f"系统参数: M={M}, Ns={Ns}, BW={BW:.2e}", flush=True)
    # --- Print K from params vs args.max_targets ---
    print(f"命令行预期最大目标数 (args.max_targets): {args.max_targets}")
    if K_data_param is not None:
        print(f"数据参数文件中的 K 值 (system_params.npz['K']): {K_data_param}")
        if K_data_param != args.max_targets:
            print(f"  注意: system_params.npz 中的 K ({K_data_param}) 与 args.max_targets ({args.max_targets}) 不同。将使用 args.max_targets ({args.max_targets}) 来处理数据集标签。")
    else:
        print("数据参数文件中未指定 K 值。")
    # ---
    print(f"准确率指标: Top-{args.top_k} 命中率 @ 容差={args.accuracy_tolerance}")


    # --- Calculate noise standard deviation (independent of Pt) ---
    noise_power_total_linear = K_BOLTZMANN * T_NOISE_KELVIN * BW
    noise_variance_per_component = noise_power_total_linear / 2.0; noise_std_dev = math.sqrt(noise_variance_per_component)
    noise_std_dev_tensor = torch.tensor(noise_std_dev, dtype=torch.float32)
    print(f"噪声标准差 (实/虚): {noise_std_dev:.3e}")

    # --- Calculate reference scaling factors for VALIDATION/TESTING using args.val_pt_dbm_list ---
    min_val_pt_dbm = min(args.val_pt_dbm_list)
    max_val_pt_dbm = max(args.val_pt_dbm_list)
    min_val_pt_linear_mw = 10**(min_val_pt_dbm / 10.0)
    max_val_pt_linear_mw = 10**(max_val_pt_dbm / 10.0)
    min_val_scaling_factor = math.sqrt(min_val_pt_linear_mw)
    max_val_scaling_factor = math.sqrt(max_val_pt_linear_mw)

    print(f"训练功率范围: [{args.min_pt_dbm:.1f}, {args.max_pt_dbm:.1f}] dBm (采样方式: {args.power_sampling})")
    print(f"验证/测试固定功率点 (dBm): {args.val_pt_dbm_list}")
    print(f"  (对应缩放因子范围: [{min_val_scaling_factor:.3f}, {max_val_scaling_factor:.3f}])")


    # --- Datasets and Dataloaders ---
    print("设置数据集和数据加载器...", flush=True)
    try:
        traj_path_check = os.path.join(data_root, 'trajectory_data.npz')
        traj_data_check = np.load(traj_path_check)
        key_to_check = 'm_peak_indices' if 'm_peak_indices' in traj_data_check else 'm_peak'
        num_total_data_available = traj_data_check[key_to_check].shape[0]
        # Get actual K dimension from loaded data if possible
        actual_K_dim_data = traj_data_check[key_to_check].shape[1] if traj_data_check[key_to_check].ndim > 1 else 1
        print(f"  在 trajectory_data.npz 中检测到 {num_total_data_available} 个总样本, K维度={actual_K_dim_data}。")
    except Exception as e:
        print(f"警告：无法从 trajectory_data.npz 确定总样本数或K维度 ({e})。使用默认值 50000 进行分割。")
        num_total_data_available = 50000 # Fallback

    test_frac = 0.15; val_frac = 0.15
    test_size = int(num_total_data_available * test_frac); val_size = int(num_total_data_available * val_frac); train_size = num_total_data_available - test_size - val_size
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        print(f"错误：无效的数据分割大小 (Train={train_size}, Val={val_size}, Test={test_size}) from Total={num_total_data_available}")
        return
    test_start_idx=0; test_end_idx=test_start_idx+test_size-1; val_start_idx=test_end_idx+1; val_end_idx=val_start_idx+val_size-1; train_start_idx=val_end_idx+1; train_end_idx=train_start_idx+train_size-1
    print(f"  目标数据分割: Train=[{train_start_idx}-{train_end_idx}] ({train_size}), Val=[{val_start_idx}-{val_end_idx}] ({val_size}), Test=[{test_start_idx}-{test_end_idx}] ({test_size})")

    try:
        # --- Use args.max_targets for expected_k ---
        test_dataset = ChunkedEchoDataset(data_root, test_start_idx, test_end_idx, expected_k=args.max_targets)
        val_dataset = ChunkedEchoDataset(data_root, val_start_idx, val_end_idx, expected_k=args.max_targets)
        train_dataset = ChunkedEchoDataset(data_root, train_start_idx, train_end_idx, expected_k=args.max_targets)
        # ---
    except Exception as e: print(f"创建数据集时出错: {e}", flush=True); traceback.print_exc(); return

    print(f"实际数据集长度: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    pin_memory=(device.type == 'cuda') # More robust check for CUDA
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)


    # --- Build Model ---
    print("构建模型 (IndexPredictionCNN)...", flush=True)
    model = IndexPredictionCNN(M_plus_1, Ns, args.hidden_dim, args.dropout).to(device)
    print(f"模型参数量: {count_parameters(model)}", flush=True)

    # --- Load Pretrained Model Logic ---
    load_path = None
    if args.load_model or args.model_dir or args.load_model_path:
        print("尝试加载预训练 CNN 模型...", flush=True)
        load_path = args.load_model_path
        # Try finding model in model_dir if load_model_path is not specified
        if not load_path and args.model_dir:
            potential_paths = []
            if os.path.isdir(args.model_dir): # Check if directory exists
                 potential_paths = [f for f in os.listdir(args.model_dir) if f.startswith('best_model') and f.endswith('.pt')]
            if potential_paths:
                 # Sort by modification time, newest first
                 potential_paths.sort(key=lambda x: os.path.getmtime(os.path.join(args.model_dir, x)), reverse=True)
                 load_path = os.path.join(args.model_dir, potential_paths[0])
                 print(f"  在 model_dir 中找到最新模型: {load_path}")
            else:
                 # Fallback to check for 'best_model.pt' specifically
                 potential_path_exact = os.path.join(args.model_dir, 'best_model.pt')
                 if os.path.exists(potential_path_exact):
                         load_path = potential_path_exact
                         print(f"  在 model_dir 中找到模型: {load_path}")
        # Try loading if a path was found or specified
        if load_path and os.path.exists(load_path):
            try:
                 model.load_state_dict(torch.load(load_path, map_location=device));
                 print(f"  已从加载模型状态字典: {load_path}", flush=True)
            except Exception as e: print(f"  警告：加载模型状态字典失败 {load_path}: {e}", flush=True); load_path = None
        elif args.load_model or args.load_model_path or args.model_dir: # Only warn if user intended to load
            print(f"  警告：未找到要加载的模型路径 '{load_path or args.load_model_path or args.model_dir}'。", flush=True)
            load_path = None


    # --- Test Only Mode ---
    if args.test_only:
        model_loaded = load_path and os.path.exists(load_path)
        if not model_loaded: print("错误：测试模式需要成功加载模型 (--load_model_path or --model_dir)。", flush=True); return
        print("\n[测试模式] 评估加载的模型...", flush=True)
        # --- MODIFIED: Call test_model with list ---
        test_results_per_pt, all_probs, all_targets_smooth = test_model(
            model, test_loader, device, args, M_plus_1,
            args.val_pt_dbm_list, # Use the specified list of powers
            noise_std_dev_tensor
        )
        # ---
        print("\n===== 仅测试结果 =====")
        # Print results for each tested power
        for pt, results in test_results_per_pt.items():
            print(f"--- Pt = {pt:.1f} dBm ---")
            print(f"  损失 ({args.loss_type.upper()}): {results['loss']:.4f}")
            print(f"  Top-{args.top_k} 命中率/召回率: {results['accuracy']:.4f}")
        visualize_predictions(all_probs, all_targets_smooth, folders, timestamp + "_test_only", M_plus_1,
                              acc_threshold=args.accuracy_threshold,
                              acc_tolerance=args.accuracy_tolerance,
                              is_target_distribution=(args.loss_type=='kldiv'))
        return

    # --- Training Setup ---
    print(f"\n优化器: AdamW (LR={args.lr:.2e}, WD={args.weight_decay:.1e})", flush=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    print(f"调度器: CosineAnnealingLR (T_max={args.epochs})", flush=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=max(1e-8, args.lr * 0.001))
    print(f"梯度裁剪范数: {args.clip_grad_norm}", flush=True)

    # --- Loss Function ---
    criterion = CombinedLoss(main_loss_type=args.loss_type, loss_sigma=args.loss_sigma, device=device)
    print(f"使用损失函数: {args.loss_type.upper()} (Sigma={args.loss_sigma})")

    # --- Training Loop Initialization ---
    epochs = args.epochs; best_val_loss = float('inf'); early_stop_counter = 0
    train_losses, val_losses_hist = [], {pt: [] for pt in args.val_pt_dbm_list} # Store val loss per pt
    train_acc_hist, val_acc_hist = [], {pt: [] for pt in args.val_pt_dbm_list} # Store val acc per pt
    model_save_path = "" ; saved_heatmap_count = 0
    snr_calculated = False; avg_snr_db_references = {} # Reference SNRs for each val power

    # Calculate min/max linear power for training sampling
    min_pt_linear_mw_train = 10**(args.min_pt_dbm / 10.0)
    max_pt_linear_mw_train = 10**(args.max_pt_dbm / 10.0)

    # --- Training Loop ---
    print(f"\n开始训练 (随机功率 [{args.min_pt_dbm}dBm, {args.max_pt_dbm}dBm], 采样: {args.power_sampling})，共 {epochs} 轮...", flush=True)
    epoch_pbar = tqdm(range(epochs), desc="Overall Progress", file=sys.stdout)

    for epoch in epoch_pbar:
        # =================== Training Phase (with Random Power) ===================
        model.train()
        epoch_train_loss = 0.0
        epoch_train_topk_accuracy_sum = 0.0; train_acc_batches = 0
        train_batch_count = 0; nan_skipped_count = 0
        train_pbar = tqdm(train_loader, desc=f"第 {epoch+1}/{epochs} 轮 [训练]", leave=False, file=sys.stdout)

        for batch_idx, batch in enumerate(train_pbar):
            clean_echo = batch['echo'].to(device)
            m_peak_targets_original = batch['m_peak'].to(device) # Shape (B, args.max_targets)
            batch_size = clean_echo.shape[0]
            optimizer.zero_grad()

            # --- START: Random Power Scaling for this Batch (MODIFIED SAMPLING) ---
            if args.power_sampling == 'linear':
                # Sample uniformly in linear scale (mW)
                current_pt_linear_mw = np.random.uniform(min_pt_linear_mw_train, max_pt_linear_mw_train)
                current_pt_dbm_for_log = 10 * math.log10(current_pt_linear_mw + 1e-9) # For logging only
            else: # args.power_sampling == 'dbm'
                # Sample uniformly in dBm scale
                current_pt_dbm_for_log = np.random.uniform(args.min_pt_dbm, args.max_pt_dbm)
                current_pt_linear_mw = 10**(current_pt_dbm_for_log / 10.0)

            current_pt_scaling_factor = math.sqrt(current_pt_linear_mw)
            current_pt_scaling_factor_tensor = torch.tensor(current_pt_scaling_factor, dtype=torch.float32, device=device)
            scaled_echo = clean_echo * current_pt_scaling_factor_tensor
            # --- END: Random Power Scaling for this Batch ---

            noise = (torch.randn_like(scaled_echo.real) + 1j * torch.randn_like(scaled_echo.imag)) * noise_std_dev_tensor.to(device)
            yecho_input = scaled_echo + noise

            # Calculate reference SNRs once (using fixed validation powers)
            if not snr_calculated and batch_idx == 0 and epoch == 0:
                print("\n--- 参考信噪比 (基于固定验证功率点) ---", flush=True)
                noise_power_watts_theoretic = 2 * (noise_std_dev_tensor**2)
                print(f"  理论噪声功率: {noise_power_watts_theoretic.item():.3e} W", flush=True)
                with torch.no_grad():
                    for ref_pt_dbm in args.val_pt_dbm_list:
                         ref_pt_linear_mw = 10**(ref_pt_dbm / 10.0)
                         ref_scaling_factor = math.sqrt(ref_pt_linear_mw)
                         ref_scaling_factor_tensor = torch.tensor(ref_scaling_factor, dtype=torch.float32, device=device)
                         scaled_echo_reference = clean_echo * ref_scaling_factor_tensor
                         signal_power_watts = (scaled_echo_reference.real**2 + scaled_echo_reference.imag**2).mean(dim=(1,2), keepdim=True)
                         snr_per_sample = signal_power_watts / (noise_power_watts_theoretic + 1e-20)
                         avg_snr_linear_reference = torch.mean(snr_per_sample).item()
                         avg_snr_db_reference = 10 * math.log10(avg_snr_linear_reference) if avg_snr_linear_reference > 1e-20 else -float('inf')
                         avg_snr_db_references[ref_pt_dbm] = avg_snr_db_reference
                         print(f"  Pt={ref_pt_dbm:.1f} dBm -> Avg Signal Power: {torch.mean(signal_power_watts).item():.3e} W, Avg SNR: {avg_snr_db_reference:.2f} dB", flush=True)
                print("  注意: 实际训练信噪比会根据随机功率变化。", flush=True)
                print("--------------------------------------\n", flush=True)
                snr_calculated = True

            pred_logits, Y_magnitude_log = model(yecho_input)

            # Save heatmaps (optional - unchanged logic)
            if saved_heatmap_count < 10 and epoch == 0 :
                B_current = Y_magnitude_log.shape[0]
                for i in range(B_current):
                    if saved_heatmap_count < 10:
                         heatmap_tensor = Y_magnitude_log[i].detach().cpu()
                         try:
                              plt.figure(figsize=(10, 4)); plt.imshow(heatmap_tensor.numpy(), aspect='auto', origin='lower', cmap='viridis'); plt.colorbar(label='Log Magnitude'); plt.xlabel('Frequency Unit (M+1)'); plt.ylabel('Doppler Unit (Ns)'); plt.title(f'Log Magnitude Spectrum Heatmap (Sample {saved_heatmap_count}, Train Pt={current_pt_dbm_for_log:.1f}dBm)'); plt.tight_layout(); # Added Pt to title
                              heatmap_filename = f'heatmap_input_sample_{saved_heatmap_count}_{timestamp}.png'; heatmap_path = os.path.join(folders['figures'], heatmap_filename); plt.savefig(heatmap_path, dpi=100); plt.close()
                         except Exception as e: print(f"  保存热力图样本 {saved_heatmap_count} 时出错: {e}", flush=True)
                         finally: saved_heatmap_count += 1
                    else: break


            # Prepare target for loss
            target_smooth_batch = torch.zeros_like(pred_logits, dtype=torch.float32)
            valid_peaks_per_sample_list_print = [] # For detailed print only
            for b in range(batch_size):
                peak_indices_b = m_peak_targets_original[b] # Shape (max_targets,)
                target_smooth_batch[b, :] = create_gaussian_target(peak_indices_b, M_plus_1, args.loss_sigma, device)
                # Store valid peaks for the detailed printout
                valid_peaks_mask = (peak_indices_b >= 0) & (peak_indices_b < M_plus_1)
                valid_peaks = peak_indices_b[valid_peaks_mask]
                valid_peaks_per_sample_list_print.append(valid_peaks.clone())

            # Calculate loss
            loss = criterion(pred_logits, target_smooth_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                tqdm.write(f"\n警告 (轮 {epoch+1}, 训练批次 {batch_idx}): 损失 NaN/Inf。跳过更新。", file=sys.stdout); nan_skipped_count += 1; optimizer.zero_grad(); continue

            loss.backward()
            if args.clip_grad_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            # --- Accumulate Metrics ---
            pred_probs_detached = torch.sigmoid(pred_logits.detach())
            batch_accuracy = calculate_accuracy_topk(
                pred_probs_detached, m_peak_targets_original, k=args.top_k, tolerance=args.accuracy_tolerance
            )
            epoch_train_loss += loss.item()
            epoch_train_topk_accuracy_sum += batch_accuracy; train_acc_batches += 1
            train_batch_count += 1

            # --- TQDM Update ---
            if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
                 postfix_dict = {'Pt': f"{current_pt_dbm_for_log:.1f}", 'L': f"{loss.item():.3f}", f'Top{args.top_k}Hit': f"{batch_accuracy:.2f}"}
                 train_pbar.set_postfix(postfix_dict)

            # --- Detailed Print --- 
            if batch_idx > 0 and batch_idx % 100 == 0: # Reduced frequency
                 tqdm.write("-" * 20, file=sys.stdout)
                 with torch.no_grad():
                      num_samples_to_print = min(2, batch_size) # Reduced number
                      for s in range(num_samples_to_print):
                           # --- Use args.top_k for k_top_print ---
                           k_top_print = min(args.top_k, pred_probs_detached.shape[1])
                           # ---
                           _, topk_indices_print = torch.topk(pred_probs_detached[s], k=k_top_print)
                           topk_indices_sorted_print, _ = torch.sort(topk_indices_print)
                           pred_peaks_str = np.array2string(topk_indices_sorted_print.cpu().numpy(), precision=0, separator=',', max_line_width=100).replace('\n', '')

                           sample_true_peaks_print = valid_peaks_per_sample_list_print[s]
                           true_pks_s_list = [p.item() for p in sample_true_peaks_print]
                           true_peaks_str = np.array2string(np.sort(np.array(true_pks_s_list)), precision=0, separator=',', max_line_width=100).replace('\n', '')
                           num_true_peaks = len(true_pks_s_list)

                           hits_s_topk_print = 0
                           if sample_true_peaks_print.numel() > 0 and topk_indices_sorted_print.numel() > 0:
                                dist_matrix_s_topk_print = torch.abs(sample_true_peaks_print.unsqueeze(1) - topk_indices_sorted_print.unsqueeze(0))
                                min_dists_s_topk_print, _ = torch.min(dist_matrix_s_topk_print, dim=1)
                                hits_s_topk_print = torch.sum(min_dists_s_topk_print <= args.accuracy_tolerance).item()
                           # --- MODIFIED: Updated print string ---
                           tqdm.write(f"  [Trn 轮 {epoch+1}, B {batch_idx}, S {s}, Pt {current_pt_dbm_for_log:.1f}dBm] Hits(Top{k_top_print}):{hits_s_topk_print}/{num_true_peaks} | T: {true_peaks_str}, P(Top{k_top_print}): {pred_peaks_str}", file=sys.stdout)
                           # ---
                 sys.stdout.flush()
                 tqdm.write("-" * 20, file=sys.stdout)


        # --- End of Training Epoch ---
        avg_train_loss = epoch_train_loss / train_batch_count if train_batch_count > 0 else float('inf')
        avg_train_acc = epoch_train_topk_accuracy_sum / train_acc_batches if train_acc_batches > 0 else 0.0
        train_losses.append(avg_train_loss)
        train_acc_hist.append(avg_train_acc)
        if nan_skipped_count > 0: tqdm.write(f"警告: 轮 {epoch+1} 训练期间跳过了 {nan_skipped_count} 个 NaN/Inf 损失批次。", file=sys.stdout)

        # =================== Validation Phase (Multi-Point Fixed Power) ===================
        # Use the modified test_model function for validation across specified power levels
        val_results_per_pt, _, _ = test_model( # Don't need viz data from validation
            model, val_loader, device, args, M_plus_1,
            args.val_pt_dbm_list, # Use the list of validation powers
            noise_std_dev_tensor
        )

        # Store history and determine loss for early stopping
        current_critical_val_loss = float('inf')
        valid_validation_run = False
        for pt, results in val_results_per_pt.items():
            val_losses_hist[pt].append(results['loss'])
            val_acc_hist[pt].append(results['accuracy'])
            if results['count'] > 0: # Check if validation ran successfully for this pt
                valid_validation_run = True # Mark validation as successful if at least one pt worked
            if pt == critical_val_pt_dbm:
                current_critical_val_loss = results['loss']


        # --- Print Epoch Summary ---
        tqdm.write(f"\nEpoch {epoch+1}/{epochs} Summary:", file=sys.stdout)
        tqdm.write(f"  Train Loss ({args.loss_type.upper()}): {avg_train_loss:.4f} | Avg Train Top-{args.top_k} Hit Rate: {avg_train_acc:.3f}", file=sys.stdout)
        tqdm.write(f"  --- Validation Results ---", file=sys.stdout)
        val_loss_summary = []
        val_acc_summary = []
        for pt in args.val_pt_dbm_list:
             loss = val_results_per_pt[pt]['loss']
             acc = val_results_per_pt[pt]['accuracy']
             tqdm.write(f"    Pt={pt:.1f}dBm: Val Loss = {loss:.4f} | Avg Val Top-{args.top_k} Hit Rate = {acc:.3f}", file=sys.stdout)
             if pt == critical_val_pt_dbm: # Highlight critical loss used for early stopping
                 tqdm.write(f"      (Loss used for early stopping: {loss:.4f})", file=sys.stdout)
             val_loss_summary.append(f"{loss:.3f}")
             val_acc_summary.append(f"{acc:.3f}")

        # Update overall progress bar description
        critical_loss_str = f"{current_critical_val_loss:.3f}" if current_critical_val_loss != float('inf') else "inf"
        epoch_pbar.set_description(f"Tr L:{avg_train_loss:.3f}, V L({critical_val_pt_dbm}dBm):{critical_loss_str}, V Top{args.top_k} Hits:[{'/'.join(val_acc_summary)}]")


        # --- Learning Rate Scheduler Step ---
        scheduler.step()
        if epoch < epochs - 1: lr_next = optimizer.param_groups[0]['lr']; tqdm.write(f"  Epoch {epoch+2} LR: {lr_next:.2e}", file=sys.stdout)

        # --- Save Best Model & Early Stopping (Based on critical low power validation loss) ---
        if valid_validation_run and current_critical_val_loss < best_val_loss:
            tqdm.write(f"  -> Validation loss at {critical_val_pt_dbm}dBm improved ({best_val_loss:.4f} -> {current_critical_val_loss:.4f}). Saving model...", file=sys.stdout)
            best_val_loss = current_critical_val_loss; early_stop_counter = 0
            model_save_path_tmp = os.path.join(folders['models'], f'best_model_{timestamp}.pt')
            try: torch.save(model.state_dict(), model_save_path_tmp); model_save_path = model_save_path_tmp; tqdm.write(f"  Model saved to {model_save_path}", file=sys.stdout)
            except Exception as e: tqdm.write(f"  Error saving model: {e}", file=sys.stdout)
        else:
            if valid_validation_run: # Only increment counter if validation ran successfully
                early_stop_counter += 1; tqdm.write(f"  Validation loss at {critical_val_pt_dbm}dBm did not improve. Counter: {early_stop_counter}/{args.patience}", file=sys.stdout)
                if early_stop_counter >= args.patience: tqdm.write(f"\n[Early Stopping] Stopped training after {args.patience} epochs without improvement at {critical_val_pt_dbm}dBm.", file=sys.stdout); break
            else:
                 # Don't increment counter if validation failed, but maybe warn
                 tqdm.write(f"  Validation did not run properly (NaN/Inf loss?). Early stopping counter remains: {early_stop_counter}.", file=sys.stdout)

        # --- Epoch Cleanup ---
        gc.collect();
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        tqdm.write("-" * 30, file=sys.stdout); sys.stdout.flush()

    # --- End of Training ---
    print("\n训练完成或触发早停。", flush=True)

    # --- Final Evaluation using Best/Last Model ---
    print("加载最佳/最终模型进行最终评估...", flush=True)
    best_model_loaded = False
    final_model_path_used = "Final state (not saved or loaded)"
    if model_save_path and os.path.exists(model_save_path):
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            print(f"已成功加载最佳模型: {model_save_path}", flush=True)
            best_model_loaded = True
            final_model_path_used = model_save_path
        except Exception as e:
            print(f"从 {model_save_path} 加载最佳模型状态字典时出错: {e}", flush=True)
            final_model_path_used = f"Final state (failed to load {os.path.basename(model_save_path)})"
    elif load_path and os.path.exists(load_path) and (args.test_only or not best_model_loaded): # If test_only or training didn't save a better model
         try:
            # Reload the initially loaded model if it exists and no better one was found during training
            model.load_state_dict(torch.load(load_path, map_location=device))
            print(f"重新加载初始模型进行最终测试: {load_path}", flush=True)
            best_model_loaded = True
            final_model_path_used = load_path
         except Exception as e:
            print(f"从 {load_path} 重新加载初始模型状态字典时出错: {e}。使用最终训练状态。", flush=True)
            final_model_path_used = f"Final state (failed to reload {os.path.basename(load_path)})"
    else:
        print("未找到保存的最佳模型或未执行保存/加载。使用最终训练状态。", flush=True)


    # --- Run Final Test (using multiple fixed test powers) ---
    # --- Call test_model with list ---
    final_test_results_per_pt, all_pred_probs_list_viz, all_targets_smooth_list_viz = test_model(
        model, test_loader, device, args, M_plus_1,
        args.val_pt_dbm_list, # Use the specified list of powers for final test
        noise_std_dev_tensor
    )
    # ---
    print("\n===== 最终测试结果 (使用已加载的最佳/最终模型) =====")
    print(f"使用模型: {final_model_path_used}")
    # Print final results for each tested power
    for pt, results in final_test_results_per_pt.items():
        print(f"--- Pt = {pt:.1f} dBm ---")
        print(f"  测试损失 ({args.loss_type.upper()}): {results['loss']:.6f}")
        print(f"  测试 Top-{args.top_k} 命中率/召回率: {results['accuracy']:.4f}")

    # --- Plotting and Saving Results ---
    print("\nGenerating plots and saving results...", flush=True)
    epochs_run = len(train_losses)
    if epochs_run == 0: print("No epochs completed. Skipping plot generation.", flush=True)
    else:
        epoch_axis = range(1, epochs_run + 1)
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True) # Increased height for more legends

        # Plot Loss
        axs[0].plot(epoch_axis, train_losses, label=f'Training Loss ({args.loss_type.upper()})', marker='.', markersize=4, alpha=0.7, color='black')
        # Plot validation loss for each power level
        colors = plt.cm.viridis(np.linspace(0, 1, len(args.val_pt_dbm_list)))
        for i, pt in enumerate(args.val_pt_dbm_list):
            label = f'Val Loss ({pt:.1f} dBm)'
            if pt == critical_val_pt_dbm:
                label += ' [Early Stop]'
            axs[0].plot(epoch_axis, val_losses_hist[pt], label=label, marker='.', markersize=4, alpha=0.7, color=colors[i])

        axs[0].set_ylabel('Loss')
        axs[0].set_title(f'Training History ({timestamp}) - Samp:{args.power_sampling}, Kmax={args.max_targets}') # Added Kmax info
        axs[0].legend(fontsize='small'); axs[0].grid(True)
        # Adjust y-axis limit for loss
        valid_losses = [l for l in train_losses if l is not None and not math.isinf(l) and not math.isnan(l)]
        for pt in args.val_pt_dbm_list:
            valid_losses.extend([l for l in val_losses_hist[pt] if l is not None and not math.isinf(l) and not math.isnan(l)])
        if valid_losses:
            min_loss_plot = max(0, min(valid_losses) - 0.1) if valid_losses else 0
            # Avoid overly high initial losses dominating the plot
            losses_after_epoch1 = [l for e, l in enumerate(valid_losses) if e > 0 or len(valid_losses) == 1] # Skip first epoch if multiple exist
            percentile_loss = np.percentile(losses_after_epoch1, 98) if losses_after_epoch1 else (valid_losses[0] if valid_losses else 1.0)
            max_loss_plot = percentile_loss * 1.5
            if max_loss_plot > min_loss_plot + 1e-3: # Add small buffer to avoid same min/max
                 axs[0].set_ylim(min_loss_plot, max_loss_plot)
            else:
                 axs[0].set_ylim(bottom=min_loss_plot)


        # Plot Top-K Hit Rate
        axs[1].plot(epoch_axis, [a * 100 for a in train_acc_hist], label=f'Training Top-{args.top_k} Hit Rate', marker='.', markersize=4, alpha=0.7, color='black')
        # Plot validation accuracy for each power level
        for i, pt in enumerate(args.val_pt_dbm_list):
            label = f'Val Top-{args.top_k} Hit Rate ({pt:.1f} dBm, Tol={args.accuracy_tolerance})'
            if pt == critical_val_pt_dbm:
                label += ' [Early Stop Ref]'
            axs[1].plot(epoch_axis, [a * 100 for a in val_acc_hist[pt]], label=label, marker='.', markersize=4, alpha=0.7, color=colors[i])

        axs[1].set_ylabel(f'Top-{args.top_k} Hit Rate / Recall (%)'); axs[1].set_xlabel('Epoch'); axs[1].legend(fontsize='small'); axs[1].grid(True); axs[1].set_ylim(-5, 105) # Start slightly below 0

        fig.tight_layout()
        metrics_curve_path = os.path.join(folders['figures'], f'training_curves_{timestamp}.png')
        try: plt.savefig(metrics_curve_path); print(f"Training curves saved to {metrics_curve_path}", flush=True)
        except Exception as e: print(f"Error saving training curves plot: {e}", flush=True)
        plt.close(fig)

    # Visualize final predictions (using data collected during the first power level test)
    visualize_predictions(all_pred_probs_list_viz, all_targets_smooth_list_viz, folders, timestamp + "_final_test", M_plus_1,
                          acc_threshold=args.accuracy_threshold,
                          acc_tolerance=args.accuracy_tolerance,
                          is_target_distribution=(args.loss_type=='kldiv'))

    # --- Save Run Summary (Updated) ---
    summary_path = os.path.join(folders['outputs'], f'summary_{timestamp}.txt')
    print(f"保存摘要至 {summary_path}", flush=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"实验: CNN - 随机功率训练 ({args.power_sampling} sampling)\n") # Added sampling info
        f.write(f"时间戳: {timestamp}\n"); f.write(f"数据根目录: {data_root}\n"); f.write(f"输出基目录: {folders['output_base']}\n"); f.write(f"设备: {device}\n")
        f.write("\n--- 参数 ---\n")
        # --- Exclude num_targets if it somehow exists ---
        args_dict = vars(args)
        args_dict.pop('num_targets', None) # Remove if it exists
        # Format list arg nicely
        args_dict['val_pt_dbm_list'] = f"[{', '.join(map(str, args.val_pt_dbm_list))}]"
        [f.write(f"  {k}: {v}\n") for k, v in sorted(args_dict.items())]
        # ---
        f.write("\n--- 系统参数 ---\n")
        # --- Report K_data_param if available ---
        f.write(f"  M={M}, Ns={Ns}, BW={BW:.2e}\n")
        if K_data_param is not None:
             f.write(f"  参数文件中的 K (K_data_param): {K_data_param}\n")
        f.write(f"  命令行设置的最大目标数 (max_targets): {args.max_targets}\n") # Added
        # ---
        f.write(f"  训练功率范围 (Pt_train): [{args.min_pt_dbm:.1f}, {args.max_pt_dbm:.1f}] dBm (采样: {args.power_sampling})\n") # Added sampling info
        f.write(f"  验证/测试功率点 (Pt_val/test): {args.val_pt_dbm_list} dBm\n")
        f.write(f"  计算出的噪声标准差 (实/虚): {noise_std_dev:.3e}\n")
        f.write(f"  验证/测试功率缩放因子范围: [{min_val_scaling_factor:.3f} at {min_val_pt_dbm}dBm, {max_val_scaling_factor:.3f} at {max_val_pt_dbm}dBm]\n")
        if snr_calculated:
            f.write(f"  计算得到的参考平均信噪比:\n")
            for pt, snr_db in avg_snr_db_references.items():
                 f.write(f"    Pt={pt:.1f}dBm -> {snr_db:.2f} dB\n")
        else: f.write(f"  参考信噪比: 未计算\n")
        f.write("\n--- 准确率参数 ---\n")
        f.write(f"  评估方法: Top-K 命中率\n"); f.write(f"  K 值 (top_k): {args.top_k}\n"); f.write(f"  容差 (Tolerance): {args.accuracy_tolerance}\n"); f.write(f"  可视化阈值 (仅标记): {args.accuracy_threshold}\n")
        f.write("\n--- 数据分割 ---\n"); f.write(f"  目标大小: Train={train_size}, Val={val_size}, Test={test_size}\n"); f.write(f"  报告长度: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}\n")
        f.write("\n--- 模型 ---\n"); f.write(f"  模型参数量: {count_parameters(model)}\n")
        f.write("\n--- 损失函数 ---\n"); f.write(f"  损失类型: {args.loss_type.upper()}\n"); f.write(f"  目标 Sigma: {args.loss_sigma:.2f}\n");
        f.write("\n--- 训练结果 ---\n");
        f.write(f"  运行轮数: {epochs_run}\n");
        f.write(f"  早停基于最低验证功率: {critical_val_pt_dbm:.1f} dBm\n")
        f.write(f"  最佳验证损失 (在 {critical_val_pt_dbm:.1f} dBm 时): {best_val_loss:.6f}\n")
        if epochs_run > 0:
            final_train_loss = train_losses[-1] if train_losses else float('nan'); final_train_acc = train_acc_hist[-1] if train_acc_hist else float('nan')
            f.write(f"  最终训练损失: {final_train_loss:.4f}\n")
            f.write(f"  最终训练 Top-{args.top_k} 命中率/召回率: {final_train_acc:.4f}\n")
            f.write(f"  最终验证结果:\n")
            for pt in args.val_pt_dbm_list:
                 final_val_loss = val_losses_hist[pt][-1] if val_losses_hist[pt] else float('nan')
                 final_val_acc = val_acc_hist[pt][-1] if val_acc_hist[pt] else float('nan')
                 f.write(f"    Pt={pt:.1f}dBm: Loss={final_val_loss:.4f}, Top-{args.top_k} Hit={final_val_acc:.4f}\n")
        else: f.write("  未完成任何训练轮次。\n")
        f.write("\n--- 最终测试结果 ---\n");
        f.write(f"  使用模型: {final_model_path_used}\n")
        for pt, results in final_test_results_per_pt.items():
            f.write(f"  Pt = {pt:.1f} dBm:\n")
            f.write(f"    测试损失 ({args.loss_type.upper()}): {results['loss']:.6f}\n")
            f.write(f"    测试 Top-{args.top_k} 命中率/召回率: {results['accuracy']:.4f}\n")
        f.write("\n--- 保存的文件 ---\n"); f.write(f"  输出目录: {folders['output_base']}\n")
        f.write(f"  最终测试使用的模型文件/状态: {final_model_path_used}\n")
        if epochs_run > 0 and 'metrics_curve_path' in locals() and os.path.exists(metrics_curve_path): f.write(f"  训练曲线: {metrics_curve_path}\n")
        plot_viz_filename = f'peak_predictions_{"dist" if args.loss_type=="kldiv" else "smooth"}_{timestamp}_final_test.png'
        if os.path.exists(os.path.join(folders['figures'], plot_viz_filename)): f.write(f"  最终测试预测图: {os.path.join(folders['figures'], plot_viz_filename)}\n")
        if saved_heatmap_count > 0: f.write(f"  示例输入热力图: {os.path.join(folders['figures'], f'heatmap_input_sample_*_{timestamp}.png')}\n")
        f.write(f"  摘要文件: {summary_path}\n")

    print(f"\n摘要已保存至 {summary_path}", flush=True)
    print("脚本执行完毕。", flush=True)


# --- 脚本入口点 ---
if __name__ == "__main__":
    main()
