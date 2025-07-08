import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import threading
import time
import functools
import itertools



import numpy as np
import torch
from torch.utils.data import Dataset
import os
import math
import gc
import traceback # For error reporting
# Removed time, OrderedDict as cache is removed

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import math
import gc
import traceback # For error reporting

class ChunkedMovingISACDataset(Dataset):
    """
    适配 V2 Chunked 数据格式的数据集 (包含 m_peak_indices)
    - 从 system_params.npz 读取配置
    - **无缓存**: 每次访问都直接从磁盘加载对应 chunk (使用 mmap)
    - **返回完整的 Ns 时间步** 的轨迹数据切片
    - 返回 'm_peak' 索引
    """
    def __init__(self, data_root, start_idx_global=0, end_idx_global=None, verbose=True):
        """
        初始化数据集 (无缓存模式, 返回完整 Ns 轨迹)

        Args:
            data_root (str): 数据根目录.
            start_idx_global (int): 要加载的全局起始样本索引 (包含).
            end_idx_global (int, optional): 要加载的全局结束样本索引 (包含). Defaults to None (加载到最后).
            verbose (bool): 是否打印详细信息.
        """
        super().__init__()
        self.data_root = data_root
        self.verbose = verbose

        # --- 1. 验证路径 ---
        self.traj_path = os.path.join(data_root, 'trajectory_data.npz')
        self.params_path = os.path.join(data_root, 'system_params.npz')
        self.channel_factors_dir = os.path.join(data_root, 'channel_factors')
        self.array_vectors_dir = os.path.join(data_root, 'array_vectors')

        if not os.path.exists(self.params_path): raise FileNotFoundError(f"System parameters missing: {self.params_path}")
        if not os.path.exists(self.traj_path): raise FileNotFoundError(f"Trajectory data missing: {self.traj_path}")
        if not os.path.isdir(self.channel_factors_dir): raise NotADirectoryError(f"Channel dir missing: {self.channel_factors_dir}")
        if not os.path.isdir(self.array_vectors_dir): raise NotADirectoryError(f"Array dir missing: {self.array_vectors_dir}")

        # --- 2. 加载系统参数 ---
        try:
            params = np.load(self.params_path)
            self.total_samples_generated = int(params['sample_num'])
            self.samples_per_chunk = int(params['samples_per_chunk'])
            self.M_saved = int(params['M'])
            self.M_dim = self.M_saved + 1
            self.Ns = int(params['Ns'])
            self.K = int(params['K'])
            self.Nt = int(params['Nt'])
            params.close()
            if self.samples_per_chunk <= 0: raise ValueError("samples_per_chunk must be positive")
        except KeyError as e:
            raise KeyError(f"Missing key '{e}' in system_params.npz.")
        except Exception as e:
            raise RuntimeError(f"Error loading system params: {e}")

        # --- 3. 确定要加载的样本范围 ---
        self.start_idx_global = max(0, start_idx_global)
        if end_idx_global is None:
            self.end_idx_global = self.total_samples_generated - 1
        else:
            self.end_idx_global = min(end_idx_global, self.total_samples_generated - 1)
        if self.start_idx_global > self.end_idx_global:
            raise ValueError(f"Start index ({self.start_idx_global}) > end index ({self.end_idx_global})")
        self.num_samples = self.end_idx_global - self.start_idx_global + 1
        if self.num_samples <= 0: print(f"Warning: No samples selected. Dataset length is 0.")

        if self.verbose:
            # ... (打印信息不变) ...
            print("-" * 30)
            print(f"ChunkedMovingISACDataset Initializing (No Cache, Full Ns Traj):") # Indicate change
            # ... (rest of prints) ...
            print("-" * 30)

        # --- 4. 加载轨迹数据 (只加载需要的切片 - 强制复制) ---
        try:
            traj_data = np.load(self.traj_path)
            req_slice = slice(self.start_idx_global, self.end_idx_global + 1)
            # Load slices and copy
            self.r_traj_slice = traj_data['r_traj'][req_slice].copy()
            self.theta_traj_slice = traj_data['theta_traj'][req_slice].copy()
            self.vr_slice = traj_data['vr'][req_slice].copy()
            vt_data = traj_data.get('vt')
            if vt_data is not None: self.vt_slice = vt_data[req_slice].copy()
            else: self.vt_slice = np.zeros_like(self.vr_slice) # Default if missing

            if 'm_peak_indices' in traj_data:
                self.m_peak_indices_slice = traj_data['m_peak_indices'][req_slice].astype(np.int64).copy()
            elif 'm_peak' in traj_data:
                self.m_peak_indices_slice = traj_data['m_peak'][req_slice].astype(np.int64).copy()
            else: raise KeyError("'m_peak_indices'/'m_peak' not found.")

            traj_data.close()
            # Verify loaded slice length
            if self.r_traj_slice.shape[0] != self.num_samples: raise ValueError("Trajectory slice length mismatch!")
            # Verify trajectory dimensions (should include Ns)
            if self.r_traj_slice.ndim != 3 or self.r_traj_slice.shape[1] != self.Ns or self.r_traj_slice.shape[2] != self.K:
                 print(f"Warning: Loaded r_traj_slice shape {self.r_traj_slice.shape} might not match expected [samples, Ns, K].")
            if self.m_peak_indices_slice.ndim != 2 or self.m_peak_indices_slice.shape[1] != self.K:
                 print(f"Warning: Loaded m_peak_indices_slice shape {self.m_peak_indices_slice.shape} might not match expected [samples, K].")

            if self.verbose: print("Trajectory slice loaded successfully (with copy).")
        except KeyError as e: raise KeyError(f"Missing key '{e}' in {self.traj_path}.")
        except Exception as e: raise RuntimeError(f"Error loading trajectory data slice from {self.traj_path}: {e}")

        # --- 5. No Cache Initialization Needed ---
        pass


    def __len__(self):
        return self.num_samples

    def _get_chunk_info(self, idx):
        if not 0 <= idx < self.num_samples: raise IndexError(f"Index {idx} out of range")
        global_sample_idx = self.start_idx_global + idx
        chunk_idx = global_sample_idx // self.samples_per_chunk
        index_within_chunk = global_sample_idx % self.samples_per_chunk
        return global_sample_idx, chunk_idx, index_within_chunk

    def _close_mmap_safe(self, mmap_obj):
        if mmap_obj is not None and hasattr(mmap_obj, '_mmap') and mmap_obj._mmap is not None:
            try: mmap_obj._mmap.close()
            except Exception: pass

    def __getitem__(self, idx):
        global_sample_idx, chunk_idx, index_within_chunk = self._get_chunk_info(idx)
        chan_chunk_mmap, array_chunk_mmap = None, None

        try:
            # Load chunk data using mmap
            chan_chunk_path = os.path.join(self.channel_factors_dir, f'chan_factors_chunk_{chunk_idx}.npy')
            array_chunk_path = os.path.join(self.array_vectors_dir, f'array_vectors_chunk_{chunk_idx}.npy')
            if not os.path.exists(chan_chunk_path): raise FileNotFoundError(f"File not found: {chan_chunk_path}")
            if not os.path.exists(array_chunk_path): raise FileNotFoundError(f"File not found: {array_chunk_path}")
            chan_chunk_mmap = np.load(chan_chunk_path, mmap_mode='r')
            array_chunk_mmap = np.load(array_chunk_path, mmap_mode='r')

            # Extract sample slices
            chan_factor = chan_chunk_mmap[index_within_chunk].copy()   # [Ns, M_dim, K]
            array_vectors = array_chunk_mmap[index_within_chunk].copy() # [M_dim, K, Nt]

            # Verify dimensions
            if chan_factor.shape != (self.Ns, self.M_dim, self.K): raise ValueError("Unexpected chan_factor shape")
            if array_vectors.shape != (self.M_dim, self.K, self.Nt): raise ValueError("Unexpected a_vectors shape")

            # --- Get corresponding FULL Ns trajectory data slice ---
            # Access the pre-copied slices for the current relative index 'idx'
            # These should have shape [Ns, K] now based on corrected loading.
            r_slice = self.r_traj_slice[idx]          # Expected [Ns, K]
            theta_slice = self.theta_traj_slice[idx]  # Expected [Ns, K]
            vr_slice = self.vr_slice[idx]          # Expected [Ns, K]
            vt_slice = self.vt_slice[idx]          # Expected [Ns, K]
            m_peak = self.m_peak_indices_slice[idx]   # Expected [K]

            # --- Expand array_vectors to match original script's expectation ---
            # (Keep this if your YOLO script expects Ns dimension in a_vectors)
            if array_vectors.ndim == 3: # [M_dim, K, Nt]
                array_vectors_expanded = np.tile(array_vectors[np.newaxis, :, :, :], (self.Ns, 1, 1, 1)) # [Ns, M_dim, K, Nt]
            else: raise ValueError(f"Loaded array_vectors unexpected shape: {array_vectors.shape}")

            # --- Convert to Tensors ---
            chan_factor_t = torch.from_numpy(chan_factor.astype(np.complex64))
            a_vectors_t = torch.from_numpy(array_vectors_expanded.astype(np.complex64))
            # ** RETURN FULL NS TRAJECTORY TENSORS **
            r_t = torch.from_numpy(r_slice.astype(np.float32))          # [Ns, K]
            theta_t = torch.from_numpy(theta_slice.astype(np.float32))  # [Ns, K]
            vr_t = torch.from_numpy(vr_slice.astype(np.float32))        # [Ns, K]
            vt_t = torch.from_numpy(vt_slice.astype(np.float32))        # [Ns, K]
            m_peak_t = torch.from_numpy(m_peak.astype(np.int64))          # [K]

            return {
                'chan_factor': chan_factor_t,    # [Ns, M_dim, K]
                'a_vectors': a_vectors_t,      # [Ns, M_dim, K, Nt] <- Expanded
                'r': r_t,                      # [Ns, K] <- Full Ns dimension
                'theta': theta_t,              # [Ns, K] <- Full Ns dimension
                'vr': vr_t,                    # [Ns, K] <- Full Ns dimension
                'vt': vt_t,                    # [Ns, K] <- Full Ns dimension
                'm_peak': m_peak_t             # [K]
            }

        except Exception as e:
            print(f"Error getting item {idx} (Global Idx: {global_sample_idx}, Chunk: {chunk_idx}, InChunk: {index_within_chunk}): {type(e).__name__} - {e}")
            traceback.print_exc()
            print("Returning dummy data...")
            # Return dummy data matching expected return shapes (with Ns for trajectory)
            return {
                'chan_factor': torch.zeros((self.Ns, self.M_dim, self.K), dtype=torch.complex64),
                'a_vectors': torch.zeros((self.Ns, self.M_dim, self.K, self.Nt), dtype=torch.complex64),
                'r': torch.zeros((self.Ns, self.K), dtype=torch.float32), # Ns dim
                'theta': torch.zeros((self.Ns, self.K), dtype=torch.float32), # Ns dim
                'vr': torch.zeros((self.Ns, self.K), dtype=torch.float32), # Ns dim
                'vt': torch.zeros((self.Ns, self.K), dtype=torch.float32), # Ns dim
                'm_peak': torch.full((self.K,), -1, dtype=torch.int64)
            }
        finally:
            # Close mmap handles
            self._close_mmap_safe(chan_chunk_mmap)
            self._close_mmap_safe(array_chunk_mmap)
            # gc.collect()

def initial_rainbow_beam_ULA_YOLO(N, d, BW, f_scs, fm_list, phi_1, phi_M):
    """
    按照MATLAB代码实现的彩虹波束参数计算
    
    Args:
        N: 天线数量
        d: 天线间距
        BW: 带宽
        M: 子载波数量
        fm_list: 子载波频率列表
        phi_1: 最低频对应的phi角（单位：度）
        phi_M: 最高频对应的phi角（单位：度）
    
    Returns:
        TTD: 时间延迟补偿
        PS: 相位补偿
    """
    c = 3e8
    
    # 按generate.py的计算逻辑
    antenna_idx = (np.arange(N) - (N - 1) / 2)  
    
    # 计算初始PS和TTD
    PS = -fm_list[0] * antenna_idx * d * np.sin(np.deg2rad(phi_1)) / c
    TTD = -PS / BW - ((fm_list[0] + BW) * antenna_idx * d * np.sin(np.deg2rad(phi_M))) / (BW * c)
    PS  = 2.0 * np.pi * PS                # [rad]
    TTD = 1e9 * TTD                        # [ns]
    PS = np.mod(PS , 2*np.pi)         # [-π, π]
    TTD = np.mod(TTD, 1e9/f_scs)    
    
    return PS,TTD

def compute_echo_from_factors_optimized(
    chan_factor,  # [B, Ns, M, K], complex
    a_vectors,    # [B, Ns, M, K, Nt], complex
    PS_T, TTD_T,  # both [B, Nt], real
    PS_R, TTD_R,  # both [B, Nt], real
    fm_list       # [M], real
):
    """
    完整 wideband + PS/TTD 版，PS/TTD 都是 per-batch per-antenna ([B,Nt])。
    """
    B, Ns, M, K = chan_factor.shape
    Nt = a_vectors.shape[-1]
    device = chan_factor.device
    dtype = torch.complex64

    # 保证在同一设备
    fm = fm_list.to(device)          # [M]
    PS_T = PS_T.to(device)           # [B, Nt]
    TTD_T = TTD_T.to(device)         # [B, Nt]
    PS_R = PS_R.to(device)
    TTD_R = TTD_R.to(device)

    f0 = fm[0]                       # 基准频率
    freq_diff = fm - f0              # [M]
    scale = 1e9

    # 1) 先构造 phase_t 和 phase_r: 形状 [B, M, Nt]
    #    phase_t[b,m,n] = -PS_T[b,n] - 2π * TTD_T[b,n] * freq_diff[m]/scale
    phase_t = (- PS_T.unsqueeze(1)                # [B,1,Nt]
               - 2*np.pi * (freq_diff.view(1,M,1) * TTD_T.unsqueeze(1)) / scale)  # [B,M,Nt]
    phase_r = (- PS_R.unsqueeze(1)
               - 2*np.pi * (freq_diff.view(1,M,1) * TTD_R.unsqueeze(1)) / scale)

    # 2) 计算 BF_t_all, BF_r_all: 形状 [B, M, Nt]
    BF_t_all = torch.exp(1j * phase_t).to(dtype)  # [B,M,Nt]
    BF_r_all = torch.exp(1j * phase_r).to(dtype)

    # 3) 初始化 echo
    echo = torch.zeros((B, Ns, M), dtype=dtype, device=device)

    # 4) 对每个子载波循环（也可改写为向量化，但这样更直观）
    for m in range(M):
        # 取出第 m 个子载波的 BF 向量：[B, Nt] -> [B,1,1,Nt]
        bf_t = BF_t_all[:, m, :].view(B, 1, 1, Nt)
        bf_r = BF_r_all[:, m, :].view(B, 1, 1, Nt)

        # 信道和 steering vectors
        h_m = chan_factor[:, :, m, :]       # [B, Ns, K]
        a_m = a_vectors[:, :, m, :, :]      # [B, Ns, K, Nt]

        # 发射端贡献: conj(a) · bf_t -> [B, Ns, K]
        tx = torch.sum(a_m.conj() * bf_t, dim=-1)
        # 接收端贡献: bf_r^H · a -> [B, Ns, K]
        rx = torch.sum(bf_r.conj() * a_m, dim=-1)
        # Pt=(1000)#mW    
        # import math
        # 整体 echo
        echo[:, :, m] = torch.sum(h_m * tx * rx, dim=-1)  # [B, Ns]

    return echo

def load_system_params(param_file):
    data = np.load(param_file)
    D_rayleigh = data['D_rayleigh']
    fc = data['fc']
    f_scs = data['f_scs']
    Delta_T = data['Delta_T']
    M = int(data['M'])
    K = int(data['K'])
    Ns = int(data['Ns'])  # OFDM符号数量
    Nt = int(data['Nt'])
    Nr = int(data['Nr'])
    d = data['d']
    lambda_c = 3e8 / fc
    fm_list = data['fm_list']
    print(f"[Info] Loaded system params: Nt={Nt}, Nr={Nr}, M={M}, Ns={Ns}, fc={fc/1e9} GHz")
    return Nt, Nr, M, Ns, fc, f_scs, Delta_T, D_rayleigh, K, d, lambda_c, fm_list

def yolo_detection(y_echo, K, fc, f_scs, Ns, M, phi_start_deg=-60, phi_end_deg=60):
    """
    YOLO算法实现，用于检测多个目标的位置和速度
    
    Args:
        y_echo: 输入波形数据 [num_samples, Ns, M]
        K: 目标数量
        fc: 中心频率
        f_scs: 子载波间隔
        Ns: OFDM符号数量
        M: 子载波数量
        phi_start_deg: 最小角度（默认-60度）
        phi_end_deg: 最大角度（默认60度）
        
    Returns:
        est_theta: 估计的角度 [num_samples, K]
        est_r: 估计的距离 [num_samples, K]
        est_v: 估计的速度 [num_samples, K]
        est_subcarriers: 估计的子载波索引 [num_samples, K]
    """
    # 系统参数
    c = 3e8         # 光速
    Ts = 1 / f_scs  # OFDM符号时长
    BW = M * f_scs  # 带宽
    
    # 2D-CFAR 参数
    guard_size_doppler = 2
    ref_size_doppler   = 4
    guard_size_angle   = 1
    ref_size_angle     = 4
    alpha = 1  # 降低阈值因子，使检测更敏感

    # YOLO/MUSIC 估计参数
    sidelobe_window = 10  # 局部窗口大小

    # 更精细的搜索范围
    v_search_range = np.linspace(-12, 12, 2001)  # 速度搜索网格
    r_search_range = np.arange(15, 200+0.01, 0.01)  # 距离搜索，步长0.01

    # 初始化结果数组
    num_samp = y_echo.shape[0]
    est_theta = np.zeros((num_samp, K))
    est_r = np.zeros((num_samp, K))
    est_v = np.zeros((num_samp, K))
    est_subcarriers = np.zeros((num_samp, K), dtype=int)

    # 主循环，对每个样本做 YOLO 检测
    for idx in range(num_samp):
        # 1) 提取该样本回波，形状 (Ns, M)
        Y_sample = y_echo[idx, :, :]

        # 2) 静态杂波滤除
        Y_dynamic = Y_sample.copy()

        # 3) 对符号维（行方向）做 FFT，并 fftshift 后得到多普勒信息
        Y_fft = np.fft.fft(Y_dynamic, axis=0)
        Y_AD = np.fft.fftshift(Y_fft, axes=0)
        G_AD = np.abs(Y_AD)  # 功率谱

        # 4) 2D-CFAR 检测峰值（多普勒×子载波平面）
        cfar_mask = np.zeros_like(G_AD, dtype=bool)
        Ns_dim, M_dim = G_AD.shape

        # 计算检测范围
        d_min = ref_size_doppler + guard_size_doppler
        d_max = Ns_dim - (ref_size_doppler + guard_size_doppler) - 1
        m_min = ref_size_angle + guard_size_angle
        m_max = M_dim - (ref_size_angle + guard_size_angle) - 1

        # 2D-CFAR检测
        for d in range(d_min, d_max+1):
            for m in range(m_min, m_max+1):
                # 参考窗口
                d_win = np.arange(d - (ref_size_doppler + guard_size_doppler),
                                d + (ref_size_doppler + guard_size_doppler) + 1)
                m_win = np.arange(m - (ref_size_angle + guard_size_angle),
                                m + (ref_size_angle + guard_size_angle) + 1)
                ref_window = G_AD[np.ix_(d_win, m_win)]
                total_ref_count = (2*ref_size_doppler + 2*guard_size_doppler + 1) * (2*ref_size_angle + 2*guard_size_angle + 1)

                # 守卫单元
                d_guard = np.arange(d - guard_size_doppler, d + guard_size_doppler + 1)
                m_guard = np.arange(m - guard_size_angle, m + guard_size_angle + 1)
                guard_cells = G_AD[np.ix_(d_guard, m_guard)]
                guard_cells_count = (2*guard_size_doppler + 1) * (2*guard_size_angle + 1)

                noise_mean = (np.sum(ref_window) - np.sum(guard_cells)) / (total_ref_count - guard_cells_count)
                threshold = alpha * noise_mean

                if G_AD[d, m] > threshold:
                    cfar_mask[d, m] = True

        # 找到候选峰值点
        cand_indices = np.argwhere(cfar_mask)
        if cand_indices.size == 0:
            continue

        cand_d = cand_indices[:, 0]
        cand_m = cand_indices[:, 1]
        cand_power = G_AD[cand_d, cand_m]
        local_window = 10
        kept_idx = np.zeros(len(cand_d), dtype=bool)

        # 局部极大值检测
        for i in range(len(cand_d)):
            d_val = cand_d[i]
            m_val = cand_m[i]
            val_ = cand_power[i]
            if (d_val <= local_window) or (d_val >= Ns_dim - local_window - 1) or \
               (m_val <= local_window) or (m_val >= M_dim - local_window - 1):
                continue

            d_neigh = np.arange(d_val - local_window, d_val + local_window + 1)
            m_neigh = np.arange(m_val - local_window, m_val + local_window + 1)
            sub_mat = G_AD[np.ix_(d_neigh, m_neigh)]
            if val_ >= np.max(sub_mat):
                kept_idx[i] = True

        # 筛选候选
        cand_d2 = cand_d[kept_idx]
        cand_m2 = cand_m[kept_idx]
        cand_power2 = cand_power[kept_idx]

        if cand_d2.size == 0:
            continue

        # 对候选点按功率降序排序，并排除相邻子载波
        sort_idx = np.argsort(-cand_power2)
        final_idx = []
        sidelobe_exclude = 150  # 排除相邻子载波

        for i in sort_idx:
            current_m = cand_m2[i]
            if len(final_idx) == 0:
                final_idx.append(i)
            else:
                too_close = False
                for j in final_idx:
                    if abs(current_m - cand_m2[j]) < sidelobe_exclude:
                        too_close = True
                        break
                if not too_close:
                    final_idx.append(i)
            if len(final_idx) >= K:
                break

        K_eff = len(final_idx)
        cand_d_final = cand_d2[final_idx]
        cand_m_final = cand_m2[final_idx]

        # 针对每个候选目标进行后续估计
        for candidate_idx in range(K_eff):
            if candidate_idx >= K:
                break

            # 5.1 角度估计
            m_peak = cand_m_final[candidate_idx]
            fm_base = m_peak * f_scs
            term1 = ((BW - fm_base) * fc / BW / (fm_base + fc)) * np.sin(np.deg2rad(phi_start_deg))
            term2 = ((BW + fc) * fm_base / BW / (fm_base + fc)) * np.sin(np.deg2rad(phi_end_deg))
            angle_value = np.rad2deg(np.arcsin(term1 + term2))
            est_theta[idx, candidate_idx] = angle_value
            est_subcarriers[idx, candidate_idx] = m_peak

            # 5.2 距离与速度估计
            m_min_local = max(0, m_peak - sidelobe_window)
            m_max_local = min(M - 1, m_peak + sidelobe_window)
            sub_index = np.arange(m_min_local, m_max_local + 1)
            Y_RD_raw = Y_dynamic[:, sub_index]
            Y_RD = Y_RD_raw / (np.abs(Y_RD_raw) + 1e-10)

            # (a) 速度方向 MUSIC 处理
            R_D = (Y_RD @ Y_RD.conj().T) / len(sub_index)
            eigvals_D, U_D = np.linalg.eig(R_D)
            idx_sort_D = np.argsort(-eigvals_D.real)
            U_D = U_D[:, idx_sort_D]
            rank_sig = K
            U_vD = U_D[:, rank_sig:]

            # MUSIC 速度谱估计
            F_v = np.zeros_like(v_search_range, dtype=float)
            for ii, v_cand in enumerate(v_search_range):
                a_v = np.exp(1j * (4 * np.pi * fc * v_cand * Ts / c) * np.arange(Ns))
                a_v = a_v / (np.linalg.norm(a_v) + 1e-10)
                P_v = np.abs(a_v.conj().T @ (U_vD @ U_vD.conj().T) @ a_v)
                F_v[ii] = 1 / (P_v + 1e-10)

            # 局部峰值检测
            v_local_window = 15
            F_v_peaks = np.zeros_like(F_v)
            for ii in range(len(F_v)):
                if ii < v_local_window or ii >= len(F_v) - v_local_window:
                    continue
                local_win = F_v[ii - v_local_window: ii + v_local_window + 1]
                if F_v[ii] >= np.max(local_win):
                    F_v_peaks[ii] = F_v[ii]
            max_idx_v = np.argmax(F_v_peaks)
            v_est = v_search_range[max_idx_v]
            est_v[idx, candidate_idx] = v_est

            # (b) 距离方向 MUSIC 处理
            R_r = (Y_RD.T @ np.conj(Y_RD)) / Ns
            eigvals_R, U_R = np.linalg.eig(R_r)
            idx_sort_R = np.argsort(-eigvals_R.real)
            U_R = U_R[:, idx_sort_R]
            rank_sig = K
            U_rR = U_R[:, rank_sig:]

            F_r = np.zeros_like(r_search_range, dtype=float)
            for ii, rr_cand in enumerate(r_search_range):
                a_r = np.exp(-1j * (4 * np.pi * rr_cand / c) * f_scs * np.arange(len(sub_index)))
                a_r = a_r / np.linalg.norm(a_r)
                F_r[ii] = 1 / (np.abs(a_r.conj().T @ (U_rR @ U_rR.conj().T) @ a_r) + 1e-10)
            max_idx_r = np.argmax(F_r)
            est_r[idx, candidate_idx] = r_search_range[max_idx_r]

    # 对各个估计结果按行排序
    sort_indices = np.argsort(est_theta, axis=1)
    est_subcarriers_sorted = np.zeros_like(est_subcarriers)
    for i in range(num_samp):
        est_subcarriers_sorted[i] = est_subcarriers[i, sort_indices[i]]

    est_theta_sorted = np.sort(est_theta, axis=1)
    est_r_sorted = np.sort(est_r, axis=1)
    est_v_sorted = np.sort(est_v, axis=1)

    return est_theta_sorted, est_r_sorted, est_v_sorted, est_subcarriers_sorted
