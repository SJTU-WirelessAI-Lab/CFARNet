# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from tqdm import tqdm
import math
from scipy.signal import find_peaks

# =================配置区域=================
# 数据根目录
DATA_ROOT = "data/1203_k3d3_nonlinear"

# 校验参数
TOLERANCE = 50       # 允许的最大误差范围 (子载波数量)
NMS_WINDOW = 10      # 峰值最小间距 (Distance)
HEIGHT_THRES = 1e-18 # [关键] 极低阈值，防止漏掉微弱信号
# ==========================================

def generate_validated_label_scipy(power_profile, geom_indices, max_targets, tolerance=40, nms_window=10):
    """
    使用 Scipy Find Peaks 生成并校验标签
    """
    # 1. 准备数据 (转 Numpy)
    if isinstance(power_profile, torch.Tensor):
        profile_np = power_profile.cpu().numpy()
    else:
        profile_np = power_profile
        
    M = len(profile_np)
    final_indices = []
    match_count = 0
    total_valid_targets = 0

    # --- 步骤 1: 使用 Scipy 寻找所有候选峰值 ---
    # height: 设置极低阈值，解决之前检测不到信号的问题
    # distance: 等同于 NMS 窗口，抑制旁瓣
    peaks, properties = find_peaks(
        profile_np, 
        height=HEIGHT_THRES, 
        distance=nms_window
    )
    
    # [关键] 按能量从大到小排序候选点
    # find_peaks 默认按 index 排序，我们必须按 peak_heights 排序
    if len(peaks) > 0:
        peak_heights = properties['peak_heights']
        sorted_args = np.argsort(peak_heights)[::-1] # 降序
        candidates = peaks[sorted_args]
    else:
        candidates = np.array([])

    # --- 步骤 2: 与几何锚点进行匹配校验 ---
    for g_idx in geom_indices:
        g_idx = int(g_idx)
        
        # 忽略无效目标
        if g_idx < 0:
            final_indices.append(-1)
            continue
            
        total_valid_targets += 1
        
        # 在候选点中找最近的一个
        best_nms_idx = -1
        min_dist = float('inf')
        
        if len(candidates) > 0:
            dists = np.abs(candidates - g_idx)
            min_dist_idx = np.argmin(dists)
            min_dist = dists[min_dist_idx]
            best_nms_idx = candidates[min_dist_idx]

        # --- 步骤 3: 判定逻辑 ---
        if min_dist <= tolerance:
            # [Case A] 匹配成功
            # 说明几何公式虽然有偏差，但在容忍范围内找到了真实的物理波峰
            # 我们采纳这个物理波峰作为 Ground Truth
            final_indices.append(best_nms_idx)
            match_count += 1
        else:
            # [Case B] 匹配失败 (距离太远)
            # 可能是 NMS 漏检了，或者偏差真的很大
            # 策略：强制在几何锚点周围局部搜索最大值 (保底)
            start = max(0, g_idx - tolerance)
            end = min(M, g_idx + tolerance + 1)
            local_region = profile_np[start:end]
            
            if len(local_region) > 0:
                local_max_offset = np.argmax(local_region)
                final_indices.append(start + local_max_offset)
            else:
                final_indices.append(g_idx)

    # 补齐长度
    while len(final_indices) < max_targets:
        final_indices.append(-1)
        
    return np.array(final_indices), match_count, total_valid_targets

def main():
    print(f"--- 开始生成修正标签 (基于 Scipy FindPeaks) ---")
    print(f"数据目录: {DATA_ROOT}")
    print(f"校验容差: ±{TOLERANCE} 子载波")
    print(f"检测阈值: {HEIGHT_THRES}")
    
    # 1. 加载参数
    params_path = os.path.join(DATA_ROOT, 'system_params.npz')
    traj_path = os.path.join(DATA_ROOT, 'trajectory_data.npz')
    
    if not os.path.exists(params_path) or not os.path.exists(traj_path):
        print("错误: 文件缺失")
        return

    sys_params = np.load(params_path)
    traj_data = np.load(traj_path)
    
    chunk_size = int(sys_params['samples_per_chunk']) if 'samples_per_chunk' in sys_params else int(sys_params['chunk_size'])
    total_samples = int(sys_params['sample_num'])
    max_targets = int(sys_params['K'])
    
    # 兼容标签名
    if 'm_peak_indices' in traj_data:
        geom_labels_all = traj_data['m_peak_indices']
    else:
        geom_labels_all = traj_data['m_peak']
        
    print(f"总样本数: {total_samples}, Chunk大小: {chunk_size}")

    # 2. 准备容器
    corrected_labels_all = np.full((total_samples, max_targets), -1, dtype=np.int32)
    global_matches = 0
    global_valid_targets = 0
    
    # 3. 处理 Chunk
    echoes_dir = os.path.join(DATA_ROOT, 'echoes')
    num_chunks = math.ceil(total_samples / chunk_size)
    
    for chunk_idx in tqdm(range(num_chunks), desc="Processing"):
        echo_file = os.path.join(echoes_dir, f'echo_chunk_{chunk_idx}.npy')
        if not os.path.exists(echo_file): continue
            
        # 读取回波
        echo_chunk = np.load(echo_file, mmap_mode='r')
        
        # 计算索引
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_samples)
        curr_bs = end_idx - start_idx
        
        geom_labels_batch = geom_labels_all[start_idx:end_idx]
        
        for i in range(curr_bs):
            # 获取信号谱
            clean_echo = torch.from_numpy(echo_chunk[i]).to(torch.complex64)
            power_profile = torch.mean(torch.abs(clean_echo)**2, dim=0) # [M]
            
            # 生成标签
            new_indices, matches, valid_cnt = generate_validated_label_scipy(
                power_profile, 
                geom_labels_batch[i], 
                max_targets, 
                tolerance=TOLERANCE, 
                nms_window=NMS_WINDOW
            )
            
            corrected_labels_all[start_idx + i] = new_indices
            global_matches += matches
            global_valid_targets += valid_cnt

    # 4. 统计与保存
    acc = global_matches / global_valid_targets if global_valid_targets > 0 else 0
    
    print("\n" + "="*50)
    print(f"统计结果 (Scipy Version)")
    print(f"总目标数: {global_valid_targets}")
    print(f"成功匹配 (Err<={TOLERANCE}): {global_matches}")
    print(f"几何公式准确率: {acc*100:.2f}%")
    print("="*50)
    
    save_path = os.path.join(DATA_ROOT, 'corrected_labels.npz')
    np.savez(save_path, 
             corrected_indices=corrected_labels_all, 
             accuracy_stat=acc)
    print(f"结果已保存: {save_path}")

if __name__ == "__main__":
    main()