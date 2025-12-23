# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm

# ================= 配置区域 =================
DATA_ROOT = "data/1203_k3d3_nonlinear"
OUTPUT_DIR = "debug_output_v2"       # 新的输出文件夹
NMS_WINDOW = 10                      # 峰值最小间距 (Distance)
HEIGHT_THRES = 1e-18                 # 极低阈值，解决之前的漏检问题
TOP_K = 3                            # 为了Debug，多显示几个候选点
# ============================================

def detect_peaks_scipy(power_profile, distance=10, height=0):
    """
    使用 scipy.signal.find_peaks 找峰值
    并按能量从大到小排序返回
    """
    # 转 numpy
    if isinstance(power_profile, torch.Tensor):
        profile_np = power_profile.cpu().numpy()
    else:
        profile_np = power_profile

    # 1. 找峰值 (关键：height设为极小值)
    peaks, properties = find_peaks(
        profile_np, 
        height=height, 
        distance=distance
    )
    
    if len(peaks) == 0:
        return [], []

    # 2. 按能量排序 (从大到小)
    peak_heights = properties['peak_heights']
    sorted_indices = np.argsort(peak_heights)[::-1]
    
    sorted_peaks = peaks[sorted_indices]
    sorted_vals = peak_heights[sorted_indices]
    
    return sorted_peaks, sorted_vals

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- 启动 Debug V2 (Scipy Find Peaks) ---")
    print(f"数据路径: {DATA_ROOT}")
    
    # 1. 加载数据
    try:
        traj_path = os.path.join(DATA_ROOT, 'trajectory_data.npz')
        traj_data = np.load(traj_path)
        # 兼容 key
        if 'm_peak_indices' in traj_data:
            geom_labels_all = traj_data['m_peak_indices']
        else:
            geom_labels_all = traj_data['m_peak']
            
        echo_file = os.path.join(DATA_ROOT, 'echoes', 'echo_chunk_0.npy')
        print(f"Loading: {echo_file}")
        # mmap_mode='r' 加速读取
        echo_chunk = np.load(echo_file, mmap_mode='r') 
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"正在生成前 10 张对比图至 {OUTPUT_DIR} ...")

    # 2. 循环处理前10个样本
    for i in tqdm(range(10)):
        # A. 信号处理
        clean_echo = torch.from_numpy(echo_chunk[i]).to(torch.complex64)
        power_profile = torch.mean(torch.abs(clean_echo)**2, dim=0)
        profile_np = power_profile.numpy()
        
        # B. 几何真值
        geom_indices = geom_labels_all[i]
        
        # C. 新的峰值检测 (Scipy)
        det_indices, det_vals = detect_peaks_scipy(
            power_profile, 
            distance=NMS_WINDOW, 
            height=HEIGHT_THRES
        )
        
        # 取前 Top-K 用于展示
        show_indices = det_indices[:TOP_K]
        show_vals = det_vals[:TOP_K]

        # D. 计算偏差信息 (用于标题)
        diff_strs = []
        for g_idx in geom_indices:
            g_idx = int(g_idx)
            if g_idx < 0: continue
            
            # 找最近的检测点
            if len(det_indices) > 0:
                dists = np.abs(det_indices - g_idx)
                min_dist = np.min(dists)
                nearest_idx = det_indices[np.argmin(dists)]
                diff_strs.append(f"G{g_idx}->D{nearest_idx}(Err:{min_dist})")
            else:
                diff_strs.append(f"G{g_idx}->Miss")
        
        title_str = f"Sample {i} | " + " | ".join(diff_strs)

        # E. 绘图
        plt.figure(figsize=(12, 6))
        
        # 1. 信号谱 (蓝线)
        plt.plot(profile_np, color='royalblue', alpha=0.6, linewidth=1, label='Signal Power')
        
        # 2. 几何公式位置 (红虚线)
        for idx, g_idx in enumerate(geom_indices):
            if g_idx >= 0:
                plt.axvline(x=g_idx, color='red', linestyle='--', alpha=0.8, label='Geometric Formula' if idx==0 else "")
        
        # 3. 检测到的峰值 (绿叉)
        if len(show_indices) > 0:
            plt.scatter(show_indices, show_vals, color='lime', marker='x', s=120, linewidth=2, zorder=10, label='Scipy FindPeaks')
            # 标注坐标
            for p_idx, p_val in zip(show_indices, show_vals):
                plt.text(p_idx, p_val, f"{p_idx}", color='green', fontweight='bold', ha='center', va='bottom')

        plt.title(title_str, fontsize=10)
        plt.xlabel("Subcarrier Index (m)")
        plt.ylabel("Power Amplitude")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(OUTPUT_DIR, f"debug_scipy_{i}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

    print(f"\n[完成] 请打开文件夹 '{OUTPUT_DIR}' 查看结果。")
    print("预期结果：绿叉(X)应该完美且准确地标记在蓝色波峰的顶点上。")

if __name__ == "__main__":
    main()