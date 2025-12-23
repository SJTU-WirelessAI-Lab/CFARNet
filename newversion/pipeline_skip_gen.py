# -*- coding: utf-8 -*-
import subprocess
import os
import sys
import queue
import re
from concurrent.futures import ThreadPoolExecutor

# ================= 配置区域 =================
# 获取当前脚本所在目录，确保能找到同目录下的其他脚本
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取当前 Python 解释器路径
PYTHON_EXEC = sys.executable

# 脚本文件名 (使用绝对路径)
SCRIPT_TRAIN = os.path.join(BASE_DIR, "train.py")
SCRIPT_TEST_YOLO = os.path.join(BASE_DIR, "YOLO_baseline.py")
SCRIPT_TEST_CFAR = os.path.join(BASE_DIR, "CFARNet.py")

# 模式开关
ONLY_TEST_YOLO = True  # <--- 【开关】：True=只测YOLO, False=训练+全测试

# 待处理的数据文件夹列表 (绝对路径)
EXISTING_DATA_DIRS = [
    "/mnt/nvme0n1/lqs/cfarnet_new/newversion/data/auto_pipeline_k3_d1_20251214_174958",
    "/mnt/nvme0n1/lqs/cfarnet_new/newversion/data/auto_pipeline_k3_d10_20251214_232659",
    "/mnt/nvme0n1/lqs/cfarnet_new/newversion/data/auto_pipeline_k3_d3_20251214_193021",
    "/mnt/nvme0n1/lqs/cfarnet_new/newversion/data/auto_pipeline_k3_d5_20251214_212326"
]

# 实验参数
PT_LIST = [40, 50, 60]
CUDA_DEVICES = [0, 1, 2, 3, 4, 5, 6, 7]

# 通用参数
EPOCHS = 60
BATCH_SIZE = 64
NUM_TEST_SAMPLES = 7500
# ===========================================

# GPU 资源队列
gpu_queue = queue.Queue()
for device_id in CUDA_DEVICES:
    gpu_queue.put(device_id)

def run_command(command, log_file_path):
    """
    执行命令并将输出实时写入指定的日志文件。
    """
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except:
            pass

    with open(log_file_path, "w", encoding='utf-8') as f:
        f.write(f"CMD: {command}\n{'='*40}\n")
        f.flush()
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        for line in process.stdout:
            f.write(line)
            f.flush()
            
        process.wait()
        
        if process.returncode != 0:
            err_msg = f"\n[ERROR] Command failed with return code {process.returncode}.\n"
            f.write(err_msg)
            print(f"Error executing: {command}. See log: {log_file_path}")
            raise RuntimeError(f"Command failed: {command}")

def worker_routine(data_path, pt, k, gpu_id):
    """
    单个 Worker 的工作流程
    """
    # 创建独立的实验子文件夹
    exp_subfolder = os.path.join(data_path, f"experiment_pt{int(pt)}")
    os.makedirs(exp_subfolder, exist_ok=True)

    task_name = "YOLO_ONLY" if ONLY_TEST_YOLO else "TRAIN_AND_TEST"
    print(f"   [Worker Start] {task_name} | Pt={pt}dBm | GPU {gpu_id} | Data: {os.path.basename(data_path)}")
    
    log_train = os.path.join(exp_subfolder, f"log_train_pt{int(pt)}.txt")
    log_yolo = os.path.join(exp_subfolder, f"log_test_yolo_pt{int(pt)}.txt")
    log_cfar = os.path.join(exp_subfolder, f"log_test_cfarnet_pt{int(pt)}.txt")
    
    try:
        model_weight_path = os.path.join(exp_subfolder, f"model_pt{int(pt)}_best.pt")

        # --- 1. 训练 (Train) ---
        if not ONLY_TEST_YOLO:
            # 检查是否已经存在模型，如果存在可以选择跳过，或者覆盖
            # 这里默认覆盖训练
            cmd_train = (
                f"\"{PYTHON_EXEC}\" \"{SCRIPT_TRAIN}\" "
                f"--data_dir \"{data_path}\" "
                f"--save_dir \"{exp_subfolder}\" "
                f"--pt_dbm {pt} "
                f"--epochs {EPOCHS} "
                f"--batch_size {BATCH_SIZE} "
                f"--cuda_device {gpu_id} "
                f"--max_targets {k} "
                f"--num_test_samples {NUM_TEST_SAMPLES} "
                f"--test_set_mode first"
            )
            run_command(cmd_train, log_train)
        
        # --- 2. YOLO Baseline 测试 ---
        # 无论是否只测 YOLO，这一步通常都跑
        cmd_yolo = (
            f"\"{PYTHON_EXEC}\" \"{SCRIPT_TEST_YOLO}\" "
            f"--data_dir \"{data_path}\" "
            f"--save_dir \"{exp_subfolder}\" "
            f"--pt_dbm {pt} "
            f"--cuda_device {gpu_id} "
            f"--num_test_samples {NUM_TEST_SAMPLES} "
            f"--max_targets {k} "
            f"--test_set_mode first"
        )
        run_command(cmd_yolo, log_yolo)

        # --- 3. CFARNet 测试 ---
        if not ONLY_TEST_YOLO:
            if not os.path.exists(model_weight_path):
                print(f"   [Worker Warning] Model not found at {model_weight_path}, skipping CFARNet test.")
            else:
                cmd_cfar = (
                    f"\"{PYTHON_EXEC}\" \"{SCRIPT_TEST_CFAR}\" "
                    f"--data_dir \"{data_path}\" "
                    f"--model_path \"{model_weight_path}\" "
                    f"--save_dir \"{exp_subfolder}\" "
                    f"--pt_dbm {pt} "
                    f"--cuda_device {gpu_id} "
                    f"--num_test_samples {NUM_TEST_SAMPLES} "
                    f"--max_targets {k} "
                    f"--test_set_mode first"
                )
                run_command(cmd_cfar, log_cfar)

        print(f"   [Worker Done] Pt={pt}dBm finished successfully.")
        return True

    except Exception as e:
        print(f"   [Worker Fail] Pt={pt}dBm failed: {e}")
        return False

def worker_wrapper(data_path, pt, k):
    """
    包装器：负责申请和释放 GPU 资源
    """
    gpu_id = gpu_queue.get() # 阻塞直到有空闲 GPU
    try:
        worker_routine(data_path, pt, k, gpu_id)
    finally:
        gpu_queue.put(gpu_id) # 归还 GPU

def main():
    executor = ThreadPoolExecutor(max_workers=len(CUDA_DEVICES))
    futures = []

    print(f"Mode: {'YOLO ONLY' if ONLY_TEST_YOLO else 'FULL TRAIN & TEST'}")
    print(f"Found {len(EXISTING_DATA_DIRS)} data directories to process.")

    for data_path in EXISTING_DATA_DIRS:
        if not os.path.exists(data_path):
            print(f"[Skip] Missing path: {data_path}")
            continue
            
        # 尝试从路径中解析 K 值 (例如 ..._k3_...)
        match = re.search(r"_k(\d+)_", os.path.basename(data_path))
        if match:
            k = int(match.group(1))
        else:
            k = 3 # 默认值，或者可以报错
            print(f"[Warn] Could not parse K from path, using default K={k}")
        
        print(f"\n>>> Submitting tasks for: {os.path.basename(data_path)} (K={k})")

        for pt in PT_LIST:
            future = executor.submit(worker_wrapper, data_path, pt, k)
            futures.append(future)

    print(f"\n{'#'*60}")
    print(f"All tasks submitted. Waiting for completion...")
    print(f"{'#'*60}")
    
    for f in futures:
        try:
            f.result()
        except Exception as e:
            print(f"Task failed with exception: {e}")

    print("\n>>> All Tasks Completed.")

if __name__ == "__main__":
    main()
