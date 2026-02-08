# -*- coding: utf-8 -*-
import subprocess
import os
import sys
import glob
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# ================= Configuration =================
SCRIPT_TRAIN = "train.py"
SCRIPT_TEST_YOLO = "YOLO_baseline.py"
SCRIPT_TEST_CFAR = "CFARNet.py"

# User Request:
# K=3: Pts [45, 50, 55, 60]
# K=1, 2, 4, 5: Pt [50]
# Output: bce0112 folder

OUTPUT_ROOT = "bce0112"
DATA_ROOT = "data"

# Hardware
CUDA_DEVICES = [ 2, 3,4, 5, 6,7]
# CUDA_DEVICES = [0] # Debug

# Training/Testing Params
EPOCHS = 60
BATCH_SIZE = 128
NUM_TEST_SAMPLES = 7500 # As per previous pipeline

# ===============================================

gpu_queue = queue.Queue()
for device_id in CUDA_DEVICES:
    gpu_queue.put(device_id)

def run_command(command, log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "w", encoding='utf-8') as f:
        f.write(f"CMD: {command}\n{'='*40}\n")
        f.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True)
        for line in process.stdout:
            f.write(line)
            f.flush()
        process.wait()
        if process.returncode != 0:
            f.write(f"\n[ERROR] Return Code {process.returncode}")
            raise RuntimeError(f"Command failed: {command}")

def get_k_from_path(path):
    # Try to parse from folder name first "auto_pipeline_k3_..."
    match = re.search(r'_k(\d+)_', os.path.basename(path))
    if match:
        return int(match.group(1))
    # Fallback to loading system_params
    try:
        p = np.load(os.path.join(path, "system_params.npz"))
        return int(p['K'])
    except:
        return None

def train_task(dataset_path, pt, gpu_id):
    dataset_name = os.path.basename(dataset_path)
    save_dir = os.path.join(OUTPUT_ROOT, dataset_name, f"pt{pt}")
    os.makedirs(save_dir, exist_ok=True)
    
    log_file = os.path.join(save_dir, "log_train.txt")
    model_path = os.path.join(save_dir, f"model_pt{pt}_best.pt")
    
    # We need K for training arg
    k = get_k_from_path(dataset_path)
    if k is None: raise ValueError(f"Could not determine K for {dataset_path}")

    print(f"[Train] Start K={k} Pt={pt} on GPU {gpu_id} | {dataset_name}")
    
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} python {SCRIPT_TRAIN} "
        f"--data_dir \"{dataset_path}\" "
        f"--save_dir \"{save_dir}\" "
        f"--pt_dbm {pt} "
        f"--epochs {EPOCHS} "
        f"--batch_size {BATCH_SIZE} "
        f"--cuda_device 0 "
        f"--max_targets {k} "
        f"--num_test_samples {NUM_TEST_SAMPLES} "
        f"--test_set_mode first"
    )
    
    try:
        run_command(cmd, log_file)
        if not os.path.exists(model_path):
             raise FileNotFoundError("Model not generated")
        print(f"[Train] Done  K={k} Pt={pt}")
        return model_path
    except Exception as e:
        print(f"[Train] Fail  K={k} Pt={pt}: {e}")
        return None

def test_task(dataset_path, model_path, pt, gpu_id):
    dataset_name = os.path.basename(dataset_path)
    save_dir = os.path.join(OUTPUT_ROOT, dataset_name, f"pt{pt}")
    k = get_k_from_path(dataset_path)
    
    print(f"[Test] Start K={k} Pt={pt} on GPU {gpu_id}")
    
    # YOLO
    log_yolo = os.path.join(save_dir, "log_test_yolo.txt")
    cmd_yolo = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} python {SCRIPT_TEST_YOLO} "
        f"--data_dir \"{dataset_path}\" "
        f"--save_dir \"{save_dir}\" "
        f"--pt_dbm {pt} "
        f"--cuda_device 0 "
        f"--num_test_samples {NUM_TEST_SAMPLES} "
        f"--max_targets {k} "
        f"--test_set_mode first"
    )
    
    # CFARNet
    log_cfar = os.path.join(save_dir, "log_test_cfarnet.txt")
    cmd_cfar = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} python {SCRIPT_TEST_CFAR} "
        f"--data_dir \"{dataset_path}\" "
        f"--model_path \"{model_path}\" "
        f"--save_dir \"{save_dir}\" "
        f"--pt_dbm {pt} "
        f"--cuda_device 0 "
        f"--num_test_samples {NUM_TEST_SAMPLES} "
        f"--max_targets {k} "
        f"--test_set_mode first"
    )
    
    try:
        run_command(cmd_yolo, log_yolo)
        run_command(cmd_cfar, log_cfar)
        print(f"[Test] Done  K={k} Pt={pt}")
    except Exception as e:
        print(f"[Test] Fail  K={k} Pt={pt}: {e}")

def pipeline_worker(dataset_path, pt):
    gpu_id = gpu_queue.get()
    try:
        # Train
        train_task(dataset_path, pt, gpu_id)
        # Test removed as per request
    finally:
        gpu_queue.put(gpu_id)

def summarize_results():
    print("\n[Summary] Generating Final Report (Training Check)...")
    report_path = os.path.join(OUTPUT_ROOT, "training_summary.txt")
    
    results = []
    
    # Check for presence of model files
    dataset_dirs = glob.glob(os.path.join(OUTPUT_ROOT, "*"))
    for d_dir in sorted(dataset_dirs):
        if not os.path.isdir(d_dir): continue
        dataset_name = os.path.basename(d_dir)
        
        pt_dirs = glob.glob(os.path.join(d_dir, "pt*"))
        for p_dir in sorted(pt_dirs):
            pt_str = os.path.basename(p_dir) # pt50
            
            model_file = os.path.join(p_dir, f"model_{pt_str}_best.pt")
            status = "Present" if os.path.exists(model_file) else "Missing"
            
            results.append({'name': dataset_name, 'pt': pt_str, 'status': status})
            
    with open(report_path, 'w') as f:
        f.write(f"{'Dataset':<50} | {'Pt':<5} | {'Model Status':<12}\n")
        f.write("-" * 75 + "\n")
        for r in results:
            f.write(f"{r['name']:<50} | {r['pt']:<5} | {r['status']:<12}\n")
            
    print(f"[Summary] Report saved to {report_path}")

def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
        
    tasks = []
    
    # 1. Discover Datasets
    data_folders = [f for f in glob.glob(os.path.join(DATA_ROOT, "auto_pipeline_*")) if os.path.isdir(f)]
    
    print(f"Found {len(data_folders)} datasets in {DATA_ROOT}")
    
    for d_path in data_folders:
        k = get_k_from_path(d_path)
        if k is None:
            print(f"Skipping {d_path}: Unknown K")
            continue
            
        pts_to_run = []
        if k == 3:
            pts_to_run = [45, 50, 55, 60]
        elif k in [1, 2, 4, 5]:
            pts_to_run = [50]
        else:
            print(f"Skipping {d_path}: K={k} not in target list")
            continue
            
        for pt in pts_to_run:
            tasks.append((d_path, pt))
            
    print(f"Total Tasks: {len(tasks)}")
    
    # 2. Execute
    with ThreadPoolExecutor(max_workers=len(CUDA_DEVICES)) as executor:
        futures = {executor.submit(pipeline_worker, d, p): (d, p) for d, p in tasks}
        
        for f in as_completed(futures):
            d, p = futures[f]
            try:
                f.result()
            except Exception as e:
                print(f"Task Failed: {os.path.basename(d)} Pt={p} -> {e}")

    # 3. Summarize
    summarize_results()

if __name__ == "__main__":
    main()
