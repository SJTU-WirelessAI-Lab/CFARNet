# -*- coding: utf-8 -*-
import subprocess
import re
import os
import sys
import time
import queue
from concurrent.futures import ThreadPoolExecutor

# ================= 配置区域 =================
# 脚本文件名
SCRIPT_GEN = "data_generation.py"
SCRIPT_TRAIN = "train.py"
SCRIPT_TEST_YOLO = "YOLO_baseline.py"      # YOLO baseline 测试脚本
SCRIPT_TEST_CFAR = "CFARNet.py"   # CFARNet 测试脚本

# 实验参数列表
K_LIST = [3]           # 用户数量列表
D_LIST = [1.5, 3.0, 5, 10]         # 最小角度间隔列表 (Delta Phi)
PT_LIST = [40, 50, 60]  # 发射功率列表

# 硬件资源
CUDA_DEVICES = [0, 1, 2, 3, 4, 5, 6, 7] # 可用的 GPU ID 列表

# 通用参数
SAMPLES = 50000        # 生成数据量
EPOCHS = 60            # 训练轮数
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
    # 确保日志文件的目录存在
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except:
            pass # 防止并发创建报错

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
        
        full_output = []
        for line in process.stdout:
            f.write(line)
            f.flush()
            full_output.append(line)
            
        process.wait()
        
        if process.returncode != 0:
            err_msg = f"\n[ERROR] Command failed with return code {process.returncode}.\n"
            f.write(err_msg)
            print(f"Error executing: {command}. See log: {log_file_path}")
            raise RuntimeError(f"Command failed: {command}")

    return "".join(full_output)

def parse_gen_path(output):
    """从生成脚本输出中抓取数据路径"""
    match = re.search(r"Dataset generation completed, path:\s*(.+)", output)
    if match: return match.group(1).strip()
    match = re.search(r"Data will be saved to:\s*(.+)", output)
    if match: return match.group(1).strip()
    raise ValueError("Critical Error: Could not capture DATA path from generation output.")

def worker_routine(k, d, pt, gpu_id, data_path):
    """
    单个 Worker 的工作流程：训练 -> YOLO测试 -> CFARNet测试
    """
    # 创建独立的实验子文件夹
    exp_subfolder = os.path.join(data_path, f"experiment_pt{int(pt)}")
    os.makedirs(exp_subfolder, exist_ok=True)

    print(f"   [Worker Start] Pt={pt}dBm on GPU {gpu_id} | Data: {os.path.basename(data_path)} | Save: {os.path.basename(exp_subfolder)}")
    
    # 定义日志文件 (保存在子文件夹下)
    log_train = os.path.join(exp_subfolder, f"log_train_pt{int(pt)}.txt")
    log_yolo = os.path.join(exp_subfolder, f"log_test_yolo_pt{int(pt)}.txt")
    log_cfar = os.path.join(exp_subfolder, f"log_test_cfarnet_pt{int(pt)}.txt")
    
    try:
        # 1. 训练 (Train)
        cmd_train = (
            f"python {SCRIPT_TRAIN} "
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
        
        model_weight_path = os.path.join(exp_subfolder, f"model_pt{int(pt)}_best.pt")

        # 2. YOLO Baseline 测试
        cmd_yolo = (
            f"python {SCRIPT_TEST_YOLO} "
            f"--data_dir \"{data_path}\" "
            f"--save_dir \"{exp_subfolder}\" "
            f"--pt_dbm {pt} "
            f"--cuda_device {gpu_id} "
            f"--num_test_samples {NUM_TEST_SAMPLES} "
            f"--max_targets {k} "
            f"--test_set_mode first"
        )
        run_command(cmd_yolo, log_yolo)

        # 3. CFARNet 测试
        cmd_cfar = (
            f"python {SCRIPT_TEST_CFAR} "
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

def worker_wrapper(k, d, pt, data_path):
    """
    包装器：负责申请和释放 GPU 资源
    """
    gpu_id = gpu_queue.get() # 阻塞直到有空闲 GPU
    try:
        worker_routine(k, d, pt, gpu_id, data_path)
    finally:
        gpu_queue.put(gpu_id) # 归还 GPU

def main():
    executor = ThreadPoolExecutor(max_workers=len(CUDA_DEVICES))
    futures = []

    for k in K_LIST:
        for d in D_LIST:
            print(f"\n{'#'*60}")
            print(f"Phase 1: Generating Data for K={k}, D={d}")
            print(f"{'#'*60}")
            
            gen_log_file = f"pipeline_gen_k{k}_d{d}.log"
            cmd_gen = (
                f"python {SCRIPT_GEN} "
                f"--samples {SAMPLES} "
                f"--chunk 500 "
                f"--num_targets {k} "
                f"--min_angle_diff {d} "
                f"--name auto_pipeline"
            )
            
            try:
                gen_output = run_command(cmd_gen, gen_log_file)
                data_path = parse_gen_path(gen_output)
                print(f">>> Data generated at: {data_path}")
                try:
                    os.rename(gen_log_file, os.path.join(data_path, "generation_log.txt"))
                except: pass

                print(f">>> Submitting {len(PT_LIST)} tasks for K={k}, D={d}...")
                for pt in PT_LIST:
                    future = executor.submit(worker_wrapper, k, d, pt, data_path)
                    futures.append(future)

            except Exception as e:
                print(f">>> Generation Failed for K={k}, D={d}. Skipping this batch. Error: {e}")
                continue

    print(f"\n{'#'*60}")
    print(f"All tasks submitted. Waiting for completion...")
    print(f"{'#'*60}")
    
    for f in futures:
        try:
            f.result()
        except Exception as e:
            print(f"Task failed with exception: {e}")

    print("\n>>> All Pipeline Tasks Completed.")

if __name__ == "__main__":
    main()

def run_command(command, log_file_path):
    """
    执行命令并将输出实时写入指定的日志文件。
    """
    # 确保日志文件的目录存在
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except:
            pass # 防止并发创建报错

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
        
        full_output = []
        for line in process.stdout:
            # 这里的 print 可以注释掉，如果不想在主控制台看到太多刷屏
            # sys.stdout.write(line) 
            f.write(line)
            f.flush()
            full_output.append(line)
            
        process.wait()
        
        if process.returncode != 0:
            err_msg = f"\n[ERROR] Command failed with return code {process.returncode}.\n"
            f.write(err_msg)
            print(f"Error executing: {command}. See log: {log_file_path}")
            raise RuntimeError(f"Command failed: {command}")

    return "".join(full_output)

def parse_gen_path(output):
    """从生成脚本输出中抓取数据路径"""
    # 匹配: Dataset generation completed, path: /path/to/data
    match = re.search(r"Dataset generation completed, path:\s*(.+)", output)
    if match: return match.group(1).strip()
    # 备用匹配
    match = re.search(r"Data will be saved to:\s*(.+)", output)
    if match: return match.group(1).strip()
    raise ValueError("Critical Error: Could not capture DATA path from generation output.")

def worker_routine(k, d, pt, gpu_id, data_path):
    """
    单个 Worker 的工作流程：训练 -> YOLO测试 -> CFARNet测试
    """
    print(f"   [Worker Start] Pt={pt}dBm on GPU {gpu_id} | Data: {os.path.basename(data_path)}")
    
    # 定义该 Pt 下的日志文件名 (全部保存在数据文件夹下)
    log_train = os.path.join(data_path, f"log_train_pt{int(pt)}.txt")
    log_yolo = os.path.join(data_path, f"log_test_yolo_pt{int(pt)}.txt")
    log_cfar = os.path.join(data_path, f"log_test_cfarnet_pt{int(pt)}.txt")
    
    try:
        # 1. 训练 (Train)
        # 注意：这里传入 --save_dir 告诉训练脚本把模型和TopHit日志保存在 data_path 下
        cmd_train = (
            f"python {SCRIPT_TRAIN} "
            f"--data_dir \"{data_path}\" "
            f"--save_dir \"{data_path}\" "  # 关键：模型保存路径
            f"--pt_dbm {pt} "
            f"--epochs {EPOCHS} "
            f"--batch_size {BATCH_SIZE} "
            f"--cuda_device {gpu_id} "
            f"--max_targets {k} "
            f"--num_test_samples {NUM_TEST_SAMPLES} "
            f"--test_set_mode first"
        )
        run_command(cmd_train, log_train)
        
        # 假设训练脚本保存模型的命名规则是固定的，或者在日志里能找到
        # 这里假设保存为 data_path/model_pt{pt}.pt (需要在 train.py 中配合修改)
        model_weight_path = os.path.join(data_path, f"model_pt{int(pt)}_best.pt")

        # 2. YOLO Baseline 测试
        # YOLO 不需要模型权重，只需要数据路径
        cmd_yolo = (
            f"python {SCRIPT_TEST_YOLO} "
            f"--data_dir \"{data_path}\" "
            f"--save_dir \"{data_path}\" " # 结果保存路径
            f"--pt_dbm {pt} "
            f"--cuda_device {gpu_id} "
            f"--num_test_samples {NUM_TEST_SAMPLES} "
            f"--max_targets {k} "
            f"--test_set_mode first"
        )
        run_command(cmd_yolo, log_yolo)

        # 3. CFARNet 测试
        # 需要加载刚才训练好的权重
        cmd_cfar = (
            f"python {SCRIPT_TEST_CFAR} "
            f"--data_dir \"{data_path}\" "
            f"--model_path \"{model_weight_path}\" " # 加载权重
            f"--save_dir \"{data_path}\" " # 结果保存路径
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

def main():
    # 外层循环：遍历 K 和 D (每次组合生成一次数据)
    for k in K_LIST:
        for d in D_LIST:
            print(f"\n{'#'*60}")
            print(f"Phase 1: Generating Data for K={k}, D={d}")
            print(f"{'#'*60}")
            
            # --- 1. 生成数据 ---
            # 这里的日志先暂存在当前目录，或者你可以指定一个 logs 文件夹
            gen_log_file = f"pipeline_gen_k{k}_d{int(d)}.log"
            
            # 构造生成命令
            # --name 用于区分文件夹前缀
            cmd_gen = (
                f"python {SCRIPT_GEN} "
                f"--samples {SAMPLES} "
                f"--chunk 500 "
                f"--num_targets {k} "
                f"--min_angle_diff {d} "
                f"--name auto_pipeline"
            )
            
            try:
                # 执行生成
                gen_output = run_command(cmd_gen, gen_log_file)
                # 获取生成的数据文件夹路径
                data_path = parse_gen_path(gen_output)
                print(f">>> Data generated at: {data_path}")
                
                # 将生成日志移动到数据文件夹内 (可选，保持整洁)
                try:
                    os.rename(gen_log_file, os.path.join(data_path, "generation_log.txt"))
                except: pass

            except Exception as e:
                print(f">>> Generation Failed for K={k}, D={d}. Skipping this batch.")
                continue

            # --- 2. 并行训练与测试 ---
            print(f"\nPhase 2: Parallel Train/Test for Pt list: {PT_LIST}")
            print(f"Using GPUs: {CUDA_DEVICES}")
            
            # 使用 ThreadPoolExecutor 来并发运行
            # 注意：Python 的 ThreadPool 足以应对 subprocess 的并发，因为 GIL 不阻塞子进程
            with ThreadPoolExecutor(max_workers=len(CUDA_DEVICES)) as executor:
                futures = []
                
                # 为每个 Pt 分配任务
                for i, pt in enumerate(PT_LIST):
                    # 简单的轮询分配 GPU
                    gpu_id = CUDA_DEVICES[i % len(CUDA_DEVICES)]
                    
                    # 提交任务
                    future = executor.submit(worker_routine, k, d, pt, gpu_id, data_path)
                    futures.append(future)
                
                # 等待所有 Pt 的任务完成
                for future in futures:
                    future.result() # 这里会抛出 worker 内部的异常（如果有）

            print(f"\n>>> Batch K={k}, D={d} Completed. Results in {data_path}")

if __name__ == "__main__":
    main()