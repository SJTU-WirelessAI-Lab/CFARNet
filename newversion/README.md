# CFARNet: Learning-Based High-Resolution Multi-Target Detection for Rainbow Beam Radar

## Related Publication

📄 **Paper**: [CFARNet: Learning-Based High-Resolution Multi-Target Detection for Rainbow Beam Radar](https://arxiv.org/abs/2505.10150)

*Qiushi Liang, Yeyue Cai, Jianhua Mo, Meixia Tao*

This repository provides the implementation for the CFARNet paper, which presents a learning-based processing framework that replaces CFAR with a convolutional neural network (CNN) for peak detection in the angle-Doppler domain.

---

## Introduction

CFARNet aims to provide a complete toolchain for radar signal processing and target detection, including simulation, data generation, deep learning training, and comparison with traditional methods. It supports high-quality echo channel parameter generation, CNN-based target detection training, and baseline comparisons with traditional methods like CFAR+MUSIC.

**Note**: All code comments, documentation, and output messages are now in English for international accessibility and collaboration.

---

## File Structure and Function Description

| Filename | Main Function Description |
|----------|---------------------------|
| `data_generation.py` | Generate echo channel parameters, target trajectories, and system parameters with batch simulation and dataset generation support. |
| `pipeline.py` | **Process Runner**: Automatically scans generated data, runs training/testing for defined Power levels (Pt), and aggregates results. |
| `trajectory.py` | Target trajectory generation module supporting various motion patterns and parameter configurations. |
| `train.py` | Main deep learning model training script supporting CNN training (BCE Loss), validation, testing, and visualization. |
| `CFARNet.py` | CFARNet neural network method implementation for high-resolution multi-target detection (main method from the paper). |
| `YOLO_baseline.py` | Traditional CFAR+MUSIC baseline method for comparison with the proposed CFARNet approach. |
| `functions.py` | Core utility function library including dataset loading, signal processing, feature extraction, and evaluation metrics. |
| `environment.yml` | Conda environment dependency configuration file containing all required packages and versions. |

---

## Quick Start

### 1. Environment Setup

It is recommended to use Anaconda/Miniconda and execute the following commands to create the environment:

```bash
conda config --add channels conda forge
conda env create -f environment.yml
conda activate isac
pip install tzdata
pip install huggingface
```

### 2. Workflow Overview

The standard workflow consists of two main steps:
1.  **Generate Data**: Create synthetic radar datasets with specific target counts (K) and angle differences (Delta).
2.  **Run Pipeline**: Execute the automated pipeline which scans for data, trains models, and outputs performance metrics.

---

### Step 1: Data Generation

Use `data_generation.py` to generate datasets. You **must** use the naming convention `auto_pipeline` for the pipeline script to recognize the folders.

**Command Syntax:**
```bash
python data_generation.py --num_targets <K> --min_angle_diff <Delta> --name auto_pipeline --samples 5000
```

**Examples:**

*   **1 Target**:
    ```bash
    python data_generation.py --num_targets 1 --min_angle_diff 0 --name auto_pipeline
    ```
*   **3 Targets, 1 degree separation**:
    ```bash
    python data_generation.py --num_targets 3 --min_angle_diff 1 --name auto_pipeline
    ```
*   **3 Targets, 10 degrees separation**:
    ```bash
    python data_generation.py --num_targets 3 --min_angle_diff 10 --name auto_pipeline
    ```
*   **5 Targets, 1 degree separation**:
    ```bash
    python data_generation.py --num_targets 5 --min_angle_diff 1 --name auto_pipeline
    ```

*Note: The script will automatically append the suffix `_k<K>_d<Delta>_<Timestamp>` to the folder name in `data/`.*

---

### Step 2: Run Pipeline

Once data generation is complete, run the `pipeline.py` script. This script acts as a manager; it does **not** generate new data. It scans the `data/` directory for folders matching `auto_pipeline_k*_d*`, and runs the training and evaluation process for pre-defined Power (Pt) levels.

**Command:**
```bash
python pipeline.py
```

**What happens:**
1.  The script identifies all valid `auto_pipeline_*` folders in `newversion/data/`.
2.  For each dataset:
    *   It determines the number of targets (K).
    *   It retrieves the target list of Pt levels (e.g., [-10, 0, 10, 20] dBm).
    *   It runs `train.py` (using BCE Loss).
    *   It evaluates performance using `CFARNet.py`.
3.  All results are aggregated.

---

### 3. Output & Results

All pipeline results are stored in the **`bce0112/`** directory.

*   **`bce0112/final_summary.txt`**: This is the primary result file. It contains a consolidated table of performance metrics (Accuracy, RMSE, etc.) for all processed K and Pt combinations.
*   **`bce0112/log_<experiment_name>.txt`**: Detailed execution logs for specific experiments.


