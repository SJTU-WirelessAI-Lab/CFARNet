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
| `trajectory.py` | Target trajectory generation module supporting various motion patterns and parameter configurations. |
| `train.py` | Main deep learning model training script supporting CNN training, validation, testing, and visualization. |
| `CFARNet.py` | CFARNet neural network method implementation for high-resolution multi-target detection (main method from the paper). |
| `YOLO_baseline.py` | Traditional CFAR+MUSIC baseline method for comparison with the proposed CFARNet approach. |
| `functions.py` | Core utility function library including dataset loading, signal processing, feature extraction, and evaluation metrics. |
| `environment.yml` | Conda environment dependency configuration file containing all required packages and versions. |
| `README.md` | Project documentation. |

---

## Quick Start

### 1. Environment Setup

It is recommended to use Anaconda/Miniconda and execute the following commands to create the environment:

```bash
conda config --add channels conda forge（新增源）
conda env create -f environment.yml（删除tzdata和huggingce，这两个包在conda源中找不到，需要额外安装，删除除python外所有的版本号，否则有些包的版本会互相不兼容导致安装失败）
conda activate isac
pip install tzdata（额外安装tzdata）
pip install huggingface（额外安装huggingface）
```

### 2. Data Generation

Generate echo channel parameters and target motion data:

```bash
python data_generation.py --sample_num 5000 --chunk_size 500 --experiment_name my_exp
python data_generation.py --samples 5000 --chunk 5 --name my_exp(上面那条指令跑不通，需要做修改）
生成的轨迹和速度是完全独立的，这里是否会有问题？速度是否应该是轨迹的导数？
```

**Main Parameter Descriptions:**
- `--sample_num`: Number of samples to generate
- `--chunk_size`: Number of samples per data chunk
- `--experiment_name`: Experiment name (for distinguishing data folders)
- Other parameters see script comments and command line help

### 3. Deep Learning Model Training

Train CNN and other deep learning models:

```bash
python train.py --data_dir ./data/my_exp --batch_size 16 --epochs 50 --max_targets 3
运行这一步的时候在train.py的1240行会出现key error，疑似是由于parser将args.val_pt_dbm_list识别为字符串引起的，因此读取出的pt实际上为"[","-","1","0"等字符，而不能读取出数字，这里的代码需要修改
```

**Main Parameter Descriptions:**
- `--data_dir`: Directory containing generated data
- `--batch_size`: Training batch size
- `--epochs`: Number of training epochs
- `--max_targets`: Maximum number of targets
- Other parameters see script comments and command line help

### 4. CFARNet Neural Network Method Testing

Run the proposed CFARNet method:

```bash
python CFARNet.py --data_dir ./data/my_exp --model_dir ./models/my_exp --top_k_cnn 3
python CFARNet.py --data_dir ./data/my_exp --model_dir ./output/my_exp/models --top_k_cnn 3(下面这条描述的路径比较符合代码生成的文件路径）


```

**Main Parameter Descriptions:**
- `--data_dir`: Data directory
- `--model_dir`: Trained model directory
- `--top_k_cnn`: Top-K peaks from CNN output
- Other parameters see script comments and command line help

### 5. Traditional CFAR+MUSIC Baseline Testing (Optional)

Compare with traditional CFAR+MUSIC method:

```bash
python YOLO_baseline.py --data_dir ./data/my_exp --num_test_samples 1000
```

---

## Dependencies

- Python 3.10+
- numpy, torch, matplotlib, tqdm, scipy, torchvision, pandas, etc.
- Detailed dependencies see `environment.yml`

---

## Data Structure Description

Generated data is saved in the specified directory, mainly including:

- `echoes/`: Chunked echo channel parameters
- `trajectory_data.npz`: Target trajectories and peak indices
- `system_params.npz`: System parameter configuration

---

## Contribution and License

We welcome anyone to submit PRs or issues to improve this project.

This project is licensed under the MIT License.

---

## Recent Updates

### Code Internationalization (Latest)
- ✅ **All Chinese comments and text have been translated to English** across all Python files
- ✅ **English-only codebase** for better international collaboration and accessibility
- ✅ **Comprehensive translation** of:
  - Function and class docstrings
  - Inline comments explaining algorithms and logic
  - Error messages and warnings
  - Print statements and user interface text
  - Parameter descriptions and help text
  - File headers and section comments

This ensures the codebase is fully accessible to international researchers and developers working with radar signal processing and deep learning.

---

## Detailed File Descriptions

### data_generation.py
Mainly used for simulating radar echo signals, target trajectories, system parameters, etc. Supports batch generation and chunked storage for large-scale dataset creation. Allows customization of target numbers, motion patterns, noise parameters, etc.

### trajectory.py
Responsible for generating 2D target trajectories, supporting circular, random, fixed-angle and other modes. Allows flexible configuration of target initial position, velocity, angle and other parameters.

### train.py
Main deep learning training script supporting CNN and other model training, validation, and testing. Built-in multiple loss functions, evaluation metrics, and visualization tools. Supports checkpoint resumption and automatic experiment result saving.

### CFARNet.py
Implements the proposed CFARNet neural network method for high-resolution multi-target detection. This is the main contribution of the paper, using CNN-based peak detection in the angle-Doppler domain to replace traditional CFAR methods.

### YOLO_baseline.py
Implements traditional CFAR+MUSIC baseline method for comparison with the proposed CFARNet approach. Provides performance benchmarks to demonstrate the superiority of the neural network-based method.

### functions.py
Provides core utility functions for dataset loading, signal processing, feature extraction, evaluation metrics, etc. Facilitates main process script calls and improves code reusability.

### environment.yml
Conda environment configuration containing all dependency packages and versions to ensure environment consistency.

---


