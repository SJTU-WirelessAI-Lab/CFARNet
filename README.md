# CFARNet: Learning-Based High-Resolution Multi-Target Detection for Rainbow Beam Radar / 基于深度学习与传统方法的雷达信号参数生成与目标检测工具链

## Related Publication

📄 **Paper**: [CFARNet: Learning-Based High-Resolution Multi-Target Detection for Rainbow Beam Radar](https://arxiv.org/abs/2505.10150)

*Qiushi Liang, Yeyue Cai, Jianhua Mo, Meixia Tao*

This repository provides the implementation for the CFARNet paper, which presents a learning-based processing framework that replaces CFAR with a convolutional neural network (CNN) for peak detection in the angle-Doppler domain.

---

## Introduction / 项目简介

CFARNet aims to provide a complete toolchain for radar signal processing and target detection, including simulation, data generation, deep learning training, and comparison with traditional methods. It supports high-quality yecho channel parameter generation, CNN-based target detection training, and baseline comparisons with traditional methods like CFAR+MUSIC.

CFARNet旨在为雷达信号处理和目标检测提供一套完整的仿真、数据生成、深度学习训练与传统方法对比的工具链。支持高质量yecho信道参数生成、基于CNN的目标检测训练，以及CFAR+MUSIC等传统方法的baseline对比。

---

## File Structure and Function Description / 文件结构与功能说明

| Filename / 文件名 | Main Function Description / 主要功能描述 |
|-------------------|-------------------------------------------|
| `data_generation.py` | Generate yecho channel parameters, target trajectories, and system parameters with batch simulation and dataset generation support.<br>生成yecho信道参数、目标运动轨迹、系统参数等，支持批量仿真和数据集生成。 |
| `trajectory.py` | Target trajectory generation module supporting various motion patterns and parameter configurations.<br>目标运动轨迹生成模块，支持多种运动模式和参数配置。 |
| `train.py` | Main deep learning model training script supporting CNN training, validation, testing, and visualization.<br>深度学习模型训练主脚本，支持CNN等模型的训练、验证、测试与可视化。 |
| `CFARNet.py` | CFARNet neural network method implementation for high-resolution multi-target detection (main method from the paper).<br>CFARNet神经网络方法实现，用于高分辨率多目标检测（论文主要方法）。 |
| `YOLO_baseline.py` | Traditional CFAR+MUSIC baseline method for comparison with the proposed CFARNet approach.<br>传统CFAR+MUSIC基线方法，用于与所提出的CFARNet方法进行对比。 |
| `functions.py` | Core utility function library including dataset loading, signal processing, feature extraction, and evaluation metrics.<br>核心工具函数库，包括数据集加载、信号处理、特征提取、评测指标等。 |
| `environment.yml` | Conda environment dependency configuration file containing all required packages and versions.<br>Conda环境依赖配置文件，包含所有运行所需的包和版本。 |
| `Readme.md` | Project documentation.<br>项目说明文档。 |

---

## Quick Start / 快速开始

### 1. Environment Setup / 环境准备

It is recommended to use Anaconda/Miniconda and execute the following commands to create the environment:

建议使用Anaconda/Miniconda，执行以下命令创建环境：

```bash
conda env create -f environment.yml
conda activate isac
```

### 2. Data Generation / 数据生成

Generate yecho channel parameters and target motion data:

生成yecho信道参数和目标运动数据：

```bash
python data_generation.py --sample_num 5000 --chunk_size 500 --experiment_name my_exp
```

**Main Parameter Descriptions / 主要参数说明:**
- `--sample_num`: Number of samples to generate / 生成样本数量
- `--chunk_size`: Number of samples per data chunk / 每个数据块的样本数
- `--experiment_name`: Experiment name (for distinguishing data folders) / 实验名称（用于区分数据文件夹）
- Other parameters see script comments and command line help / 其他参数详见脚本内注释和命令行帮助

### 3. Deep Learning Model Training / 深度学习模型训练

Train CNN and other deep learning models:

训练CNN等深度学习模型：

```bash
python train.py --data_dir ./data/my_exp --batch_size 16 --epochs 50 --max_targets 3
```

**Main Parameter Descriptions / 主要参数说明:**
- `--data_dir`: Directory containing generated data / 包含生成数据的目录
- `--batch_size`: Training batch size / 训练批次大小
- `--epochs`: Number of training epochs / 训练轮数
- `--max_targets`: Maximum number of targets / 最大目标数
- Other parameters see script comments and command line help / 其他参数详见脚本内注释和命令行帮助

### 4. CFARNet Neural Network Method Testing / CFARNet神经网络方法测试

Run the proposed CFARNet method:

运行所提出的CFARNet方法：

```bash
python CFARNet.py --data_dir ./data/my_exp --model_dir ./models/my_exp --top_k_cnn 3
```

**Main Parameter Descriptions / 主要参数说明:**
- `--data_dir`: Data directory / 数据目录
- `--model_dir`: Trained model directory / 训练好的模型目录
- `--top_k_cnn`: Top-K peaks from CNN output / CNN输出的Top-K峰值
- Other parameters see script comments and command line help / 其他参数详见脚本内注释和命令行帮助

### 5. Traditional CFAR+MUSIC Baseline Testing (Optional) / 传统CFAR+MUSIC基线测试（可选）

Compare with traditional CFAR+MUSIC method:

与传统CFAR+MUSIC方法对比：

```bash
python YOLO_baseline.py --data_dir ./data/my_exp --num_test_samples 1000
```

---

## Dependencies / 依赖环境

- Python 3.10+
- numpy, torch, matplotlib, tqdm, scipy, torchvision, pandas, etc.
- Detailed dependencies see `environment.yml` / 详细依赖见 `environment.yml`

---

## Data Structure Description / 数据结构说明

Generated data is saved in the specified directory, mainly including:

生成的数据保存在指定目录下，主要包括：

- `echoes/`: Chunked yecho channel parameters / 分块保存的yecho信道参数
- `trajectory_data.npz`: Target trajectories and peak indices / 目标轨迹与峰值索引
- `system_params.npz`: System parameter configuration / 系统参数配置

---

## Contribution and License / 贡献与许可

We welcome anyone to submit PRs or issues to improve this project.

欢迎任何人提交PR或issue改进本项目。

This project is licensed under the MIT License.

本项目采用MIT开源协议。

---

## Detailed File Descriptions / 各文件详细说明

### data_generation.py
Mainly used for simulating radar echo signals (yecho), target trajectories, system parameters, etc. Supports batch generation and chunked storage for large-scale dataset creation. Allows customization of target numbers, motion patterns, noise parameters, etc.

主要用于仿真生成雷达回波信号（yecho）、目标运动轨迹、系统参数等。支持批量生成、分块存储，便于大规模数据集制作。可自定义目标数量、运动模式、噪声参数等。

### trajectory.py
Responsible for generating 2D target trajectories, supporting circular, random, fixed-angle and other modes. Allows flexible configuration of target initial position, velocity, angle and other parameters.

负责生成目标的二维运动轨迹，支持圆形、随机、固定角度等多种模式。可灵活配置目标初始位置、速度、角度等参数。

### train.py
Main deep learning training script supporting CNN and other model training, validation, and testing. Built-in multiple loss functions, evaluation metrics, and visualization tools. Supports checkpoint resumption and automatic experiment result saving.

深度学习训练主脚本，支持CNN等模型的训练、验证、测试。内置多种损失函数、评测指标、可视化工具。支持断点续训、实验结果自动保存。

### CFARNet.py
Implements the proposed CFARNet neural network method for high-resolution multi-target detection. This is the main contribution of the paper, using CNN-based peak detection in the angle-Doppler domain to replace traditional CFAR methods.

实现所提出的CFARNet神经网络方法，用于高分辨率多目标检测。这是论文的主要贡献，使用基于CNN的角度-多普勒域峰值检测来替代传统CFAR方法。

### YOLO_baseline.py
Implements traditional CFAR+MUSIC baseline method for comparison with the proposed CFARNet approach. Provides performance benchmarks to demonstrate the superiority of the neural network-based method.

实现传统CFAR+MUSIC基线方法，用于与所提出的CFARNet方法进行对比。提供性能基准，以证明基于神经网络方法的优越性。

### functions.py
Provides core utility functions for dataset loading, signal processing, feature extraction, evaluation metrics, etc. Facilitates main process script calls and improves code reusability.

提供数据集加载、信号处理、特征提取、评测指标等核心工具函数。便于主流程脚本调用，提升代码复用性。

### environment.yml
Conda environment configuration containing all dependency packages and versions to ensure environment consistency.

Conda环境配置，包含所有依赖包及版本，确保环境一致性。

---


