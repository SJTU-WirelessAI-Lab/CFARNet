# CFARNet: 基于深度学习与传统方法的雷达信号参数生成与目标检测工具链

## 项目简介

CFARNet旨在为雷达信号处理和目标检测提供一套完整的仿真、数据生成、深度学习训练与传统方法对比的工具链。支持高质量yecho信道参数生成、基于CNN的目标检测训练，以及CFAR+MUSIC等传统方法的baseline对比。

---

## 文件结构与功能说明

| 文件名                | 主要功能描述                                                                 |
|----------------------|------------------------------------------------------------------------------|
| `data_generation.py` | 生成yecho信道参数、目标运动轨迹、系统参数等，支持批量仿真和数据集生成。         |
| `trajectory.py`      | 目标运动轨迹生成模块，支持多种运动模式和参数配置。                             |
| `train.py`           | 深度学习模型训练主脚本，支持CNN等模型的训练、验证、测试与可视化。               |
| `CFARNet.py`         | 基于CNN+MUSIC的参数估计与传统CFAR方法对比测试脚本。                            |
| `YOLO_baseline.py`   | YOLO方法的目标检测推理与评测脚本，支持多种噪声和参数配置。                     |
| `functions.py`       | 核心工具函数库，包括数据集加载、信号处理、特征提取、评测指标等。               |
| `environment.yml`    | Conda环境依赖配置文件，包含所有运行所需的包和版本。                            |
| `Readme.md`          | 项目说明文档（即本文件）。                                                    |

---

## 快速开始

### 1. 环境准备

建议使用Anaconda/Miniconda，执行以下命令创建环境：

```bash
conda env create -f environment.yml
conda activate isac
```

### 2. 数据生成

生成yecho信道参数和目标运动数据：

```bash
python data_generation.py --sample_num 5000 --chunk_size 500 --experiment_name my_exp
```

**主要参数说明：**
- `--sample_num`：生成样本数量
- `--chunk_size`：每个数据块的样本数
- `--experiment_name`：实验名称（用于区分数据文件夹）
- 其他参数详见脚本内注释和命令行帮助

### 3. 深度学习模型训练

训练CNN等深度学习模型：

```bash
python train.py --data_dir ./data/my_exp --batch_size 16 --epochs 50 --max_targets 3
```

**主要参数说明：**
- `--data_dir`：包含生成数据的目录
- `--batch_size`：训练批次大小
- `--epochs`：训练轮数
- `--max_targets`：最大目标数
- 其他参数详见脚本内注释和命令行帮助

### 4. 传统CFAR+MUSIC baseline测试

对比传统方法：

```bash
python CFARNet.py --data_dir ./data/my_exp --model_dir ./models/my_exp --top_k_cnn 3
```

**主要参数说明：**
- `--data_dir`：数据目录
- `--model_dir`：训练好的模型目录
- `--top_k_cnn`：CNN输出的Top-K峰值
- 其他参数详见脚本内注释和命令行帮助

### 5. YOLO方法推理/测试（可选）

```bash
python YOLO_baseline.py --data_dir ./data/my_exp --num_test_samples 1000
```

---

## 依赖环境

- Python 3.10+
- numpy, torch, matplotlib, tqdm, scipy, torchvision, pandas 等
- 详细依赖见 `environment.yml`

---

## 数据结构说明

- 生成的数据保存在指定目录下，主要包括：
  - `echoes/`：分块保存的yecho信道参数
  - `trajectory_data.npz`：目标轨迹与峰值索引
  - `system_params.npz`：系统参数配置

---

## 贡献与许可

欢迎任何人提交PR或issue改进本项目。  
本项目采用MIT开源协议。

---

## 各文件详细说明

### data_generation.py
- 主要用于仿真生成雷达回波信号（yecho）、目标运动轨迹、系统参数等。
- 支持批量生成、分块存储，便于大规模数据集制作。
- 可自定义目标数量、运动模式、噪声参数等。

### trajectory.py
- 负责生成目标的二维运动轨迹，支持圆形、随机、固定角度等多种模式。
- 可灵活配置目标初始位置、速度、角度等参数。

### train.py
- 深度学习训练主脚本，支持CNN等模型的训练、验证、测试。
- 内置多种损失函数、评测指标、可视化工具。
- 支持断点续训、实验结果自动保存。

### CFARNet.py
- 实现基于CNN+MUSIC的参数估计方法，并与传统CFAR方法进行对比。
- 支持批量测试、结果可视化、性能评估。

### YOLO_baseline.py
- 实现YOLO方法的目标检测推理与评测。
- 支持多种噪声、参数配置，便于与深度学习方法对比。

### functions.py
- 提供数据集加载、信号处理、特征提取、评测指标等核心工具函数。
- 便于主流程脚本调用，提升代码复用性。

### environment.yml
- Conda环境配置，包含所有依赖包及版本，确保环境一致性。

---



