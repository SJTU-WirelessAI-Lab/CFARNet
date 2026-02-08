# CFARNet: Learning-Based High-Resolution Multi-Target Detection for Rainbow Beam Radar

## Related Publication

📄 **Paper**: [CFARNet: Learning-Based High-Resolution Multi-Target Detection for Rainbow Beam Radar](https://arxiv.org/abs/2505.10150)

*Qiushi Liang, Yeyue Cai, Jianhua Mo, Meixia Tao*

This repository provides the implementation for **CFARNet**, a deep learning-based framework for radar signal processing. CFARNet replaces the traditional CFAR (Constant False Alarm Rate) detector with a Convolutional Neural Network (CNN) to detect peaks in the angle-Doppler domain, significantly improving multi-target detection performance in high-resolution rainbow beam radar systems.

---

## 🌟 Key Features

*   **Deep Learning Pipeline**: End-to-end training framework replacing traditional CFAR with CNN-based index prediction.
*   **High-Fidelity Simulation**: Includes `data_generation.py` for generating radar echo data with configurable system parameters, target trajectories, and SNRs.
*   **Comprehensive Evaluation**:
    *   **2D RMSE & Percentile Errors**: Metrics for Range, Angle, Velocity, and 2D Position.
    *   **Baseline Comparison**: Built-in comparison with "YOLO" (a Grid-based CFAR + MUSIC baseline).
    *   **Parameter Sweeps**: Scripts to analyze performance across Tranmit Power ($P_t$), Angle Separation ($\Delta\phi$), and Number of Targets ($K$).
*   **Visualization**: Specialized plotting scripts for paper-quality figures (e.g., 90th percentile error curves).

---

## 🛠️ Environment Setup

We recommended using **Anaconda/Miniconda**.

1.  **Create Environment**:
    ```bash
    conda create -n cfarnet python=3.9
    conda activate cfarnet
    ```

2.  **Install Dependencies**:
    ```bash
    # Install PyTorch (adjust cuda version as needed)
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

    # Install other requirements
    pip install numpy scipy matplotlib tqdm
    ```

---

## 📂 File Structure

The project is organized as follows:

| Component | File | Description |
| :--- | :--- | :--- |
| **Core** | `CFARNet.py` | Main model architecture and inference logic. |
| | `train.py` | Training script for the CNN model (BCE Loss). |
| | `dataset.py` / `functions.py` | Dataset loaders and signal processing utilities (FFT, MUSIC, etc.). |
| **Simulation** | `data_generation.py` | Generates synthetic radar datasets (Echoes, Trajectories). |
| | `trajectory.py` | Helper for simulating target motion. |
| **Automation** | `pipeline.py` | **Train-Only Pipeline**: Automatically scans recursively for datasets and trains models. |
| **Evaluation** | `calculate_2d_rmse.py` | **Main Evaluation**: Computes metrics vs $P_t$ and $\Delta\phi$. Compares CFARNet vs Baseline. |
| | `calculate_k_sweep.py` | **K-Sweep**: Computes metrics vs Number of Targets ($K$). |
| | `results_k_sweep.txt` | Output logs for K-sweep experiments. |
| | `results_2d_rmse.txt` | Output logs for P_t and Delta experiments. |
| **Plotting** | `plot_p90_multi.py` | Generates 90th percentile error plots for different $\Delta\phi$. |
| | `plot_k_sweep_p90.py` | Generates performance plots vs $K$. |

---

## 🚀 Usage Guide

### 1. Data Generation

Generate synthetic radar data for training or testing.

```bash
# Generate training data (e.g., K=3, min_angle_diff=1.0)
python data_generation.py --num_targets 3 --min_angle_diff 1.0 --name auto_pipeline --samples 5000

# Generate test data (e.g., K=3, min_angle_diff=1.5, 3.0, 5.0, etc.)
python data_generation.py --num_targets 3 --min_angle_diff 1.5 --name auto_pipeline --samples 2000
python data_generation.py --num_targets 3 --min_angle_diff 3.0 --name auto_pipeline --samples 2000
```
*Data will be saved in `data/`.*

### 2. Training (Pipeline)

Run the automated pipeline to scan for datasets and train models.

```bash
python pipeline.py
```
*This script will scan for generated datasets in `data/` and train a model for each configuration. Trained models are saved in `bce0112/` (or similar output directory).*

### 3. Evaluation & Benchmarking

We provide two primary scripts to evaluate the trained models against the Baseline (YOLO/CFAR+MUSIC).

**A. Performance vs. Signal Power ($P_t$) & Angle Separation ($\Delta\phi$):**
This script loads the trained model and test datasets, runs inference, and calculates RMSE and P90/P95 errors.
```bash
python calculate_2d_rmse.py
```
*   **Output**: `newversion/results_2d_rmse.txt`

**B. Performance vs. Number of Targets ($K$):**
This script evaluates how performance degrades as $K$ increases.
```bash
python newversion/calculate_k_sweep.py
```
*   **Output**: `newversion/results_k_sweep.txt`

### 4. Visualization

Generate paper-ready figures from the evaluation results.

**Plot P90 Error vs Pt (grouped by Delta):**
```bash
# Reads results_2d_rmse.txt
python newversion/plot_p90_multi.py
```
*   **Output**: `newversion/plots/90th_p90_styled.png` (Legend at top, style matched to paper).

**Plot P90 Error vs K:**
```bash
# Reads results_k_sweep.txt
python newversion/plot_k_sweep_p90.py
```
*   **Output**: `newversion/plots/plot_k_sweep_p90.png`.

---

## 📊 Results Summary

The evaluation scripts output text files containing detailed tables. Example format (`results_2d_rmse.txt`):

```text
K | Dataset_D | Pt(dBm) | Model | RMSE_2D | 90%_2D | ... 
--------------------------------------------------------
3 | 1.0       | 45      | YOLO  | 14.12   | 0.73   | ...
3 | 1.0       | 45      | CFAR  | 11.77   | 0.52   | ...
```

---

## 🤝 Citation

If you find this code useful, please cite our paper:

```bibtex
@article{liang2026cfarnet,
  title={CFARNet: Learning-Based High-Resolution Multi-Target Detection for Rainbow Beam Radar},
  author={Liang, Qiushi and Cai, Yeyue and Mo, Jianhua and Tao, Meixia},
  journal={arXiv preprint arXiv:2505.10150},
  year={2026}
}
```

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for more details.

