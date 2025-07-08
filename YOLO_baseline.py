import numpy as np
# import scipy.io as sio # Not needed as we load .npz
import matplotlib.pyplot as plt
import torch # Only needed for noise generation tensor
import os
# import glob # Not needed
import argparse
# from functions_new import load_system_params, initial_rainbow_beam_ULA_YOLO # YOLO function not used here
from functions import load_system_params # Keep load_system_params if used
from torch.utils.data import Dataset, DataLoader # Dataset used now
import time # 引入时间库
import math # Added for noise calculations
import datetime # Added for CDF plot timestamp
import traceback # Added for dataset error handling
import sys # Added for tqdm output redirection
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment # <<< Import Hungarian Algorithm
from itertools import islice # For limiting dataloader

# ==============================================================================
# 0. 初始化与配置区域
# ==============================================================================
print("--- Initialization and Configuration ---")

# --- Noise Constants ---
K_BOLTZMANN = 1.38e-23
T_NOISE_KELVIN = 290 # Standard noise temperature
c = 3e8 # Speed of light

# --- Dataset Definition (Copied from previous version) ---
class ChunkedEchoDataset(Dataset):
    """
    Loads pre-computed *noiseless* echo signals (yecho),
    target peak indices (m_peak), and corresponding ground truth target parameters (theta, r, vr)
    from chunked .npy files.
    """
    # expected_k now directly controlled by effective_k (derived from args.max_targets)
    def __init__(self, data_root, start_idx, end_idx, expected_k):
        """
        Initializes the dataset.

        Args:
            data_root (str): Root directory containing 'echoes' subdirectory,
                             trajectory_data.npz, and system_params.npz.
            start_idx (int): Absolute start index for this dataset slice.
            end_idx (int): Absolute end index for this dataset slice.
            expected_k (int): Expected maximum number of targets K (for padding/truncation).
        """
        super().__init__()
        self.data_root = data_root
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_samples = end_idx - start_idx + 1
        self.expected_k = expected_k # Now set by effective_k

        print(f"  Dataset Init: Root='{data_root}', Range=[{start_idx}, {end_idx}], Num Samples={self.num_samples}, Expected K={self.expected_k}") # Added expected_k

        self.echoes_dir = os.path.join(data_root, 'echoes')
        if not os.path.isdir(self.echoes_dir):
            raise FileNotFoundError(f"Echoes directory not found: {self.echoes_dir}")

        params_path = os.path.join(data_root, 'system_params.npz')
        if not os.path.isfile(params_path):
            raise FileNotFoundError(f"System parameters file not found: {params_path}")
        try:
            params_data = np.load(params_path)
            if 'samples_per_chunk' not in params_data:
                if 'chunk_size' in params_data: self.chunk_size = int(params_data['chunk_size'])
                else: raise KeyError("Missing 'samples_per_chunk' or 'chunk_size' in system_params.npz.")
            else: self.chunk_size = int(params_data['samples_per_chunk'])
            self.M_plus_1 = int(params_data['M']) + 1 if 'M' in params_data else None
            self.Ns = int(params_data['Ns']) if 'Ns' in params_data else None
            # print(f"  Loaded chunk_size: {self.chunk_size} from params") # Less verbose
            # if self.M_plus_1: print(f"  Loaded M+1: {self.M_plus_1} from params")
            # if self.Ns: print(f"  Loaded Ns: {self.Ns} from params")
        except Exception as e: raise IOError(f"Error loading or parsing system_params.npz: {e}")
        if self.chunk_size <= 0: raise ValueError("samples_per_chunk must be positive.")

        traj_path = os.path.join(data_root, 'trajectory_data.npz')
        if not os.path.isfile(traj_path): raise FileNotFoundError(f"Trajectory data file not found: {traj_path}")
        try:
            traj_data = np.load(traj_path)
            if 'm_peak_indices' not in traj_data:
                if 'm_peak' in traj_data: m_peak_all = traj_data['m_peak']
                else: raise KeyError("Missing 'm_peak_indices' or 'm_peak' in trajectory_data.npz")
            else: m_peak_all = traj_data['m_peak_indices']

            if 'theta_traj' not in traj_data or 'r_traj' not in traj_data or 'vr' not in traj_data:
                raise KeyError("trajectory_data.npz missing 'theta_traj', 'r_traj', or 'vr' data.")
            theta_all = traj_data['theta_traj']
            r_all = traj_data['r_traj']
            vr_all = traj_data['vr']

            if theta_all.ndim == 3 and r_all.ndim == 3 and vr_all.ndim == 3:
                # print("  Detected GT params include Ns dimension, will average during loading.") # Less verbose
                self.gt_needs_averaging = True
            elif theta_all.ndim == 2 and r_all.ndim == 2 and vr_all.ndim == 2:
                # print("  Detected GT params do not include Ns dimension.") # Less verbose
                self.gt_needs_averaging = False
            else:
                raise ValueError("Unsupported dimensions for 'theta', 'r', 'vr' in trajectory data. Expect (N_total, K) or (N_total, Ns, K).")

            total_samples_in_file = m_peak_all.shape[0]
            if self.end_idx >= total_samples_in_file:
                print(f"Warning: Requested end_idx ({self.end_idx}) exceeds available samples ({total_samples_in_file}) in trajectory_data.npz.")
                self.end_idx = total_samples_in_file - 1; self.num_samples = self.end_idx - self.start_idx + 1
                if self.num_samples <= 0: raise ValueError(f"Adjusted sample range is invalid [{self.start_idx}, {self.end_idx}]")
                print(f"  Adjusted dataset range: [{self.start_idx}, {self.end_idx}], Num Samples={self.num_samples}")

            self.m_peak_targets = m_peak_all[self.start_idx : self.end_idx + 1]
            self.theta_targets = theta_all[self.start_idx : self.end_idx + 1]
            self.r_targets = r_all[self.start_idx : self.end_idx + 1]
            self.vr_targets = vr_all[self.start_idx : self.end_idx + 1]

            # Pad or truncate K dimension to expected_k for all targets
            # print(f"  Adjusting loaded target arrays to expected K={self.expected_k}...") # Verbose
            for name, arr_ref in [('m_peak', 'm_peak_targets'), ('theta', 'theta_targets'),
                                  ('r', 'r_targets'), ('vr', 'vr_targets')]:
                arr = getattr(self, arr_ref)
                # Handle potential 1D array if K=1 in source data
                current_k = arr.shape[-1] if arr.ndim >= 2 else 1
                if arr.ndim == 1: # Reshape K=1 data to have K dimension
                    arr = arr.reshape(-1, 1)
                    current_k = 1
                if arr.ndim == 3 and self.gt_needs_averaging: # Check last dim if Ns is present
                    current_k = arr.shape[-1]

                if current_k < self.expected_k:
                    # print(f"  Warning: {name} K dim ({current_k}) < expected_k ({self.expected_k}). Padding.")
                    if arr.ndim == 2: pad_config = ((0, 0), (0, self.expected_k - current_k))
                    elif arr.ndim == 3: pad_config = ((0, 0), (0, 0), (0, self.expected_k - current_k))
                    else: raise ValueError(f"{name} has unsupported dimensions {arr.ndim}.")
                    fill_value = -1 if name == 'm_peak' else np.nan
                    padded_arr = np.pad(arr, pad_config, 'constant', constant_values=fill_value)
                    setattr(self, arr_ref, padded_arr)
                elif current_k > self.expected_k:
                    # print(f"  Warning: {name} K dim ({current_k}) > expected_k ({self.expected_k}). Truncating.")
                    if arr.ndim == 2: truncated_arr = arr[:, :self.expected_k]
                    elif arr.ndim == 3: truncated_arr = arr[:, :, :self.expected_k]
                    else: raise ValueError(f"{name} has unsupported dimensions {arr.ndim}.")
                    setattr(self, arr_ref, truncated_arr)

            # print(f"  Loaded m_peak_targets shape: {self.m_peak_targets.shape}") # Less verbose
            # print(f"  Loaded theta_targets final shape: {self.theta_targets.shape}")
            # print(f"  Loaded r_targets shape: {self.r_targets.shape}")
            # print(f"  Loaded vr_targets shape: {self.vr_targets.shape}")

        except KeyError as e: raise IOError(f"Missing key while loading trajectory_data.npz: {e}")
        except Exception as e: raise IOError(f"Error loading or processing trajectory_data.npz: {e}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if index < 0 or index >= self.num_samples: raise IndexError(f"Index {index} out of bounds [0, {self.num_samples - 1}]")
        try:
            absolute_idx = self.start_idx + index
            chunk_idx = absolute_idx // self.chunk_size
            index_in_chunk = absolute_idx % self.chunk_size
            echo_file_path = os.path.join(self.echoes_dir, f'echo_chunk_{chunk_idx}.npy')
            if not os.path.isfile(echo_file_path): raise FileNotFoundError(f"Echo data file not found: {echo_file_path} (for chunk {chunk_idx})")

            echo_chunk = np.load(echo_file_path)
            # Check dimensions if M_plus_1 and Ns are known
            if self.Ns and self.M_plus_1 and echo_chunk.ndim >= 3 and echo_chunk.shape[1:] != (self.Ns, self.M_plus_1):
                print(f"Warning (idx={absolute_idx}, chunk={chunk_idx}): Echo chunk shape {echo_chunk.shape} mismatch expected ({(-1, self.Ns, self.M_plus_1)})")
            if index_in_chunk >= echo_chunk.shape[0]: raise IndexError(f"Index within chunk {index_in_chunk} out of bounds for chunk size {echo_chunk.shape[0]} (file: echo_chunk_{chunk_idx}.npy, absolute index: {absolute_idx})")

            clean_echo_signal = echo_chunk[index_in_chunk]
            m_peak = self.m_peak_targets[index] # Shape (expected_k,)
            theta = self.theta_targets[index]   # Shape (Ns, expected_k) or (expected_k,)
            r = self.r_targets[index]           # Shape (Ns, expected_k) or (expected_k,)
            vr = self.vr_targets[index]         # Shape (Ns, expected_k) or (expected_k,)

            # Average over Ns dimension if needed
            if self.gt_needs_averaging:
                # Use nanmean to handle potential NaNs from padding
                theta = np.nanmean(theta, axis=0) # Shape (expected_k,)
                r = np.nanmean(r, axis=0)         # Shape (expected_k,)
                vr = np.nanmean(vr, axis=0)       # Shape (expected_k,)

            # Convert to tensors
            echo_tensor = torch.from_numpy(clean_echo_signal).to(torch.complex64)
            m_peak_tensor = torch.from_numpy(m_peak).to(torch.long)
            theta_tensor = torch.from_numpy(theta).to(torch.float32)
            r_tensor = torch.from_numpy(r).to(torch.float32)
            vr_tensor = torch.from_numpy(vr).to(torch.float32)

            sample = {'echo': echo_tensor, 'm_peak': m_peak_tensor,
                      'theta': theta_tensor, 'r': r_tensor, 'vr': vr_tensor}
            return sample
        except FileNotFoundError as e: print(f"Error: File not found when loading index {index} (abs: {self.start_idx + index}): {e}", flush=True); raise
        except IndexError as e: print(f"Error: Index out of bounds when loading index {index} (abs: {self.start_idx + index}): {e}", flush=True); raise
        except Exception as e: print(f"Error: Unexpected error loading index {index} (abs: {self.start_idx + index}): {e}", flush=True); traceback.print_exc(); raise
# --- End Dataset Definition ---

# --- 0.1 Command Line Argument Parsing ---
parser = argparse.ArgumentParser(description='YOLO method evaluation script testing multiple Pt levels.')
# --- MODIFIED: Removed pt_dbm ---
# parser.add_argument('--pt_dbm', type=float, default=30.0, help='Transmit Power (dBm)')
parser.add_argument('--cuda_device', type=str, default='cpu', help='CUDA device ID (set for env var, but processing uses CPU)')
parser.add_argument('--num_test_samples', type=int, default=None, help='Number of test samples to process (default: all available)') # Default None
parser.add_argument('--num_print_details', type=int, default=2, help='Number of samples per Pt for which to print detailed processing info (default: 2)') # Reduced default
parser.add_argument('--data_dir', type=str, default=None, help='Directory containing data and system_params.npz (default: read from latest_experiment.txt)')
parser.add_argument('--max_targets', type=int, default=None, help='Expected maximum number of targets (K_max) to handle. Overrides K from params file if set.')

args = parser.parse_args()

# --- MODIFIED: Define Power Levels to Test ---
pt_dbm_test_list = [-10.0, 0.0, 10.0, 20.0, 30.0]
print(f"Will test the following Pt levels: {pt_dbm_test_list} dBm")
# ---

# --- 0.2 Core Configuration ---
BATCH_SIZE = 1 # Process one sample at a time for this script
NUM_WORKERS = 0 # Use 0 for numpy based processing compatibility

# Test and Print Control
# NUM_SAMPLES_TO_PROCESS set later after loading dataset
MAX_SAMPLES_TO_PRINT_DETAILS = args.num_print_details

# 2D-CFAR Parameters (Keep as before)
GUARD_SIZE_DOPPLER = 2
REF_SIZE_DOPPLER   = 4
GUARD_SIZE_ANGLE   = 1
REF_SIZE_ANGLE     = 4
alpha = 0.8

# YOLO/MUSIC Estimation Parameters (Keep as before)
SIDELOBE_WINDOW = 10
LOCAL_MAX_WINDOW = 1
SIDELOBE_EXCLUDE_SUBCARRIERS = 50

# Search Ranges (Keep as before)
V_SEARCH_RANGE = np.linspace(-10.5, 10.5, 2001)
R_SEARCH_RANGE = np.arange(34.5, 200.5, 0.01)

# Beamforming Angle Range (Keep as before)
PHI_START_DEG = -60
PHI_END_DEG = 60

# --- 0.2.1 Determine Data Root Directory ---
# (Function unchanged)
def get_latest_experiment_path():
    """Tries to find the latest experiment path from standard locations."""
    try:
        try: script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError: script_dir = os.getcwd() # Fallback for interactive environments
        paths_to_check = [
            '/mnt/sda/liangqiushi/moving_target/latest_experiment.txt', # Specific absolute path
            os.path.join(script_dir, 'latest_experiment.txt'),         # Path relative to script
            os.path.join(os.getcwd(), 'latest_experiment.txt')         # Path relative to current working dir
        ]
        file_path = next((p for p in paths_to_check if os.path.exists(p)), None)
        if file_path is None: raise FileNotFoundError("未在标准位置找到 latest_experiment.txt。")
        with open(file_path, 'r') as f: return f.read().strip()
    except FileNotFoundError: print("错误：未找到 'latest_experiment.txt'。", flush=True); raise
    except Exception as e: print(f"读取 latest_experiment.txt 时发生错误: {e}", flush=True); raise

if args.data_dir:
    DATA_ROOT = args.data_dir
else:
    print("Data directory (--data_dir) not specified, attempting to read from latest_experiment.txt...")
    try:
        DATA_ROOT = get_latest_experiment_path()
    except Exception: # Catch both FileNotFoundError and other read errors
        print("Error: --data_dir not specified and failed to read latest_experiment.txt. Please provide data directory.")
        exit(1)
print(f"Using data root directory: {DATA_ROOT}")
PARAMS_FILE = os.path.join(DATA_ROOT, 'system_params.npz')


# --- 0.3 Device Setup ---
if args.cuda_device.lower() != 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    print(f"Environment CUDA_VISIBLE_DEVICES set to '{args.cuda_device}' (Processing will still use CPU).")
else:
    print("CUDA device set to 'cpu'. Processing will use CPU.")
device = torch.device('cpu') # Force CPU


# --- 0.4 Load System Parameters ---
print(f"Loading system parameters from {PARAMS_FILE}...")
K_data_param = None # Variable to store K specifically from the params file
try:
    params_data = np.load(PARAMS_FILE)
    Nt = int(params_data['Nt']) if 'Nt' in params_data else 1
    Nr = int(params_data['Nr']) if 'Nr' in params_data else 1
    M = int(params_data['M'])
    Ns = int(params_data['Ns'])
    fc = float(params_data['fc'])
    f_scs = float(params_data['f_scs'])
    if 'K' in params_data and params_data['K'] is not None:
        K_data_param = int(params_data['K'])
    lambda_c = float(params_data['lambda_c']) if 'lambda_c' in params_data else c/fc
    if 'BW' in params_data: BW = float(params_data['BW'])
    else: BW = M * f_scs; print(f"  Calculated BW from M and f_scs: {BW:.2e} Hz")
    if 'fm_list' in params_data: f0 = params_data['fm_list'][0]
    else: f0 = fc - BW / 2; print(f"  Warning: 'fm_list' missing. Approximating f0 = fc - BW/2 = {f0/1e9:.4f} GHz")
    Ts = float(params_data['Delta_T']) if 'Delta_T' in params_data else (1/f_scs)

except FileNotFoundError: print(f"Error: Parameter file {PARAMS_FILE} not found."); exit(1)
except KeyError as e: print(f"Error: Parameter file {PARAMS_FILE} missing required key: {e}"); exit(1)
except Exception as e: print(f"Error loading parameter file: {e}"); exit(1)

# --- Determine effective K to use ---
if args.max_targets is not None:
    effective_k = args.max_targets
    print(f"Using K = {effective_k} from command line argument (--max_targets).")
    if K_data_param is not None and K_data_param != effective_k:
        print(f"  Warning: This differs from K={K_data_param} found in {PARAMS_FILE}.")
elif K_data_param is not None:
    effective_k = K_data_param
    print(f"Using K = {effective_k} from parameter file ({PARAMS_FILE}).")
else:
    effective_k = 3 # Fallback default if not specified anywhere
    print(f"Warning: K not specified in {PARAMS_FILE} or via --max_targets. Using default K = {effective_k}.")
# ---

print("System Parameter Overview:")
print(f"  Nt={Nt}, Nr={Nr}, M={M}, Ns={Ns}, K (Effective)={effective_k}") # Updated K print
print(f"  fc={fc/1e9:.2f} GHz, f_scs={f_scs/1e3:.2f} kHz, Ts={Ts*1e6:.2f} us, BW={BW/1e6:.2f} MHz")
print(f"  f0 (for Vel MUSIC)={f0/1e9:.4f} GHz")

# --- 0.5 Calculate Constant Noise Standard Deviation ---
# Noise std dev depends only on BW and Temp, not Pt
noise_power = K_BOLTZMANN * T_NOISE_KELVIN * BW
noise_std_dev = math.sqrt(noise_power / 2.0)
print(f"Thermal Noise Power (kTB): {noise_power:.2e} W -> Noise Std Dev (per component): {noise_std_dev:.2e}")
noise_std_dev_tensor = torch.tensor(noise_std_dev, dtype=torch.float32).to(device)


# ==============================================================================
# 1. Data Loading (Load ONCE)
# ==============================================================================
print("\n--- Data Loading ---")
print(f"Target dataset: {DATA_ROOT}")
if not os.path.exists(DATA_ROOT): print(f"Error: Data directory {DATA_ROOT} does not exist."); exit(1)

print("Initializing dataset object (ChunkedEchoDataset)...")
try:
    # Check trajectory_data.npz size first to get available samples
    try:
        traj_path_check = os.path.join(DATA_ROOT, 'trajectory_data.npz')
        traj_data_check = np.load(traj_path_check)
        key_to_check = 'm_peak_indices' if 'm_peak_indices' in traj_data_check else 'm_peak'
        total_samples_in_datafile = 7500
        print(f"  Found {total_samples_in_datafile} total samples in trajectory data file.")
    except Exception as e:
        print(f"  Warning: Could not determine total samples from trajectory file ({e}). Assuming large number.")
        total_samples_in_datafile = 7500 # Fallback

    dataset_end_idx = total_samples_in_datafile - 1
    test_dataset = ChunkedEchoDataset(DATA_ROOT, 0, dataset_end_idx, expected_k=effective_k)
    total_samples_in_dataset = len(test_dataset)

    num_samples_to_run = total_samples_in_dataset
    if args.num_test_samples is not None and args.num_test_samples >= 0:
        num_samples_to_run = min(total_samples_in_dataset, args.num_test_samples)
        print(f"Limiting processing to first {num_samples_to_run} samples from dataset (Total available: {total_samples_in_dataset}).")
    else:
        print(f"Processing all {num_samples_to_run} available samples in the dataset.")
    NUM_SAMPLES_TO_PROCESS = num_samples_to_run

except FileNotFoundError as e: print(f"Error: File not found during dataset initialization: {e}"); exit(1)
except Exception as e: print(f"Error initializing dataset: {e}"); traceback.print_exc(); exit(1)

if num_samples_to_run == 0: print("No samples available to process. Exiting."); exit()

# Create DataLoader ONCE
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=False)

# ==============================================================================
# --- MODIFIED: Outer Loop for Power Levels ---
# ==============================================================================
all_metrics = {} # Dictionary to store metrics for each Pt level

for current_pt_dbm in pt_dbm_test_list:
    print(f"\n\n{'='*20} Testing with Pt = {current_pt_dbm:.1f} dBm {'='*20}")

    # --- Recalculate Scaling Factor for Current Power ---
    pt_linear_mw = 10**(current_pt_dbm / 10.0)
    pt_scaling_factor = math.sqrt(pt_linear_mw)
    print(f"  Amplitude Scaling Factor: {pt_scaling_factor:.3f}")

    # ==============================================================================
    # 2. Result Storage Initialization (Re-initialize for each Pt)
    # ==============================================================================
    print("\n  --- Initializing Result Storage for current Pt ---")
    est_theta_raw = np.full((num_samples_to_run, effective_k), np.nan)
    est_r_raw = np.full((num_samples_to_run, effective_k), np.nan)
    est_v_raw = np.full((num_samples_to_run, effective_k), np.nan)
    est_subcarriers_raw = np.full((num_samples_to_run, effective_k), -1, dtype=int)
    true_theta = np.full((num_samples_to_run, effective_k), np.nan)
    true_r = np.full((num_samples_to_run, effective_k), np.nan)
    true_v = np.full((num_samples_to_run, effective_k), np.nan)
    matched_est_theta = np.full((num_samples_to_run, effective_k), np.nan)
    matched_est_r = np.full((num_samples_to_run, effective_k), np.nan)
    matched_est_v = np.full((num_samples_to_run, effective_k), np.nan)


    # ==============================================================================
    # 3. Main Processing Loop (Iterate through samples for current Pt)
    # ==============================================================================
    print(f"\n  --- Starting processing for {num_samples_to_run} samples at {current_pt_dbm:.1f} dBm ---")
    start_time_pt = time.time()

    # Use islice to limit the loader iteration (Recreate iterator for each Pt)
    limited_loader = islice(test_loader, num_samples_to_run)
    data_iterator = tqdm(enumerate(limited_loader), total=num_samples_to_run,
                         desc=f"Pt={current_pt_dbm:.0f}dBm Samples", file=sys.stdout, leave=False) # leave=False for inner loop

    for batch_idx, batch in data_iterator:
        current_sample_index = batch_idx # Index within 0 to num_samples_to_run-1

        print_details_this_sample = (current_sample_index < MAX_SAMPLES_TO_PRINT_DETAILS)

        # --- 3.1 Data Preparation ---
        try:
            clean_echo = batch['echo'].to(device)       # [1, Ns, M+1]
            gt_theta = batch['theta'].cpu().numpy()     # [1, effective_k]
            gt_r = batch['r'].cpu().numpy()             # [1, effective_k]
            gt_vr = batch['vr'].cpu().numpy()           # [1, effective_k]
        except KeyError as e: print(f"Error: Missing key in sample {current_sample_index + 1} data: {e}. Skipping."); continue
        except Exception as e: print(f"Error processing sample {current_sample_index + 1} data: {e}. Skipping."); continue

        # Store ground truth (squeeze batch dim)
        true_theta[current_sample_index, :] = gt_theta[0, :]
        true_r[current_sample_index, :] = gt_r[0, :]
        true_v[current_sample_index, :] = gt_vr[0, :]

        # --- 3.2 Add Noise (using current_pt_dbm's scaling factor) ---
        scaled_echo = clean_echo * pt_scaling_factor # Use factor calculated for this Pt loop
        noise = (torch.randn_like(scaled_echo.real) + 1j * torch.randn_like(scaled_echo.imag)) * noise_std_dev_tensor
        y_echo_noisy = scaled_echo + noise # Still a tensor [1, Ns, M+1]

        # --- 3.3 Signal Processing Pipeline (YOLO/CFAR/MUSIC - Numpy based) ---
        Y_sample = y_echo_noisy[0, :, :].cpu().numpy() # Convert to numpy [Ns, M+1]
        Y_dynamic = Y_sample.copy()
        Y_fft = np.fft.fft(Y_dynamic, axis=0)
        Y_AD = np.fft.fftshift(Y_fft, axes=0) # Angle-Doppler Domain
        G_AD = np.abs(Y_AD) # Magnitude

        # --- CFAR Detection ---
        cfar_mask = np.zeros_like(G_AD, dtype=bool)
        Ns_dim, M_dim = G_AD.shape
        d_min = REF_SIZE_DOPPLER + GUARD_SIZE_DOPPLER
        d_max = Ns_dim - (REF_SIZE_DOPPLER + GUARD_SIZE_DOPPLER) - 1
        m_min = REF_SIZE_ANGLE + GUARD_SIZE_ANGLE
        m_max = M_dim - (REF_SIZE_ANGLE + GUARD_SIZE_ANGLE) - 1

        for d_idx in range(max(0, d_min), min(Ns_dim, d_max + 1)):
             for m_idx in range(max(0, m_min), min(M_dim, m_max + 1)):
                 d_win = np.arange(d_idx - (REF_SIZE_DOPPLER + GUARD_SIZE_DOPPLER), d_idx + (REF_SIZE_DOPPLER + GUARD_SIZE_DOPPLER) + 1)
                 m_win = np.arange(m_idx - (REF_SIZE_ANGLE + GUARD_SIZE_ANGLE), m_idx + (REF_SIZE_ANGLE + GUARD_SIZE_ANGLE) + 1)
                 d_guard = np.arange(d_idx - GUARD_SIZE_DOPPLER, d_idx + GUARD_SIZE_DOPPLER + 1)
                 m_guard = np.arange(m_idx - GUARD_SIZE_ANGLE, m_idx + GUARD_SIZE_ANGLE + 1)
                 d_win_valid = d_win[(d_win >= 0) & (d_win < Ns_dim)]
                 m_win_valid = m_win[(m_win >= 0) & (m_win < M_dim)]
                 d_guard_valid = d_guard[(d_guard >= 0) & (d_guard < Ns_dim)]
                 m_guard_valid = m_guard[(m_guard >= 0) & (m_guard < M_dim)]
                 ref_window = G_AD[np.ix_(d_win_valid, m_win_valid)]
                 guard_cells = G_AD[np.ix_(d_guard_valid, m_guard_valid)]
                 effective_ref_count = ref_window.size - guard_cells.size
                 if effective_ref_count <= 0: continue
                 noise_sum = np.sum(ref_window) - np.sum(guard_cells)
                 noise_mean = noise_sum / effective_ref_count
                 threshold = alpha * noise_mean
                 if G_AD[d_idx, m_idx] > threshold: cfar_mask[d_idx, m_idx] = True

        cand_indices = np.argwhere(cfar_mask)
        if cand_indices.size == 0:
            if print_details_this_sample: print(f"    Sample {current_sample_index + 1}: No peaks detected by CFAR.");
            continue

        # --- NMS and Peak Selection ---
        cand_d = cand_indices[:, 0]; cand_m = cand_indices[:, 1]
        cand_power = G_AD[cand_d, cand_m]
        kept_idx = np.zeros(len(cand_d), dtype=bool)
        for i in range(len(cand_d)):
            d_val = cand_d[i]; m_val = cand_m[i]; val_ = cand_power[i]
            if (d_val <= LOCAL_MAX_WINDOW) or (d_val >= Ns_dim - LOCAL_MAX_WINDOW - 1) or \
               (m_val <= LOCAL_MAX_WINDOW) or (m_val >= M_dim - LOCAL_MAX_WINDOW - 1): continue
            d_neigh = np.arange(d_val - LOCAL_MAX_WINDOW, d_val + LOCAL_MAX_WINDOW + 1)
            m_neigh = np.arange(m_val - LOCAL_MAX_WINDOW, m_val + LOCAL_MAX_WINDOW + 1)
            sub_mat = G_AD[np.ix_(d_neigh, m_neigh)]
            if val_ >= np.max(sub_mat): kept_idx[i] = True

        cand_d2 = cand_d[kept_idx]; cand_m2 = cand_m[kept_idx]; cand_power2 = cand_power[kept_idx]
        if cand_d2.size == 0:
            if print_details_this_sample: print(f"    Sample {current_sample_index + 1}: No local maxima found after CFAR.");
            continue

        sort_idx = np.argsort(-cand_power2)
        final_idx = []
        for i in sort_idx:
            current_m = cand_m2[i]; is_close = False
            for j in final_idx:
                if abs(current_m - cand_m2[j]) < SIDELOBE_EXCLUDE_SUBCARRIERS: is_close = True; break
            if not is_close: final_idx.append(i)
            if len(final_idx) >= effective_k: break # Limit by effective_k

        if not final_idx:
            if print_details_this_sample: print(f"    Sample {current_sample_index + 1}: No peaks remaining after NMS.");
            continue

        K_eff_detected = len(final_idx)
        cand_d_final = cand_d2[final_idx]
        cand_m_final = cand_m2[final_idx]

        if print_details_this_sample:
            print(f"    Sample {current_sample_index + 1}: Detected {K_eff_detected} target candidates after NMS.")

        # --- 3.4 Parameter Estimation for Each Detected Peak ---
        for k_idx in range(K_eff_detected): # Iterate through detected targets
            peak_m_idx = cand_m_final[k_idx]

            # (a) Angle Estimation (Formula)
            angle_deg = np.nan
            fm_base = peak_m_idx * f_scs
            denom = BW * (fm_base + fc)
            if abs(denom) > 1e-9:
                term1 = ((BW - fm_base) * fc / denom) * np.sin(np.deg2rad(PHI_START_DEG))
                term2 = ((BW + fc) * fm_base / denom) * np.sin(np.deg2rad(PHI_END_DEG))
                asin_arg = np.clip(term1 + term2, -1.0, 1.0)
                try: angle_rad = np.arcsin(asin_arg); angle_deg = np.rad2deg(angle_rad)
                except ValueError: pass

            # Store raw estimate up to effective_k
            if k_idx < effective_k:
                est_theta_raw[current_sample_index, k_idx] = angle_deg
                est_subcarriers_raw[current_sample_index, k_idx] = peak_m_idx
            # else: # Should not be reached due to NMS break
            #      print(f"Warning: Sample {current_sample_index+1}: Attempting to store more targets ({k_idx+1}) than effective_k ({effective_k}). Discarding extra.")
            #      break

            if print_details_this_sample:
                 print(f"      Candidate {k_idx+1}: m-index={peak_m_idx}, Est Angle={angle_deg:.2f}°")

            # (b) Distance & Velocity Estimation (MUSIC)
            m_min_local = max(0, peak_m_idx - SIDELOBE_WINDOW)
            m_max_local = min(M, peak_m_idx + SIDELOBE_WINDOW)
            sub_index = np.arange(m_min_local, m_max_local + 1)
            sub_index = sub_index[sub_index < Y_dynamic.shape[1]]
            if len(sub_index) == 0:
                if k_idx < effective_k: est_v_raw[current_sample_index, k_idx] = np.nan; est_r_raw[current_sample_index, k_idx] = np.nan
                continue

            Y_RD_raw = Y_dynamic[:, sub_index]
            Ns_local, Sub_local = Y_RD_raw.shape
            if Ns_local == 0 or Sub_local == 0:
                if k_idx < effective_k: est_v_raw[current_sample_index, k_idx] = np.nan; est_r_raw[current_sample_index, k_idx] = np.nan
                continue

            # Y_RD = Y_RD_raw / (np.abs(Y_RD_raw) + 1e-10)
            Y_RD = Y_RD_raw

            # Velocity MUSIC
            v_est = np.nan
            rank_sig_v = 1
            if Ns_local > rank_sig_v:
                try:
                    R_D = (Y_RD @ np.conjugate(Y_RD.T)) / Sub_local
                    eigvals_D, U_D = np.linalg.eigh(R_D)
                    idx_sort_D = np.argsort(-eigvals_D); U_D = U_D[:, idx_sort_D]
                    U_vD = U_D[:, rank_sig_v:]
                    F_v = np.zeros_like(V_SEARCH_RANGE, dtype=float)
                    v_time_indices = np.arange(Ns)
                    for ii, v_cand in enumerate(V_SEARCH_RANGE):
                        a_v = np.exp(1j * (4 * np.pi * f0 * v_cand * Ts / c) * v_time_indices)
                        a_v_norm = np.linalg.norm(a_v);
                        if a_v_norm < 1e-10: continue
                        a_v = a_v / a_v_norm
                        P_v = np.linalg.norm(np.conjugate(a_v) @ U_vD)**2
                        F_v[ii] = 1 / (P_v + 1e-10)
                    if np.any(F_v > 0): max_idx_v = np.argmax(F_v); v_est = V_SEARCH_RANGE[max_idx_v]
                except np.linalg.LinAlgError: pass
            if k_idx < effective_k: est_v_raw[current_sample_index, k_idx] = v_est

            # Distance MUSIC
            r_est = np.nan
            rank_sig_r = 1
            if Sub_local > rank_sig_r:
                try:
                    R_r = (Y_RD.T @ np.conjugate(Y_RD)) / Ns
                    eigvals_R, U_R = np.linalg.eigh(R_r)
                    idx_sort_R = np.argsort(-eigvals_R); U_R = U_R[:, idx_sort_R]
                    U_rR = U_R[:, rank_sig_r:]
                    F_r = np.zeros_like(R_SEARCH_RANGE, dtype=float)
                    r_freq_indices = np.arange(Sub_local)
                    for ii, rr_cand in enumerate(R_SEARCH_RANGE):
                        a_r = np.exp(-1j * (4 * np.pi * rr_cand / c) * f_scs * r_freq_indices)
                        a_r_norm = np.linalg.norm(a_r);
                        if a_r_norm < 1e-10: continue
                        a_r = a_r / a_r_norm
                        P_r = np.linalg.norm(np.conjugate(a_r) @ U_rR)**2
                        F_r[ii] = 1 / (P_r + 1e-10)
                    if np.any(F_r > 0): max_idx_r = np.argmax(F_r); r_est = R_SEARCH_RANGE[max_idx_r]
                except np.linalg.LinAlgError: pass
            if k_idx < effective_k: est_r_raw[current_sample_index, k_idx] = r_est

            # if print_details_this_sample and k_idx < effective_k: # Moved print outside MUSIC
            #     v_str = f"{est_v_raw[current_sample_index, k_idx]:.2f} m/s" if not np.isnan(est_v_raw[current_sample_index, k_idx]) else "NaN"
            #     r_str = f"{est_r_raw[current_sample_index, k_idx]:.2f} m" if not np.isnan(est_r_raw[current_sample_index, k_idx]) else "NaN"
            #     print(f"        Est Velocity={v_str}, Est Distance={r_str}")

    # End Sample Loop
    data_iterator.close()
    end_time_pt = time.time()
    print(f"\n  --- Processing for Pt={current_pt_dbm:.1f} dBm Complete ---")
    print(f"  Processing time: {end_time_pt - start_time_pt:.2f} seconds")
    if num_samples_to_run > 0:
        print(f"  Average time per sample: {(end_time_pt - start_time_pt) / num_samples_to_run:.3f} seconds")

    # ==============================================================================
    # 4. Performance Evaluation (Run for each Pt level)
    # ==============================================================================
    print(f"\n  --- Performance Evaluation for Pt={current_pt_dbm:.1f} dBm ---")

    # --- 4.1 Apply Hungarian Matching ---
    print("  Applying Hungarian matching...")
    num_total_matched_pairs_pt = 0
    for i in range(num_samples_to_run): # No tqdm here, happens inside Pt loop
        sample_est_theta = est_theta_raw[i, :]
        sample_est_r = est_r_raw[i, :]
        sample_est_v = est_v_raw[i, :]
        sample_true_theta = true_theta[i, :]
        sample_true_r = true_r[i, :]
        sample_true_v = true_v[i, :]
        valid_est_indices = np.where(~np.isnan(sample_est_theta))[0]
        valid_true_indices = np.where(~np.isnan(sample_true_theta))[0]
        num_valid_est = len(valid_est_indices)
        num_valid_true = len(valid_true_indices)
        if num_valid_est == 0 or num_valid_true == 0: continue
        valid_est_angles = sample_est_theta[valid_est_indices]
        valid_true_angles = sample_true_theta[valid_true_indices]
        cost_matrix = np.abs(valid_est_angles[:, np.newaxis] - valid_true_angles[np.newaxis, :])
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            num_total_matched_pairs_pt += len(row_ind)
            for r, c in zip(row_ind, col_ind):
                original_est_idx = valid_est_indices[r]
                original_true_idx = valid_true_indices[c]
                if original_true_idx < effective_k:
                    matched_est_theta[i, original_true_idx] = sample_est_theta[original_est_idx]
                    matched_est_r[i, original_true_idx] = sample_est_r[original_est_idx]
                    matched_est_v[i, original_true_idx] = sample_est_v[original_est_idx]
        except ValueError as e: print(f"Error during Hungarian matching for sample {i} at Pt={current_pt_dbm}: {e}")
    print(f"  Matching complete for Pt={current_pt_dbm}. Total matched pairs: {num_total_matched_pairs_pt}")

    # --- 4.2 Calculate Errors and RMSE ---
    print("  Calculating errors and RMSE...")
    abs_error_theta = np.abs(matched_est_theta - true_theta)
    abs_error_r = np.abs(matched_est_r - true_r)
    abs_error_v = np.abs(matched_est_v - true_v)
    valid_errors_theta = abs_error_theta[~np.isnan(abs_error_theta)]
    valid_errors_r = abs_error_r[~np.isnan(abs_error_r)]
    valid_errors_v = abs_error_v[~np.isnan(abs_error_v)]
    num_valid_pairs_theta = len(valid_errors_theta)
    num_valid_pairs_r = len(valid_errors_r)
    num_valid_pairs_v = len(valid_errors_v)
    rmse_theta = np.sqrt(np.mean(valid_errors_theta**2)) if num_valid_pairs_theta > 0 else np.nan
    rmse_r = np.sqrt(np.mean(valid_errors_r**2)) if num_valid_pairs_r > 0 else np.nan
    rmse_v = np.sqrt(np.mean(valid_errors_v**2)) if num_valid_pairs_v > 0 else np.nan
    percentile_95_theta = np.percentile(valid_errors_theta, 95) if num_valid_pairs_theta > 0 else np.nan
    percentile_95_r = np.percentile(valid_errors_r, 95) if num_valid_pairs_r > 0 else np.nan
    percentile_95_v = np.percentile(valid_errors_v, 95) if num_valid_pairs_v > 0 else np.nan

    # --- Store metrics for this Pt ---
    all_metrics[current_pt_dbm] = {
        'num_matched': num_total_matched_pairs_pt,
        'num_valid_err_theta': num_valid_pairs_theta,
        'num_valid_err_r': num_valid_pairs_r,
        'num_valid_err_v': num_valid_pairs_v,
        'rmse_theta': rmse_theta, 'p95_theta': percentile_95_theta,
        'rmse_r': rmse_r, 'p95_r': percentile_95_r,
        'rmse_v': rmse_v, 'p95_v': percentile_95_v,
    }

    # --- 4.4 Print Matched Detailed Results (Optional, for current Pt) ---
    if MAX_SAMPLES_TO_PRINT_DETAILS > 0:
        print(f"\n  --------- Matched Target Comparison (First {MAX_SAMPLES_TO_PRINT_DETAILS} Samples, Pt={current_pt_dbm}dBm) ---------")
        num_samples_to_print_pt = min(num_samples_to_run, MAX_SAMPLES_TO_PRINT_DETAILS)
        for idx in range(num_samples_to_print_pt):
            print(f"\n  Sample {idx+1} Matched Results (Pt={current_pt_dbm}):")
            raw_m_indices = est_subcarriers_raw[idx, est_subcarriers_raw[idx,:] != -1] # Show valid detected m-indices
            print(f"    Detected M-Indices (this Pt): {raw_m_indices}")
            print(f"    {'True Idx':<8} {'Matched Est Ang':<15} {'True Ang':<10} | {'Matched Est Dist':<16} {'True Dist':<11} | {'Matched Est Vel':<15} {'True Vel':<10}")
            print( "    " + "-" * 90)
            num_printed_targets_pt = 0
            for k in range(effective_k): # Iterate through potential true target slots
                current_true_a = true_theta[idx, k]
                if not np.isnan(current_true_a): # Only print rows for valid true targets
                    matched_a = matched_est_theta[idx, k]
                    matched_d = matched_est_r[idx, k]
                    matched_v = matched_est_v[idx, k]
                    current_true_d = true_r[idx, k]
                    current_true_v = true_v[idx, k]
                    est_a_str = f"{matched_a:.2f}°" if not np.isnan(matched_a) else "NaN (Unmatched)"
                    est_d_str = f"{matched_d:.2f}m" if not np.isnan(matched_d) else "NaN (Unmatched)"
                    est_ve_str = f"{matched_v:.2f}m/s" if not np.isnan(matched_v) else "NaN (Unmatched)"
                    true_a_str = f"{current_true_a:.2f}°"
                    true_d_str = f"{current_true_d:.2f}m"
                    true_ve_str = f"{current_true_v:.2f}m/s"
                    print(f"    {k:<8} {est_a_str:<15} {true_a_str:<10} | {est_d_str:<16} {true_d_str:<11} | {est_ve_str:<15} {true_ve_str:<10}")
                    num_printed_targets_pt += 1
            if num_printed_targets_pt == 0: print("    (No valid true targets found for this sample)")
        print("  " + "-" * 95)


    # --- 4.5 Plot CDF Curves (Generate one plot per Pt) ---
    print(f"\n  Plotting error CDF curves for Pt={current_pt_dbm} dBm...")
    fig_cdf_pt, axs_cdf_pt = plt.subplots(1, 3, figsize=(18, 5))

    # Define CDF plotting function (ensure English labels) - Copied here for completeness
    def plot_cdf(ax, data, label, unit):
        """Helper function to plot CDF."""
        if data is None or len(data) == 0: # Check for None added
            ax.text(0.5, 0.5, 'No Valid Error Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{label} Error CDF")
            ax.set_xlabel(f"Absolute Error ({unit})")
            ax.set_ylabel('CDF')
            ax.grid(True)
            return
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf, marker='.', linestyle='-', markersize=2)
        p95_val = np.percentile(sorted_data, 95)
        ax.axvline(p95_val, color='r', linestyle='--', label=f"95th Perc. ({p95_val:.2f} {unit})")
        ax.axhline(0.95, color='r', linestyle=':', alpha=0.7)
        ax.set_title(f"{label} Error CDF (Matched)")
        ax.set_xlabel(f"Absolute Error ({unit})")
        ax.set_ylabel('CDF')
        ax.legend()
        ax.grid(True)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(left=0)

    # Use the valid (matched) errors for plotting
    plot_cdf(axs_cdf_pt[0], valid_errors_theta, "Angle", "degrees")
    plot_cdf(axs_cdf_pt[1], valid_errors_r, "Distance", "m")
    plot_cdf(axs_cdf_pt[2], valid_errors_v, "Velocity", "m/s")

    plt.suptitle(f"YOLO-CFAR-MUSIC Error CDF - Pt={current_pt_dbm:.1f}dBm (Matched, K_eff={effective_k})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save CDF plot
    timestamp_cdf = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # --- MODIFIED: Include Pt in filename ---
    cdf_filename = f"yolo_cfarmusic_matched_error_cdf_pt{current_pt_dbm:.0f}dBm_k{effective_k}_{timestamp_cdf}.png"
    # ---
    try: script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: script_dir = os.getcwd()
    cdf_save_path = os.path.join(script_dir, cdf_filename) # Save in script dir for simplicity
    try:
        plt.savefig(cdf_save_path)
        print(f"  CDF plot saved to: {cdf_save_path}")
    except Exception as e: print(f"  Error saving CDF plot: {e}")
    plt.close(fig_cdf_pt) # Close the figure for this Pt

# --- End Pt Loop ---

# ==============================================================================
# 5. Final Summary Report
# ==============================================================================
print("\n\n" + "=" * 80)
print(" " * 20 + "Overall Performance Summary Across Pt Levels")
print("=" * 80)
print(f"Effective Max Targets (K_eff) used: {effective_k}")
print(f"Total Samples Processed per Pt: {num_samples_to_run}")
print("-" * 80)
print(f"{'Pt (dBm)':<10} | {'RMSE Angle':<12} | {'P95 Angle':<12} | {'RMSE Dist':<12} | {'P95 Dist':<12} | {'RMSE Vel':<12} | {'P95 Vel':<12}")
print("-" * 80)

for pt_dbm in pt_dbm_test_list:
    if pt_dbm in all_metrics:
        metrics = all_metrics[pt_dbm]
        print(f"{pt_dbm:<10.1f} | "
              f"{metrics.get('rmse_theta', np.nan):<12.4f} | {metrics.get('p95_theta', np.nan):<12.4f} | "
              f"{metrics.get('rmse_r', np.nan):<12.4f} | {metrics.get('p95_r', np.nan):<12.4f} | "
              f"{metrics.get('rmse_v', np.nan):<12.4f} | {metrics.get('p95_v', np.nan):<12.4f}")
    else:
        print(f"{pt_dbm:<10.1f} | {'Not Run':<12} | {'Not Run':<12} | {'Not Run':<12} | {'Not Run':<12} | {'Not Run':<12} | {'Not Run':<12}")
print("-" * 80)

print("\nScript finished.")