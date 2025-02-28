import wfdb
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import resample
import os
import logging
from multiprocessing import Pool, cpu_count

# Set up logging to track errors and progress
logging.basicConfig(filename='preprocess.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def process_file(file_info):
    """Function to process a single ECG file."""
    dataset_path, file, output_dir = file_info
    record_name = os.path.join(dataset_path, file.split('.')[0])

    try:
        # Read ECG record
        record = wfdb.rdrecord(record_name)
        signal = record.p_signal[:, 0]  # Use first channel

        # Ensure signal has data
        if len(signal) == 0:
            raise ValueError("Empty signal detected")

        # Resample to 250 Hz
        signal_resampled = resample(signal, int(len(signal) * 250 / record.fs))

        # Normalize to [-1, 1]
        signal_normalized = (signal_resampled - signal_resampled.min()) / \
                            (signal_resampled.max() - signal_resampled.min()) * 2 - 1

        # Extract ECG features
        try:
            ecg_signals, info = nk.ecg_process(signal_normalized, sampling_rate=250)
            features = nk.ecg_analyze(ecg_signals, sampling_rate=250)
        except Exception as e:
            logging.warning(f"Failed to extract features for {file}: {e}. Using minimal features.")
            r_peaks = nk.ecg_peaks(signal_normalized, sampling_rate=250)['ECG_R_Peaks']
            hrv = np.diff(r_peaks) if len(r_peaks) > 1 else np.nan
            features = pd.DataFrame({
                'HRV': [hrv.mean() if not np.isnan(hrv) else 0],
                'R_peaks': [len(r_peaks)],
                'amplitude': [signal_normalized.max() - signal_normalized.min()]
            })

        # Save processed signal and features
        np.save(os.path.join(output_dir, f"{file.split('.')[0]}_processed.npy"), signal_normalized)
        features.to_csv(os.path.join(output_dir, f"{file.split('.')[0]}_features.csv"), index=False)

        return f"Successfully processed {file}"

    except Exception as e:
        return f"Error processing {file}: {e}"

def preprocess_ecg(data_dir, output_dir, max_files=None):
    """Preprocess ECG data using parallel processing."""
    os.makedirs(output_dir, exist_ok=True)

    # Collect all files from both datasets
    file_list = []
    for dataset in ["mitdb", "ptb-xl"]:
        dataset_path = os.path.join(data_dir, dataset)
        if not os.path.exists(dataset_path):
            logging.warning(f"Dataset path {dataset_path} not found, skipping.")
            continue
        
        files = [f for f in os.listdir(dataset_path) if f.endswith(".dat")]
        if max_files:
            files = files[:max_files]  # Limit for testing

        file_list.extend([(dataset_path, f, output_dir) for f in files])

    logging.info(f"Processing {len(file_list)} files in parallel")

    # Use multiprocessing to speed up processing
    num_workers = min(cpu_count(), 8)  # Use up to 8 cores for efficiency
    with Pool(num_workers) as pool:
        results = pool.map(process_file, file_list)

    for res in results:
        logging.info(res)

    logging.info("Preprocessing completed.")

if __name__ == "__main__":
    preprocess_ecg("C:/Users/akshi/OneDrive/Desktop/HeartGuardAI/data", 
                   "C:/Users/akshi/OneDrive/Desktop/HeartGuardAI/data/processed")
