import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import wfdb

# === CONFIGURATION ===
DATA_DIR = r"C:/Users/bhavn/Downloads/Physionet"
record_ids = [
    "bidmc01", "bidmc02", "bidmc03", "bidmc53", "bidmc09", "bidmc10", "bidmc14", 
    "bidmc17", "bidmc18", "bidmc20", "bidmc19", "bidmc23", "bidmc25", "bidmc28", 
    "bidmc29", "bidmc30", "bidmc32", "bidmc33", "bidmc06", "bidmc11"
]

# === PROCESSING ===
mean_intervals = []
std_intervals = []
processed_ids = []

for rec_id in record_ids:
    rec_path = os.path.join(DATA_DIR, rec_id)

    try:
        record = wfdb.rdrecord(rec_path, physical=True)
    except Exception as e:
        print(f"[SKIP] {rec_id}: can't read ({e})")
        continue

    # Clean channel names by removing commas and whitespace
    clean_channels = [ch.strip().rstrip(',') for ch in record.sig_name]
    
    # Look for PLETH channel
    if "PLETH" not in clean_channels:
        print(f"[SKIP] {rec_id}: no PLETH channel found. Available: {record.sig_name}")
        continue

    # Find the original channel index (before cleaning)
    pleth_index = None
    for i, ch in enumerate(record.sig_name):
        if ch.strip().rstrip(',') == "PLETH":
            pleth_index = i
            break
    
    ppg = record.p_signal[:, pleth_index]
    fs = record.fs

    # Check for valid PPG signal
    if np.isnan(ppg).all() or len(ppg) == 0:
        print(f"[SKIP] {rec_id}: invalid PPG signal")
        continue

    # Bandpass filter: 0.5–8 Hz
    try:
        b, a = butter(2, [0.5 / (fs / 2), 8 / (fs / 2)], btype='band')
        ppg_filtered = filtfilt(b, a, ppg)
    except Exception as e:
        print(f"[SKIP] {rec_id}: filtering failed ({e})")
        continue

    # Peak detection with adaptive parameters
    distance = int(0.5 * fs)  # Minimum 0.5 seconds between peaks
    height = np.percentile(ppg_filtered, 70)  # Use 70th percentile as height threshold
    peaks, _ = find_peaks(ppg_filtered, distance=distance, height=height)

    if len(peaks) < 2:
        # Try with lower threshold if no peaks found
        height = np.percentile(ppg_filtered, 50)
        peaks, _ = find_peaks(ppg_filtered, distance=distance, height=height)
        
    if len(peaks) < 2:
        print(f"[SKIP] {rec_id}: too few peaks detected ({len(peaks)})")
        continue

    intervals = np.diff(peaks / fs)
    # Filter out unrealistic intervals (0.3-2.0 seconds = 30-200 bpm)
    valid_intervals = intervals[(intervals >= 0.3) & (intervals <= 2.0)]
    
    if len(valid_intervals) < 2:
        print(f"[SKIP] {rec_id}: too few valid intervals")
        continue

    mean_int = np.mean(valid_intervals)
    std_int = np.std(valid_intervals)
    hr = 60 / mean_int
    # Corrected HR standard deviation calculation
    hr_std = 60 * std_int / (mean_int ** 2) 

    print(f"{rec_id}: HR = {hr:.1f} ± {hr_std:.1f} bpm ({len(valid_intervals)} intervals)")

    mean_intervals.append(mean_int)
    std_intervals.append(std_int)
    processed_ids.append(rec_id)

# === PLOTTING ===
if len(processed_ids) > 0:
    x = np.arange(len(processed_ids))
    plt.figure(figsize=(14, 8))
    plt.bar(x, mean_intervals, yerr=std_intervals, capsize=5, color='skyblue', edgecolor='black')
    plt.xticks(x, processed_ids, rotation=45)
    plt.xlabel("Recording ID")
    plt.ylabel("Mean Peak Intervals (s)")
    plt.title("Mean PPG Intervals ± SD across BIDMC-PPG Recordings")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    print(f"\nProcessed {len(processed_ids)} records successfully!")
else:
    print("No records were processed successfully!")
