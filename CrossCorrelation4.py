# === ECG & PPG Cross-Correlation: All in One File ===

import wfdb
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, correlate

# ----------------------------------
# Signal Helper Functions
# ----------------------------------
def bandpass_filter(signal, freq_range, fs, order=2):
    """Apply a Butterworth bandpass filter to the signal."""
    nyquist = 0.5 * fs
    low = freq_range[0] / nyquist
    high = freq_range[1] / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def extract_hr_hrv(peaks, fs):
    """Compute HR and HRV metrics from peak indices."""
    if len(peaks) < 2:
        return None, None, None, None

    rr_intervals_sec = np.diff(peaks) / fs
    rr_intervals_ms = rr_intervals_sec * 1000
    hr_values = 60 / rr_intervals_sec

    sdnn = np.std(rr_intervals_ms)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals_ms) ** 2))

    return hr_values, rr_intervals_ms, sdnn, rmssd

# ----------------------------------
# ECG Signal Processing
# ----------------------------------
def run_ecg_work():
    ecg_channels = ["II", "MLII", "ECG"]
    path = "C:/Users/bhavn/Downloads/Physionet/bidmc53"



    record = wfdb.rdrecord(path)
    print("Available channels:", record.sig_name)

    ch_names = [label.upper().replace(',', '').strip() for label in record.sig_name]
    ecg_channels = ["II", "MLII", "ECG"]

    idx = next((i for i, name in enumerate(ch_names) if name in ecg_channels), None)
    if idx is None:
        print("No ECG channel found.")
        return {}

    raw_ecg = record.p_signal[:, idx]
    fs = record.fs

    filtered = bandpass_filter(raw_ecg, (5, 15), fs)
    mean_val = np.mean(filtered)
    std_val = np.std(filtered)

    peaks, _ = find_peaks(
        filtered,
        distance=int(fs * 0.3),
        prominence=std_val * 0.6,
        height=mean_val + std_val * 0.5
    )

    hr, rr, sdnn, rmssd = extract_hr_hrv(peaks, fs)
    return {
        "type": "ECG",
        "fs": fs,
        "hr_data": hr,
        "rr_vals": rr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "peaks": peaks,
        "filtered": filtered
    }

# ----------------------------------
# PPG Signal Processing
# ----------------------------------
def run_ppg_work():
    ppg_channels = ["PPG", "PLETH"]
    path = "C:/Users/bhavn/Downloads/Physionet/bidmc53"


    record = wfdb.rdrecord(path)
    print("Available channels:", record.sig_name)

    ch_names = [label.upper().replace(',', '').strip() for label in record.sig_name]
    ppg_channels = ["PPG", "PLETH"]


    idx = next((i for i, name in enumerate(ch_names) if name in ppg_channels), None)
    if idx is None:
        print("No PPG channel found.")
        return {}

    raw_ppg = record.p_signal[:, idx]
    fs = record.fs

    filtered = bandpass_filter(raw_ppg, (0.5, 5.0), fs)
    mean_val = np.mean(filtered)
    std_val = np.std(filtered)

    peaks, _ = find_peaks(
        filtered,
        distance=int(fs * 0.4),
        prominence=std_val * 0.5,
        height=mean_val + std_val * 0.3
    )

    hr, rr, sdnn, rmssd = extract_hr_hrv(peaks, fs)
    return {
        "type": "PPG",
        "fs": fs,
        "hr_data": hr,
        "rr_vals": rr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "peaks": peaks,
        "filtered": filtered
    }

# ----------------------------------
# Main Comparison & Cross-Correlation
# ----------------------------------
def compare_signals_with_crosscorr():
    ecg = run_ecg_work()
    ppg = run_ppg_work()

    if not ecg or not ppg:
        print("One or both signals are missing. Exiting.")
        return

    print("=== HR & HRV Comparison ===")
    if ecg['hr_data'] is not None and ppg['hr_data'] is not None:
        print(f"→ ECG Average HR: {np.mean(ecg['hr_data']):.2f} BPM")
        print(f"→ PPG Average HR: {np.mean(ppg['hr_data']):.2f} BPM")

    if ecg['sdnn'] is not None and ppg['sdnn'] is not None:
        print(f"→ ECG SDNN: {ecg['sdnn']:.2f} ms")
        print(f"→ PPG SDNN: {ppg['sdnn']:.2f} ms")

    if ecg['rmssd'] is not None and ppg['rmssd'] is not None:
        print(f"→ ECG RMSSD: {ecg['rmssd']:.2f} ms")
        print(f"→ PPG RMSSD: {ppg['rmssd']:.2f} ms")

    # --- Cross-correlation analysis ---
    print("\n=== Cross-Correlation Analysis ===")
    ecg_filtered = ecg['filtered'][:2000]
    ppg_filtered = ppg['filtered'][:2000]

    min_len = min(len(ecg_filtered), len(ppg_filtered))
    ecg_cut = ecg_filtered[:min_len]
    ppg_cut = ppg_filtered[:min_len]

    cross_corr = correlate(ppg_cut, ecg_cut, mode='full')
    lags = np.arange(-min_len + 1, min_len)
    best_lag = lags[np.argmax(cross_corr)]
    similarity = np.max(cross_corr)

    print(f"→ Max Cross-Correlation: {similarity:.2f}")
    print(f"→ Optimal Lag (samples): {best_lag}")
    print(f"→ Time Delay: {best_lag / ecg['fs']:.3f} seconds")

# ----------------------------------
# Run Main
# ----------------------------------
if __name__ == "__main__":
    compare_signals_with_crosscorr()
