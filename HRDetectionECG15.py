import wfdb  # For reading physiological waveform data files
import numpy as np  # For numerical operations and array handling
import matplotlib.pyplot as plt  # For plotting and visualization
from scipy.signal import butter, filtfilt, find_peaks  # For signal filtering and peak detection

# --- Configuration ---
DATA_PATH = "C:/Users/bhavn/Downloads/Physionet/bidmc29"  # Path to the physiological data record
ECG_CHANNEL_NAMES = ["ECG", "II", "MLII", "V", "I"]  # Common ECG channel names
FILTER_RANGE = (5.0, 15.0)  # Bandpass filter frequency range (Hz) for ECG

def load_physio_record(path):
    record = wfdb.rdrecord(path)
    channels = [ch.strip().upper().replace(",", "") for ch in record.sig_name]
    return record, channels

def create_bandpass(low, high, fs, order=2):
    nyquist = 0.5 * fs
    low_norm = low / nyquist
    high_norm = high / nyquist
    return butter(order, [low_norm, high_norm], btype='band')

def filter_signal(signal, filter_params, fs):
    b, a = create_bandpass(*filter_params, fs)
    return filtfilt(b, a, signal)

def compute_hr_and_hrv(peaks, fs):
    rr_intervals_sec = np.diff(peaks) / fs
    rr_intervals_ms = rr_intervals_sec * 1000

    if len(rr_intervals_ms) < 2:
        return None, None, None, None

    hr_matrix = 60 / rr_intervals_sec
    hrv_sdnn = np.std(rr_intervals_ms)
    successive_diffs = np.diff(rr_intervals_ms)
    hrv_rmssd = np.sqrt(successive_diffs ** 2)

    return hr_matrix, rr_intervals_ms, hrv_sdnn, hrv_rmssd

if __name__ == "__main__":
    physio_record, cleaned_channels = load_physio_record(DATA_PATH)
    print(f"Available channels: {cleaned_channels}")

    try:
        ecg_index = next(i for i, ch in enumerate(cleaned_channels) if ch in ECG_CHANNEL_NAMES)
    except StopIteration:
        raise RuntimeError("No valid ECG channel found in recording")

    raw_ecg = physio_record.p_signal[:, ecg_index]
    time = np.arange(len(raw_ecg)) / physio_record.fs

    filtered_ecg = filter_signal(raw_ecg, FILTER_RANGE, physio_record.fs)

    # Peak detection (R-peaks)
    signal_std = np.std(filtered_ecg)
    signal_mean = np.mean(filtered_ecg)

    min_distance = int(physio_record.fs * 0.3)  # ~200 BPM
    min_prominence = signal_std * 0.6
    min_height = signal_mean + signal_std * 0.5

    peaks, _ = find_peaks(filtered_ecg, distance=min_distance, prominence=min_prominence, height=min_height)

    num_cycles = 5
    if len(peaks) < num_cycles + 1:
        raise RuntimeError("Not enough detected R-peaks to extract 5 cycles")

    start_idx = peaks[0]
    end_idx = peaks[num_cycles]
    time_segment = time[start_idx:end_idx]
    raw_segment = raw_ecg[start_idx:end_idx]
    filtered_segment = filtered_ecg[start_idx:end_idx]
    peaks_segment = peaks[(peaks >= start_idx) & (peaks < end_idx)] - start_idx

    # Heart rate
    if len(peaks_segment) >= 2:
        beat_intervals = np.diff(peaks[0:num_cycles + 1]) / physio_record.fs
        avg_hr = 60 / np.mean(beat_intervals)
        hr_text = f"Average HR over 5 cycles: {avg_hr:.1f} BPM"
    else:
        hr_text = "Insufficient peaks for HR calculation"

    print(hr_text)

    # HRV metrics
    hr_matrix, rr_matrix, hrv_sdnn, hrv_rmssd_per_beat = compute_hr_and_hrv(peaks[:num_cycles + 1], physio_record.fs)

    if hr_matrix is not None:
        print("\n--- Heart Rate Matrix (BPM) ---")
        print(np.round(hr_matrix, 2))

        print("\n--- RR Intervals Matrix (ms) ---")
        print(np.round(rr_matrix, 2))

        print("\n--- HRV (RMSSD) per beat (ms) ---")
        print(np.round(hrv_rmssd_per_beat, 2))

    # --- Plot 1: Entire ECG ---
    plt.figure(figsize=(14, 4))
    plt.plot(time, raw_ecg, color='#999999', alpha=0.4, label='Raw ECG')
    plt.plot(time, filtered_ecg, color='blue', linewidth=1, label='Filtered ECG')
    plt.scatter(time[peaks], filtered_ecg[peaks], color='red', s=15, label='R-peaks')
    plt.title("ECG Signal: Entire Recording")
    plt.xlabel("Time (seconds)")
    plt.ylabel("ECG Amplitude")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # --- Plot 2: Zoomed Segment (5 Cycles) ---
    plt.figure(figsize=(14, 5))
    plt.plot(time_segment, raw_segment, color='#999999', alpha=0.4, label='Raw ECG')
    plt.plot(time_segment, filtered_segment, color='blue', linewidth=1.5, label='Filtered ECG')
    plt.scatter(time_segment[peaks_segment], filtered_segment[peaks_segment], color='red', s=35, label='R-peaks')

    y_arrow = max(filtered_segment) + 0.2
    for i in range(1, len(peaks_segment)):
        idx1, idx2 = peaks_segment[i - 1], peaks_segment[i]
        t1, t2 = time_segment[idx1], time_segment[idx2]
        rr_interval = t2 - t1
        mid = (t1 + t2) / 2

        plt.annotate("",
                     xy=(t2, y_arrow), xytext=(t1, y_arrow),
                     arrowprops=dict(arrowstyle="<->", color='green', lw=1.5))

        plt.text(mid, y_arrow + 0.05, f"{rr_interval:.2f}s",
                 ha='center', va='bottom', fontsize=9, color='green')

    plt.title("ECG Signal: 5 Cardiac Cycles with RR Intervals", fontsize=13)
    plt.xlabel("Time (seconds)")
    plt.ylabel("ECG Amplitude")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.text(0.98, 0.95, hr_text, transform=plt.gca().transAxes,
             fontsize=12, color='green', ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='green'))
    plt.show()

    # --- Plot 3: RR Interval Bars ---
    rr_intervals_zoom = np.diff(time_segment[peaks_segment])

    plt.figure(figsize=(10, 5))
    bar_positions = np.arange(len(rr_intervals_zoom))

    for i, interval in enumerate(rr_intervals_zoom):
        plt.vlines(x=bar_positions[i], ymin=0, ymax=interval, color='teal', linewidth=3)
        plt.text(bar_positions[i], interval + 0.01, f"{interval:.2f}s", ha='center', color='teal', fontsize=9)

    plt.title("RR Intervals from 5 R-R Cycles (Zoomed Segment)", fontsize=13)
    plt.xlabel("Interval Index (between peaks)")
    plt.ylabel("Duration (seconds)")
    plt.ylim(0, max(rr_intervals_zoom) + 0.1)
    plt.xticks(bar_positions, [f"{i+1}" for i in bar_positions])
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
