import wfdb  # For reading physiological waveform data files
import numpy as np  # For numerical operations and array handling
import matplotlib.pyplot as plt  # For plotting and visualization
import math 
from scipy.signal import butter, filtfilt, find_peaks  # For signal filtering and peak detection

# --- Configuration ---
DATA_PATH = "C:/Users/bhavn/Downloads/Physionet/bidmc33"  # Path to the physiological data record
PPG_CHANNEL_NAMES = ["PPG", "PLETH"]  # Possible names for the PPG channel
FILTER_RANGE = (0.5, 5.0)  # Bandpass filter frequency range (Hz), typical for PPG

def load_physio_record(path):
    """Load a WFDB record and return the record and cleaned channel names."""
    record = wfdb.rdrecord(path)  # Read the record using WFDB
    channels = [ch.strip().upper().replace(",", "") for ch in record.sig_name]
    return record, channels

def create_bandpass(low, high, fs, order=2):
    """Design a Butterworth bandpass filter for the given frequency range."""
    nyquist = 0.5 * fs
    low_norm = low / nyquist
    high_norm = high / nyquist
    return butter(order, [low_norm, high_norm], btype='band')

def filter_signal(signal, filter_params, fs):
    """Apply a zero-phase Butterworth bandpass filter to the signal."""
    b, a = create_bandpass(*filter_params, fs)
    return filtfilt(b, a, signal)

def compute_hr_and_hrv(peaks, fs):
    """Compute HR, RR intervals, SDNN, and RMSSD for each heartbeat."""
    rr_intervals_sec = np.diff(peaks) / fs  # RR intervals in seconds
    rr_intervals_ms = rr_intervals_sec * 1000  # Convert to ms

    if len(rr_intervals_ms) < 2:
        return None, None, None, None

    hr_matrix = 60 / rr_intervals_sec

    # SDNN = standard deviation of RR intervals
    hrv_sdnn = np.std(rr_intervals_ms)

    # RMSSD = root mean square of successive RR interval differences
    successive_diffs = np.diff(rr_intervals_ms)
    hrv_rmssd_per_beat = np.sqrt(successive_diffs ** 2)  # Instantaneous RMSSD (not windowed)

    return hr_matrix, rr_intervals_ms, hrv_sdnn, hrv_rmssd_per_beat


if __name__ == "__main__":
    # Load data and channels
    physio_record, cleaned_channels = load_physio_record(DATA_PATH)
    print(f"Available channels: {cleaned_channels}")

    # Find PPG channel index
    try:
        ppg_index = next(i for i, ch in enumerate(cleaned_channels) if ch in PPG_CHANNEL_NAMES)
    except StopIteration:
        raise RuntimeError("No valid PPG channel found in recording")

    # Extract raw PPG and time axis
    raw_ppg = physio_record.p_signal[:, ppg_index]
    time = np.arange(len(raw_ppg)) / physio_record.fs

    # Filter the PPG signal
    filtered_ppg = filter_signal(raw_ppg, FILTER_RANGE, physio_record.fs)

    # --- Adaptive peak detection ---
    signal_std = np.std(filtered_ppg)
    signal_mean = np.mean(filtered_ppg)

    min_distance = int(physio_record.fs * 0.4)  # 0.4s between beats = ~150 BPM
    min_prominence = signal_std * 0.5
    min_height = signal_mean + signal_std * 0.3

    peaks, properties = find_peaks(
        filtered_ppg,
        distance=min_distance,
        prominence=min_prominence,
        height=min_height
    )

    num_cycles = 5
    if len(peaks) < num_cycles + 1:
        raise RuntimeError("Not enough detected peaks to extract 5 cycles")


    start_idx = peaks[0]
    end_idx = peaks[num_cycles]

    # Slice signals and time for the selected segment
    time_segment = time[start_idx:end_idx]
    raw_segment = raw_ppg[start_idx:end_idx]
    filtered_segment = filtered_ppg[start_idx:end_idx]
    peaks_segment = peaks[(peaks >= start_idx) & (peaks < end_idx)] - start_idx

    # Calculate average heart rate
    if len(peaks_segment) >= 2:
        beat_intervals = np.diff(peaks[0:num_cycles + 1]) / physio_record.fs
        avg_hr = 60 / np.mean(beat_intervals)
        hr_text = f"Average HR over 5 cycles: {avg_hr:.1f} BPM"
    else:
        hr_text = "Insufficient peaks for HR calculation"

    print(hr_text)
     # --- HR and HRV Calculation ---
    hr_matrix, rr_matrix, hrv_sdnn, hrv_rmssd_per_beat = compute_hr_and_hrv(peaks[:num_cycles + 1], physio_record.fs)


    if hr_matrix is not None:
        avg_hr = np.mean(hr_matrix)
        hr_text = f"Average HR: {avg_hr:.1f} BPM"
        hrv_text = f"HRV (SDNN): {hrv_sdnn:.2f} ms"
    else:
        hr_text = "Insufficient data for HR"
        hrv_text = "Insufficient data for HRV"

    # --- Print HR and HRV matrices ---
    print("\n--- Heart Rate Matrix (BPM) ---")
    print(np.round(hr_matrix, 2))

    print("\n--- RR Intervals Matrix (ms) ---")
    print(np.round(rr_matrix, 2))

    print("\n--- HRV (RMSSD) per beat (ms) ---")
    if hrv_rmssd_per_beat is not None:
        print(np.round(hrv_rmssd_per_beat, 2))

    print("\n--- HR Summary ---")
    print(hr_text)
    print(hrv_text)

    # --- Plot 1: Entire signal ---
    plt.figure(figsize=(14, 4))
    plt.plot(time, raw_ppg, color="#E2E53C", alpha=0.4, label='Raw PPG (Noisy)')
    plt.plot(time, filtered_ppg, color='blue', linewidth=1, label='Filtered PPG')
    plt.scatter(time[peaks], filtered_ppg[peaks], color='red', s=15, label='Detected Heartbeats')
    plt.title("PPG Signal: Entire Recording")
    plt.xlabel("Time (seconds)")
    plt.ylabel("PPG Amplitude")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

            # --- Plot 2: Zoomed-in segment (5 cycles) with RR intervals ---
plt.figure(figsize=(14, 5))
plt.plot(time_segment, raw_segment, color='#999999', alpha=0.4, label='Raw PPG (Noisy)')
plt.plot(time_segment, filtered_segment, color='blue', linewidth=1.5, label='Filtered PPG')
plt.scatter(time_segment[peaks_segment], filtered_segment[peaks_segment], color='red', s=35, label='Detected Heartbeats')

# Draw RR intervals with arrows and labels
y_arrow = max(filtered_segment) + 0.2  # position arrows above the signal
for i in range(1, len(peaks_segment)):
    idx1 = peaks_segment[i - 1]
    idx2 = peaks_segment[i]
    t1 = time_segment[idx1]
    t2 = time_segment[idx2]
    rr_interval = t2 - t1
    mid = (t1 + t2) / 2

    # Horizontal arrow
    plt.annotate("",
                 xy=(t2, y_arrow), xytext=(t1, y_arrow),
                 arrowprops=dict(arrowstyle="<->", color='green', lw=1.5))

    # RR interval label
    plt.text(mid, y_arrow + 0.05, f"{rr_interval:.2f}s",
             ha='center', va='bottom', fontsize=9, color='green')

plt.title("PPG Signal: 5 Cardiac Cycles (Raw, Filtered, RR Intervals)", fontsize=13)
plt.xlabel("Time (seconds)")
plt.ylabel("PPG Amplitude")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.text(
    0.98, 0.95, hr_text, transform=plt.gca().transAxes,
    fontsize=12, color='green', ha='right', va='top',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='green')
)
plt.show()

# --- Plot 3: Vertical Bars of RR Intervals (Only for 5 Zoomed Cycles) ---

# Compute RR intervals (in seconds) for the zoomed 5 cycles
rr_intervals_zoom = np.diff(time_segment[peaks_segment])

plt.figure(figsize=(10, 5))
bar_positions = np.arange(len(rr_intervals_zoom))

# Plot vertical bars
for i, interval in enumerate(rr_intervals_zoom):
    plt.vlines(x=bar_positions[i], ymin=0, ymax=interval, color='teal', linewidth=3)
    plt.text(bar_positions[i], interval + 0.01, f"{interval:.2f}s", ha='center', color='teal', fontsize=9)

plt.title("RR Intervals from 5 Cardiac Cycles (Zoomed Segment)", fontsize=13)
plt.xlabel("Interval Index (between peaks)")
plt.ylabel("Duration (seconds)")
plt.ylim(0, max(rr_intervals_zoom) + 0.1)
plt.xticks(bar_positions, [f"{i+1}" for i in bar_positions])
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
