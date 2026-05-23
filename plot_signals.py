import numpy as np
import matplotlib.pyplot as plt

# 1. Generate dummy time vector
t = np.linspace(0, 10, 1000)

# 2. Simulate an ECG signal (simplified peak pattern) and a lagging PPG wave
ecg = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 2 * t)
ppg = np.sin(2 * np.pi * 1 * (t - 0.2)) + 0.5 * np.sin(2 * np.pi * 2 * (t - 0.2)) # 0.2s lag

# 3. Create the plot
plt.figure(figsize=(10, 5))
plt.plot(t[:300], ecg[:300], label='Raw ECG (Source)', color='crimson', linewidth=2)
plt.plot(t[:300], ppg[:300], label='Lagging PPG (Target)', color='navy', linewidth=2, linestyle='--')

plt.title('Biomedical Waveform Alignment: ECG vs. PPG')
plt.xlabel('Time (seconds)')
plt.ylabel('Normalized Amplitude')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# 4. Save the plot to the folder
plt.savefig('signal_alignment_chart.png')
print("📊 Success! Chart saved as 'signal_alignment_chart.png'")