import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def extract_and_plot_spectral_features(time_series: np.ndarray, sampling_rate: float):
    """
    Extracts time-frequency features and plots signal analysis graphics.
    """
    # 1. Compute Power Spectral Density via Welch's method
    freqs, psd = signal.welch(time_series, fs=sampling_rate, nperseg=1024)
    
    # 2. Spectral Centroid
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
    
    # 3. Peak Frequencies
    peaks, _ = signal.find_peaks(psd, height=np.max(psd)*0.1)
    dominant_peak_freq = freqs[peaks[0]] if len(peaks) > 0 else 0.0
    
    # 4. Diagnostic Frequency Band Powers
    mask_low = (freqs >= 1e3) & (freqs <= 10e3)
    mask_high = (freqs > 10e3) & (freqs <= 100e3)
    
    features = {
        "rms_amplitude": np.sqrt(np.mean(np.square(time_series))),
        "spectral_centroid_hz": spectral_centroid,
        "dominant_peak_hz": dominant_peak_freq,
        "power_ratio_high_low": np.sum(psd[mask_high]) / (np.sum(psd[mask_low]) + 1e-12)
    }

    # --- GRAPHICAL VISUALIZATION ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    
    # Time-Domain Plot
    time_ms = np.linspace(0, len(time_series) / sampling_rate * 1000, len(time_series))
    axes[0].plot(time_ms[:1000], time_series[:1000], color='#1f77b4', lw=1.2, label='Raw Sensor Signal')
    axes[0].set_title("Raw Sensor Signal (Time Domain)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude (V)")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='upper right')

    # Frequency-Domain PSD Plot
    axes[1].semilogy(freqs / 1e3, psd, color='#2ca02c', lw=1.5, label='Power Spectral Density (PSD)')
    axes[1].axvline(spectral_centroid / 1e3, color='red', linestyle='--', lw=1.5, 
                    label=f'Spectral Centroid ({spectral_centroid/1e3:.1f} kHz)')
    
    if len(peaks) > 0:
        axes[1].plot(freqs[peaks] / 1e3, psd[peaks], "x", color='darkorange', markersize=8, markeredgewidth=2, label='Detected Peaks')

    # Highlight diagnostic bands
    axes[1].fill_between(freqs / 1e3, 0, psd, where=mask_low, color='blue', alpha=0.15, label='Band Low (1–10 kHz)')
    axes[1].fill_between(freqs / 1e3, 0, psd, where=mask_high, color='green', alpha=0.15, label='Band High (10–100 kHz)')

    axes[1].set_title("Spectral Analysis & Feature Extraction (Frequency Domain)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Frequency (kHz)")
    axes[1].set_ylabel("Power / Frequency (dB/Hz)")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    
    return features

# Execute Demo
fs = 1e6  # 1 MHz sampling
t = np.linspace(0, 0.01, int(fs * 0.01))
raw_signal = np.sin(2 * np.pi * 20000 * t) + 0.5 * np.random.normal(size=len(t))

features = extract_and_plot_spectral_features(raw_signal, fs)