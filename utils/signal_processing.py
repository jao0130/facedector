"""
Signal processing utilities for rPPG.
Ported from rPPG-Toolbox evaluation/post_process.py.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.sparse import spdiags


def detrend(signal: np.ndarray, lambda_value: float = 100.0) -> np.ndarray:
    """
    Second-order detrending (removes slow baseline drift).
    Uses sparse matrix approach from Tarvainen et al. 2002.
    """
    T = len(signal)
    I = np.eye(T)
    ones = np.ones(T)
    D2 = spdiags([ones, -2 * ones, ones], [0, 1, 2], T - 2, T).toarray()
    detrended = (I + lambda_value ** 2 * D2.T @ D2)
    return np.linalg.solve(detrended, signal)


def bandpass_filter(signal: np.ndarray, fs: float,
                    low: float = 0.75, high: float = 2.5,
                    order: int = 6) -> np.ndarray:
    """Butterworth bandpass filter."""
    nyq = fs / 2.0
    low_norm = low / nyq
    high_norm = high / nyq
    low_norm = max(low_norm, 0.001)
    high_norm = min(high_norm, 0.999)
    b, a = butter(order, [low_norm, high_norm], btype='band')
    return filtfilt(b, a, signal)


def estimate_hr_fft(signal: np.ndarray, fs: float,
                    freq_low: float = 0.75, freq_high: float = 2.5) -> float:
    """
    FFT-based heart rate estimation.

    Args:
        signal: 1D rPPG signal.
        fs: sampling frequency.
        freq_low: min frequency (Hz).
        freq_high: max frequency (Hz).

    Returns:
        Estimated heart rate in BPM.
    """
    N = len(signal)
    fft_vals = np.fft.rfft(signal)
    fft_freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # Mask to cardiac frequency band
    mask = (fft_freqs >= freq_low) & (fft_freqs <= freq_high)
    if not np.any(mask):
        return 0.0

    magnitudes = np.abs(fft_vals)
    magnitudes_masked = magnitudes * mask

    peak_idx = np.argmax(magnitudes_masked)
    peak_freq = fft_freqs[peak_idx]

    return peak_freq * 60.0  # Hz -> BPM


def compute_snr(pred_signal: np.ndarray, gt_hr_bpm: float, fs: float,
                hr_tolerance_bpm: float = 6.0) -> float:
    """
    Signal-to-Noise Ratio: power at HR ± tolerance vs rest of band.
    """
    N = len(pred_signal)
    fft_vals = np.fft.rfft(pred_signal)
    fft_freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    power = np.abs(fft_vals) ** 2

    gt_hr_hz = gt_hr_bpm / 60.0
    tolerance_hz = hr_tolerance_bpm / 60.0

    # Signal: first and second harmonics within tolerance
    signal_mask = np.zeros(len(fft_freqs), dtype=bool)
    for harmonic in [1, 2]:
        center = gt_hr_hz * harmonic
        signal_mask |= (fft_freqs >= center - tolerance_hz) & (fft_freqs <= center + tolerance_hz)

    # Noise: rest of 0.75-2.5 Hz band
    band_mask = (fft_freqs >= 0.75) & (fft_freqs <= 2.5)
    noise_mask = band_mask & ~signal_mask

    signal_power = np.sum(power[signal_mask]) + 1e-10
    noise_power = np.sum(power[noise_mask]) + 1e-10

    return 10 * np.log10(signal_power / noise_power)


def diff_normalize(frames: np.ndarray) -> np.ndarray:
    """
    DiffNormalized preprocessing: frame differences normalized by intensity.

    Args:
        frames: [T, H, W, C] uint8 or float (0-255 range).

    Returns:
        [T, H, W, C] float32 diff-normalized frames.
    """
    frames = frames.astype(np.float32)
    T = frames.shape[0]
    result = np.zeros_like(frames)
    for t in range(1, T):
        denom = frames[t] + 1e-7
        result[t] = (frames[t] - frames[t - 1]) / denom
    return result
