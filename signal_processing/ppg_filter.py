from scipy.signal import butter, sosfiltfilt
import numpy as np


def bandpass_iir(x: np.ndarray, fs: float,
                 low_hz: float = 0.5,
                 high_hz: float = 7.0,
                 order: int = 4) -> np.ndarray:
    """PPG band-pass filter. Default: 0.5 ~ 7 Hz, order 4."""
    nyq = 0.5 * fs
    if high_hz >= nyq:
        raise ValueError(f"PPG high_hz={high_hz} must be lower than Nyquist={nyq:.3f} Hz")
    if low_hz <= 0:
        raise ValueError(f"PPG low_hz={low_hz} must be greater than 0")
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def filter_ppg(ppg: np.ndarray, fs: float,
               low_hz: float = 0.5,
               high_hz: float = 7.0,
               order: int = 4) -> np.ndarray:
    return bandpass_iir(ppg, fs, low_hz, high_hz, order)