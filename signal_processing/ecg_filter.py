from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos
import numpy as np


# ============================================================
# ECG 사용자 설정
# ============================================================
# 단방향 ADC를 위한 DC offset (단위: V)
ECG_DC_OFFSET_V = 1.65

# 다단(cascade) 필터 파라미터
ECG_NOTCH_FREQ_HZ = 60.0      # 전원 노이즈 (한국 60 Hz). 50 Hz 환경이면 50으로.
ECG_NOTCH_Q = 30.0            # Notch quality factor (좁을수록 ↑)
ECG_HPF_HZ = 0.5              # baseline wander 차단 (low edge)
ECG_LPF_HZ = 40.0             # muscle/EMG 억제 (high edge)
ECG_BAND_ORDER = 3            # Butterworth 차수


def _notch_sos(freq_hz: float, Q: float, fs: float):
    """60 Hz 같은 단일 주파수 제거용 IIR notch (SOS 형태)."""
    b, a = iirnotch(w0=freq_hz, Q=Q, fs=fs)
    return tf2sos(b, a)


def _bandpass_sos(low_hz: float, high_hz: float, order: int, fs: float):
    nyq = 0.5 * fs
    if high_hz >= nyq:
        raise ValueError(
            f"ECG highcut={high_hz} must be lower than Nyquist={nyq:.3f} Hz"
        )
    return butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")


def filter_ecg(
    ecg: np.ndarray,
    fs: float,
    dc_offset_v: float = ECG_DC_OFFSET_V,
    notch_freq_hz: float = ECG_NOTCH_FREQ_HZ,
    notch_q: float = ECG_NOTCH_Q,
    hpf_hz: float = ECG_HPF_HZ,
    lpf_hz: float = ECG_LPF_HZ,
    order: int = ECG_BAND_ORDER,
) -> np.ndarray:
    """ECG 다단 필터:
    1) DC offset 1.65 V 제거
    2) Notch 60 Hz (전원 노이즈)
    3) Bandpass 0.5 ~ 40 Hz (baseline + muscle artifact)
    """
    x = ecg.astype(float) - dc_offset_v

    # 단 1: Notch 60 Hz
    sos_notch = _notch_sos(notch_freq_hz, notch_q, fs)
    x = sosfiltfilt(sos_notch, x)

    # 단 2: Bandpass 0.5 ~ 40 Hz
    sos_bp = _bandpass_sos(hpf_hz, lpf_hz, order, fs)
    x = sosfiltfilt(sos_bp, x)

    return x


# 외부 호환성 (다른 모듈에서 직접 import 할 가능성 대비)
def bandpass_iir(
    x: np.ndarray,
    fs: float,
    lowcut_hz: float = 0.5,
    highcut_hz: float = 40.0,
    order: int = 3,
) -> np.ndarray:
    return sosfiltfilt(_bandpass_sos(lowcut_hz, highcut_hz, order, fs), x)