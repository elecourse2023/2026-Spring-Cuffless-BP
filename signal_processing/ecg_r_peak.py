import numpy as np
from scipy.signal import find_peaks


def detect_rpeaks(
    ecg_filtered: np.ndarray,
    fs: float,
    r_mwi_window_sec: float = 0.12,
    r_min_distance_sec: float = 0.30,
    r_search_radius_sec: float = 0.08,
    r_threshold_scale: float = 0.3,
    min_rpeaks_required: int = 5,
):
    """ECG R-peak detection using derivative, square, MWI, and local-max refinement."""
    if len(ecg_filtered) == 0:
        return {
            "r_idx": np.array([], dtype=int),
            "r_time": np.array([]),
            "rr_interval": np.array([]),
            "qrs_energy": np.array([]),
            "valid": False,
            "reason": "Empty ECG signal",
        }

    diff_sig = np.diff(ecg_filtered, prepend=ecg_filtered[0])
    sq = diff_sig ** 2

    win = max(1, int(r_mwi_window_sec * fs))
    qrs_energy = np.convolve(sq, np.ones(win) / win, mode="same")

    distance = max(1, int(r_min_distance_sec * fs))
    threshold = np.mean(qrs_energy) + r_threshold_scale * np.std(qrs_energy)
    candidate_idx, _ = find_peaks(qrs_energy, distance=distance, height=threshold)

    refined_idx = []
    search_radius = max(1, int(r_search_radius_sec * fs))
    for c in candidate_idx:
        s = max(0, c - search_radius)
        e = min(len(ecg_filtered), c + search_radius + 1)
        refined_idx.append(s + np.argmax(ecg_filtered[s:e]))

    refined_idx = np.array(sorted(set(refined_idx)), dtype=int)
    r_time = refined_idx / fs
    rr_interval = np.diff(r_time) if len(r_time) >= 2 else np.array([])

    valid = len(refined_idx) >= min_rpeaks_required
    reason = "OK" if valid else "Too few R-peaks detected"

    return {
        "r_idx": refined_idx,
        "r_time": r_time,
        "rr_interval": rr_interval,
        "qrs_energy": qrs_energy,
        "valid": valid,
        "reason": reason,
    }
