import numpy as np


def compute_vpg_apg(ppg: np.ndarray, fs: float):
    vpg = np.gradient(ppg) * fs
    apg = np.gradient(vpg) * fs
    return vpg, apg


def detect_ppg_foots(
    ppg_filtered: np.ndarray,
    r_idx: np.ndarray,
    fs: float,
    foot_search_min_sec: float = 0.10,
    foot_search_max_sec: float = 0.45,
    foot_left_search_sec: float = 0.06,
    foot_min_slope: float = 1e-6,
    min_foots_required: int = 5,
):
    """PPG foot detection using VPG max, preceding minimum, and tangent intersection."""
    if len(ppg_filtered) == 0:
        return {
            "foot_idx": np.array([], dtype=int),
            "foot_time": np.array([]),
            "matched_r_idx": np.array([], dtype=int),
            "matched_r_time": np.array([]),
            "vpg": np.array([]),
            "apg": np.array([]),
            "valid": False,
            "reason": "Empty PPG signal",
        }

    vpg, apg = compute_vpg_apg(ppg_filtered, fs)
    foot_list = []
    matched_r_list = []

    for r in r_idx:
        start = r + int(foot_search_min_sec * fs)
        end = min(r + int(foot_search_max_sec * fs), len(ppg_filtered) - 1)
        if start >= len(ppg_filtered) or start >= end:
            continue

        local_vpg = vpg[start:end + 1]
        if len(local_vpg) < 3:
            continue

        t_s = start + np.argmax(local_vpg)
        left_start = max(r, t_s - int(foot_left_search_sec * fs))
        left_end = t_s
        if left_start >= left_end:
            continue

        local_ppg = ppg_filtered[left_start:left_end + 1]
        t_min = left_start + np.argmin(local_ppg)

        slope = vpg[t_s]  # amplitude / second
        if abs(slope) < foot_min_slope:
            t_foot = t_min
        else:
            dt_sec = (ppg_filtered[t_s] - ppg_filtered[t_min]) / slope
            t_foot = int(round(t_s - dt_sec * fs))
            t_foot = max(t_min, min(t_s, t_foot))

        foot_list.append(t_foot)
        matched_r_list.append(r)

    foot_idx = np.array(foot_list, dtype=int)
    matched_r_idx = np.array(matched_r_list, dtype=int)
    foot_time = foot_idx / fs
    matched_r_time = matched_r_idx / fs

    valid = len(foot_idx) >= min_foots_required
    reason = "OK" if valid else "Too few PPG foots detected"

    return {
        "foot_idx": foot_idx,
        "foot_time": foot_time,
        "matched_r_idx": matched_r_idx,
        "matched_r_time": matched_r_time,
        "vpg": vpg,
        "apg": apg,
        "valid": valid,
        "reason": reason,
    }
