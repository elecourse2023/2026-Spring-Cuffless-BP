import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "r_index", "foot_index",
    "r_time_sec", "foot_time_sec",
    "pep_sec", "pep_ms",
    "pat_sec", "pat_ms",
    "ptt_sec", "ptt_ms",
    "rr_interval_sec", "rr_interval_ms",
]


def build_rr_map(r_idx: np.ndarray, time_array: np.ndarray):
    rr_map = {}
    if len(r_idx) == 0:
        return rr_map

    r_time = time_array[r_idx]
    for i, r in enumerate(r_idx):
        rr_map[int(r)] = np.nan if i == 0 else float(r_time[i] - r_time[i - 1])
    return rr_map


def calculate_pat_ptt_fixed_pep(
    time_array: np.ndarray,
    foot_result: dict,
    rr_map: dict,
    pep_ms: float = 70.0,
) -> pd.DataFrame:
    """Build beat-level feature table using fixed PEP.

    PEP is a user-configurable constant in milliseconds.
    PAT = foot - R
    PTT = PAT - PEP
    """
    foot_map = {int(r): int(f) for r, f in zip(foot_result["matched_r_idx"], foot_result["foot_idx"])}
    common_r = sorted(foot_map.keys())
    rows = []

    pep_sec = pep_ms / 1000.0

    for r_idx in common_r:
        f_idx = foot_map[r_idx]

        r_time = float(time_array[r_idx])
        foot_time = float(time_array[f_idx])

        pat_sec = foot_time - r_time
        ptt_sec = pat_sec - pep_sec
        rr_sec = rr_map.get(int(r_idx), np.nan)

        rows.append({
            "r_index": int(r_idx),
            "foot_index": int(f_idx),
            "r_time_sec": r_time,
            "foot_time_sec": foot_time,
            "pep_sec": float(pep_sec),
            "pep_ms": float(pep_ms),
            "pat_sec": float(pat_sec),
            "pat_ms": float(pat_sec * 1000.0),
            "ptt_sec": float(ptt_sec),
            "ptt_ms": float(ptt_sec * 1000.0),
            "rr_interval_sec": float(rr_sec) if np.isfinite(rr_sec) else np.nan,
            "rr_interval_ms": float(rr_sec * 1000.0) if np.isfinite(rr_sec) else np.nan,
        })

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
