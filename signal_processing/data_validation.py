import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "pep_ms", "pat_ms", "ptt_ms", "rr_interval_sec", "rr_interval_ms"
]


def validate_feature_table(
    df: pd.DataFrame,
    pep_min_ms: float = 20.0,
    pep_max_ms: float = 180.0,
    #
    pat_min_ms: float = 150.0, ##수정 : 100 -> 150
    pat_max_ms: float = 450.0, 
    #
    ptt_min_ms: float = 80.0, ##수정 : 50 -> 80
    ptt_max_ms: float = 380.0, ##수정 : 400 -> 380
    # ▼ 완화: 40 → 60 ms (PAT/PTT jump). PEP는 그대로.
    pep_jump_ms: float = 30.0,
    pat_jump_ms: float = 100.0, ##수정 : 60 -> 100
    ptt_jump_ms: float = 100.0,  ##수정 : 60 -> 100
    # ▼ 완화: 0.20 → 0.30 (RR ±30%)
    rr_outlier_ratio: float = 0.30,
    # ▼ 신규: dev 임계도 외부에서 조정 가능하게 노출
    pep_dev_ratio: float = 0.20,
    pep_dev_floor_ms: float = 20.0,
    pat_dev_ratio: float = 0.15,        # 0.15 → 0.20 -> 0.15
    pat_dev_floor_ms: float = 30.0,
    ptt_dev_ratio: float = 0.20,        # 0.15 → 0.20
    ptt_dev_floor_ms: float = 30.0,
) -> pd.DataFrame:
    out = df.copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"feature table missing required columns: {missing}")

    if out.empty:
        out["valid_pep_range"] = pd.Series(dtype=bool)
        out["valid_pat_range"] = pd.Series(dtype=bool)
        out["valid_ptt_range"] = pd.Series(dtype=bool)
        out["valid_pep_dev"] = pd.Series(dtype=bool)
        out["valid_pat_dev"] = pd.Series(dtype=bool)
        out["valid_ptt_dev"] = pd.Series(dtype=bool)
        out["pep_diff_ms"] = pd.Series(dtype=float)
        out["pat_diff_ms"] = pd.Series(dtype=float)
        out["ptt_diff_ms"] = pd.Series(dtype=float)
        out["valid_pep_jump"] = pd.Series(dtype=bool)
        out["valid_pat_jump"] = pd.Series(dtype=bool)
        out["valid_ptt_jump"] = pd.Series(dtype=bool)
        out["valid_rr"] = pd.Series(dtype=bool)
        out["is_valid"] = pd.Series(dtype=bool)
        return out

    out["valid_pep_range"] = (out["pep_ms"] >= pep_min_ms) & (out["pep_ms"] <= pep_max_ms)
    out["valid_pat_range"] = (out["pat_ms"] >= pat_min_ms) & (out["pat_ms"] <= pat_max_ms)
    out["valid_ptt_range"] = (out["ptt_ms"] >= ptt_min_ms) & (out["ptt_ms"] <= ptt_max_ms)

    pep_valid = out[out["valid_pep_range"]]
    if len(pep_valid) > 0:
        pep_med = pep_valid["pep_ms"].median()
        pep_dev_thr = max(pep_dev_floor_ms, pep_dev_ratio * pep_med)
        out["valid_pep_dev"] = np.abs(out["pep_ms"] - pep_med) <= pep_dev_thr
    else:
        out["valid_pep_dev"] = False

    pat_valid = out[out["valid_pat_range"]]
    if len(pat_valid) > 0:
        pat_med = pat_valid["pat_ms"].median()
        pat_dev_thr = max(pat_dev_floor_ms, pat_dev_ratio * pat_med)
        out["valid_pat_dev"] = np.abs(out["pat_ms"] - pat_med) <= pat_dev_thr
    else:
        out["valid_pat_dev"] = False

    ptt_valid = out[out["valid_ptt_range"]]
    if len(ptt_valid) > 0:
        ptt_med = ptt_valid["ptt_ms"].median()
        ptt_dev_thr = max(ptt_dev_floor_ms, ptt_dev_ratio * ptt_med)
        out["valid_ptt_dev"] = np.abs(out["ptt_ms"] - ptt_med) <= ptt_dev_thr
    else:
        out["valid_ptt_dev"] = False

    out["pep_diff_ms"] = out["pep_ms"].diff().abs()
    out["pat_diff_ms"] = out["pat_ms"].diff().abs()
    out["ptt_diff_ms"] = out["ptt_ms"].diff().abs()

    out["valid_pep_jump"] = out["pep_diff_ms"].fillna(0.0) <= pep_jump_ms
    out["valid_pat_jump"] = out["pat_diff_ms"].fillna(0.0) <= pat_jump_ms
    out["valid_ptt_jump"] = out["ptt_diff_ms"].fillna(0.0) <= ptt_jump_ms

    rr_valid = out["rr_interval_sec"].dropna()
    if len(rr_valid) > 0:
        rr_med = rr_valid.median()
        rr_low = (1.0 - rr_outlier_ratio) * rr_med
        rr_high = (1.0 + rr_outlier_ratio) * rr_med
        out["valid_rr"] = out["rr_interval_sec"].isna() | ((out["rr_interval_sec"] >= rr_low) & (out["rr_interval_sec"] <= rr_high))
    else:
        out["valid_rr"] = True

    out["is_valid"] = (
        out["valid_pep_range"] &
        out["valid_pat_range"] &
        out["valid_ptt_range"] &
        out["valid_pep_dev"] &
        out["valid_pat_dev"] &
        out["valid_ptt_dev"] &
        out["valid_pep_jump"] &
        out["valid_pat_jump"] &
        out["valid_ptt_jump"] &
        out["valid_rr"]
    )
    return out


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{
            "num_total_beats": 0,
            "num_valid_beats": 0,
            "rejection_rate_percent": np.nan,
            "mean_pep_ms": np.nan,
            "median_pep_ms": np.nan,
            "std_pep_ms": np.nan,
            "mean_pat_ms": np.nan,
            "median_pat_ms": np.nan,
            "std_pat_ms": np.nan,
            "mean_ptt_ms": np.nan,
            "median_ptt_ms": np.nan,
            "std_ptt_ms": np.nan,
            "mean_rr_ms": np.nan,
            "median_rr_ms": np.nan,
        }])

    valid = df[df["is_valid"]]
    return pd.DataFrame([{
        "num_total_beats": len(df),
        "num_valid_beats": len(valid),
        "rejection_rate_percent": (1.0 - len(valid) / len(df)) * 100.0 if len(df) > 0 else np.nan,
        "mean_pep_ms": valid["pep_ms"].mean() if len(valid) > 0 else np.nan,
        "median_pep_ms": valid["pep_ms"].median() if len(valid) > 0 else np.nan,
        "std_pep_ms": valid["pep_ms"].std(ddof=1) if len(valid) > 1 else np.nan,
        "mean_pat_ms": valid["pat_ms"].mean() if len(valid) > 0 else np.nan,
        "median_pat_ms": valid["pat_ms"].median() if len(valid) > 0 else np.nan,
        "std_pat_ms": valid["pat_ms"].std(ddof=1) if len(valid) > 1 else np.nan,
        "mean_ptt_ms": valid["ptt_ms"].mean() if len(valid) > 0 else np.nan,
        "median_ptt_ms": valid["ptt_ms"].median() if len(valid) > 0 else np.nan,
        "std_ptt_ms": valid["ptt_ms"].std(ddof=1) if len(valid) > 1 else np.nan,
        "mean_rr_ms": valid["rr_interval_ms"].mean() if len(valid) > 0 else np.nan,
        "median_rr_ms": valid["rr_interval_ms"].median() if len(valid) > 0 else np.nan,
    }])
