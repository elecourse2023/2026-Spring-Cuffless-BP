from pathlib import Path
import sys

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ecg_filter import filter_ecg
from ppg_filter import filter_ppg
from ecg_r_peak import detect_rpeaks
from ppg_foot import detect_ppg_foots
from ptt_pat_calculation import build_rr_map, calculate_pat_ptt_fixed_pep
from data_validation import validate_feature_table, summarize_results
from final_data_output import save_filtered_signals, save_detection_results, plot_filtered_overview, extract_final_output


# ============================================================
# User settings
# 이 숫자만 바꾸면 고정 PEP 값이 바뀝니다.
# 예: PEP = 80.0
# ============================================================
PEP = 70.0  # ms

# 하드웨어 팀 데이터 주기
# 13 ms -> fs ≈ 76.923 Hz
SAMPLE_PERIOD_MS = 4.0
# HW 팀에서 들어오는대로 주기 변경. 최대한 정수값으로.

# ============================================================
# 검증 임계 (strict / relaxed 토글)
# ============================================================
VALIDATION_PARAMS = dict(
    # 절대 범위 (생리적 한계 — 잘 안 만짐)
    pep_min_ms=20.0,  pep_max_ms=180.0,
    pat_min_ms=100.0, pat_max_ms=450.0,
    ptt_min_ms=50.0,  ptt_max_ms=400.0,
    # beat-to-beat jump (완화: 40 → 60)
    pep_jump_ms=30.0,
    pat_jump_ms=150.0,
    ptt_jump_ms=150.0,
    # 중앙값 편차 비율 (완화: 0.15 → 0.20)
    pep_dev_ratio=0.20, pep_dev_floor_ms=20.0,
    pat_dev_ratio=0.20, pat_dev_floor_ms=30.0,
    ptt_dev_ratio=0.20, ptt_dev_floor_ms=30.0,
    # RR 이상치 (완화: 0.20 → 0.30)
    rr_outlier_ratio=0.30,
)

# True면 time 컬럼이 있어도 SAMPLE_PERIOD_MS 기준으로 시간축을 다시 만듭니다.
# 하드웨어 팀 데이터가 "13 ms 고정 주기"라면 True 사용을 권장합니다.
USE_FIXED_SAMPLE_PERIOD = True

# False로 바꾸면 CSV의 time 컬럼을 그대로 사용합니다.
TIME_COLUMN = "time"


def estimate_fs_from_time(time: np.ndarray) -> float:
    if len(time) < 2:
        raise ValueError("time array length must be at least 2")
    dt = np.diff(time)
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError("time column must be strictly increasing and finite")
    return 1.0 / float(np.median(dt))


# ============================================================
# HW 포맷 호환 유틸 (SP팀 추가)
# - 새 HW 포맷: time_ms, sample_idx, ecg_v, ppg, lead_off
# - 기존 포맷: time, ecg, ppg
# ============================================================
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """새 HW 포맷(time_ms / ecg_v 등)을 기존 컬럼 이름으로 매핑."""
    df = df.copy()
    if "ecg" not in df.columns and "ecg_v" in df.columns:
        df["ecg"] = pd.to_numeric(df["ecg_v"], errors="coerce")
    if "time" not in df.columns and "time_ms" in df.columns:
        df["time"] = pd.to_numeric(df["time_ms"], errors="coerce") / 1000.0
    return df


def _split_segments_by_idx(df: pd.DataFrame,
                           gap_threshold: int = 10,
                           min_len: int = 200) -> list:
    """sample_idx 의 큰 갭(>gap_threshold)이 있으면 그 지점에서 세그먼트로 분할.
       sample_idx 컬럼이 없으면 단일 세그먼트로 반환.
       min_len 미만 세그먼트는 제외.
    """
    if "sample_idx" not in df.columns:
        return [df.reset_index(drop=True)]
    idx = pd.to_numeric(df["sample_idx"], errors="coerce").to_numpy()
    if not np.all(np.isfinite(idx)):
        return [df.reset_index(drop=True)]
    gaps = np.diff(idx.astype(np.int64))
    cuts = np.where(gaps > gap_threshold)[0] + 1
    if len(cuts) == 0:
        return [df.reset_index(drop=True)]
    segs, prev = [], 0
    for cp in cuts:
        segs.append(df.iloc[prev:cp].reset_index(drop=True))
        prev = cp
    segs.append(df.iloc[prev:].reset_index(drop=True))
    return [s for s in segs if len(s) >= min_len]


def _detect_sample_period_ms(df: pd.DataFrame, fallback_ms: float) -> float:
    """time 컬럼이 신뢰가능하면 거기서 dt(ms) 중앙값을 반환, 아니면 fallback."""
    if "time" in df.columns:
        t = pd.to_numeric(df["time"], errors="coerce").dropna().to_numpy()
        if len(t) >= 50:
            dt = float(np.median(np.diff(t)))
            if np.isfinite(dt) and dt > 0:
                return dt * 1000.0
    return fallback_ms


def build_time_and_fs(df: pd.DataFrame):
    if USE_FIXED_SAMPLE_PERIOD:
        dt_sec = SAMPLE_PERIOD_MS / 1000.0
        fs = 1.0 / dt_sec
        time = np.arange(len(df), dtype=float) * dt_sec

        if TIME_COLUMN in df.columns:
            raw_time = df[TIME_COLUMN].to_numpy(dtype=float)
            if len(raw_time) >= 2 and np.all(np.isfinite(raw_time)):
                raw_dt = np.median(np.diff(raw_time))
                if raw_dt > 0:
                    rel_err = abs(raw_dt - dt_sec) / dt_sec
                    if rel_err > 0.05:
                        print(
                            f"[알림] CSV time 컬럼 간격({raw_dt*1000:.3f} ms)과 "
                            f"설정값 SAMPLE_PERIOD_MS({SAMPLE_PERIOD_MS:.3f} ms)가 다릅니다. "
                            "현재는 SAMPLE_PERIOD_MS 기준 시간축을 사용합니다."
                        )
        return time, fs

    if TIME_COLUMN not in df.columns:
        raise ValueError(
            f"USE_FIXED_SAMPLE_PERIOD=False 인데 '{TIME_COLUMN}' 컬럼이 없습니다."
        )
    time = df[TIME_COLUMN].to_numpy(dtype=float)
    fs = estimate_fs_from_time(time)
    return time, fs


def _process_one_segment(seg_df: pd.DataFrame, stem: str, results_dir: Path,
                         orig_file_name: str) -> dict:
    """세그먼트 단위 처리. 기존 process_one_file 의 본체."""
    global SAMPLE_PERIOD_MS  # 자동 감지값을 모듈 상수에도 동기화

    # time 컬럼이 신뢰 가능하면 SAMPLE_PERIOD_MS 자동 감지
    detected_ms = _detect_sample_period_ms(seg_df, fallback_ms=SAMPLE_PERIOD_MS)
    if abs(detected_ms - SAMPLE_PERIOD_MS) / SAMPLE_PERIOD_MS > 0.05:
        print(f"[자동감지] SAMPLE_PERIOD_MS {SAMPLE_PERIOD_MS:.3f} -> {detected_ms:.3f} ms"
              f" (data 의 time dt 중앙값 기준)")
        SAMPLE_PERIOD_MS = float(detected_ms)

    time, fs = build_time_and_fs(seg_df)

    df_filtered = seg_df.copy()
    df_filtered["time"] = time
    df_filtered["ecg_filtered"] = filter_ecg(seg_df["ecg"].to_numpy(dtype=float), fs)
    df_filtered["ppg_filtered"] = filter_ppg(seg_df["ppg"].to_numpy(dtype=float), fs)

    r_result = detect_rpeaks(df_filtered["ecg_filtered"].to_numpy(dtype=float), fs)
    foot_result = detect_ppg_foots(
        df_filtered["ppg_filtered"].to_numpy(dtype=float),
        r_result["r_idx"], fs)

    rr_map = build_rr_map(r_result["r_idx"], time)
    feature_df = calculate_pat_ptt_fixed_pep(time, foot_result, rr_map, pep_ms=PEP)
    validated_df = validate_feature_table(feature_df)
    summary_df = summarize_results(validated_df)

    print(f"[설정] PEP={PEP:.1f} ms, SAMPLE_PERIOD_MS={SAMPLE_PERIOD_MS:.3f} ms, fs={fs:.4f} Hz")
    print(f"[R-peak] valid={r_result['valid']}, reason={r_result['reason']}, num={len(r_result['r_idx'])}")
    print(f"[PPG foot] valid={foot_result['valid']}, reason={foot_result['reason']}, num={len(foot_result['foot_idx'])}")

    save_dir = results_dir / stem
    save_dir.mkdir(exist_ok=True, parents=True)
    save_filtered_signals(df_filtered, save_dir, stem)
    save_detection_results(r_result, foot_result, save_dir, stem)
    plot_filtered_overview(df_filtered, r_result, foot_result, fs, save_dir, stem)

    validated_df.to_csv(save_dir / f"{stem}_pat_ptt_beats.csv", index=False)
    summary_df.to_csv(save_dir / f"{stem}_pat_ptt_summary.csv", index=False)
    extract_final_output(validated_df, save_dir / f"{stem}_final_output.csv")

    print(summary_df)
    print(f"최종 출력 저장 완료: {save_dir / f'{stem}_final_output.csv'}")

    return {
        "file_name": orig_file_name,
        "segment_stem": stem,
        "fs": fs,
        "sample_period_ms": SAMPLE_PERIOD_MS,
        "pep_ms": PEP,
        "n_samples": int(len(seg_df)),
        "duration_s": float(time[-1] - time[0]) if len(time) >= 2 else 0.0,
        "num_rpeaks": len(r_result["r_idx"]),
        "num_foots": len(foot_result["foot_idx"]),
        "num_total_beats": len(validated_df),
        "num_valid_beats": int(validated_df["is_valid"].sum()) if len(validated_df) > 0 else 0,
        "final_output": str(save_dir / f"{stem}_final_output.csv"),
    }


def process_one_file(csv_path: Path, results_dir: Path):
    if not csv_path.exists():
        print(f"[경고] 파일 없음: {csv_path}")
        return None

    print("\n" + "=" * 80)
    print(f"처리 중: {csv_path.name}")
    print("=" * 80)

    df_raw = pd.read_csv(csv_path)
    df = _normalize_columns(df_raw)
    required_cols = {"ecg", "ppg"}
    if not required_cols.issubset(df.columns):
        print(f"[경고] 필요한 컬럼 없음: {csv_path.name} -> {list(df_raw.columns)}")
        return None

    segments = _split_segments_by_idx(df)
    print(f"[세그먼트] sample_idx 갭 검사 결과 {len(segments)}개 세그먼트 사용")
    if len(segments) == 0:
        print(f"[경고] 사용할 수 있는 세그먼트가 없습니다: {csv_path.name}")
        return None

    if len(segments) == 1:
        return _process_one_segment(segments[0], csv_path.stem, results_dir, csv_path.name)

    # 다중 세그먼트: 각각 독립적으로 처리하고 결과 리스트로 반환
    results = []
    for i, seg in enumerate(segments, start=1):
        stem = f"{csv_path.stem}_seg{i}"
        print(f"\n--- 세그먼트 {i}/{len(segments)} : {len(seg)} samples ---")
        try:
            r = _process_one_segment(seg, stem, results_dir, csv_path.name)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"[ERR] segment {i}: {e}")
    return results if results else None


if __name__ == "__main__":
    DATA_DIR = BASE_DIR / "data"
    RESULTS_DIR = Path("/Users/jshyun/Downloads/HWSP_pipeline/results")## 저장 파일 위치
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    CSV_LIST = [
        "data_20260604_193905.csv"
    ] 

    print("현재 작업 폴더:", BASE_DIR)
    print("데이터 폴더:", DATA_DIR)
    print("결과 폴더:", RESULTS_DIR)

    all_rows = []
    for csv_name in CSV_LIST:
        result = process_one_file(DATA_DIR / csv_name, RESULTS_DIR)
        if result is not None:
            all_rows.append(result)

    if len(all_rows) > 0:
        all_df = pd.DataFrame(all_rows)
        all_df.to_csv(RESULTS_DIR / "all_file_processing_summary.csv", index=False)
        print("\n전체 파일 요약 저장 완료:", RESULTS_DIR / "all_file_processing_summary.csv")
        print(all_df)
    else:
        print("\n처리된 파일이 없습니다. data 폴더와 CSV_LIST를 확인하세요.")
