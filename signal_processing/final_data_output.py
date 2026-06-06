from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def save_filtered_signals(df_filtered: pd.DataFrame, save_dir: Path, stem: str):
    cols = ["time", "ecg", "ecg_filtered", "ppg", "ppg_filtered"]
    df_filtered[cols].to_csv(save_dir / f"{stem}_filtered_signals.csv", index=False)


def save_detection_results(r_result: dict, foot_result: dict, save_dir: Path, stem: str):
    pd.DataFrame({
        "r_index": r_result["r_idx"],
        "r_time_sec": r_result["r_time"],
    }).to_csv(save_dir / f"{stem}_r_peaks.csv", index=False)

    if len(r_result["rr_interval"]) > 0:
        pd.DataFrame({
            "rr_interval_sec": r_result["rr_interval"],
        }).to_csv(save_dir / f"{stem}_rr_intervals.csv", index=False)

    pd.DataFrame({
        "matched_r_index": foot_result["matched_r_idx"],
        "matched_r_time_sec": foot_result["matched_r_time"],
        "foot_index": foot_result["foot_idx"],
        "foot_time_sec": foot_result["foot_time"],
    }).to_csv(save_dir / f"{stem}_ppg_foots.csv", index=False)


def _plot_overlay(ax, time, raw_sig, filt_sig, title, ylabel):
    ax.plot(time, raw_sig, label="raw", alpha=0.60)
    ax.plot(time, filt_sig, label="filtered", linewidth=1.2)
    ax.set_xlim(0, min(10, time[-1]))
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_filtered_overview(df_filtered: pd.DataFrame, r_result: dict, foot_result: dict, fs: float, save_dir: Path, stem: str):
    time = df_filtered["time"].to_numpy()
    ecg_raw = df_filtered["ecg"].to_numpy()
    ecg_f = df_filtered["ecg_filtered"].to_numpy()
    ppg_raw = df_filtered["ppg"].to_numpy()
    ppg_f = df_filtered["ppg_filtered"].to_numpy()

    for raw_sig, filt_sig, title, fname, ylabel in [
        (ecg_raw, ecg_f, f"{stem} | ECG raw vs filtered", f"{stem}_ecg_filtered.png", "ECG"),
        (ppg_raw, ppg_f, f"{stem} | PPG raw vs filtered", f"{stem}_ppg_filtered.png", "PPG"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 4))
        _plot_overlay(ax, time, raw_sig, filt_sig, title, ylabel)
        fig.tight_layout()
        fig.savefig(save_dir / fname, dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    _plot_overlay(axes[0], time, ecg_raw, ecg_f, f"{stem} | ECG raw vs filtered", "ECG")
    _plot_overlay(axes[1], time, ppg_raw, ppg_f, f"{stem} | PPG raw vs filtered", "PPG")
    fig.tight_layout()
    fig.savefig(save_dir / f"{stem}_all_filters_overlay.png", dpi=200)
    plt.close(fig)

    start_sec, end_sec = 0, min(10, time[-1])
    start_idx = max(0, int(start_sec * fs))
    end_idx = min(len(ecg_f), int(end_sec * fs))
    time_zoom = time[start_idx:end_idx]

    r_idx = r_result["r_idx"]
    foot_idx = foot_result["foot_idx"]

    r_zoom = r_idx[(r_idx >= start_idx) & (r_idx < end_idx)]
    foot_zoom = foot_idx[(foot_idx >= start_idx) & (foot_idx < end_idx)]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_zoom, ecg_f[start_idx:end_idx], label="filtered ECG")
    if len(r_zoom) > 0:
        ax.scatter(time[r_zoom], ecg_f[r_zoom], s=20, label="R-peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{stem} | ECG R-peaks (0~10s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / f"{stem}_zoom_ecg_rpeaks.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_zoom, ppg_f[start_idx:end_idx], label="filtered PPG")
    if len(foot_zoom) > 0:
        ax.scatter(time[foot_zoom], ppg_f[foot_zoom], s=20, label="PPG foots")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{stem} | PPG foots (0~10s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / f"{stem}_zoom_ppg_foots.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(time_zoom, ecg_f[start_idx:end_idx], label="filtered ECG")
    if len(r_zoom) > 0:
        axes[0].scatter(time[r_zoom], ecg_f[r_zoom], s=20, label="R-peaks")
    axes[0].set_ylabel("ECG")
    axes[0].set_title(f"{stem} | Detection summary (0~10s)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_zoom, ppg_f[start_idx:end_idx], label="filtered PPG")
    if len(foot_zoom) > 0:
        axes[1].scatter(time[foot_zoom], ppg_f[foot_zoom], s=20, label="PPG foots")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("PPG")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / f"{stem}_detection_summary.png", dpi=200)
    plt.close(fig)


def extract_final_output(processed_df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    result_df = processed_df[[
        "r_index",
        "foot_index",
        "r_time_sec",
        "foot_time_sec",
        "pep_sec",
        "pep_ms",
        "pat_sec",
        "pat_ms",
        "ptt_sec",
        "ptt_ms",
        "rr_interval_sec",
        "rr_interval_ms",
        "is_valid",
    ]].copy()

    result_df.to_csv(save_path, index=False)
    return result_df
