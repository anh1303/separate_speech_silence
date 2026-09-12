"""
plotting.py – Module trực quan hóa cho bài toán phân đoạn Speech/Silence.

Chứa các hàm vẽ biểu đồ:
- Figure 1: Waveform + Ground Truth shading
- Figure 2: Waveform → logMA (xếp chồng)
- Figure 2b: Signal → logMA + Vùng Overlap & Ngưỡng T
- Figure 3: Histogram phân phối Speech vs Silence
- Figure 4: Quá trình hội tụ binary search
- Figure 5: Phân đoạn cuối cùng (Ground Truth vs Predicted)
- Figure 6: So sánh trước / sau lọc silence 300ms
- Test Evaluation Figure: Đáp ứng chuẩn yêu cầu chấm thi của GV (Đường xanh pred, đường đỏ GT)
- Noise Comparison Figures: So sánh dạng sóng, phân phối và đường cong F1/MAE/RMSE theo SNR.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

try:
    from src.config import (
        BOUNDARY_MATCH_TOLERANCE_MS, FRAME_SHIFT_MS, MIN_SILENCE_DURATION_MS,
        BinarySearchState, Segment,
    )
    from src.segmentation import extract_boundaries
except ImportError:
    from config import (
        BOUNDARY_MATCH_TOLERANCE_MS, FRAME_SHIFT_MS, MIN_SILENCE_DURATION_MS,
        BinarySearchState, Segment,
    )
    from segmentation import extract_boundaries

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mkdirs(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _shade(ax, segments: List[Segment], alpha: float = 0.20):
    """Tô màu nền speech/silence lên ax."""
    for seg in segments:
        ax.axvspan(seg.start, seg.end, alpha=alpha,
                   color="royalblue" if seg.label == "speech" else "lightcoral",
                   linewidth=0)


def _gt_legend():
    return [mpatches.Patch(color="royalblue",  alpha=0.5, label="Speech (GT)"),
            mpatches.Patch(color="lightcoral", alpha=0.5, label="Silence (GT)")]


def _save(fig, path: str):
    _mkdirs(path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path)


# ── Figure 1 ───────────────────────────────────────────────────────────────────

def plot_waveform_and_ground_truth(
    signal: np.ndarray, sample_rate: int,
    gt_segments: List[Segment], title: str, save_path: str,
):
    """Dạng sóng âm thanh với vùng Speech (xanh) và Silence (hồng) từ .lab."""
    t = np.arange(len(signal)) / sample_rate
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, signal, color="#2c3e50", lw=0.6, alpha=0.9)
    ax.set(xlabel="Time (s)", ylabel="Amplitude",
           title=f"Figure 1 – Waveform & Ground Truth: {title}")
    ax.grid(alpha=0.3)
    _shade(ax, gt_segments, alpha=0.25)
    ax.legend(handles=_gt_legend(), loc="upper right", fontsize=9)
    plt.tight_layout()
    _save(fig, save_path)


# ── Figure 2 ───────────────────────────────────────────────────────────────────

def plot_feature(
    signal: np.ndarray, sample_rate: int,
    feature_vals: np.ndarray, frame_centers: np.ndarray,
    gt_segments: List[Segment], feature_name: str, title: str, save_path: str,
):
    """Waveform (trên) và feature (dưới), xếp chồng cùng trục thời gian."""
    t = np.arange(len(signal)) / sample_rate
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(t, signal, color="#2c3e50", lw=0.5, alpha=0.85)
    axes[0].set(ylabel="Amplitude", title=f"Figure 2 – Waveform & {feature_name}: {title}")
    axes[0].grid(alpha=0.3)
    _shade(axes[0], gt_segments, alpha=0.2)

    axes[1].plot(frame_centers, feature_vals, color="#e67e22", lw=1.2, label=feature_name)
    axes[1].set(xlabel="Time (s)", ylabel=feature_name)
    axes[1].grid(alpha=0.3)
    _shade(axes[1], gt_segments, alpha=0.2)

    handles = _gt_legend() + [mpatches.Patch(color="#e67e22", alpha=0.8, label=feature_name)]
    axes[1].legend(handles=handles, loc="upper right", fontsize=8)
    plt.tight_layout()
    _save(fig, save_path)


# ── Figure 2b ──────────────────────────────────────────────────────────────────

def plot_signal_to_feature_with_overlap(
    signal:        np.ndarray,
    sample_rate:   int,
    feature_vals:  np.ndarray,
    frame_centers: np.ndarray,
    overlap_info:  Dict[str, Any],
    threshold:     Optional[float] = None,
    gt_segments:   Optional[List[Segment]] = None,
    feature_name:  str = "log(MA)",
    title:         str = "",
    save_path:     str = "",
    overlap_label: str = "Overlap",
    threshold_label: str = "Optimal T",
):
    """
    Minh hoạ chuỗi biến đổi: Tín hiệu gốc → log(MA), kết hợp hiển thị
    vùng overlap (và ngưỡng tối ưu T nếu có).
    """
    t = np.arange(len(signal)) / sample_rate
    ov_min = overlap_info["overlap_min"]
    ov_max = overlap_info["overlap_max"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax0 = axes[0]
    ax0.plot(t, signal, color="#2c3e50", lw=0.5, alpha=0.85)
    ax0.set(ylabel="Amplitude",
            title=f"Figure 2b – Signal → {feature_name} + Overlap Region: {title}")
    ax0.grid(alpha=0.3)
    if gt_segments is not None:
        _shade(ax0, gt_segments, alpha=0.22)
        ax0.legend(handles=_gt_legend(), loc="upper right", fontsize=9)

    ax1 = axes[1]
    ax1.plot(frame_centers, feature_vals, color="#e67e22", lw=1.3, label=feature_name, zorder=3)
    ax1.set(xlabel="Time (s)", ylabel=feature_name)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(t[0], t[-1])

    if overlap_info["has_overlap"]:
        ax1.axhspan(ov_min, ov_max, alpha=0.15, color="gold", zorder=1,
                    label=f"{overlap_label} [{ov_min:.3f}, {ov_max:.3f}]")

    ax1.axhline(ov_min, color="darkorange", ls="--", lw=1.5,
                label=f"Overlap min = {ov_min:.4f}", zorder=4)
    ax1.axhline(ov_max, color="darkorange", ls=":",  lw=1.5,
                label=f"Overlap max = {ov_max:.4f}", zorder=4)
    if threshold is not None:
        ax1.axhline(threshold, color="#27ae60", ls="-", lw=2.2,
                    label=f"{threshold_label} = {threshold:.4f}", zorder=5)

    x_right = t[-1] * 0.98
    yrange  = ax1.get_ylim()
    offset  = (yrange[1] - yrange[0]) * 0.015

    labels_to_draw = [
        (ov_min, f"Tmin={ov_min:.4f}", "darkorange"),
        (ov_max, f"Tmax={ov_max:.4f}", "darkorange"),
    ]
    if threshold is not None:
        labels_to_draw.append((threshold, f"T={threshold:.4f}", "#27ae60"))

    for y, text, color in labels_to_draw:
        ax1.text(x_right, y + offset, text, ha="right", va="bottom",
                 fontsize=8, color=color, fontweight="bold")

    ax1.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    _save(fig, save_path)


# ── Figure 3 ───────────────────────────────────────────────────────────────────

def plot_feature_distribution(
    speech_vals: np.ndarray, silence_vals: np.ndarray,
    overlap_info: Dict[str, Any], threshold: Optional[float] = None,
    feature_name: str = "log(MA)", title: str = "", save_path: str = "",
):
    """Histogram phân phối Speech vs Silence, đánh dấu overlap (và T tối ưu nếu có)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = min(80, max(20, int((len(speech_vals) + len(silence_vals)) ** 0.4)))

    ax.hist(speech_vals,  bins=bins, alpha=0.55, color="royalblue",
            label="Speech", density=True, edgecolor="none")
    ax.hist(silence_vals, bins=bins, alpha=0.55, color="lightcoral",
            label="Silence", density=True, edgecolor="none")

    ov_min, ov_max = overlap_info["overlap_min"], overlap_info["overlap_max"]
    if overlap_info["has_overlap"]:
        ax.axvspan(ov_min, ov_max, alpha=0.15, color="gold", label="Overlap region")

    ax.axvline(ov_min,    color="darkorange", ls="--", lw=1.3, label=f"Tmin={ov_min:.4f}")
    ax.axvline(ov_max,    color="darkorange", ls=":",  lw=1.3, label=f"Tmax={ov_max:.4f}")
    if threshold is not None:
        ax.axvline(threshold, color="#27ae60",    ls="-",  lw=2.0, label=f"T={threshold:.4f}")
        fig_title = f"Figure 3 – Feature Distribution & Threshold: {title}"
    else:
        fig_title = f"Figure 3 – Feature Distribution & Overlap Region: {title}"

    ax.set(xlabel=feature_name, ylabel="Density", title=fig_title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path)


# ── Figure 4 ───────────────────────────────────────────────────────────────────

def plot_binary_search_history(
    history: List[BinarySearchState], title: str, save_path: str,
):
    """Hội tụ binary search: T, confusion và difference theo iteration."""
    iters  = [s.iteration       for s in history]
    thres  = [s.threshold       for s in history]
    c_sil  = [s.confusion_sil   for s in history]
    c_spch = [s.confusion_speech for s in history]
    diffs  = [s.difference      for s in history]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(iters, thres,  "o-", color="#8e44ad", lw=1.5)
    axes[0].set(ylabel="Threshold T", title=f"Figure 4 – Binary Search Convergence: {title}")
    axes[0].grid(alpha=0.3)

    axes[1].plot(iters, c_sil,  "s-", color="lightcoral", lw=1.5, label="Confusion Silence")
    axes[1].plot(iters, c_spch, "^-", color="royalblue",  lw=1.5, label="Confusion Speech")
    axes[1].set(ylabel="Confusion")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(iters, diffs, "D-", color="#27ae60", lw=1.5)
    axes[2].axhline(0, color="black", ls="--", lw=0.8)
    axes[2].set(xlabel="Iteration", ylabel="Difference\n(C_sil − C_spch)")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)


# ── Figure 5 ───────────────────────────────────────────────────────────────────

def plot_final_segmentation(
    signal: np.ndarray, sample_rate: int,
    feature_vals: np.ndarray, frame_centers: np.ndarray,
    pred_segments: List[Segment], gt_segments: List[Segment],
    threshold: float, feature_name: str, title: str, save_path: str,
    metrics: Optional[Dict[str, Any]] = None,
):
    """
    Waveform (trên) + Feature (dưới).
    Đường đỏ = GT boundary, đường xanh = predicted boundary.
    """
    t = np.arange(len(signal)) / sample_rate
    gt_bounds   = extract_boundaries(gt_segments)
    pred_bounds = extract_boundaries(pred_segments)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax0 = axes[0]
    ax0.plot(t, signal, color="#2c3e50", lw=0.5, alpha=0.8)
    sub_title = f"Figure 5 – Final Segmentation: {title}"
    if metrics:
        sub_title += f" (F1={metrics.get('boundary_f1', 0):.2f}, MAE={metrics.get('mae_ms', float('nan')):.1f}ms)"
    ax0.set(ylabel="Amplitude", title=sub_title)
    ax0.grid(alpha=0.3)
    _shade(ax0, pred_segments, alpha=0.15)
    for b in gt_bounds:
        ax0.axvline(b, color="crimson", lw=1.8, label="Ground Truth" if b == gt_bounds[0] else "")
    for b in pred_bounds:
        ax0.axvline(b, color="royalblue", lw=1.8, ls="--", label="Predicted" if b == pred_bounds[0] else "")
    ax0.legend(loc="upper right", fontsize=8)

    ax1 = axes[1]
    ax1.plot(frame_centers, feature_vals, color="#e67e22", lw=1.2, label=feature_name)
    ax1.axhline(threshold, color="#27ae60", lw=2.0, label=f"T={threshold:.4f}")
    ax1.set(xlabel="Time (s)", ylabel=feature_name)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(t[0], t[-1])
    for b in gt_bounds:
        ax1.axvline(b, color="crimson", lw=1.8)
    for b in pred_bounds:
        ax1.axvline(b, color="royalblue", lw=1.8, ls="--")
    ax1.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    _save(fig, save_path)


# ── Figure 6 ───────────────────────────────────────────────────────────────────

def plot_before_after_filter(
    signal: np.ndarray, sample_rate: int,
    seg_before: List[Segment], seg_after: List[Segment],
    title: str, save_path: str,
    min_silence_ms: float = MIN_SILENCE_DURATION_MS,
):
    """Segmentation trước/sau khi lọc bỏ silence ngắn."""
    t = np.arange(len(signal)) / sample_rate

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    def _draw(ax, segs, subtitle):
        ax.plot(t, signal, color="#2c3e50", lw=0.5, alpha=0.7)
        _shade(ax, segs, alpha=0.3)
        for b in extract_boundaries(segs):
            ax.axvline(b, color="#2c3e50", lw=1.0, alpha=0.6)
        ax.set(ylabel="Amplitude", title=subtitle)
        ax.grid(alpha=0.3)
        ylim = ax.get_ylim()
        for seg in segs:
            if seg.label == "silence":
                ax.text((seg.start + seg.end) / 2, ylim[1] * 0.85,
                        f"{seg.duration_ms:.0f}ms",
                        ha="center", fontsize=7, color="darkred")
        ax.legend(handles=[
            mpatches.Patch(color="royalblue",  alpha=0.5, label="Speech"),
            mpatches.Patch(color="lightcoral", alpha=0.5, label="Silence"),
        ], loc="upper right", fontsize=8)

    _draw(axes[0], seg_before, f"Before filter – {title}")
    _draw(axes[1], seg_after,  f"After {min_silence_ms:.0f}ms filter – {title}")
    axes[1].set(xlabel="Time (s)")
    axes[1].set_xlim(t[0], t[-1])

    plt.suptitle(f"Figure 6 – Before/After {min_silence_ms:.0f}ms Silence Filter: {title}",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    _save(fig, save_path)


# ── TEST EVALUATION FIGURE (Dành cho buổi chấm thi của GV) ────────────────────

def plot_test_evaluation(
    signal:         np.ndarray,
    sample_rate:    int,
    feature_vals:   np.ndarray,
    frame_centers:  np.ndarray,
    pred_segments:  List[Segment],
    gt_segments:    List[Segment],
    threshold:      float,
    feature_name:   str,
    filename:       str,
    metrics:        Dict[str, Any],
    save_path:      str,
):
    """
    Vẽ Figure kiểm thử chuẩn theo đúng yêu cầu đề tài HuongDan.md:
    - Đường dọc màu xanh: ranh giới do thuật toán xuất ra (predicted)
    - Đường dọc màu đỏ: ranh giới chuẩn ground truth (từ file .lab)
    - Hiển thị định lượng: MAE, RMSE (ms), Precision, Recall, F1.
    """
    t = np.arange(len(signal)) / sample_rate
    gt_bounds   = extract_boundaries(gt_segments)
    pred_bounds = extract_boundaries(pred_segments)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Subplot 1: Waveform với vạch biên xanh/đỏ
    ax0 = axes[0]
    ax0.plot(t, signal, color="#2c3e50", lw=0.5, alpha=0.85)
    _shade(ax0, pred_segments, alpha=0.15)
    ax0.set(ylabel="Amplitude", title=f"Test Evaluation – Waveform & Segmentation: {filename}")
    ax0.grid(alpha=0.3)

    for b in gt_bounds:
        ax0.axvline(b, color="crimson", lw=2.0, ls="-", zorder=5)
    for b in pred_bounds:
        ax0.axvline(b, color="dodgerblue", lw=2.0, ls="--", zorder=6)

    # Subplot 2: Feature curve
    ax1 = axes[1]
    ax1.plot(frame_centers, feature_vals, color="#e67e22", lw=1.2, label=feature_name, zorder=3)
    ax1.axhline(threshold, color="#27ae60", lw=2.0, label=f"Global T = {threshold:.4f}", zorder=4)
    ax1.set(xlabel="Time (s)", ylabel=feature_name)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(t[0], t[-1])

    for b in gt_bounds:
        ax1.axvline(b, color="crimson", lw=2.0, ls="-", zorder=5)
    for b in pred_bounds:
        ax1.axvline(b, color="dodgerblue", lw=2.0, ls="--", zorder=6)

    # Legends & Metrics box
    mae_str  = f"{metrics.get('mae_ms', float('nan')):.1f} ms"
    rmse_str = f"{metrics.get('rmse_ms', float('nan')):.1f} ms"
    f1_val   = metrics.get("boundary_f1", 0.0)
    p_val    = metrics.get("boundary_precision", 0.0)
    r_val    = metrics.get("boundary_recall", 0.0)

    metric_text = (f"MAE: {mae_str}\n"
                   f"RMSE: {rmse_str}\n"
                   f"Precision: {p_val:.2f}\n"
                   f"Recall: {r_val:.2f}\n"
                   f"F1-Score: {f1_val:.2f}\n"
                   f"Matched: {metrics.get('n_matched', 0)}/{metrics.get('n_gt_boundaries', 0)}")

    props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#bdc3c7")
    ax0.text(0.02, 0.95, metric_text, transform=ax0.transAxes, fontsize=9,
             verticalalignment="top", bbox=props)

    custom_handles = [
        plt.Line2D([0], [0], color="crimson", lw=2.0, label="Ground Truth boundary (Red)"),
        plt.Line2D([0], [0], color="dodgerblue", lw=2.0, ls="--", label="Predicted boundary (Blue)"),
        mpatches.Patch(color="royalblue", alpha=0.4, label="Predicted Speech"),
        mpatches.Patch(color="lightcoral", alpha=0.4, label="Predicted Silence"),
    ]
    ax0.legend(handles=custom_handles, loc="upper right", fontsize=8)
    ax1.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    _save(fig, save_path)


# ── NOISE EXPERIMENT FIGURES ───────────────────────────────────────────────────

def plot_noisy_waveforms_comparison(
    waveform_dict: Dict[str, Tuple[np.ndarray, int]],
    filename: str,
    save_path: str,
):
    """So sánh dạng sóng 1 file qua các mức SNR (Original, 20dB, 10dB, 5dB, 0dB)."""
    n_plots = len(waveform_dict)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.5 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    colors = ["#2c3e50", "#2980b9", "#16a085", "#d35400", "#c0392b"]
    for idx, (condition, (sig, sr)) in enumerate(waveform_dict.items()):
        t = np.arange(len(sig)) / sr
        ax = axes[idx]
        col = colors[idx % len(colors)]
        ax.plot(t, sig, color=col, lw=0.5, alpha=0.85)
        ax.set(ylabel="Amplitude", title=f"{condition} – {filename}")
        ax.grid(alpha=0.3)

    axes[-1].set(xlabel="Time (s)")
    plt.suptitle(f"Waveform Degradation Under Controlled Noise: {filename}", fontsize=12, y=1.005)
    plt.tight_layout()
    _save(fig, save_path)


def plot_snr_distribution_comparison(
    distributions_by_snr: Dict[str, Dict[str, np.ndarray]],
    thresholds: Dict[str, float],
    feature_name: str,
    save_path: str,
):
    """
    So sánh Histogram phân phối Speech vs Silence qua các mức SNR.
    Thể hiện trực quan cách nhiễu làm tăng vùng overlap.
    """
    n_conditions = len(distributions_by_snr)
    fig, axes = plt.subplots(1, n_conditions, figsize=(4.5 * n_conditions, 4), sharey=True)
    if n_conditions == 1:
        axes = [axes]

    for ax, (cond_name, dist_data) in zip(axes, distributions_by_snr.items()):
        spk = dist_data["speech"]
        sil = dist_data["silence"]
        bins = 40
        ax.hist(spk, bins=bins, alpha=0.5, color="royalblue", label="Speech", density=True)
        ax.hist(sil, bins=bins, alpha=0.5, color="lightcoral", label="Silence", density=True)

        if cond_name in thresholds:
            t_val = thresholds[cond_name]
            ax.axvline(t_val, color="#27ae60", lw=1.8, label=f"T={t_val:.3f}")

        ax.set(xlabel=feature_name, title=f"Condition: {cond_name}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    axes[0].set(ylabel="Density")
    plt.suptitle(f"Feature Distribution Shift Under Noise ({feature_name})", fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, save_path)


def plot_snr_metrics_curves(
    snr_levels: List[float],
    fixed_summary: Dict[str, List[float]],
    retrained_summary: Dict[str, List[float]],
    save_path: str,
):
    """
    Vẽ 3 biểu đồ so sánh Fixed Threshold vs Retrained Threshold theo SNR:
    - F1-score vs SNR
    - MAE vs SNR
    - RMSE vs SNR
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    snr_labels = [f"{s} dB" for s in snr_levels]
    x = np.arange(len(snr_levels))

    # 1. F1-score
    ax = axes[0]
    ax.plot(x, fixed_summary.get("f1", []), "o-", color="#e74c3c", lw=2, label="Fixed T (Robustness)")
    ax.plot(x, retrained_summary.get("f1", []), "s--", color="#27ae60", lw=2, label="Retrained T (Adaptation)")
    ax.set_xticks(x)
    ax.set_xticklabels(snr_labels)
    ax.set(xlabel="SNR", ylabel="Boundary F1-Score", title="F1-Score vs SNR", ylim=(-0.05, 1.05))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # 2. MAE
    ax = axes[1]
    ax.plot(x, fixed_summary.get("mae", []), "o-", color="#e74c3c", lw=2, label="Fixed T")
    ax.plot(x, retrained_summary.get("mae", []), "s--", color="#27ae60", lw=2, label="Retrained T")
    ax.set_xticks(x)
    ax.set_xticklabels(snr_labels)
    ax.set(xlabel="SNR", ylabel="MAE (ms)", title="Boundary MAE vs SNR")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # 3. RMSE
    ax = axes[2]
    ax.plot(x, fixed_summary.get("rmse", []), "o-", color="#e74c3c", lw=2, label="Fixed T")
    ax.plot(x, retrained_summary.get("rmse", []), "s--", color="#27ae60", lw=2, label="Retrained T")
    ax.set_xticks(x)
    ax.set_xticklabels(snr_labels)
    ax.set(xlabel="SNR", ylabel="RMSE (ms)", title="Boundary RMSE vs SNR")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.suptitle("Performance Comparison: Fixed Threshold vs Retrained Threshold Under Noise",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
