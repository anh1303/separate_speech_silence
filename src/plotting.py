"""
plotting.py – Module trực quan hóa toàn diện cho bài toán phân đoạn Speech/Silence.

Chứa các hàm vẽ biểu đồ:
- plot_waveform_and_ground_truth: Dạng sóng âm thanh kết hợp vùng nhãn chuẩn Ground Truth.
- plot_feature: Waveform xếp chồng với hàm đặc trưng ngắn hạn (logMA / STE).
- plot_signal_to_feature_with_overlap: Chuỗi biến đổi tín hiệu sang feature, thể hiện vùng overlap và ngưỡng T.
- plot_feature_distribution: Histogram phân phối đặc trưng Speech vs Silence và ngưỡng tối ưu T.
- plot_binary_search_history: Lịch sử hội tụ của thuật toán tìm kiếm nhị phân qua từng bước lặp.
- plot_final_segmentation: Đồ thị phân đoạn cuối cùng (so sánh ranh giới chuẩn đỏ và ranh giới thuật toán xanh).
- plot_before_after_filter: So sánh kết quả phân đoạn trước và sau khi lọc bỏ khoảng lặng ngắn (< 300 ms).
- plot_test_evaluation: Đồ thị đánh giá bộ kiểm thử chuẩn theo quy định của Giáo viên.
- position_figures_4_corners: Tự động tính toán và sắp xếp 4 cửa sổ Figure vào 4 góc màn hình.
- Các biểu đồ khảo sát ảnh hưởng của mức nhiễu nền (SNR).
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
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


# ── Cấu hình Backend & Định vị Cửa sổ ──────────────────────────────────────────

def setup_matplotlib_backend(interactive: bool = True) -> str:
    """
    Cấu hình backend cho thư viện Matplotlib.

    Nếu interactive=True, ưu tiên sử dụng TkAgg để hỗ trợ mở cửa sổ đồ họa và
    tự động định vị cửa sổ trên các hệ điều hành (macOS, Linux, Windows).
    Nếu interactive=False, sử dụng Agg để chạy ngầm và chỉ xuất file ảnh.

    Args:
        interactive (bool, optional): Có kích hoạt giao diện cửa sổ tương tác hay không. Mặc định là True.

    Returns:
        str: Tên backend Matplotlib đang được áp dụng.
    """
    if interactive:
        try:
            matplotlib.use("TkAgg")
            logger.debug("Sử dụng backend Matplotlib tương tác: TkAgg")
        except Exception:
            logger.debug("Không nạp được TkAgg, sử dụng backend mặc định: %s", matplotlib.get_backend())
    else:
        matplotlib.use("Agg")
        logger.debug("Sử dụng backend Matplotlib không tương tác: Agg")

    return matplotlib.get_backend()


def position_figures_4_corners(figures: List[plt.Figure]) -> None:
    """
    Tự động tính toán kích thước màn hình và sắp xếp tối đa 4 cửa sổ Figure vào 4 góc của màn hình.

    Đáp ứng chính xác yêu cầu của Giáo viên trong file hướng dẫn:
    - Figure 1: Góc trên - trái (Top-Left)
    - Figure 2: Góc trên - phải (Top-Right)
    - Figure 3: Góc dưới - trái (Bottom-Left)
    - Figure 4: Góc dưới - phải (Bottom-Right)

    Args:
        figures (List[plt.Figure]): Danh sách các đối tượng Figure cần sắp xếp vị trí hiển thị.

    Returns:
        None
    """
    if not figures:
        return

    # Bước 1: Xác định độ phân giải màn hình hiển thị
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
    except Exception:
        # Độ phân giải dự phòng thông dụng nếu không truy vấn được Tkinter
        screen_w, screen_h = 1440, 900

    # Bước 2: Tính toán kích thước cho mỗi cửa sổ (chia đôi màn hình theo 2 chiều ngang và dọc)
    # Trừ lề khoảng 60px cho thanh Taskbar/Dock và thanh Menu
    usable_h = max(screen_h - 60, 400)
    w = int(screen_w // 2)
    h = int(usable_h // 2)

    # Bước 3: Định nghĩa toạ độ (x, y) cho 4 góc màn hình
    # Top-Left, Top-Right, Bottom-Left, Bottom-Right
    corner_coords = [
        (0, 25),
        (w, 25),
        (0, 25 + h),
        (w, 25 + h),
    ]

    # Bước 4: Áp dụng vị trí hình học cho từng cửa sổ Figure
    for idx, fig in enumerate(figures[:4]):
        x, y = corner_coords[idx]
        try:
            mngr = fig.canvas.manager
            if mngr is not None:
                # Định vị hình học đối với Backend TkAgg
                if hasattr(mngr, "window") and hasattr(mngr.window, "wm_geometry"):
                    mngr.window.wm_geometry(f"{w}x{h}+{x}+{y}")
                # Định vị hình học đối với Backend Qt
                elif hasattr(mngr, "window") and hasattr(mngr.window, "setGeometry"):
                    mngr.window.setGeometry(int(x), int(y), int(w), int(h))
                # Hỗ trợ co giãn kích thước chung
                elif hasattr(mngr, "resize"):
                    mngr.resize(int(w), int(h))
        except Exception as e:
            logger.debug("Không thể tự động đặt vị trí cửa sổ cho Figure %d: %s", idx, e)


# ── Hàm Trợ năng (Helpers) ─────────────────────────────────────────────────────

def _mkdirs(path: str) -> None:
    """
    Tạo thư mục cha chứa file nếu chưa tồn tại.

    Args:
        path (str): Đường dẫn file cần tạo thư mục cha.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _shade(ax: plt.Axes, segments: List[Segment], alpha: float = 0.20) -> None:
    """
    Tô màu nền cho các vùng tiếng nói (speech - màu xanh) và khoảng lặng (silence - màu hồng).

    Args:
        ax (plt.Axes): Trục vẽ đồ thị cần tô màu nền.
        segments (List[Segment]): Danh sách các đoạn phân loại.
        alpha (float, optional): Độ mờ đục của màu nền. Mặc định 0.20.

    Returns:
        None
    """
    for seg in segments:
        ax.axvspan(seg.start, seg.end, alpha=alpha,
                   color="royalblue" if seg.label == "speech" else "lightcoral",
                   linewidth=0)


def _gt_legend() -> List[mpatches.Patch]:
    """
    Tạo danh sách các chú thích màu sắc cho nhãn Ground Truth.

    Args:
        None (sử dụng màu sắc mặc định).

    Returns:
        List[mpatches.Patch]: Danh sách các đối tượng Patch chú thích màu sắc.
    """
    return [
        mpatches.Patch(color="royalblue",  alpha=0.5, label="Speech (GT)"),
        mpatches.Patch(color="lightcoral", alpha=0.5, label="Silence (GT)"),
    ]


def _save(fig: plt.Figure, path: str, close_fig: bool = True) -> None:
    """
    Lưu đối tượng Figure ra đĩa với độ phân giải cao và tùy chọn đóng Figure để giải phóng bộ nhớ.

    Args:
        fig (plt.Figure): Đối tượng Figure cần lưu.
        path (str): Đường dẫn file ảnh đầu ra.
        close_fig (bool, optional): Có đóng Figure sau khi lưu hay không. Mặc định True.

    Returns:
        None
    """
    _mkdirs(path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if close_fig:
        plt.close(fig)
    logger.info("  Đã lưu đồ thị: %s", path)


# ── Figure 1 ───────────────────────────────────────────────────────────────────

def plot_waveform_and_ground_truth(
    signal: np.ndarray,
    sample_rate: int,
    gt_segments: List[Segment],
    title: str,
    save_path: str,
    close_fig: bool = True,
) -> plt.Figure:
    """
    Vẽ dạng sóng tín hiệu âm thanh kết hợp vùng nhãn chuẩn Ground Truth (Speech: xanh, Silence: hồng).

    Args:
        signal (np.ndarray): Mảng 1D float chứa tín hiệu âm thanh.
        sample_rate (int): Tần số lấy mẫu (Hz).
        gt_segments (List[Segment]): Danh sách phân đoạn chuẩn từ file .lab.
        title (str): Tiêu đề file âm thanh.
        save_path (str): Đường dẫn lưu file ảnh.
        close_fig (bool, optional): Có đóng figure sau khi lưu không. Mặc định True.

    Returns:
        plt.Figure: Đối tượng Figure đã vẽ.
    """
    # Bước 1: Khởi tạo trục thời gian và cửa sổ đồ họa
    t = np.arange(len(signal)) / sample_rate
    fig, ax = plt.subplots(figsize=(14, 4))

    # Bước 2: Vẽ dạng sóng tín hiệu và tô màu nền phân đoạn
    ax.plot(t, signal, color="#2c3e50", lw=0.6, alpha=0.9)
    ax.set(xlabel="Time (s)", ylabel="Amplitude",
           title=f"Figure 1 – Waveform & Ground Truth: {title}")
    ax.grid(alpha=0.3)
    _shade(ax, gt_segments, alpha=0.25)
    ax.legend(handles=_gt_legend(), loc="upper right", fontsize=9)

    plt.tight_layout()
    _save(fig, save_path, close_fig=close_fig)
    return fig


# ── Figure 2 ───────────────────────────────────────────────────────────────────

def plot_feature(
    signal: np.ndarray,
    sample_rate: int,
    feature_vals: np.ndarray,
    frame_centers: np.ndarray,
    gt_segments: List[Segment],
    feature_name: str,
    title: str,
    save_path: str,
    close_fig: bool = True,
) -> plt.Figure:
    """
    Vẽ dạng sóng tín hiệu và đường đặc trưng ngắn hạn (logMA / STE) xếp chồng cùng trục thời gian.

    Args:
        signal (np.ndarray): Tín hiệu âm thanh gốc.
        sample_rate (int): Tần số lấy mẫu (Hz).
        feature_vals (np.ndarray): Mảng giá trị đặc trưng ngắn hạn.
        frame_centers (np.ndarray): Mảng mốc thời gian tâm khung (giây).
        gt_segments (List[Segment]): Phân đoạn chuẩn Ground Truth.
        feature_name (str): Tên loại đặc trưng (ví dụ: 'logMA', 'STE').
        title (str): Tiêu đề file âm thanh.
        save_path (str): Đường dẫn lưu file ảnh.
        close_fig (bool, optional): Có đóng figure sau khi lưu không. Mặc định True.

    Returns:
        plt.Figure: Đối tượng Figure đã vẽ.
    """
    t = np.arange(len(signal)) / sample_rate
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Subplot 1: Dạng sóng âm thanh
    axes[0].plot(t, signal, color="#2c3e50", lw=0.5, alpha=0.85)
    axes[0].set(ylabel="Amplitude", title=f"Figure 2 – Waveform & {feature_name}: {title}")
    axes[0].grid(alpha=0.3)
    _shade(axes[0], gt_segments, alpha=0.2)

    # Subplot 2: Đường đặc trưng ngắn hạn theo thời gian
    axes[1].plot(frame_centers, feature_vals, color="#e67e22", lw=1.2, label=feature_name)
    axes[1].set(xlabel="Time (s)", ylabel=feature_name)
    axes[1].grid(alpha=0.3)
    _shade(axes[1], gt_segments, alpha=0.2)

    handles = _gt_legend() + [mpatches.Patch(color="#e67e22", alpha=0.8, label=feature_name)]
    axes[1].legend(handles=handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    _save(fig, save_path, close_fig=close_fig)
    return fig


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
    close_fig:     bool = True,
) -> plt.Figure:
    """
    Minh hoạ chuỗi biến đổi tín hiệu sang đường đặc trưng kết hợp hiển thị vùng overlap và ngưỡng T.

    Args:
        signal (np.ndarray): Mảng tín hiệu gốc.
        sample_rate (int): Tần số lấy mẫu.
        feature_vals (np.ndarray): Giá trị đặc trưng.
        frame_centers (np.ndarray): Vị trí tâm khung.
        overlap_info (Dict[str, Any]): Thông tin vùng giao thoa giữa speech và silence.
        threshold (Optional[float], optional): Giá trị ngưỡng T nếu có.
        gt_segments (Optional[List[Segment]], optional): Nhãn chuẩn ground truth.
        feature_name (str, optional): Tên đặc trưng.
        title (str, optional): Tiêu đề đồ thị.
        save_path (str, optional): Đường dẫn lưu file ảnh.
        overlap_label (str, optional): Nhãn vùng overlap.
        threshold_label (str, optional): Nhãn ngưỡng tối ưu.
        close_fig (bool, optional): Có đóng figure sau khi lưu không. Mặc định True.

    Returns:
        plt.Figure: Đối tượng Figure đã vẽ.
    """
    t = np.arange(len(signal)) / sample_rate
    ov_min = overlap_info["overlap_min"]
    ov_max = overlap_info["overlap_max"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Subplot 1: Dạng sóng tín hiệu gốc
    ax0 = axes[0]
    ax0.plot(t, signal, color="#2c3e50", lw=0.5, alpha=0.85)
    ax0.set(ylabel="Amplitude", title=f"Figure 2b – Signal → {feature_name} + Overlap Region: {title}")
    ax0.grid(alpha=0.3)
    if gt_segments is not None:
        _shade(ax0, gt_segments, alpha=0.22)
        ax0.legend(handles=_gt_legend(), loc="upper right", fontsize=9)

    # Subplot 2: Đường đặc trưng kết hợp dải vùng overlap màu vàng
    ax1 = axes[1]
    ax1.plot(frame_centers, feature_vals, color="#e67e22", lw=1.3, label=feature_name, zorder=3)
    ax1.set(xlabel="Time (s)", ylabel=feature_name)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(t[0], t[-1])

    if overlap_info["has_overlap"]:
        ax1.axhspan(ov_min, ov_max, alpha=0.15, color="gold", zorder=1,
                    label=f"{overlap_label} [{ov_min:.3f}, {ov_max:.3f}]")

    ax1.axhline(ov_min, color="darkorange", ls="--", lw=1.5, label=f"Overlap min = {ov_min:.4f}", zorder=4)
    ax1.axhline(ov_max, color="darkorange", ls=":",  lw=1.5, label=f"Overlap max = {ov_max:.4f}", zorder=4)
    if threshold is not None:
        ax1.axhline(threshold, color="#27ae60", ls="-", lw=2.2, label=f"{threshold_label} = {threshold:.4f}", zorder=5)

    ax1.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    _save(fig, save_path, close_fig=close_fig)
    return fig


# ── Figure 3 ───────────────────────────────────────────────────────────────────

def plot_feature_distribution(
    speech_vals:  np.ndarray,
    silence_vals: np.ndarray,
    overlap_info: Dict[str, Any],
    threshold:    Optional[float] = None,
    feature_name: str = "log(MA)",
    title:        str = "",
    save_path:    str = "",
    close_fig:    bool = True,
) -> plt.Figure:
    """
    Vẽ Histogram phân phối xác suất đặc trưng của Speech và Silence, đánh dấu vùng overlap và ngưỡng T.

    Args:
        speech_vals (np.ndarray): Mảng giá trị đặc trưng của tiếng nói.
        silence_vals (np.ndarray): Mảng giá trị đặc trưng của khoảng lặng.
        overlap_info (Dict[str, Any]): Thông tin vùng giao thoa.
        threshold (Optional[float], optional): Ngưỡng phân loại T.
        feature_name (str, optional): Tên đặc trưng.
        title (str, optional): Tiêu đề đồ thị.
        save_path (str, optional): Đường dẫn lưu file ảnh.
        close_fig (bool, optional): Có đóng figure sau khi lưu không. Mặc định True.

    Returns:
        plt.Figure: Đối tượng Figure đã vẽ.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = min(80, max(20, int((len(speech_vals) + len(silence_vals)) ** 0.4)))

    # Vẽ histogram phân phối mật độ xác suất của hai lớp
    ax.hist(speech_vals,  bins=bins, alpha=0.55, color="royalblue", label="Speech", density=True, edgecolor="none")
    ax.hist(silence_vals, bins=bins, alpha=0.55, color="lightcoral", label="Silence", density=True, edgecolor="none")

    ov_min, ov_max = overlap_info["overlap_min"], overlap_info["overlap_max"]
    if overlap_info["has_overlap"]:
        ax.axvspan(ov_min, ov_max, alpha=0.15, color="gold", label="Overlap region")

    ax.axvline(ov_min, color="darkorange", ls="--", lw=1.3, label=f"Tmin={ov_min:.4f}")
    ax.axvline(ov_max, color="darkorange", ls=":",  lw=1.3, label=f"Tmax={ov_max:.4f}")
    if threshold is not None:
        ax.axvline(threshold, color="#27ae60", ls="-", lw=2.0, label=f"T={threshold:.4f}")
        fig_title = f"Figure 3 – Feature Distribution & Threshold: {title}"
    else:
        fig_title = f"Figure 3 – Feature Distribution & Overlap Region: {title}"

    ax.set(xlabel=feature_name, ylabel="Density", title=fig_title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path, close_fig=close_fig)
    return fig


# ── Figure 4 ───────────────────────────────────────────────────────────────────

def plot_binary_search_history(
    history:   List[BinarySearchState],
    title:     str,
    save_path: str,
    close_fig: bool = True,
) -> plt.Figure:
    """
    Vẽ 3 đồ thị mô tả quá trình hội tụ của thuật toán Binary Search:
    - Biểu đồ 1: Sự biến thiên của ngưỡng T qua các lần lặp.
    - Biểu đồ 2: Độ nhầm lẫn Confusion Silence và Confusion Speech.
    - Biểu đồ 3: Hiệu số Difference (C_sil - C_spch) tiệm cận về 0.

    Args:
        history (List[BinarySearchState]): Lịch sử biến thiên qua các lần lặp.
        title (str): Tiêu đề đồ thị.
        save_path (str): Đường dẫn lưu file ảnh.
        close_fig (bool, optional): Có đóng figure sau khi lưu không. Mặc định True.

    Returns:
        plt.Figure: Đối tượng Figure đã vẽ.
    """
    iters  = [s.iteration        for s in history]
    thres  = [s.threshold        for s in history]
    c_sil  = [s.confusion_sil    for s in history]
    c_spch = [s.confusion_speech for s in history]
    diffs  = [s.difference       for s in history]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Subplot 1: Ngưỡng T qua các bước lặp
    axes[0].plot(iters, thres, "o-", color="#8e44ad", lw=1.5)
    axes[0].set(ylabel="Threshold T", title=f"Figure 4 – Binary Search Convergence: {title}")
    axes[0].grid(alpha=0.3)

    # Subplot 2: Hai hàm mất mát Confusion
    axes[1].plot(iters, c_sil,  "s-", color="lightcoral", lw=1.5, label="Confusion Silence")
    axes[1].plot(iters, c_spch, "^-", color="royalblue",  lw=1.5, label="Confusion Speech")
    axes[1].set(ylabel="Confusion")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # Subplot 3: Hiệu số chênh lệch tiệm cận về đường y = 0
    axes[2].plot(iters, diffs, "D-", color="#27ae60", lw=1.5)
    axes[2].axhline(0, color="black", ls="--", lw=0.8)
    axes[2].set(xlabel="Iteration", ylabel="Difference\n(C_sil − C_spch)")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path, close_fig=close_fig)
    return fig


# ── Figure 5 ───────────────────────────────────────────────────────────────────

def plot_final_segmentation(
    signal:        np.ndarray,
    sample_rate:   int,
    feature_vals:  np.ndarray,
    frame_centers: np.ndarray,
    pred_segments: List[Segment],
    gt_segments:   List[Segment],
    threshold:     float,
    feature_name:  str,
    title:         str,
    save_path:     str,
    metrics:       Optional[Dict[str, Any]] = None,
    close_fig:     bool = True,
) -> plt.Figure:
    """
    Vẽ kết quả phân đoạn cuối cùng: Waveform (trên) và Feature (dưới) với ranh giới chuẩn (đỏ) và dự đoán (xanh).

    Args:
        signal (np.ndarray): Dữ liệu tín hiệu gốc.
        sample_rate (int): Tần số lấy mẫu.
        feature_vals (np.ndarray): Giá trị đặc trưng.
        frame_centers (np.ndarray): Tâm khung thời gian.
        pred_segments (List[Segment]): Đoạn thuật toán tìm được.
        gt_segments (List[Segment]): Đoạn chuẩn Ground Truth.
        threshold (float): Ngưỡng phân loại T.
        feature_name (str): Tên đặc trưng.
        title (str): Tiêu đề đồ thị.
        save_path (str): Đường dẫn lưu file ảnh.
        metrics (Optional[Dict[str, Any]], optional): Các chỉ số đánh giá F1, MAE,...
        close_fig (bool, optional): Có đóng figure sau khi lưu không. Mặc định True.

    Returns:
        plt.Figure: Đối tượng Figure đã vẽ.
    """
    t = np.arange(len(signal)) / sample_rate
    gt_bounds   = extract_boundaries(gt_segments)
    pred_bounds = extract_boundaries(pred_segments)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Subplot 1: Dạng sóng với các vạch biên dọc
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

    # Subplot 2: Đường đặc trưng kết hợp ngưỡng T và vạch biên
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
    _save(fig, save_path, close_fig=close_fig)
    return fig


# ── Figure 6 ───────────────────────────────────────────────────────────────────

def plot_before_after_filter(
    signal:         np.ndarray,
    sample_rate:    int,
    seg_before:     List[Segment],
    seg_after:      List[Segment],
    title:          str,
    save_path:      str,
    min_silence_ms: float = MIN_SILENCE_DURATION_MS,
    close_fig:      bool = True,
) -> plt.Figure:
    """
    Vẽ so sánh phân đoạn trước và sau khi áp dụng thuật toán lọc bỏ khoảng lặng ngắn (< 300 ms).

    Args:
        signal (np.ndarray): Tín hiệu âm thanh.
        sample_rate (int): Tần số lấy mẫu.
        seg_before (List[Segment]): Danh sách đoạn trước khi lọc.
        seg_after (List[Segment]): Danh sách đoạn sau khi lọc.
        title (str): Tiêu đề đồ thị.
        save_path (str): Đường dẫn lưu file ảnh.
        min_silence_ms (float, optional): Thời lượng tối thiểu lọc silence (ms). Mặc định 300ms.
        close_fig (bool, optional): Có đóng figure sau khi lưu không. Mặc định True.

    Returns:
        plt.Figure: Đối tượng Figure đã vẽ.
    """
    t = np.arange(len(signal)) / sample_rate
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    def _draw(ax: plt.Axes, segs: List[Segment], subtitle: str) -> None:
        """
        Vẽ dạng sóng và các phân đoạn cho 1 trạng thái (trước hoặc sau khi lọc).

        Args:
            ax (plt.Axes): Trục vẽ.
            segs (List[Segment]): Danh sách các đoạn phân loại.
            subtitle (str): Tiêu đề phụ của subplot.

        Returns:
            None
        """
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
    _save(fig, save_path, close_fig=close_fig)
    return fig


# ── TEST EVALUATION FIGURE (Dành cho buổi chấm thi của GV) ────────────────────

def plot_test_evaluation(
    signal:        np.ndarray,
    sample_rate:   int,
    feature_vals:  np.ndarray,
    frame_centers: np.ndarray,
    pred_segments: List[Segment],
    gt_segments:   List[Segment],
    threshold:     float,
    feature_name:  str,
    filename:      str,
    metrics:       Dict[str, Any],
    save_path:     str,
    close_fig:     bool = True,
) -> plt.Figure:
    """
    Vẽ Figure kiểm thử đáp ứng chuẩn 100% yêu cầu đề thi của Giáo viên:
    - Subplot 1 (kết quả cuối cùng): Dạng sóng âm thanh với các đường kẻ dọc thể hiện:
      + Ranh giới chuẩn Ground Truth (đường dọc màu đỏ) từ file .lab.
      + Ranh giới do thuật toán tự động xác định được (đường dọc nét đứt màu xanh).
      + Bảng thông số định lượng: MAE (ms), RMSE (ms), Precision, Recall, F1-Score.
    - Subplot 2 (kết quả trung gian): Đường hàm đặc trưng ngắn hạn (logMA) xếp chồng cùng trục thời gian
      kèm theo đường ngang biểu thị ngưỡng tối ưu toàn cục (Global Threshold T).
    - Mỗi plot đều có đầy đủ title và axis label (Time (s), Amplitude, logMA).

    Args:
        signal (np.ndarray): Tín hiệu âm thanh đầu vào.
        sample_rate (int): Tần số lấy mẫu (Hz).
        feature_vals (np.ndarray): Mảng giá trị đặc trưng ngắn hạn trung gian.
        frame_centers (np.ndarray): Mảng mốc thời gian tâm khung (giây).
        pred_segments (List[Segment]): Danh sách phân đoạn do thuật toán xác định.
        gt_segments (List[Segment]): Danh sách phân đoạn chuẩn Ground Truth từ .lab.
        threshold (float): Giá trị ngưỡng phân loại toàn cục T.
        feature_name (str): Tên hàm đặc trưng (ví dụ: 'logMA').
        filename (str): Tên file tín hiệu đang đánh giá.
        metrics (Dict[str, Any]): Từ điển chứa các kết quả đo đạc sai số định lượng.
        save_path (str): Đường dẫn lưu file ảnh trên đĩa.
        close_fig (bool, optional): Có đóng figure sau khi lưu hay giữ lại để mở cửa sổ GUI. Mặc định True.

    Returns:
        plt.Figure: Đối tượng Figure Matplotlib chứa 2 subplot hoàn chỉnh.
    """
    t = np.arange(len(signal)) / sample_rate
    gt_bounds   = extract_boundaries(gt_segments)
    pred_bounds = extract_boundaries(pred_segments)

    # Khởi tạo cửa sổ đồ họa với 2 hàng subplot dùng chung trục hoành thời gian
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    fig.canvas.manager.set_window_title(f"Evaluation – {filename}")

    # Subplot 1: Dạng sóng âm thanh với vạch biên xanh/đỏ (Kết quả cuối cùng)
    ax0 = axes[0]
    ax0.plot(t, signal, color="#2c3e50", lw=0.5, alpha=0.85)
    _shade(ax0, pred_segments, alpha=0.15)
    ax0.set(ylabel="Amplitude", title=f"Test Evaluation – Waveform & Segmentation: {filename}")
    ax0.grid(alpha=0.3)

    for b in gt_bounds:
        ax0.axvline(b, color="crimson", lw=2.0, ls="-", zorder=5)
    for b in pred_bounds:
        ax0.axvline(b, color="dodgerblue", lw=2.0, ls="--", zorder=6)

    # Subplot 2: Đường đặc trưng ngắn hạn và ngưỡng T (Kết quả trung gian)
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

    # Hộp thông số định lượng (Metrics Box) trên Subplot 1
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

    # Chú thích màu sắc ranh giới và vùng phân loại
    custom_handles = [
        plt.Line2D([0], [0], color="crimson", lw=2.0, label="Ground Truth boundary (Red)"),
        plt.Line2D([0], [0], color="dodgerblue", lw=2.0, ls="--", label="Predicted boundary (Blue)"),
        mpatches.Patch(color="royalblue", alpha=0.4, label="Predicted Speech"),
        mpatches.Patch(color="lightcoral", alpha=0.4, label="Predicted Silence"),
    ]
    ax0.legend(handles=custom_handles, loc="upper right", fontsize=8)
    ax1.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    _save(fig, save_path, close_fig=close_fig)
    return fig


# ── NOISE EXPERIMENT FIGURES ───────────────────────────────────────────────────

def plot_noisy_waveforms_comparison(
    waveform_dict: Dict[str, Tuple[np.ndarray, int]],
    filename:      str,
    save_path:     str,
    close_fig:     bool = True,
) -> plt.Figure:
    """
    Vẽ so sánh dạng sóng của 1 file âm thanh qua các mức nhiễu nền SNR khác nhau (Original, 20dB, 10dB, 5dB, 0dB).

    Args:
        waveform_dict (Dict[str, Tuple[np.ndarray, int]]): Từ điển ánh xạ mức SNR sang cặp (signal, sample_rate).
        filename (str): Tên file âm thanh.
        save_path (str): Đường dẫn lưu file ảnh.
        close_fig (bool, optional): Có đóng figure sau khi lưu không. Mặc định True.

    Returns:
        plt.Figure: Đối tượng Figure đã vẽ.
    """
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
    _save(fig, save_path, close_fig=close_fig)
    return fig


def plot_snr_distribution_comparison(
    distributions_by_snr: Dict[str, Dict[str, np.ndarray]],
    thresholds:           Dict[str, float],
    feature_name:         str,
    save_path:            str,
    close_fig:            bool = True,
) -> plt.Figure:
    """
    Vẽ so sánh Histogram phân phối Speech vs Silence qua các mức SNR để minh họa sự mở rộng vùng overlap.

    Args:
        distributions_by_snr (Dict[str, Dict[str, np.ndarray]]): Dữ liệu phân phối theo từng điều kiện SNR.
        thresholds (Dict[str, float]): Ngưỡng tối ưu tương ứng của từng điều kiện.
        feature_name (str): Tên đặc trưng.
        save_path (str): Đường dẫn lưu file ảnh.
        close_fig (bool, optional): Có đóng figure sau khi lưu không. Mặc định True.

    Returns:
        plt.Figure: Đối tượng Figure đã vẽ.
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
    _save(fig, save_path, close_fig=close_fig)
    return fig


def plot_snr_metrics_curves(
    snr_levels:        List[float],
    fixed_summary:     Dict[str, List[float]],
    retrained_summary: Dict[str, List[float]],
    save_path:         str,
    close_fig:         bool = True,
) -> plt.Figure:
    """
    Vẽ 3 biểu đồ đường so sánh hiệu năng giữa Fixed Threshold và Retrained Threshold theo các mức SNR:
    - F1-Score vs SNR
    - MAE vs SNR
    - RMSE vs SNR

    Args:
        snr_levels (List[float]): Danh sách các mức SNR kiểm nghiệm (dB).
        fixed_summary (Dict[str, List[float]]): Chỉ số khi dùng ngưỡng cố định gốc.
        retrained_summary (Dict[str, List[float]]): Chỉ số khi huấn luyện lại ngưỡng theo mức nhiễu.
        save_path (str): Đường dẫn lưu file ảnh.
        close_fig (bool, optional): Có đóng figure sau khi lưu không. Mặc định True.

    Returns:
        plt.Figure: Đối tượng Figure đã vẽ.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    snr_labels = [f"{s} dB" for s in snr_levels]
    x = np.arange(len(snr_levels))

    # 1. Đồ thị F1-score theo SNR
    ax = axes[0]
    ax.plot(x, fixed_summary.get("f1", []), "o-", color="#e74c3c", lw=2, label="Fixed T (Robustness)")
    ax.plot(x, retrained_summary.get("f1", []), "s--", color="#27ae60", lw=2, label="Retrained T (Adaptation)")
    ax.set_xticks(x)
    ax.set_xticklabels(snr_labels)
    ax.set(xlabel="SNR", ylabel="Boundary F1-Score", title="F1-Score vs SNR", ylim=(-0.05, 1.05))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # 2. Đồ thị MAE theo SNR
    ax = axes[1]
    ax.plot(x, fixed_summary.get("mae", []), "o-", color="#e74c3c", lw=2, label="Fixed T")
    ax.plot(x, retrained_summary.get("mae", []), "s--", color="#27ae60", lw=2, label="Retrained T")
    ax.set_xticks(x)
    ax.set_xticklabels(snr_labels)
    ax.set(xlabel="SNR", ylabel="MAE (ms)", title="Boundary MAE vs SNR")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # 3. Đồ thị RMSE theo SNR
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
    _save(fig, save_path, close_fig=close_fig)
    return fig
