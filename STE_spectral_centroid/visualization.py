"""
visualization.py
Ve hinh minh hoa ket qua cho 1 file tin hieu, gom 5 subplot xep theo luoi
3 hang x 2 cot:
    (hang 1) STE theo thoi gian + nguong T1      | Histogram cua STE + T1
    (hang 2) Spectral centroid theo thoi gian + T2 | Histogram centroid + T2
    (hang 3) Dang song tin hieu + bien du doan (xanh) / groundtruth (do), chiem ca 2 cot
"""

import numpy as np
import matplotlib.pyplot as plt

from segmentation import get_frame_times


def _plot_histogram_with_threshold(ax, debug_info, threshold, xlabel, bar_color):
    """
    Ve 1 subplot histogram: cot histogram goc, duong histogram da lam muot,
    2 vach M1/M2 (cuc dai cuc bo dung de tinh T), va vach nguong T cuoi cung.

    Input:
        ax (matplotlib.axes.Axes): truc de ve len
        debug_info (dict): output thu 2 cua estimate_threshold() -- chua
            'hist_values', 'smoothed', 'bin_centers', 'M1', 'M2'
        threshold (float): gia tri nguong T (= (W*M1+M2)/(W+1)) can ve vach do
        xlabel (str): nhan truc x (ten dac trung, vd 'Năng lượng (STE)')
        bar_color (str): mau cho cot histogram goc

    Output:
        None (ve truc tiep len ax)
    """
    bin_centers = debug_info["bin_centers"]
    hist_values = debug_info["hist_values"]
    smoothed = debug_info["smoothed"]

    # Do rong cot = khoang cach giua 2 tam bin lien tiep (gia dinh bin deu nhau)
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0

    ax.bar(bin_centers, hist_values, width=bin_width * 0.9,
           color=bar_color, alpha=0.35, label="Histogram gốc")
    ax.plot(bin_centers, smoothed, color="black", linewidth=1.3, label="Đã làm mượt")
    ax.axvline(debug_info["M1"], color="green", linestyle=":", linewidth=1.5, label="M1")
    ax.axvline(debug_info["M2"], color="purple", linestyle=":", linewidth=1.5, label="M2")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label="Ngưỡng T")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tần suất (số khung)")
    ax.legend(fontsize=7, loc="upper right")


def plot_result_for_one_file(
    signal, fs, energy, centroid, energy_threshold, centroid_threshold,
    energy_debug_info, centroid_debug_info,
    pred_boundaries, gt_boundaries, filename, rmse_ms=None, mae_ms=None,
):
    """
    Ve 1 figure gom 5 subplot cho 1 file tin hieu (luoi 3 hang x 2 cot):
        (1) STE theo thoi gian + nguong T1
        (2) Histogram cua STE + M1/M2/T1
        (3) Spectral centroid theo thoi gian + nguong T2
        (4) Histogram cua spectral centroid + M1/M2/T2
        (5) Dang song tin hieu + bien du doan (xanh) / groundtruth (do)

    Input:
        signal (np.ndarray): tin hieu am thanh goc
        fs (int): tan so lay mau
        energy (np.ndarray): STE tung khung
        centroid (np.ndarray): spectral centroid tung khung
        energy_threshold (float): nguong T1
        centroid_threshold (float): nguong T2
        energy_debug_info (dict): debug_info tra ve tu estimate_threshold(energy,...)
        centroid_debug_info (dict): debug_info tra ve tu estimate_threshold(centroid,...)
        pred_boundaries (list of float): bien (giay) thuat toan tu tim
        gt_boundaries (list of float): bien (giay) groundtruth tu file .lab
        filename (str): ten file tin hieu, dung lam tieu de figure
        rmse_ms (float, optional): RMSE (ms) de ghi chu tren figure
        mae_ms (float, optional): MAE (ms) de ghi chu tren figure

    Output:
        fig (matplotlib.figure.Figure): doi tuong figure da ve, de main.py
            tu quyet dinh show() hoac save
    """
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])

    ax_energy_t = fig.add_subplot(gs[0, 0])
    ax_energy_hist = fig.add_subplot(gs[0, 1])
    ax_centroid_t = fig.add_subplot(gs[1, 0])
    ax_centroid_hist = fig.add_subplot(gs[1, 1])
    ax_wave = fig.add_subplot(gs[2, :])

    frame_times = get_frame_times(len(energy), frame_length_ms=50.0)

    # --- (1) Short-Time Energy theo thoi gian + nguong T1 ---
    ax_energy_t.plot(frame_times, energy, color="tab:blue", label="STE")
    ax_energy_t.axhline(energy_threshold, color="black", linestyle="--", label="Ngưỡng T1")
    ax_energy_t.set_title("Short-Time Energy theo thời gian")
    ax_energy_t.set_xlabel("Thời gian (s)")
    ax_energy_t.set_ylabel("Năng lượng")
    ax_energy_t.legend(fontsize=8, loc="upper right")

    # --- (2) Histogram cua STE + M1/M2/T1 ---
    _plot_histogram_with_threshold(
        ax_energy_hist, energy_debug_info, energy_threshold,
        xlabel="Năng lượng (STE)", bar_color="tab:blue",
    )
    ax_energy_hist.set_title("Histogram STE và ngưỡng T1")

    # --- (3) Spectral Centroid theo thoi gian + nguong T2 ---
    ax_centroid_t.plot(frame_times, centroid, color="tab:orange", label="Spectral Centroid")
    ax_centroid_t.axhline(centroid_threshold, color="black", linestyle="--", label="Ngưỡng T2")
    ax_centroid_t.set_title("Spectral Centroid theo thời gian")
    ax_centroid_t.set_xlabel("Thời gian (s)")
    ax_centroid_t.set_ylabel("Centroid (chỉ số k)")
    ax_centroid_t.legend(fontsize=8, loc="upper right")

    # --- (4) Histogram cua Spectral Centroid + M1/M2/T2 ---
    _plot_histogram_with_threshold(
        ax_centroid_hist, centroid_debug_info, centroid_threshold,
        xlabel="Spectral Centroid (chỉ số k)", bar_color="tab:orange",
    )
    ax_centroid_hist.set_title("Histogram Centroid và ngưỡng T2")

    # --- (5) Dang song tin hieu + bien du doan (xanh) / groundtruth (do) ---
    time_axis = np.arange(len(signal)) / fs
    ax_wave.plot(time_axis, signal, color="gray", linewidth=0.5)
    for b in pred_boundaries:
        ax_wave.axvline(b, color="blue", linestyle="-", linewidth=1.2)
    for b in gt_boundaries:
        ax_wave.axvline(b, color="red", linestyle="--", linewidth=1.2)
    ax_wave.set_title("Tín hiệu + biên (xanh=thuật toán, đỏ=groundtruth)")
    ax_wave.set_xlabel("Thời gian (s)")
    ax_wave.set_ylabel("Biên độ")

    # --- Tieu de tong the: ten file + RMSE/MAE neu co ---
    title = filename
    if rmse_ms is not None and mae_ms is not None and not np.isnan(rmse_ms):
        title += f"  |  RMSE = {rmse_ms:.1f} ms, MAE = {mae_ms:.1f} ms"
    fig.suptitle(title)
    fig.tight_layout()

    return fig
