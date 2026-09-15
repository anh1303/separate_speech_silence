#!/usr/bin/env python3
"""
train.py – Pipeline huấn luyện ngưỡng tối ưu toàn cục (Global Threshold) bằng Binary Search.

Quy trình thực hiện:
    1. Quét 4 file tín hiệu âm thanh và nhãn chuẩn (.wav + .lab) trong TinHieuHuanLuyen/.
    2. Phân khung và trích xuất đặc trưng ngắn hạn (mặc định logMA).
    3. Gán nhãn cấp độ khung (Speech / Silence) dựa trên file .lab.
    4. Tìm ngưỡng tối ưu cho từng file riêng biệt (Per-file Threshold) để tham khảo.
    5. Gộp phân phối của cả 4 file để huấn luyện Ngưỡng tối ưu dùng chung (Global Threshold T).
    6. Đánh giá kiểm thử lại Global Threshold trên từng file (MAE, RMSE, Precision, Recall, F1).
    7. Lưu trữ siêu dữ liệu (JSON, TXT), bảng số liệu (CSV) và xuất các đồ thị trực quan hóa.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Đảm bảo đường dẫn gốc được ưu tiên trong sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    BOUNDARY_MATCH_TOLERANCE_MS, FEATURE_TYPE, FRAME_LENGTH_MS,
    FRAME_SHIFT_MS, MIN_SILENCE_DURATION_MS, ORIGINAL_TRAINING_DIR,
    OUTPUT_DIR, BinarySearchState,
)
from src.features import assign_frame_labels, compute_short_time_feature
from src.io_utils import (
    load_audio, load_ground_truth, parse_lab_file,
    save_binary_search_history_csv, save_boundary_errors_csv,
    save_global_threshold_json, save_global_threshold_txt,
    save_threshold_summary_csv,
)
from src.plotting import (
    plot_before_after_filter, plot_binary_search_history,
    plot_feature, plot_feature_distribution, plot_final_segmentation,
    plot_signal_to_feature_with_overlap, plot_waveform_and_ground_truth,
    setup_matplotlib_backend,
)
from src.segmentation import (
    classify_frames, compute_boundary_errors, frames_to_segments,
    remove_short_silence_segments,
)
from src.threshold import find_optimal_threshold_binary_search

# Thiết lập hệ thống ghi log
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def process_training_file(
    wav_path: str,
    lab_path: str,
    output_dir: str,
) -> Optional[Dict[str, Any]]:
    """
    Xử lý trích xuất đặc trưng, huấn luyện ngưỡng riêng lẻ và đánh giá cho 1 file âm thanh huấn luyện.

    Args:
        wav_path (str): Đường dẫn đến file âm thanh .wav.
        lab_path (str): Đường dẫn đến file nhãn chuẩn .lab tương ứng.
        output_dir (str): Thư mục lưu các đồ thị của file.

    Returns:
        Optional[Dict[str, Any]]: Từ điển chứa tín hiệu, đặc trưng, nhãn và kết quả đánh giá (hoặc None nếu lỗi).
    """
    basename = Path(wav_path).stem
    logger.info("Đang xử lý file huấn luyện: %s", Path(wav_path).name)
    os.makedirs(output_dir, exist_ok=True)

    # Bước 1: Nạp tín hiệu âm thanh và kiểm tra thời lượng
    try:
        signal, sr = load_audio(wav_path)
    except Exception as e:
        logger.error("  Lỗi đọc file WAV %s: %s", wav_path, e)
        return None

    duration_s = len(signal) / sr

    # Bước 2: Nạp và phân tích file nhãn chuẩn Ground Truth .lab
    try:
        lab_segs = parse_lab_file(lab_path)
    except Exception as e:
        logger.error("  Lỗi đọc file .lab %s: %s", lab_path, e)
        return None

    gt_segments = load_ground_truth(lab_segs)

    # Bước 3: Tính toán chuỗi đặc trưng ngắn hạn theo khung thời gian
    feature_vals, frame_centers = compute_short_time_feature(signal, sr)
    if len(feature_vals) == 0:
        logger.error("  Không thể tính đặc trưng cho file: %s", basename)
        return None

    # Bước 4: Gán nhãn cho từng khung và tách phân phối Speech / Silence
    frame_labels = assign_frame_labels(frame_centers, lab_segs)
    speech_vals  = feature_vals[frame_labels == 1]
    silence_vals = feature_vals[frame_labels == 0]
    n_speech, n_sil = len(speech_vals), len(silence_vals)

    if n_speech == 0 or n_sil == 0:
        logger.warning("  %s: Thiếu khung Speech hoặc Silence để huấn luyện.", basename)
        return None

    # Bước 5: Tìm kiếm nhị phân ngưỡng tối ưu cho riêng file hiện tại (Per-file)
    try:
        threshold, history, overlap_info = find_optimal_threshold_binary_search(speech_vals, silence_vals)
    except Exception as e:
        logger.error("  Lỗi Binary search trên file %s: %s", basename, e)
        return None

    fig_dir = os.path.join(output_dir, "figures")
    feat_name = FEATURE_TYPE.value

    # Bước 6: Vẽ và lưu các đồ thị trực quan hóa quá trình huấn luyện per-file
    plot_waveform_and_ground_truth(
        signal, sr, gt_segments, basename,
        os.path.join(fig_dir, f"{basename}_fig1_waveform_gt.png"),
    )
    plot_feature(
        signal, sr, feature_vals, frame_centers, gt_segments, feat_name, basename,
        os.path.join(fig_dir, f"{basename}_fig2_feature.png"),
    )
    plot_signal_to_feature_with_overlap(
        signal, sr, feature_vals, frame_centers, overlap_info, threshold=None,
        gt_segments=gt_segments, feature_name=feat_name, title=basename,
        save_path=os.path.join(fig_dir, f"{basename}_fig2b_signal_to_feature_overlap.png"),
        overlap_label=f"{basename} Overlap",
    )
    plot_feature_distribution(
        speech_vals, silence_vals, overlap_info, threshold=None,
        feature_name=feat_name, title=basename,
        save_path=os.path.join(fig_dir, f"{basename}_fig3_distribution.png"),
    )
    plot_binary_search_history(
        history, basename,
        os.path.join(fig_dir, f"{basename}_fig4_binary_search.png"),
    )

    # Bước 7: Phân đoạn tín hiệu với ngưỡng riêng và đánh giá sai số
    pred_labels = classify_frames(feature_vals, threshold)
    pred_segs   = frames_to_segments(pred_labels, frame_centers, signal_duration_s=duration_s)
    pred_after  = remove_short_silence_segments(pred_segs)

    err = compute_boundary_errors(pred_after, gt_segments)
    logger.info("  Per-file T=%.6f -> F1=%.2f | P=%.2f | R=%.2f | MAE=%.1f ms | RMSE=%.1f ms",
                threshold, err["boundary_f1"], err["boundary_precision"],
                err["boundary_recall"], err["mae_ms"], err["rmse_ms"])

    plot_final_segmentation(
        signal, sr, feature_vals, frame_centers,
        pred_after, gt_segments, threshold, feat_name, basename,
        os.path.join(fig_dir, f"{basename}_fig5_final_segmentation.png"), metrics=err,
    )
    plot_before_after_filter(
        signal, sr, pred_segs, pred_after, basename,
        os.path.join(fig_dir, f"{basename}_fig6_before_after_filter.png"),
    )

    return {
        "basename": basename, "sr": sr, "signal": signal,
        "n_frames": len(feature_vals), "n_speech": n_speech, "n_silence": n_sil,
        "speech_vals": speech_vals, "silence_vals": silence_vals,
        "threshold": threshold, "history": history, "overlap_info": overlap_info,
        "gt_segments": gt_segments, "pred_segments": pred_segs, "pred_after_filter": pred_after,
        "error_metrics": err, "feature_vals": feature_vals, "frame_centers": frame_centers,
        "duration_s": duration_s,
    }


def train_global_threshold(
    all_speech: np.ndarray,
    all_silence: np.ndarray,
    output_dir: str,
) -> Tuple[float, List[BinarySearchState], Dict[str, Any]]:
    """
    Huấn luyện ngưỡng tối ưu toàn cục (Global Threshold) trên dữ liệu gộp của cả 4 file huấn luyện.

    Args:
        all_speech (np.ndarray): Mảng 1D float chứa toàn bộ khung Speech của các file.
        all_silence (np.ndarray): Mảng 1D float chứa toàn bộ khung Silence của các file.
        output_dir (str): Thư mục lưu trữ đồ thị huấn luyện toàn cục.

    Returns:
        Tuple[float, List[BinarySearchState], Dict[str, Any]]:
            - g_threshold (float): Ngưỡng tối ưu toàn cục.
            - g_history (List[BinarySearchState]): Lịch sử hội tụ Binary Search.
            - g_overlap (Dict[str, Any]): Thông tin vùng giao thoa gộp.
    """
    logger.info("\n" + "=" * 65)
    logger.info("TIẾN HÀNH HUẤN LUYỆN GLOBAL THRESHOLD TRÊN DỮ LIỆU GỘP (4 FILES)")
    logger.info("  Tổng số Speech frames: %d | Tổng số Silence frames: %d", len(all_speech), len(all_silence))
    logger.info("=" * 65)

    # Bước 1: Tìm kiếm nhị phân trên phân phối gộp
    g_threshold, g_history, g_overlap = find_optimal_threshold_binary_search(all_speech, all_silence)
    logger.info("  >>> KẾT QUẢ GLOBAL THRESHOLD T_original = %.8f <<<", g_threshold)

    # Bước 2: Vẽ đồ thị phân phối gộp trước khi train (chưa vẽ T global để thể hiện trạng thái ban đầu)
    fig_dir = os.path.join(output_dir, "figures")
    plot_feature_distribution(
        all_speech, all_silence, g_overlap, threshold=None,
        feature_name=FEATURE_TYPE.value, title="Global Training Set",
        save_path=os.path.join(fig_dir, "global_fig3_distribution.png"),
    )
    # Đồ thị phân phối sau khi train (có đường ngưỡng tối ưu T_global)
    plot_feature_distribution(
        all_speech, all_silence, g_overlap, threshold=g_threshold,
        feature_name=FEATURE_TYPE.value, title="Global Training Set (with T_global)",
        save_path=os.path.join(fig_dir, "global_fig3_distribution_with_threshold.png"),
    )
    plot_binary_search_history(
        g_history, "Global Training Set",
        os.path.join(fig_dir, "global_fig4_binary_search.png"),
    )

    return g_threshold, g_history, g_overlap


def evaluate_with_global_threshold(
    file_results: List[Dict[str, Any]],
    g_threshold: float,
    output_dir: str,
    g_overlap: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Đánh giá hiệu năng của Global Threshold dùng chung trên từng file âm thanh huấn luyện.

    Args:
        file_results (List[Dict[str, Any]]): Danh sách kết quả xử lý của các file.
        g_threshold (float): Ngưỡng toàn cục cần đánh giá.
        output_dir (str): Thư mục lưu đồ thị đánh giá.
        g_overlap (Optional[Dict[str, Any]], optional): Vùng overlap toàn cục.

    Returns:
        List[Dict[str, Any]]: Danh sách kết quả định lượng chi tiết của từng file.
    """
    logger.info("\n" + "=" * 65)
    logger.info("ĐÁNH GIÁ GLOBAL THRESHOLD TRÊN TỪNG FILE HUẤN LUYỆN")
    logger.info("=" * 65)

    fig_dir = os.path.join(output_dir, "figures")
    feat_name = FEATURE_TYPE.value
    results = []

    # Bước 1: Quét từng file và áp dụng Global Threshold
    for res in file_results:
        if res is None or res.get("feature_vals") is None:
            continue

        name  = res["basename"]
        fv    = res["feature_vals"]
        fc    = res["frame_centers"]
        gt    = res["gt_segments"]
        sig   = res["signal"]
        sr    = res["sr"]
        dur_s = res["duration_s"]

        # Phân loại khung và lọc khoảng lặng ngắn
        pred_labels = classify_frames(fv, g_threshold)
        pred_segs   = frames_to_segments(pred_labels, fc, signal_duration_s=dur_s)
        pred_after  = remove_short_silence_segments(pred_segs)
        err         = compute_boundary_errors(pred_after, gt)

        logger.info("  %-12s (T=%.6f) -> F1=%.2f | P=%.2f | R=%.2f | MAE=%.1f ms | Matched=%d/%d",
                    name, g_threshold, err["boundary_f1"], err["boundary_precision"],
                    err["boundary_recall"], err.get("mae_ms", float("nan")),
                    err["n_matched"], err["n_gt_boundaries"])

        # Vẽ đồ thị phân đoạn cuối cùng theo Global Threshold
        plot_final_segmentation(
            sig, sr, fv, fc, pred_after, gt, g_threshold,
            feat_name, f"{name}_global",
            os.path.join(fig_dir, f"{name}_global_fig5_segmentation.png"), metrics=err,
        )

        plot_signal_to_feature_with_overlap(
            sig, sr, fv, fc, res["overlap_info"], g_threshold,
            gt, feat_name, f"{name}_global",
            os.path.join(fig_dir, f"{name}_global_fig2b_overlap.png"),
            overlap_label=f"Per-file Overlap ({name})",
            threshold_label="Global T",
        )

        # Vẽ đồ thị so sánh trước và sau khi lọc silence < 300ms với Global Threshold
        plot_before_after_filter(
            sig, sr, pred_segs, pred_after, f"{name}_global",
            os.path.join(fig_dir, f"{name}_global_fig6_before_after_filter.png"),
        )

        results.append({
            "basename": name, "global_threshold": g_threshold,
            "pred_after_filter": pred_after, "error_metrics": err,
        })

    return results


def run_training_pipeline(
    training_dir: str = ORIGINAL_TRAINING_DIR,
    output_base_dir: str = OUTPUT_DIR,
) -> float:
    """
    Hàm thực thi toàn bộ pipeline huấn luyện: đọc file, tính per-file, tìm global threshold và xuất báo cáo.

    Args:
        training_dir (str, optional): Thư mục chứa 4 file huấn luyện .wav và .lab. Mặc định ORIGINAL_TRAINING_DIR.
        output_base_dir (str, optional): Thư mục gốc lưu trữ toàn bộ output. Mặc định OUTPUT_DIR.

    Returns:
        float: Giá trị Global Threshold tối ưu tìm được.
    """
    # Bước 1: Kiểm tra thư mục dữ liệu huấn luyện
    train_path = Path(training_dir)
    if not train_path.exists():
        logger.error("Thư mục huấn luyện không tồn tại: %s", training_dir)
        sys.exit(1)

    wav_files = sorted(train_path.glob("*.wav"))
    if not wav_files:
        logger.error("Không tìm thấy file WAV nào trong %s", training_dir)
        sys.exit(1)

    logger.info("Tìm thấy %d file WAV trong %s", len(wav_files), training_dir)

    # Bước 2: Tạo các thư mục lưu trữ kết quả đầu ra
    out_training = os.path.join(output_base_dir, "training")
    out_per_file = os.path.join(output_base_dir, "per_file")
    out_final    = os.path.join(output_base_dir, "final")
    for d in [out_training, out_per_file, out_final]:
        os.makedirs(d, exist_ok=True)

    # Bước 3: Huấn luyện và phân tích per-file cho từng file âm thanh
    file_results: List[Optional[Dict[str, Any]]] = []
    for wav_path in wav_files:
        lab_path = wav_path.with_suffix(".lab")
        if not lab_path.exists():
            logger.warning("Không tìm thấy file .lab cho %s -> Bỏ qua", wav_path.name)
            continue
        res = process_training_file(str(wav_path), str(lab_path), os.path.join(out_per_file, wav_path.stem))
        file_results.append(res)

    valid = [r for r in file_results if r is not None]
    if not valid:
        logger.error("Không có file âm thanh nào được xử lý thành công!")
        sys.exit(1)

    # Bước 4: Gộp phân phối và tìm Global Threshold tối ưu
    all_speech  = np.concatenate([r["speech_vals"]  for r in valid])
    all_silence = np.concatenate([r["silence_vals"] for r in valid])
    g_thresh, g_history, g_overlap = train_global_threshold(all_speech, all_silence, out_training)

    # Bước 5: Đánh giá kiểm nghiệm Global Threshold trên toàn bộ tập dữ liệu
    eval_results = evaluate_with_global_threshold(valid, g_thresh, out_final, g_overlap)

    # Bước 6: Lưu trữ siêu dữ liệu và bảng số liệu
    save_global_threshold_json(
        threshold=g_thresh, output_dir=out_training,
        training_data=training_dir, threshold_method="binary_search",
        noise_condition="original",
    )
    save_global_threshold_txt(g_thresh, out_training)
    df_pf, df_gl = save_threshold_summary_csv(valid, g_thresh, out_training, eval_results)
    save_binary_search_history_csv(valid, g_history, out_training)
    save_boundary_errors_csv(eval_results, out_training)

    # Bước 7: In bảng số liệu ra terminal
    print("\n" + "=" * 80)
    print(f"BẢNG ĐÁNH GIÁ GLOBAL THRESHOLD = {g_thresh:.8f} (DỮ LIỆU HUẤN LUYỆN GỐC)")
    print("=" * 80)
    cols = ["filename", "global_threshold", "precision", "recall", "f1", "mae_ms", "rmse_ms", "n_matched", "n_gt_boundaries"]
    print(df_gl[cols].to_string(index=False))
    print("=" * 80)

    return g_thresh


def main() -> None:
    """
    Hàm thực thi chính khi chạy train.py từ dòng lệnh terminal.

    Args:
        None (sử dụng sys.argv hoặc tham số mặc định).

    Returns:
        None
    """
    # Khi chạy huấn luyện hàng loạt, dùng Agg để xuất file nhanh
    setup_matplotlib_backend(interactive=False)

    parser = argparse.ArgumentParser(description="Huấn luyện ngưỡng tối ưu Binary Search trên tập TinHieuHuanLuyen.")
    parser.add_argument("--training-dir", default=ORIGINAL_TRAINING_DIR, help="Thư mục chứa file WAV/LAB huấn luyện")
    parser.add_argument("--output-dir",   default=OUTPUT_DIR,            help="Thư mục lưu trữ kết quả đầu ra")
    args = parser.parse_args()

    run_training_pipeline(training_dir=args.training_dir, output_base_dir=args.output_dir)


if __name__ == "__main__":
    main()
