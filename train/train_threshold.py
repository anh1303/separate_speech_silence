"""
train_threshold.py – Pipeline huấn luyện Binary Search trên tập dữ liệu gốc (Original Training Set).

Quy trình:
    1. Quét các file WAV + LAB trong TinHieuHuanLuyen/
    2. Rút trích đặc trưng ngắn hạn (mặc định logMA)
    3. Gán nhãn frame-level dựa trên LAB
    4. Tìm ngưỡng Per-file cho từng file
    5. Gộp toàn bộ frame để tìm Global Threshold T_original
    6. Đánh giá Global Threshold trên từng file (MAE, RMSE, Precision, Recall, F1)
    7. Lưu JSON, TXT, CSV và xuất các figures trực quan hóa.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Cho phép import src khi chạy script từ bất kỳ đâu
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BOUNDARY_MATCH_TOLERANCE_MS, FEATURE_TYPE, FRAME_LENGTH_MS,
    FRAME_SHIFT_MS, MIN_SILENCE_DURATION_MS, ORIGINAL_TRAINING_DIR,
    OUTPUT_DIR,
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
)
from src.segmentation import (
    classify_frames, compute_boundary_errors, frames_to_segments,
    remove_short_silence_segments,
)
from src.threshold import find_optimal_threshold_binary_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def process_training_file(
    wav_path: str, lab_path: str, output_dir: str,
) -> Optional[Dict[str, Any]]:
    basename = Path(wav_path).stem
    logger.info("Processing: %s", Path(wav_path).name)
    os.makedirs(output_dir, exist_ok=True)

    try:
        signal, sr = load_audio(wav_path)
    except Exception as e:
        logger.error("  Lỗi WAV %s: %s", wav_path, e); return None

    duration_s = len(signal) / sr

    try:
        lab_segs = parse_lab_file(lab_path)
    except Exception as e:
        logger.error("  Lỗi .lab %s: %s", lab_path, e); return None

    gt_segments = load_ground_truth(lab_segs)

    feature_vals, frame_centers = compute_short_time_feature(signal, sr)
    if len(feature_vals) == 0:
        logger.error("  Không tính được feature cho %s", basename); return None

    frame_labels = assign_frame_labels(frame_centers, lab_segs)
    speech_vals  = feature_vals[frame_labels == 1]
    silence_vals = feature_vals[frame_labels == 0]
    n_speech, n_sil = len(speech_vals), len(silence_vals)

    if n_speech == 0 or n_sil == 0:
        logger.warning("  %s: Thiếu speech hoặc silence frame", basename)
        return None

    try:
        threshold, history, overlap_info = find_optimal_threshold_binary_search(speech_vals, silence_vals)
    except Exception as e:
        logger.error("  Lỗi binary search %s: %s", basename, e); return None

    fig_dir   = os.path.join(output_dir, "figures")
    feat_name = FEATURE_TYPE.value

    # Figures per-file
    plot_waveform_and_ground_truth(signal, sr, gt_segments, basename,
        os.path.join(fig_dir, f"{basename}_fig1_waveform_gt.png"))
    plot_feature(signal, sr, feature_vals, frame_centers, gt_segments, feat_name, basename,
        os.path.join(fig_dir, f"{basename}_fig2_feature.png"))
    plot_signal_to_feature_with_overlap(
        signal, sr, feature_vals, frame_centers, overlap_info, threshold=None,
        gt_segments=gt_segments, feature_name=feat_name, title=basename,
        save_path=os.path.join(fig_dir, f"{basename}_fig2b_signal_to_feature_overlap.png"),
        overlap_label=f"{basename} Overlap")
    plot_feature_distribution(speech_vals, silence_vals, overlap_info, threshold=None,
        feature_name=feat_name, title=basename,
        save_path=os.path.join(fig_dir, f"{basename}_fig3_distribution.png"))
    plot_binary_search_history(history, basename,
        os.path.join(fig_dir, f"{basename}_fig4_binary_search.png"))

    # Phân đoạn per-file
    pred_labels = classify_frames(feature_vals, threshold)
    pred_segs   = frames_to_segments(pred_labels, frame_centers, signal_duration_s=duration_s)
    pred_after  = remove_short_silence_segments(pred_segs)

    err = compute_boundary_errors(pred_after, gt_segments)
    logger.info("  Per-file T=%.6f -> F1=%.2f | P=%.2f | R=%.2f | MAE=%.1f ms | RMSE=%.1f ms",
                threshold, err["boundary_f1"], err["boundary_precision"],
                err["boundary_recall"], err["mae_ms"], err["rmse_ms"])

    plot_final_segmentation(signal, sr, feature_vals, frame_centers,
        pred_after, gt_segments, threshold, feat_name, basename,
        os.path.join(fig_dir, f"{basename}_fig5_final_segmentation.png"), metrics=err)
    plot_before_after_filter(signal, sr, pred_segs, pred_after, basename,
        os.path.join(fig_dir, f"{basename}_fig6_before_after_filter.png"))

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
    all_speech: np.ndarray, all_silence: np.ndarray, output_dir: str,
):
    logger.info("\n" + "=" * 60)
    logger.info("HUẤN LUYỆN GLOBAL THRESHOLD TRÊN DỮ LIỆU GỘP")
    logger.info("  Speech frames=%d | Silence frames=%d", len(all_speech), len(all_silence))
    logger.info("=" * 60)

    g_threshold, g_history, g_overlap = find_optimal_threshold_binary_search(all_speech, all_silence)
    logger.info("  >>> OPTIMAL GLOBAL THRESHOLD T_original = %.8f <<<", g_threshold)

    fig_dir = os.path.join(output_dir, "figures")
    plot_feature_distribution(all_speech, all_silence, g_overlap, g_threshold,
        FEATURE_TYPE.value, "Global Training Set",
        os.path.join(fig_dir, "global_fig3_distribution.png"))
    plot_binary_search_history(g_history, "Global Training Set",
        os.path.join(fig_dir, "global_fig4_binary_search.png"))

    return g_threshold, g_history, g_overlap


def evaluate_with_global_threshold(
    file_results: List[Dict[str, Any]], g_threshold: float, output_dir: str,
    g_overlap: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    logger.info("\n" + "=" * 60)
    logger.info("ĐÁNH GIÁ GLOBAL THRESHOLD TRÊN TỪNG FILE TRAINING")
    logger.info("=" * 60)

    fig_dir   = os.path.join(output_dir, "figures")
    feat_name = FEATURE_TYPE.value
    results   = []

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

        pred_labels = classify_frames(fv, g_threshold)
        pred_segs   = frames_to_segments(pred_labels, fc, signal_duration_s=dur_s)
        pred_after  = remove_short_silence_segments(pred_segs)
        err         = compute_boundary_errors(pred_after, gt)

        logger.info("  %-12s (T=%.6f) -> F1=%.2f | P=%.2f | R=%.2f | MAE=%.1f ms | Matched=%d/%d",
                    name, g_threshold, err["boundary_f1"], err["boundary_precision"],
                    err["boundary_recall"], err.get("mae_ms", float("nan")),
                    err["n_matched"], err["n_gt_boundaries"])

        plot_final_segmentation(sig, sr, fv, fc, pred_after, gt, g_threshold,
            feat_name, f"{name}_global",
            os.path.join(fig_dir, f"{name}_global_fig5_segmentation.png"), metrics=err)

        # Fig 2b minh hoạ Global T vs Overlap của file đó
        plot_signal_to_feature_with_overlap(
            sig, sr, fv, fc, res["overlap_info"], g_threshold,
            gt, feat_name, f"{name}_global",
            os.path.join(fig_dir, f"{name}_global_fig2b_overlap.png"),
            overlap_label=f"Per-file Overlap ({name})",
            threshold_label="Global T")

        results.append({
            "basename": name, "global_threshold": g_threshold,
            "pred_after_filter": pred_after, "error_metrics": err,
        })
    return results


def run_training_pipeline(
    training_dir: str = ORIGINAL_TRAINING_DIR,
    output_base_dir: str = OUTPUT_DIR,
) -> float:
    train_path = Path(training_dir)
    if not train_path.exists():
        logger.error("Thư mục training không tồn tại: %s", training_dir); sys.exit(1)

    wav_files = sorted(train_path.glob("*.wav"))
    if not wav_files:
        logger.error("Không tìm thấy file WAV nào trong %s", training_dir); sys.exit(1)

    logger.info("Tìm thấy %d file WAV trong %s", len(wav_files), training_dir)

    out_training = os.path.join(output_base_dir, "training")
    out_per_file = os.path.join(output_base_dir, "per_file")
    out_final    = os.path.join(output_base_dir, "final")
    for d in [out_training, out_per_file, out_final]:
        os.makedirs(d, exist_ok=True)

    # 1. Per-file
    file_results: List[Optional[Dict[str, Any]]] = []
    for wav_path in wav_files:
        lab_path = wav_path.with_suffix(".lab")
        if not lab_path.exists():
            logger.warning("Không có .lab cho %s -> bỏ qua", wav_path.name)
            continue
        res = process_training_file(str(wav_path), str(lab_path), os.path.join(out_per_file, wav_path.stem))
        file_results.append(res)

    valid = [r for r in file_results if r is not None]
    if not valid:
        logger.error("Không có file nào xử lý thành công!"); sys.exit(1)

    # 2. Global
    all_speech  = np.concatenate([r["speech_vals"]  for r in valid])
    all_silence = np.concatenate([r["silence_vals"] for r in valid])
    g_thresh, g_history, g_overlap = train_global_threshold(all_speech, all_silence, out_training)

    # 3. Evaluate Global
    eval_results = evaluate_with_global_threshold(valid, g_thresh, out_final, g_overlap)

    # 4. Save artifacts
    save_global_threshold_json(
        threshold=g_thresh, output_dir=out_training,
        training_data=training_dir, threshold_method="binary_search",
        noise_condition="original",
    )
    save_global_threshold_txt(g_thresh, out_training)
    df_pf, df_gl = save_threshold_summary_csv(valid, g_thresh, out_training, eval_results)
    save_binary_search_history_csv(valid, g_history, out_training)
    save_boundary_errors_csv(eval_results, out_training)

    # In bảng đẹp ra terminal
    print("\n" + "=" * 80)
    print(f"BẢNG ĐÁNH GIÁ GLOBAL THRESHOLD = {g_thresh:.8f} (ORIGINAL TRAINING DATA)")
    print("=" * 80)
    cols = ["filename", "global_threshold", "precision", "recall", "f1", "mae_ms", "rmse_ms", "n_matched", "n_gt_boundaries"]
    print(df_gl[cols].to_string(index=False))
    print("=" * 80)

    return g_thresh


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Binary Search trên dữ liệu gốc.")
    parser.add_argument("--training-dir", default=ORIGINAL_TRAINING_DIR, help="Thư mục training gốc")
    parser.add_argument("--output-dir",   default=OUTPUT_DIR,            help="Thư mục output chính")
    args = parser.parse_args()

    run_training_pipeline(training_dir=args.training_dir, output_base_dir=args.output_dir)


if __name__ == "__main__":
    main()
