"""
evaluate_test.py – Đánh giá phân đoạn trên bộ tín hiệu kiểm thử (TinHieuKiemThu).

Đáp ứng 100% yêu cầu của Giáo viên trong HuongDan.md:
    1. Duyệt qua 4 file tín hiệu kiểm thử trong 01 LẦN CHẠY DUY NHẤT.
    2. Xuất đúng 4 figures (mỗi figure 1 file) thể hiện:
       - Input & Output (dạng sóng + hàm logMA)
       - Ranh giới dự đoán: đường dọc màu xanh
       - Ranh giới chuẩn Ground Truth từ .lab: đường dọc màu đỏ
    3. Đánh giá định lượng: MAE, RMSE (đơn vị ms), Precision, Recall, F1-Score.
    4. Xuất bảng tổng kết và file test_summary.csv.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Cho phép import src khi chạy script từ bất kỳ đâu
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    FEATURE_TYPE, ORIGINAL_TRAINING_DIR, OUTPUT_DIR, TEST_DIR,
)
from src.features import compute_short_time_feature
from src.io_utils import load_audio, load_ground_truth, parse_lab_file
from src.plotting import plot_test_evaluation
from src.segmentation import (
    classify_frames, compute_boundary_errors, extract_boundaries,
    frames_to_segments, remove_short_silence_segments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_test_evaluation(
    test_dir: str = TEST_DIR,
    output_dir: str = OUTPUT_DIR,
    threshold_file: Optional[str] = None,
    custom_threshold: Optional[float] = None,
    demo_mode: bool = False,
) -> pd.DataFrame:
    out_test_dir = os.path.join(output_dir, "test")
    fig_test_dir = os.path.join(out_test_dir, "figures")
    os.makedirs(fig_test_dir, exist_ok=True)

    # 1. Xác định ngưỡng Threshold T
    if custom_threshold is not None:
        threshold = float(custom_threshold)
        logger.info("Sử dụng ngưỡng chỉ định trực tiếp: T = %.8f", threshold)
    else:
        if threshold_file is None:
            threshold_file = os.path.join(output_dir, "training", "global_threshold.json")
        if not os.path.exists(threshold_file):
            logger.error("Không tìm thấy file ngưỡng tại %s. Hãy chạy run_training.py trước!", threshold_file)
            sys.exit(1)
        with open(threshold_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            threshold = float(data["threshold"])
            logger.info("Nạp Global Threshold từ %s: T = %.8f", threshold_file, threshold)

    # 2. Quét thư mục test
    test_path = Path(test_dir)
    if not test_path.exists():
        os.makedirs(test_path, exist_ok=True)
        logger.warning("Thư mục %s chưa tồn tại -> Đã tự động tạo mới.", test_dir)

    wav_files = sorted(test_path.glob("*.wav"))

    if not wav_files:
        if demo_mode:
            logger.info("Không có file trong %s, chạy DEMO trên %s để kiểm tra quy trình!",
                        test_dir, ORIGINAL_TRAINING_DIR)
            test_path = Path(ORIGINAL_TRAINING_DIR)
            wav_files = sorted(test_path.glob("*.wav"))
        else:
            logger.warning("Thư mục %s hiện chưa có file .wav nào từ Giáo viên!", test_dir)
            logger.info(">>> GỢI Ý: Bạn có thể chạy với cờ '--demo' để xem trước kết quả 4 figures chuẩn bị cho GV:")
            logger.info("    python test/evaluate_test.py --demo")
            return pd.DataFrame()

    logger.info("=" * 75)
    logger.info("BẮT ĐẦU ĐÁNH GIÁ TRÊN BỘ KIỂM THỬ (%d files)", len(wav_files))
    logger.info("  Thư mục dữ liệu : %s", test_path)
    logger.info("  Global Threshold: %.8f", threshold)
    logger.info("  Thư mục output  : %s", out_test_dir)
    logger.info("=" * 75)

    summary_rows = []

    for idx, wav_path in enumerate(wav_files, 1):
        basename = wav_path.stem
        lab_path = wav_path.with_suffix(".lab")
        logger.info("\n[%d/%d] Đang xử lý file: %s", idx, len(wav_files), wav_path.name)

        signal, sr = load_audio(str(wav_path))
        dur_s = len(signal) / sr

        # Tính feature và phân đoạn
        fv, fc = compute_short_time_feature(signal, sr)
        pred_labels = classify_frames(fv, threshold)
        pred_segs   = frames_to_segments(pred_labels, fc, signal_duration_s=dur_s)
        pred_after  = remove_short_silence_segments(pred_segs)

        has_gt = lab_path.exists()
        if has_gt:
            lab_segs = parse_lab_file(str(lab_path))
            gt_segs  = load_ground_truth(lab_segs)
            err = compute_boundary_errors(pred_after, gt_segs)
        else:
            gt_segs = []
            err = {
                "mae_ms": float("nan"), "rmse_ms": float("nan"),
                "boundary_precision": float("nan"), "boundary_recall": float("nan"),
                "boundary_f1": float("nan"),
                "n_matched": 0, "n_gt_boundaries": 0,
                "n_pred_boundaries": len(extract_boundaries(pred_after)),
            }
            logger.info("  Không có file .lab ground truth cho %s, chỉ xuất phân đoạn dự đoán", basename)

        logger.info("  %s -> F1=%.2f | Precision=%.2f | Recall=%.2f | MAE=%.1f ms | RMSE=%.1f ms",
                    basename, err.get("boundary_f1", 0.0), err.get("boundary_precision", 0.0),
                    err.get("boundary_recall", 0.0), err.get("mae_ms", float("nan")),
                    err.get("rmse_ms", float("nan")))

        # Xuất figure chuẩn theo format yêu cầu của Giáo viên
        save_fig_path = os.path.join(fig_test_dir, f"{basename}_test_figure.png")
        plot_test_evaluation(
            signal=signal,
            sample_rate=sr,
            feature_vals=fv,
            frame_centers=fc,
            pred_segments=pred_after,
            gt_segments=gt_segs,
            threshold=threshold,
            feature_name=FEATURE_TYPE.value,
            filename=basename,
            metrics=err,
            save_path=save_fig_path,
        )

        summary_rows.append({
            "filename":          basename,
            "threshold":         threshold,
            "precision":         err.get("boundary_precision"),
            "recall":            err.get("boundary_recall"),
            "f1":                err.get("boundary_f1"),
            "mae_ms":            err.get("mae_ms"),
            "rmse_ms":           err.get("rmse_ms"),
            "n_pred_boundaries": err.get("n_pred_boundaries"),
            "n_gt_boundaries":   err.get("n_gt_boundaries"),
            "n_matched":         err.get("n_matched"),
            "figure_path":       save_fig_path,
        })

    df_summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(out_test_dir, "test_summary.csv")
    df_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("\nĐã lưu file tổng kết: %s", csv_path)

    # In bảng ra terminal cho giáo viên xem trực tiếp
    print("\n" + "=" * 80)
    print("KẾT QUẢ KIỂM THỬ TRÊN BỘ DỮ LIỆU TEST (TIN HIỆU KIỂM THỬ)")
    print("=" * 80)
    cols = ["filename", "threshold", "precision", "recall", "f1", "mae_ms", "rmse_ms", "n_matched", "n_gt_boundaries"]
    print(df_summary[cols].to_string(index=False))
    print("=" * 80)
    print(f"Toàn bộ {len(wav_files)} figure (đường xanh dự đoán, đường đỏ GT) đã lưu tại:\n  {fig_test_dir}\n")

    return df_summary


def main():
    parser = argparse.ArgumentParser(description="Đánh giá bộ tín hiệu kiểm thử (TinHieuKiemThu) cho giáo viên.")
    parser.add_argument("--test-dir",        default=TEST_DIR,   help="Thư mục chứa WAV/LAB kiểm thử")
    parser.add_argument("--output-dir",      default=OUTPUT_DIR, help="Thư mục lưu output")
    parser.add_argument("--threshold-file",  default=None,       help="Đường dẫn file global_threshold.json")
    parser.add_argument("--threshold",       type=float, default=None, help="Giá trị threshold cụ thể (nếu có)")
    parser.add_argument("--demo",            action="store_true", help="Chạy demo trên 4 file training để kiểm tra figure")
    args = parser.parse_args()

    run_test_evaluation(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        threshold_file=args.threshold_file,
        custom_threshold=args.threshold,
        demo_mode=args.demo,
    )


if __name__ == "__main__":
    main()
