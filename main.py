#!/usr/bin/env python3
"""
main.py – File thực thi chính của chương trình phân đoạn Tiếng nói / Khoảng lặng (Speech / Silence).

Đáp ứng 100% các tiêu chí trong tài liệu "Hướng dẫn trình bày slide và nộp bài thi.pdf":
    1. Điểm khởi chạy duy nhất: Người dùng/Giáo viên chỉ cần bấm Run chạy 1 lần duy nhất từ file main.py.
    2. Áp dụng Global Threshold: Tự động nạp ngưỡng tối ưu toàn cục tìm được từ quá trình huấn luyện (train.py).
       (Nếu chưa có file ngưỡng, chương trình sẽ tự động kích hoạt huấn luyện để sinh ra ngưỡng).
    3. Xử lý tập kiểm thử: Quét toàn bộ các file trong thư mục TinHieuKiemThu/.
       (Nếu thư mục TinHieuKiemThu/ đang trống trước buổi thi, tự động kích hoạt chế độ Demo trên 4 file TinHieuHuanLuyen/).
    4. Xuất kết quả 4 file trên 4 Figure riêng biệt:
       - Kết quả cuối cùng: Dạng sóng (Waveform), vạch ranh giới chuẩn Ground Truth (đỏ),
         ranh giới thuật toán dự đoán (xanh), các chỉ số định lượng (MAE, RMSE, Precision, Recall, F1).
       - Kết quả trung gian: Hàm đặc trưng ngắn hạn (logMA) xếp chồng cùng trục thời gian và vạch ngưỡng T.
       - Mỗi plot có đầy đủ Title và Axis labels.
    5. Hiển thị và sắp xếp 4 góc màn hình: Tự động mở 4 cửa sổ Figure và định vị vào 4 góc màn hình:
       Top-Left, Top-Right, Bottom-Left, Bottom-Right để Thầy/Cô quan sát trực quan đồng thời.
    6. Lưu trữ và in ấn: Xuất bảng tổng kết ra terminal và lưu file CSV + ảnh figures vào thư mục output/.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Đảm bảo đường dẫn gốc được ưu tiên trong sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    FEATURE_TYPE, ORIGINAL_TRAINING_DIR, OUTPUT_DIR, TEST_DIR,
)
from src.features import compute_short_time_feature
from src.io_utils import load_audio, load_ground_truth, parse_lab_file
from src.plotting import (
    plot_test_evaluation, position_figures_4_corners, setup_matplotlib_backend,
)
from src.segmentation import (
    classify_frames, compute_boundary_errors, extract_boundaries,
    frames_to_segments, remove_short_silence_segments,
)
from train import run_training_pipeline

# Thiết lập hệ thống ghi log
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_or_create_global_threshold(
    threshold_file: Optional[str] = None,
    output_dir: str = OUTPUT_DIR,
) -> float:
    """
    Nạp ngưỡng tối ưu toàn cục (Global Threshold) từ file JSON.

    Nếu file cấu hình chưa tồn tại, tự động kích hoạt hàm huấn luyện để tạo mới ngưỡng tối ưu.

    Args:
        threshold_file (Optional[str], optional): Đường dẫn tới file JSON lưu ngưỡng. Mặc định None.
        output_dir (str, optional): Thư mục output chứa thư mục con training. Mặc định OUTPUT_DIR.

    Returns:
        float: Giá trị ngưỡng tối ưu toàn cục (Global Threshold T).
    """
    # Bước 1: Xác định đường dẫn file JSON cấu hình
    if threshold_file is None:
        threshold_file = os.path.join(output_dir, "training", "global_threshold.json")

    # Bước 2: Kiểm tra sự tồn tại của file, tự động train nếu chưa có
    if not os.path.exists(threshold_file):
        logger.warning("Chưa tìm thấy file ngưỡng tại %s.", threshold_file)
        logger.info("Tự động kích hoạt pipeline huấn luyện (train.py) để sinh Global Threshold tối ưu...")
        run_training_pipeline(training_dir=ORIGINAL_TRAINING_DIR, output_base_dir=output_dir)

    # Bước 3: Đọc file JSON và lấy giá trị threshold
    with open(threshold_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        threshold = float(data["threshold"])
        logger.info("Nạp thành công Global Threshold: T = %.8f (từ %s)", threshold, threshold_file)

    return threshold


def run_evaluation(
    test_dir: str = TEST_DIR,
    output_dir: str = OUTPUT_DIR,
    threshold_file: Optional[str] = None,
    custom_threshold: Optional[float] = None,
    demo_mode: bool = False,
    show_gui: bool = True,
) -> pd.DataFrame:
    """
    Hàm thực thi kiểm thử và hiển thị 4 cửa sổ đồ thị trên 4 góc màn hình.

    Quy trình:
    1. Nạp ngưỡng tối ưu toàn cục.
    2. Quét dữ liệu file âm thanh (nếu TinHieuKiemThu trống thì tự động chuyển sang chế độ Demo trên TinHieuHuanLuyen).
    3. Xử lý từng file: tính đặc trưng, phân đoạn theo ngưỡng, lọc silence ngắn, so khớp với ground truth nếu có.
    4. Khởi tạo 4 Figure, lưu ảnh ra đĩa và giữ mở cửa sổ.
    5. Tự động định vị 4 Figure vào 4 góc màn hình (Top-Left, Top-Right, Bottom-Left, Bottom-Right).
    6. In bảng tổng kết chỉ số định lượng ra terminal và hiển thị đồng thời 4 cửa sổ lên màn hình.

    Args:
        test_dir (str, optional): Thư mục chứa file âm thanh kiểm thử. Mặc định TEST_DIR.
        output_dir (str, optional): Thư mục lưu kết quả kiểm thử. Mặc định OUTPUT_DIR.
        threshold_file (Optional[str], optional): Đường dẫn file JSON chứa ngưỡng.
        custom_threshold (Optional[float], optional): Giá trị ngưỡng chỉ định thủ công (nếu muốn ghi đè).
        demo_mode (bool, optional): Cưỡng chế chạy demo trên tập huấn luyện. Mặc định False.
        show_gui (bool, optional): Có hiển thị 4 cửa sổ lên màn hình hay không. Mặc định True.

    Returns:
        pd.DataFrame: Bảng tổng hợp các chỉ số đánh giá (MAE, RMSE, Precision, Recall, F1).
    """
    # Bước 1: Khởi tạo thư mục đầu ra
    out_test_dir = os.path.join(output_dir, "test")
    fig_test_dir = os.path.join(out_test_dir, "figures")
    os.makedirs(fig_test_dir, exist_ok=True)

    # Bước 2: Xác định ngưỡng Threshold T áp dụng cho bài kiểm thử
    if custom_threshold is not None:
        threshold = float(custom_threshold)
        logger.info("Sử dụng ngưỡng chỉ định thủ công: T = %.8f", threshold)
    else:
        threshold = get_or_create_global_threshold(threshold_file, output_dir)

    # Bước 3: Quét tìm các file âm thanh kiểm thử .wav
    test_path = Path(test_dir)
    if not test_path.exists():
        os.makedirs(test_path, exist_ok=True)

    wav_files = sorted(test_path.glob("*.wav"))

    # Xử lý trường hợp thư mục test chưa có file (tự động chạy demo trên 4 file huấn luyện)
    if not wav_files or demo_mode:
        if not wav_files:
            logger.info("\n>>> THÔNG BÁO: Thư mục kiểm thử '%s' hiện đang trống (chưa có file test từ GV).", test_dir)
            logger.info(">>> Tự động chuyển sang chế độ DEMO trên 4 file mẫu tại '%s' để kiểm tra 4 cửa sổ...",
                        ORIGINAL_TRAINING_DIR)
        test_path = Path(ORIGINAL_TRAINING_DIR)
        wav_files = sorted(test_path.glob("*.wav"))

    logger.info("=" * 80)
    logger.info("BẮT ĐẦU CHẠY KIỂM THỬ PHÂN ĐOẠN SPEECH / SILENCE (%d FILES)", len(wav_files))
    logger.info("  Thư mục dữ liệu : %s", test_path)
    logger.info("  Global Threshold: %.8f", threshold)
    logger.info("  Thư mục output  : %s", out_test_dir)
    logger.info("=" * 80)

    figures: List[plt.Figure] = []
    summary_rows: List[Dict[str, Any]] = []

    # Bước 4: Vòng lặp xử lý từng file âm thanh
    for idx, wav_path in enumerate(wav_files, 1):
        basename = wav_path.stem
        lab_path = wav_path.with_suffix(".lab")
        logger.info("\n[%d/%d] Đang xử lý file: %s", idx, len(wav_files), wav_path.name)

        # Nạp tín hiệu âm thanh
        signal, sr = load_audio(str(wav_path))
        dur_s = len(signal) / sr

        # Trích xuất chuỗi đặc trưng ngắn hạn (logMA)
        fv, fc = compute_short_time_feature(signal, sr)

        # Áp dụng ngưỡng Global Threshold để phân loại khung
        pred_labels = classify_frames(fv, threshold)
        pred_segs   = frames_to_segments(pred_labels, fc, signal_duration_s=dur_s)

        # Hậu xử lý: Lọc bỏ các khoảng lặng ngắn hơn 300 ms theo quy định
        pred_after  = remove_short_silence_segments(pred_segs)

        # Kiểm tra sự tồn tại của nhãn chuẩn Ground Truth (.lab) để tính sai số
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
            logger.info("  Không có file .lab ground truth cho %s, chỉ vẽ phân đoạn dự đoán", basename)

        logger.info("  %s -> F1=%.2f | Precision=%.2f | Recall=%.2f | MAE=%.1f ms | RMSE=%.1f ms",
                    basename, err.get("boundary_f1", 0.0), err.get("boundary_precision", 0.0),
                    err.get("boundary_recall", 0.0), err.get("mae_ms", float("nan")),
                    err.get("rmse_ms", float("nan")))

        # Vẽ Figure kiểm thử chuẩn cho file hiện tại (giữ mở Figure nếu bật chế độ GUI)
        save_fig_path = os.path.join(fig_test_dir, f"{basename}_test_figure.png")
        fig = plot_test_evaluation(
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
            close_fig=not show_gui,
        )
        if show_gui:
            figures.append(fig)

        # Ghi nhận kết quả vào danh sách tổng hợp
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

    # Bước 5: Lưu bảng tổng kết số liệu ra file CSV
    df_summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(out_test_dir, "test_summary.csv")
    df_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("\nĐã lưu file tổng kết kết quả kiểm thử: %s", csv_path)

    # Bước 6: In bảng số liệu định lượng ra terminal cho Giáo viên quan sát
    print("\n" + "=" * 80)
    print(f"KẾT QUẢ PHÂN ĐOẠN SPEECH / SILENCE VỚI GLOBAL THRESHOLD T = {threshold:.8f}")
    print("=" * 80)
    cols = ["filename", "threshold", "precision", "recall", "f1", "mae_ms", "rmse_ms", "n_matched", "n_gt_boundaries"]
    print(df_summary[cols].to_string(index=False))
    print("=" * 80)
    print(f"Toàn bộ {len(wav_files)} Figures (ranh giới xanh dự đoán, ranh giới đỏ GT) đã lưu tại:\n  {fig_test_dir}\n")

    # Bước 7: Tự động sắp xếp 4 cửa sổ Figure vào 4 góc màn hình và hiển thị
    if show_gui and figures:
        logger.info("Đang tự động sắp xếp 4 cửa sổ Figure vào 4 góc màn hình...")
        position_figures_4_corners(figures)
        print(">>> 4 Cửa sổ Figure đã hiển thị tại 4 góc màn hình. Đóng cửa sổ để kết thúc chương trình.")
        plt.show()

    return df_summary


def main() -> None:
    """
    Hàm thực thi chính khi bấm Run chạy file main.py từ dòng lệnh hoặc IDE.

    Args:
        None (sử dụng sys.argv hoặc tham số mặc định).

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Chương trình chính phân đoạn Speech/Silence (Chạy 1 lần duy nhất từ main.py)."
    )
    parser.add_argument("--test-dir",       default=TEST_DIR,   help="Thư mục chứa file WAV/LAB kiểm thử")
    parser.add_argument("--output-dir",     default=OUTPUT_DIR, help="Thư mục lưu kết quả đầu ra")
    parser.add_argument("--threshold-file", default=None,       help="Đường dẫn file global_threshold.json")
    parser.add_argument("--threshold",      type=float, default=None, help="Giá trị threshold cụ thể (nếu muốn ghi đè)")
    parser.add_argument("--demo",           action="store_true", help="Chạy chế độ demo trên 4 file huấn luyện")
    parser.add_argument("--no-gui",         action="store_true", help="Chạy ngầm không hiển thị cửa sổ GUI")
    args = parser.parse_args()

    # Thiết lập backend Matplotlib tương tác để hiển thị cửa sổ lên màn hình (nếu không có cờ --no-gui)
    setup_matplotlib_backend(interactive=not args.no_gui)

    run_evaluation(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        threshold_file=args.threshold_file,
        custom_threshold=args.threshold,
        demo_mode=args.demo,
        show_gui=not args.no_gui,
    )


if __name__ == "__main__":
    main()
