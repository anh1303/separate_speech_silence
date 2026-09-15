"""
threshold.py – Module xác định ngưỡng tối ưu phân tách Speech/Silence bằng Binary Search.

Dựa trên thuật toán của Hodgkinson (2012, Mục 2.1):
- Xác định vùng giao nhau (overlap region) giữa phân phối đặc trưng của Tiếng nói và Khoảng lặng.
- Dùng tìm kiếm nhị phân (Binary Search) để cân bằng độ nhầm lẫn (confusion) giữa hai lớp.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from src.config import (
        MAX_ITERATIONS, THRESHOLD_TOLERANCE,
        BinarySearchState,
    )
except ImportError:
    from config import (
        MAX_ITERATIONS, THRESHOLD_TOLERANCE,
        BinarySearchState,
    )

logger = logging.getLogger(__name__)


def find_overlap_region(
    speech_values:  np.ndarray,
    silence_values: np.ndarray,
) -> Dict[str, Any]:
    """
    Xác định vùng giao thoa (overlap region) giữa phân phối đặc trưng của Speech và Silence.

    Công thức giới hạn vùng overlap:
        overlap_min = max(min(Speech), min(Silence))
        overlap_max = min(max(Speech), max(Silence))

    Chỉ các khung có giá trị đặc trưng nằm trong khoảng [overlap_min, overlap_max] mới gây ra
    sự mơ hồ khi phân loại và cần được tối ưu hóa bằng Binary Search.

    Args:
        speech_values (np.ndarray): Mảng 1D float chứa giá trị đặc trưng của các khung tiếng nói.
        silence_values (np.ndarray): Mảng 1D float chứa giá trị đặc trưng của các khung khoảng lặng.

    Returns:
        Dict[str, Any]: Từ điển chứa các thông tin về vùng overlap:
            - "overlap_min" (float): Cận dưới của vùng giao nhau.
            - "overlap_max" (float): Cận trên của vùng giao nhau.
            - "has_overlap" (bool): True nếu tồn tại vùng giao thoa hợp lệ (min < max).
            - "speech_overlap_vals" (np.ndarray): Các giá trị speech nằm trong vùng overlap.
            - "silence_overlap_vals" (np.ndarray): Các giá trị silence nằm trong vùng overlap.
            - "speech_all_vals" (np.ndarray): Toàn bộ mảng speech ban đầu.
            - "silence_all_vals" (np.ndarray): Toàn bộ mảng silence ban đầu.
    """
    # Bước 1: Kiểm tra điều kiện dữ liệu đầu vào không được rỗng
    if len(speech_values) == 0 or len(silence_values) == 0:
        raise ValueError("Yêu cầu tối thiểu 1 frame Speech và 1 frame Silence để tìm vùng overlap.")

    # Bước 2: Tìm giá trị cực tiểu và cực đại của từng phân phối đặc trưng
    min_s, max_s   = float(np.min(speech_values)),  float(np.max(speech_values))
    min_sl, max_sl = float(np.min(silence_values)), float(np.max(silence_values))

    # Bước 3: Xác định giao điểm của hai khoảng giá trị
    ov_min = max(min_s, min_sl)
    ov_max = min(max_s, max_sl)
    has_overlap = ov_min < ov_max

    # Bước 4: Xử lý trường hợp đặc biệt khi hai phân phối tách biệt hoàn toàn (không giao nhau)
    if not has_overlap:
        logger.warning(
            "  Hai phân phối không overlap: Speech=[%.4f, %.4f], Silence=[%.4f, %.4f]",
            min_s, max_s, min_sl, max_sl,
        )
        # Lấy trung điểm giữa cận trên của phân phối thấp hơn và cận dưới của phân phối cao hơn
        ov_min = ov_max = (min(max_s, max_sl) + max(min_s, min_sl)) / 2.0
        logger.warning("  Sử dụng giá trị trung điểm làm ngưỡng: %.6f", ov_min)

    # Bước 5: Lọc các mẫu rơi vào bên trong vùng overlap phục vụ tìm kiếm nhị phân
    spk_ov = speech_values[(speech_values >= ov_min) & (speech_values <= ov_max)]
    sil_ov = silence_values[(silence_values >= ov_min) & (silence_values <= ov_max)]

    return {
        "overlap_min":          ov_min,
        "overlap_max":          ov_max,
        "has_overlap":          has_overlap,
        "speech_overlap_vals":  spk_ov,
        "silence_overlap_vals": sil_ov,
        "speech_all_vals":      speech_values,
        "silence_all_vals":     silence_values,
    }


def compute_confusion(
    silence_vals: np.ndarray,
    speech_vals:  np.ndarray,
    T: float,
) -> Tuple[float, float, float]:
    """
    Tính toán hàm độ nhầm lẫn (confusion) cho Silence và Speech tại ngưỡng T (Hodgkinson 2012).

    Công thức:
        C_sil   = mean(max(silence - T, 0))  -> Độ nhầm lẫn của khoảng lặng khi vượt quá ngưỡng T.
        C_spch  = mean(max(T - speech, 0))   -> Độ nhầm lẫn của tiếng nói khi rơi xuống dưới ngưỡng T.
        diff    = C_sil - C_spch             -> Độ lệch giữa hai độ nhầm lẫn.

    Mục tiêu tối ưu: Tìm T sao cho diff xấp xỉ 0 (cân bằng lỗi phân loại giữa 2 lớp).

    Args:
        silence_vals (np.ndarray): Mảng giá trị đặc trưng của các khung khoảng lặng trong vùng overlap.
        speech_vals (np.ndarray): Mảng giá trị đặc trưng của các khung tiếng nói trong vùng overlap.
        T (float): Ngưỡng phân loại ứng viên đang được đánh giá.

    Returns:
        Tuple[float, float, float]:
            - c_sil (float): Độ nhầm lẫn của khoảng lặng tại ngưỡng T.
            - c_spch (float): Độ nhầm lẫn của tiếng nói tại ngưỡng T.
            - diff (float): Hiệu số c_sil - c_spch.
    """
    # Bước 1: Kiểm tra mảng rỗng để tránh chia cho 0
    if len(silence_vals) == 0 or len(speech_vals) == 0:
        return 0.0, 0.0, 0.0

    # Bước 2: Tính toán độ nhầm lẫn theo hàm max(x, 0)
    c_sil  = float(np.mean(np.maximum(silence_vals - T, 0.0)))
    c_spch = float(np.mean(np.maximum(T - speech_vals,  0.0)))
    diff   = c_sil - c_spch

    return c_sil, c_spch, diff


def find_optimal_threshold_binary_search(
    speech_values:  np.ndarray,
    silence_values: np.ndarray,
    max_iterations: int   = MAX_ITERATIONS,
    tolerance:      float = THRESHOLD_TOLERANCE,
) -> Tuple[float, List[BinarySearchState], Dict[str, Any]]:
    """
    Tìm ngưỡng phân loại tối ưu T bằng thuật toán tìm kiếm nhị phân (Binary Search).

    Nguyên lý cập nhật ngưỡng:
    - Nếu diff > 0: C_sil > C_spch -> Ngưỡng T đang quá thấp làm nhiều silence vượt T -> Tăng cận dưới (tmin = T).
    - Nếu diff < 0: C_spch > C_sil -> Ngưỡng T đang quá cao làm nhiều speech dưới T -> Giảm cận trên (tmax = T).
    - Ngưỡng mới là trung điểm: T = (tmin + tmax) / 2.0.

    Args:
        speech_values (np.ndarray): Mảng 1D float chứa giá trị đặc trưng tiếng nói.
        silence_values (np.ndarray): Mảng 1D float chứa giá trị đặc trưng khoảng lặng.
        max_iterations (int, optional): Số lần lặp tối đa. Mặc định 100.
        tolerance (float, optional): Ngưỡng sai số hội tụ (|diff| < tol hoặc (tmax - tmin) < tol).

    Returns:
        Tuple[float, List[BinarySearchState], Dict[str, Any]]:
            - T_optimal (float): Giá trị ngưỡng tối ưu tìm được.
            - history (List[BinarySearchState]): Lịch sử biến thiên của T và confusion qua các bước lặp.
            - overlap (Dict[str, Any]): Thông tin vùng overlap được sử dụng.
    """
    # Bước 1: Xác định vùng giao nhau giữa hai phân phối
    overlap = find_overlap_region(speech_values, silence_values)
    ov_min, ov_max = overlap["overlap_min"], overlap["overlap_max"]
    sil_ov, spk_ov = overlap["silence_overlap_vals"], overlap["speech_overlap_vals"]

    # Bước 2: Kiểm tra dữ liệu vùng overlap trước khi chạy vòng lặp
    if not overlap["has_overlap"] or len(sil_ov) == 0 or len(spk_ov) == 0:
        logger.warning("  Không đủ dữ liệu trong vùng overlap -> dùng midpoint = %.6f", ov_min)
        dummy_state = BinarySearchState(0, ov_min, ov_max, ov_min, 0.0, 0.0, 0.0, 0, 0)
        return ov_min, [dummy_state], overlap

    # Bước 3: Khởi tạo các biên tìm kiếm nhị phân và danh sách lưu vết
    tmin, tmax = ov_min, ov_max
    T = (tmin + tmax) / 2.0
    history: List[BinarySearchState] = []

    logger.info("  Bắt đầu Binary search: vùng overlap = [%.6f, %.6f]", tmin, tmax)
    logger.info("  %-5s %-14s %-14s %-14s %-14s %-14s %-14s",
                "Iter", "Tmin", "Tmax", "T", "C_sil", "C_spch", "Diff")

    # Bước 4: Vòng lặp tìm kiếm nhị phân điều chỉnh ngưỡng T
    for i in range(1, max_iterations + 1):
        # Tính toán độ nhầm lẫn và hiệu số diff tại ngưỡng T hiện tại
        c_sil, c_spch, diff = compute_confusion(sil_ov, spk_ov, T)

        # Lưu lại trạng thái của bước lặp hiện tại để vẽ biểu đồ phân tích hội tụ
        state = BinarySearchState(
            iteration=i, tmin=tmin, tmax=tmax, threshold=T,
            confusion_sil=c_sil, confusion_speech=c_spch, difference=diff,
            speech_below_T=int(np.sum(spk_ov < T)),
            silence_above_T=int(np.sum(sil_ov > T)),
        )
        history.append(state)
        logger.info("  %-5d %-14.8f %-14.8f %-14.8f %-14.8f %-14.8f %-14.8f",
                    i, tmin, tmax, T, c_sil, c_spch, diff)

        # Kiểm tra điều kiện dừng: sai số diff đủ nhỏ
        if abs(diff) < tolerance:
            logger.info("  Binary search hội tụ theo sai số (|diff| < %.1e) sau %d bước lặp", tolerance, i)
            break

        # Kiểm tra điều kiện dừng: khoảng tìm kiếm đã co cụm dưới ngưỡng tolerance
        if (tmax - tmin) < tolerance:
            logger.info("  Khoảng [Tmin, Tmax] nhỏ hơn tolerance sau %d bước lặp", i)
            break

        # Bước 5: Cập nhật khoảng tìm kiếm cho bước lặp kế tiếp
        if diff > 0:
            tmin = T
        else:
            tmax = T
        T = (tmin + tmax) / 2.0
    else:
        logger.warning("  Đạt giới hạn số lần lặp max_iterations=%d mà chưa hội tụ tuyệt đối.", max_iterations)

    return T, history, overlap
