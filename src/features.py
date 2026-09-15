"""
features.py – Module tính toán đặc trưng ngắn hạn và gán nhãn khung thời gian.

Cung cấp các hàm:
- compute_short_time_feature: Phân khung và tính đặc trưng ngắn hạn (MA, logMA, STE, logSTE).
- assign_frame_labels: Gán nhãn khung (Speech / Silence / Unknown) dựa trên thời gian thực từ file .lab.
"""

import logging
from typing import List, Tuple

import numpy as np

try:
    from src.config import (
        FEATURE_TYPE, FRAME_LENGTH_MS, FRAME_SHIFT_MS, LOG_EPSILON,
        FeatureType, LabSegment,
    )
except ImportError:
    from config import (
        FEATURE_TYPE, FRAME_LENGTH_MS, FRAME_SHIFT_MS, LOG_EPSILON,
        FeatureType, LabSegment,
    )

logger = logging.getLogger(__name__)


def compute_short_time_feature(
    signal:          np.ndarray,
    sample_rate:     int,
    frame_length_ms: int         = FRAME_LENGTH_MS,
    frame_shift_ms:  int         = FRAME_SHIFT_MS,
    feature_type:    FeatureType = FEATURE_TYPE,
    log_epsilon:     float       = LOG_EPSILON,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính đặc trưng biên độ ngắn hạn (Short-Time Feature) và vị trí tâm thời gian của từng khung.

    Các đặc trưng hỗ trợ:
    - MA (Short-time Magnitude): mean(|x|)
    - logMA (Log Short-time Magnitude): log(mean(|x|) + ε)
    - STE (Short-time Energy): mean(x²)
    - logSTE (Log Short-time Energy): log(mean(x²) + ε)

    Args:
        signal (np.ndarray): Mảng 1D float chứa tín hiệu âm thanh đầu vào.
        sample_rate (int): Tần số lấy mẫu của tín hiệu (Hz).
        frame_length_ms (int, optional): Độ dài mỗi khung tín hiệu (mili-giây). Mặc định 20ms.
        frame_shift_ms (int, optional): Độ dịch giữa hai khung liên tiếp (mili-giây). Mặc định 10ms.
        feature_type (FeatureType, optional): Kiểu đặc trưng cần tính toán. Mặc định là FeatureType.LOG_MA.
        log_epsilon (float, optional): Hằng số cực tiểu để tránh lỗi log(0). Mặc định 1e-10.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - feature_values (np.ndarray): Mảng 1D float64 chứa giá trị đặc trưng của từng khung.
            - frame_centers (np.ndarray): Mảng 1D float64 chứa vị trí tâm của từng khung (đơn vị: giây).
    """
    # Bước 1: Quy đổi độ dài khung và bước nhảy từ mili-giây sang số mẫu tín hiệu (samples)
    frame_len  = int(sample_rate * frame_length_ms / 1000)
    frame_step = int(sample_rate * frame_shift_ms  / 1000)

    # Kiểm tra tính hợp lệ của tham số cấu hình khung
    if frame_len <= 0 or frame_step <= 0:
        raise ValueError(f"Tham số khung không hợp lệ: frame_len={frame_len}, frame_step={frame_step}")

    # Bước 2: Kiểm tra chiều dài tín hiệu so với độ dài tối thiểu của một khung
    n_samples = len(signal)
    if n_samples < frame_len:
        logger.warning("Tín hiệu ngắn hơn 1 frame (%d < %d samples)", n_samples, frame_len)
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Bước 3: Tính toán tổng số khung và khởi tạo trước bộ nhớ cho các mảng kết quả
    n_frames       = 1 + (n_samples - frame_len) // frame_step
    feature_values = np.empty(n_frames, dtype=np.float64)
    frame_centers  = np.empty(n_frames, dtype=np.float64)

    # Bước 4: Vòng lặp trích xuất đặc trưng cho từng khung thời gian
    for i in range(n_frames):
        # Xác định chỉ số mẫu bắt đầu của khung thứ i
        start_sample = i * frame_step
        frame = signal[start_sample : start_sample + frame_len]

        # Tính toán giá trị đặc trưng cơ bản (MA: trung bình trị tuyệt đối, STE: năng lượng trung bình)
        if feature_type in (FeatureType.MA, FeatureType.LOG_MA):
            val = np.mean(np.abs(frame))
        else:
            val = np.mean(frame.astype(np.float64) ** 2)

        # Áp dụng hàm log tự nhiên nếu yêu cầu đặc trưng ở thang đo logarithmic
        if feature_type in (FeatureType.LOG_MA, FeatureType.LOG_STE):
            val = np.log(val + log_epsilon)

        # Lưu giá trị đặc trưng và thời gian tâm khung tương ứng
        feature_values[i] = val
        frame_centers[i]  = (start_sample + frame_len / 2.0) / sample_rate

    return feature_values, frame_centers


def assign_frame_labels(
    frame_centers: np.ndarray,
    lab_segments:  List[LabSegment],
) -> np.ndarray:
    """
    Gán nhãn cho từng khung thời gian dựa trên vị trí tâm khung và phân đoạn Ground Truth từ file .lab.

    Quy ước nhãn:
    - 1: Tiếng nói (Speech - nhãn 'v' hoặc 'uv' trong file .lab)
    - 0: Khoảng lặng (Silence - nhãn 'sil' trong file .lab)
    - -1: Không xác định (Unknown - nằm ngoài tất cả khoảng của file .lab)

    Args:
        frame_centers (np.ndarray): Mảng 1D float64 chứa thời điểm tâm của từng khung (giây).
        lab_segments (List[LabSegment]): Danh sách các đoạn chuẩn đọc từ file .lab.

    Returns:
        np.ndarray: Mảng 1D int32 chứa nhãn phân loại tương ứng của từng khung.
    """
    # Bước 1: Khởi tạo mảng nhãn mặc định là -1 (chưa xác định)
    labels = np.full(len(frame_centers), -1, dtype=np.int32)

    # Bước 2: Quét qua từng đoạn nhãn chuẩn ground truth trong file .lab
    for seg in lab_segments:
        # Khung được gán nhãn nếu thời gian tâm nằm trong nửa khoảng [start, end)
        mask = (frame_centers >= seg.start) & (frame_centers < seg.end)
        labels[mask] = 0 if seg.is_silence() else 1

    # Bước 3: Kiểm tra và cảnh báo nếu tồn tại khung không rơi vào khoảng nhãn nào
    n_unk = int(np.sum(labels == -1))
    if n_unk > 0:
        logger.warning("  %d frame không thuộc đoạn lab nào (unknown)", n_unk)

    return labels
