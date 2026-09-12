"""features.py – Tính short-time feature và gán nhãn frame."""

import logging
from typing import Tuple

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
    Trả về (feature_values, frame_centers_in_seconds).

    MA    = mean(|x|),  STE = mean(x²) trên mỗi frame.
    logMA = log(MA + ε), tương tự logSTE.
    """
    frame_len  = int(sample_rate * frame_length_ms / 1000)
    frame_step = int(sample_rate * frame_shift_ms  / 1000)

    if frame_len <= 0 or frame_step <= 0:
        raise ValueError(f"frame_len={frame_len}, frame_step={frame_step}")

    n_samples = len(signal)
    if n_samples < frame_len:
        logger.warning("Tín hiệu ngắn hơn 1 frame (%d < %d samples)", n_samples, frame_len)
        return np.array([]), np.array([])

    n_frames       = 1 + (n_samples - frame_len) // frame_step
    feature_values = np.empty(n_frames, dtype=np.float64)
    frame_centers  = np.empty(n_frames, dtype=np.float64)

    for i in range(n_frames):
        s = i * frame_step
        frame = signal[s : s + frame_len]

        if feature_type in (FeatureType.MA, FeatureType.LOG_MA):
            val = np.mean(np.abs(frame))
        else:
            val = np.mean(frame.astype(np.float64) ** 2)

        if feature_type in (FeatureType.LOG_MA, FeatureType.LOG_STE):
            val = np.log(val + log_epsilon)

        feature_values[i] = val
        frame_centers[i]  = (s + frame_len / 2.0) / sample_rate

    return feature_values, frame_centers


def assign_frame_labels(
    frame_centers: np.ndarray,
    lab_segments:  list,   # List[LabSegment]
) -> np.ndarray:
    """
    Gán nhãn cho mỗi frame dựa theo thời gian trung tâm:
        1 = speech, 0 = silence, -1 = unknown

    Dùng interval [start, end) → frame tại đúng boundary thuộc đoạn bên phải.
    """
    labels = np.full(len(frame_centers), -1, dtype=np.int32)
    for seg in lab_segments:
        mask = (frame_centers >= seg.start) & (frame_centers < seg.end)
        labels[mask] = 0 if seg.is_silence() else 1

    n_unk = int(np.sum(labels == -1))
    if n_unk:
        logger.warning("  %d frame không thuộc đoạn lab nào (unknown)", n_unk)
    return labels
