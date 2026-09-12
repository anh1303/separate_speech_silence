"""threshold.py – Binary search để tìm ngưỡng tối ưu (Hodgkinson 2012, Sec. 2.1)."""

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
    Xác định vùng giao nhau giữa phân phối Speech và Silence:
        overlap = [max(min_s, min_sil), min(max_s, max_sil)]

    Chỉ các giá trị trong vùng này được dùng để binary search.
    """
    if len(speech_values) == 0 or len(silence_values) == 0:
        raise ValueError("Cần ít nhất 1 frame Speech và 1 frame Silence.")

    min_s, max_s   = float(np.min(speech_values)),  float(np.max(speech_values))
    min_sl, max_sl = float(np.min(silence_values)), float(np.max(silence_values))

    ov_min = max(min_s, min_sl)
    ov_max = min(max_s, max_sl)
    has_overlap = ov_min < ov_max

    if not has_overlap:
        logger.warning(
            "  Hai phân phối không overlap: Speech=[%.4f,%.4f], Silence=[%.4f,%.4f]",
            min_s, max_s, min_sl, max_sl,
        )
        ov_min = ov_max = (min(max_s, max_sl) + max(min_s, min_sl)) / 2.0
        logger.warning("  Dùng midpoint: %.6f", ov_min)

    return {
        "overlap_min":          ov_min,
        "overlap_max":          ov_max,
        "has_overlap":          has_overlap,
        "speech_overlap_vals":  speech_values[(speech_values >= ov_min) & (speech_values <= ov_max)],
        "silence_overlap_vals": silence_values[(silence_values >= ov_min) & (silence_values <= ov_max)],
        "speech_all_vals":      speech_values,
        "silence_all_vals":     silence_values,
    }


def compute_confusion(
    silence_vals: np.ndarray,
    speech_vals:  np.ndarray,
    T: float,
) -> Tuple[float, float, float]:
    """
    Công thức confusion (Hodgkinson 2012):
        C_sil   = mean(max(silence - T, 0))  → lượng silence vượt T (nhầm thành speech)
        C_spch  = mean(max(T - speech,  0))  → lượng speech dưới T  (nhầm thành silence)
        diff    = C_sil - C_spch

    Mục tiêu: tìm T để diff = 0 (cân bằng hai vùng nhầm lẫn).
    """
    if len(silence_vals) == 0 or len(speech_vals) == 0:
        return 0.0, 0.0, 0.0
    c_sil  = float(np.mean(np.maximum(silence_vals - T, 0.0)))
    c_spch = float(np.mean(np.maximum(T - speech_vals,  0.0)))
    return c_sil, c_spch, c_sil - c_spch


def find_optimal_threshold_binary_search(
    speech_values:  np.ndarray,
    silence_values: np.ndarray,
    max_iterations: int   = MAX_ITERATIONS,
    tolerance:      float = THRESHOLD_TOLERANCE,
) -> Tuple[float, List[BinarySearchState], Dict[str, Any]]:
    """
    Binary search trên vùng overlap để tìm T tối ưu.

    Quy tắc cập nhật:
        diff > 0  →  C_sil > C_spch  →  T quá thấp  →  Tmin = T  (tìm ở nửa trên)
        diff < 0  →  C_spch > C_sil  →  T quá cao   →  Tmax = T  (tìm ở nửa dưới)
    """
    overlap = find_overlap_region(speech_values, silence_values)
    ov_min, ov_max  = overlap["overlap_min"], overlap["overlap_max"]
    sil_ov, spk_ov  = overlap["silence_overlap_vals"], overlap["speech_overlap_vals"]

    if not overlap["has_overlap"] or len(sil_ov) == 0 or len(spk_ov) == 0:
        logger.warning("  Không đủ dữ liệu overlap → dùng midpoint = %.6f", ov_min)
        dummy = BinarySearchState(0, ov_min, ov_max, ov_min, 0., 0., 0., 0, 0)
        return ov_min, [dummy], overlap

    tmin, tmax = ov_min, ov_max
    T = (tmin + tmax) / 2.0
    history: List[BinarySearchState] = []

    logger.info("  Binary search: overlap=[%.6f, %.6f]", tmin, tmax)
    logger.info("  %-5s %-14s %-14s %-14s %-14s %-14s %-14s",
                "Iter", "Tmin", "Tmax", "T", "C_sil", "C_spch", "Diff")

    for i in range(1, max_iterations + 1):
        c_sil, c_spch, diff = compute_confusion(sil_ov, spk_ov, T)
        history.append(BinarySearchState(
            iteration=i, tmin=tmin, tmax=tmax, threshold=T,
            confusion_sil=c_sil, confusion_speech=c_spch, difference=diff,
            speech_below_T=int(np.sum(spk_ov < T)),
            silence_above_T=int(np.sum(sil_ov > T)),
        ))
        logger.info("  %-5d %-14.8f %-14.8f %-14.8f %-14.8f %-14.8f %-14.8f",
                    i, tmin, tmax, T, c_sil, c_spch, diff)

        if abs(diff) < tolerance:
            logger.info("  Hội tụ sau %d iters", i); break
        if (tmax - tmin) < tolerance:
            logger.info("  [Tmin,Tmax] < tol, dừng sau %d iters", i); break

        if diff > 0:
            tmin = T
        else:
            tmax = T
        T = (tmin + tmax) / 2.0
    else:
        logger.warning("  Đạt max_iterations=%d, chưa hội tụ.", max_iterations)

    return T, history, overlap
