"""segmentation.py – Phân loại frame, tạo segment, lọc silence ngắn và đánh giá boundary."""

import logging
from typing import List

import numpy as np

try:
    from src.config import (
        BOUNDARY_MATCH_TOLERANCE_MS, FRAME_SHIFT_MS,
        MIN_SILENCE_DURATION_MS, SPEECH_IS_ABOVE_THRESHOLD, Segment,
    )
except ImportError:
    from config import (
        BOUNDARY_MATCH_TOLERANCE_MS, FRAME_SHIFT_MS,
        MIN_SILENCE_DURATION_MS, SPEECH_IS_ABOVE_THRESHOLD, Segment,
    )

logger = logging.getLogger(__name__)


def classify_frames(
    feature_values:  np.ndarray,
    threshold:       float,
    speech_is_above: bool = SPEECH_IS_ABOVE_THRESHOLD,
) -> np.ndarray:
    """
    speech_is_above=True (mặc định với MA/logMA):
        feature >= T → speech (1), feature < T → silence (0)
    """
    if speech_is_above:
        return (feature_values >= threshold).astype(np.int32)
    return (feature_values <= threshold).astype(np.int32)


def frames_to_segments(
    labels:            np.ndarray,
    frame_centers:     np.ndarray,
    frame_shift_ms:    int   = FRAME_SHIFT_MS,
    signal_duration_s: float = 0.0,
) -> List[Segment]:
    """
    Chuyển nhãn frame-level thành danh sách Segment.
    - Segment đầu bắt đầu tại 0.0.
    - Segment cuối kết thúc tại signal_duration_s (nếu truyền).
    - Boundary nội bộ = (center[i-1] + center[i]) / 2.
    """
    if len(labels) == 0:
        return []

    segs: List[Segment] = []
    cur_label = labels[0]
    cur_start = 0.0

    for i in range(1, len(labels)):
        if labels[i] != cur_label:
            boundary = (frame_centers[i - 1] + frame_centers[i]) / 2.0
            if cur_label != -1:
                segs.append(Segment(cur_start, boundary, "speech" if cur_label == 1 else "silence"))
            cur_label = labels[i]
            cur_start = boundary

    last_end = signal_duration_s if signal_duration_s > 0.0 else frame_centers[-1] + frame_shift_ms / 2000.0
    if cur_label != -1:
        segs.append(Segment(cur_start, last_end, "speech" if cur_label == 1 else "silence"))

    return segs


def remove_short_silence_segments(
    segments:                List[Segment],
    min_silence_duration_ms: float = MIN_SILENCE_DURATION_MS,
) -> List[Segment]:
    """
    Xóa silence ngắn hơn min_silence_duration_ms bằng cách merge segment cấp độ:
        Silence ngắn giữa 2 speech → merge cả 3 thành speech
        Silence ngắn ở đầu/cuối   → merge vào segment kề
    Lặp cho đến khi không còn silence ngắn.
    """
    if not segments:
        return []

    min_s = min_silence_duration_ms / 1000.0
    changed = True
    result = list(segments)

    while changed:
        changed = False
        out: List[Segment] = []
        i = 0
        while i < len(result):
            seg = result[i]
            if seg.label == "silence" and seg.duration_s < min_s:
                changed  = True
                has_prev = i > 0
                has_next = i < len(result) - 1
                if has_prev and has_next:
                    prev = out.pop()
                    merged = Segment(prev.start, result[i + 1].end, "speech")
                    out.append(merged); i += 2
                elif has_prev:
                    prev = out.pop()
                    out.append(Segment(prev.start, seg.end, prev.label)); i += 1
                elif has_next:
                    nxt = result[i + 1]
                    out.append(Segment(seg.start, nxt.end, nxt.label)); i += 2
                else:
                    out.append(seg); i += 1
            else:
                out.append(seg); i += 1
        result = out

    merged: List[Segment] = []
    for seg in result:
        if merged and merged[-1].label == seg.label:
            merged[-1] = Segment(merged[-1].start, seg.end, seg.label)
        else:
            merged.append(Segment(seg.start, seg.end, seg.label))
    return merged


def extract_boundaries(segments: List[Segment]) -> List[float]:
    """Lấy các thời điểm boundary nội bộ (không kể 2 đầu 0 và duration)."""
    return [s.end for s in segments[:-1]] if segments else []


def match_boundaries(
    pred: List[float],
    gt:   List[float],
    tolerance_s: float = BOUNDARY_MATCH_TOLERANCE_MS / 1000.0,
):
    """
    Nearest-neighbor matching với tolerance.
    Mỗi GT boundary khớp với predicted boundary chưa dùng gần nhất trong tolerance.
    Trả về (matched_pairs, unmatched_pred, unmatched_gt).
    """
    used = [False] * len(pred)
    matched, unmatched_gt = [], []

    for g in gt:
        best_i, best_d = -1, float("inf")
        for j, p in enumerate(pred):
            if not used[j] and abs(p - g) < best_d:
                best_d, best_i = abs(p - g), j
        if best_i >= 0 and best_d <= tolerance_s:
            matched.append((pred[best_i], g))
            used[best_i] = True
        else:
            unmatched_gt.append(g)

    unmatched_pred = [pred[j] for j in range(len(pred)) if not used[j]]
    return matched, unmatched_pred, unmatched_gt


def compute_boundary_errors(
    pred_segments: List[Segment],
    gt_segments:   List[Segment],
    tolerance_s:   float = BOUNDARY_MATCH_TOLERANCE_MS / 1000.0,
) -> dict:
    """
    Tính đầy đủ metrics đánh giá ranh giới:
    - MAE, RMSE (đơn vị ms trên các cặp matched)
    - Precision = N_matched / N_pred
    - Recall    = N_matched / N_gt
    - F1-score  = 2 * P * R / (P + R)
    - Đếm số lượng matched, unmatched_pred, unmatched_gt
    """
    pred_bounds = extract_boundaries(pred_segments)
    gt_bounds   = extract_boundaries(gt_segments)

    matched, unmatched_pred, unmatched_gt = match_boundaries(pred_bounds, gt_bounds, tolerance_s)

    n_pred = len(pred_bounds)
    n_gt   = len(gt_bounds)
    n_m    = len(matched)

    precision = (n_m / n_pred) if n_pred > 0 else (1.0 if n_gt == 0 else 0.0)
    recall    = (n_m / n_gt)   if n_gt > 0   else (1.0 if n_pred == 0 else 0.0)
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

    if not matched:
        return {
            "mae_ms":             float("nan"),
            "rmse_ms":            float("nan"),
            "boundary_precision": float(precision),
            "boundary_recall":    float(recall),
            "boundary_f1":        float(f1),
            "matched_pairs":      [],
            "unmatched_pred":     unmatched_pred,
            "unmatched_gt":       unmatched_gt,
            "errors_ms":          [],
            "n_pred_boundaries":  n_pred,
            "n_gt_boundaries":    n_gt,
            "n_matched":          0,
            "n_unmatched_pred":   len(unmatched_pred),
            "n_unmatched_gt":     len(unmatched_gt),
        }

    errors_ms = [abs(p - g) * 1000.0 for p, g in matched]
    arr = np.array(errors_ms)
    return {
        "mae_ms":             float(np.mean(arr)),
        "rmse_ms":            float(np.sqrt(np.mean(arr ** 2))),
        "boundary_precision": float(precision),
        "boundary_recall":    float(recall),
        "boundary_f1":        float(f1),
        "matched_pairs":      matched,
        "unmatched_pred":     unmatched_pred,
        "unmatched_gt":       unmatched_gt,
        "errors_ms":          errors_ms,
        "n_pred_boundaries":  n_pred,
        "n_gt_boundaries":    n_gt,
        "n_matched":          n_m,
        "n_unmatched_pred":   len(unmatched_pred),
        "n_unmatched_gt":     len(unmatched_gt),
    }
