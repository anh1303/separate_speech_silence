"""io_utils.py – Đọc/ghi WAV, đọc/validate .lab và lưu kết quả."""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import wavfile

try:
    from src.config import (
        BOUNDARY_MATCH_TOLERANCE_MS, FEATURE_TYPE, FRAME_LENGTH_MS,
        FRAME_SHIFT_MS, LOG_EPSILON, MIN_SILENCE_DURATION_MS,
        SPEECH_IS_ABOVE_THRESHOLD, BinarySearchState, FeatureType,
        LabSegment, Segment,
    )
except ImportError:
    from config import (
        BOUNDARY_MATCH_TOLERANCE_MS, FEATURE_TYPE, FRAME_LENGTH_MS,
        FRAME_SHIFT_MS, LOG_EPSILON, MIN_SILENCE_DURATION_MS,
        SPEECH_IS_ABOVE_THRESHOLD, BinarySearchState, FeatureType,
        LabSegment, Segment,
    )

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

logger = logging.getLogger(__name__)


def validate_lab_segments(segments: List[LabSegment], lab_path: str) -> bool:
    """
    Kiểm tra tính nhất quán và hợp lệ của danh sách LabSegment:
    - start >= 0
    - start < end
    - Sắp xếp tăng dần theo start
    - Không bị overlap (segment i kết thúc trước khi segment i+1 bắt đầu)
    - Cảnh báo nếu có khoảng hở (gap) giữa 2 segment liên tiếp.
    """
    if not segments:
        logger.warning("  [Validation] %s: Không có segment nào!", lab_path)
        return False

    is_valid = True
    for i, seg in enumerate(segments):
        if seg.start < 0.0:
            logger.warning("  [Validation] %s [seg %d]: start < 0 (%.4f)", lab_path, i, seg.start)
            is_valid = False
        if seg.start >= seg.end:
            logger.warning("  [Validation] %s [seg %d]: start >= end (%.4f >= %.4f)", lab_path, i, seg.start, seg.end)
            is_valid = False

        if i > 0:
            prev = segments[i - 1]
            if seg.start < prev.end:
                logger.warning("  [Validation] %s: Overlap giữa seg %d (%.4f-%.4f) và seg %d (%.4f-%.4f)",
                               lab_path, i - 1, prev.start, prev.end, i, seg.start, seg.end)
                is_valid = False
            elif seg.start > prev.end:
                gap = seg.start - prev.end
                logger.warning("  [Validation] %s: Có khoảng trống (gap = %.4f s) giữa seg %d và seg %d",
                               lab_path, gap, i - 1, i)

    return is_valid


def parse_lab_file(lab_path: str) -> List[LabSegment]:
    """
    Đọc file .lab với format: start<TAB>end<TAB>label
    Nhãn hợp lệ: sil, v, uv. Hai dòng cuối F0mean/F0std bị bỏ qua.
    Tự động thực hiện validation sau khi đọc.
    """
    VALID_LABELS = {"sil", "v", "uv"}
    METADATA_KEYS = {"f0mean", "f0std"}

    if not os.path.exists(lab_path):
        raise FileNotFoundError(f"Không tìm thấy file .lab: {lab_path}")

    segments: List[LabSegment] = []
    with open(lab_path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            parts = [p.strip() for p in raw.strip().split("\t") if p.strip()]
            if not parts:
                continue
            if parts[0].lower() in METADATA_KEYS:
                continue
            if len(parts) < 3:
                logger.warning("%s dòng %d: thiếu cột -> '%s'", lab_path, lineno, raw.strip())
                continue
            try:
                start, end, label = float(parts[0]), float(parts[1]), parts[2].lower()
            except ValueError:
                logger.warning("%s dòng %d: không parse được -> '%s'", lab_path, lineno, raw.strip())
                continue
            if label not in VALID_LABELS:
                logger.warning("%s dòng %d: nhãn '%s' không hợp lệ", lab_path, lineno, label)
                continue
            if start >= end:
                logger.warning("%s dòng %d: start >= end", lab_path, lineno)
                continue
            segments.append(LabSegment(start=start, end=end, label=label))

    validate_lab_segments(segments, lab_path)
    return segments


def load_ground_truth(lab_segments: List[LabSegment]) -> List[Segment]:
    """Gộp sil→silence, v/uv→speech; merge các segment liền kề cùng nhãn."""
    if not lab_segments:
        return []

    to_bin = lambda lbl: "silence" if lbl == "sil" else "speech"

    merged: List[Segment] = []
    cur_start = lab_segments[0].start
    cur_label = to_bin(lab_segments[0].label)
    cur_end   = lab_segments[0].end

    for seg in lab_segments[1:]:
        lbl = to_bin(seg.label)
        if lbl == cur_label:
            cur_end = seg.end
        else:
            merged.append(Segment(cur_start, cur_end, cur_label))
            cur_start, cur_end, cur_label = seg.start, seg.end, lbl

    merged.append(Segment(cur_start, cur_end, cur_label))
    return merged


def load_audio(wav_path: str) -> Tuple[np.ndarray, int]:
    """Đọc WAV → mono float32 ∈ [-1, 1]. Stereo tự động average hai kênh."""
    try:
        sr, data = wavfile.read(wav_path)
    except Exception as exc:
        if HAS_SOUNDFILE:
            data, sr = sf.read(wav_path, always_2d=False)
            data = data.astype(np.float32)
        else:
            raise RuntimeError(f"Không đọc được WAV: {wav_path}") from exc

    dtype_map = {
        np.dtype("int16"): 32768.0,
        np.dtype("int32"): 2147483648.0,
        np.dtype("uint8"): None,
    }
    if data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    elif data.dtype in dtype_map:
        data = data.astype(np.float32) / dtype_map[data.dtype]
    else:
        data = data.astype(np.float32)

    if data.ndim == 2:
        logger.info("  Stereo → mono (%d channels)", data.shape[1])
        data = data.mean(axis=1)

    return data, int(sr)


def save_audio(wav_path: str, signal: np.ndarray, sample_rate: int):
    """Lưu mảng float [-1, 1] thành WAV int16 16-bit PCM chuẩn."""
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    clipped = np.clip(signal, -1.0, 1.0)
    int16_data = (clipped * 32767.0).astype(np.int16)
    wavfile.write(wav_path, sample_rate, int16_data)


# ── Save functions ─────────────────────────────────────────────────────────────

def save_global_threshold_json(
    threshold:        float,
    output_dir:       str,
    feature_type:     FeatureType = FEATURE_TYPE,
    training_data:    str = "TinHieuHuanLuyen",
    threshold_method: str = "binary_search",
    noise_condition:  str = "original",
    filename:         str = "global_threshold.json",
):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    json.dump({
        "threshold":                   threshold,
        "feature":                     feature_type.value,
        "frame_length_ms":             FRAME_LENGTH_MS,
        "frame_shift_ms":              FRAME_SHIFT_MS,
        "log_epsilon":                 LOG_EPSILON,
        "min_silence_duration_ms":     MIN_SILENCE_DURATION_MS,
        "boundary_match_tolerance_ms": BOUNDARY_MATCH_TOLERANCE_MS,
        "speech_is_above_threshold":   SPEECH_IS_ABOVE_THRESHOLD,
        "training_data":               training_data,
        "threshold_method":            threshold_method,
        "noise_condition":             noise_condition,
    }, open(path, "w", encoding="utf-8"), indent=2)
    logger.info("  Saved: %s", path)


def save_global_threshold_txt(threshold: float, output_dir: str, filename: str = "global_threshold.txt"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Feature      : {FEATURE_TYPE.value}\n"
                f"Frame length : {FRAME_LENGTH_MS} ms\n"
                f"Frame shift  : {FRAME_SHIFT_MS} ms\n"
                f"Min silence  : {MIN_SILENCE_DURATION_MS} ms\n"
                f"Threshold T  : {threshold:.10f}\n")
    logger.info("  Saved: %s", path)


def save_threshold_summary_csv(
    file_results:     List[Dict[str, Any]],
    global_threshold: float,
    output_dir:       str,
    eval_results:     Optional[List[Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lưu 2 CSV: per-file threshold (tham khảo) và global threshold (kết quả chính) với đầy đủ P, R, F1."""
    os.makedirs(output_dir, exist_ok=True)

    rows_pf = []
    for res in file_results:
        if res is None:
            continue
        err = res.get("error_metrics", {})
        rows_pf.append({
            "filename":             res["basename"],
            "speech_frames":        res["n_speech"],
            "silence_frames":       res["n_silence"],
            "per_file_threshold":   res.get("threshold"),
            "mae_ms":               err.get("mae_ms"),
            "rmse_ms":              err.get("rmse_ms"),
            "precision":            err.get("boundary_precision"),
            "recall":               err.get("boundary_recall"),
            "f1":                   err.get("boundary_f1"),
            "n_pred_boundaries":    err.get("n_pred_boundaries"),
            "n_gt_boundaries":      err.get("n_gt_boundaries"),
            "n_matched":            err.get("n_matched"),
            "n_unmatched_pred":     err.get("n_unmatched_pred"),
            "n_unmatched_gt":       err.get("n_unmatched_gt"),
        })
    df_pf = pd.DataFrame(rows_pf)
    pf_path = os.path.join(output_dir, "threshold_summary_perfile.csv")
    df_pf.to_csv(pf_path, index=False, encoding="utf-8-sig")
    logger.info("  Saved: %s", pf_path)

    eval_map = {er["basename"]: er.get("error_metrics", {}) for er in (eval_results or [])}
    rows_gl = []
    for res in file_results:
        if res is None:
            continue
        err_gl = eval_map.get(res["basename"], {})
        rows_gl.append({
            "filename":          res["basename"],
            "speech_frames":     res["n_speech"],
            "silence_frames":    res["n_silence"],
            "global_threshold":  global_threshold,
            "mae_ms":            err_gl.get("mae_ms"),
            "rmse_ms":           err_gl.get("rmse_ms"),
            "precision":         err_gl.get("boundary_precision"),
            "recall":            err_gl.get("boundary_recall"),
            "f1":                err_gl.get("boundary_f1"),
            "n_pred_boundaries": err_gl.get("n_pred_boundaries"),
            "n_gt_boundaries":   err_gl.get("n_gt_boundaries"),
            "n_matched":         err_gl.get("n_matched"),
            "n_unmatched_pred":  err_gl.get("n_unmatched_pred"),
            "n_unmatched_gt":    err_gl.get("n_unmatched_gt"),
        })
    df_gl = pd.DataFrame(rows_gl)
    gl_path = os.path.join(output_dir, "threshold_summary_global.csv")
    df_gl.to_csv(gl_path, index=False, encoding="utf-8-sig")
    logger.info("  Saved: %s", gl_path)

    return df_pf, df_gl


def save_binary_search_history_csv(
    file_results:   List[Dict[str, Any]],
    global_history: List[BinarySearchState],
    output_dir:     str,
    filename:       str = "binary_search_history.csv",
):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for res in file_results or []:
        if res is None:
            continue
        for s in res.get("history", []):
            rows.append({
                "source": res["basename"], "iteration": s.iteration,
                "tmin": s.tmin, "tmax": s.tmax, "threshold": s.threshold,
                "confusion_sil": s.confusion_sil, "confusion_speech": s.confusion_speech,
                "difference": s.difference,
                "speech_below_T": s.speech_below_T, "silence_above_T": s.silence_above_T,
            })
    for s in global_history:
        rows.append({
            "source": "GLOBAL", "iteration": s.iteration,
            "tmin": s.tmin, "tmax": s.tmax, "threshold": s.threshold,
            "confusion_sil": s.confusion_sil, "confusion_speech": s.confusion_speech,
            "difference": s.difference,
            "speech_below_T": s.speech_below_T, "silence_above_T": s.silence_above_T,
        })
    path = os.path.join(output_dir, filename)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("  Saved: %s", path)


def save_boundary_errors_csv(eval_results: List[Dict[str, Any]], output_dir: str, filename: str = "boundary_errors.csv"):
    """Lưu chi tiết lỗi boundary từng cặp matched và unmatched."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for er in eval_results or []:
        if er is None:
            continue
        err = er.get("error_metrics", {})
        name = er.get("basename", "?")
        for pb, gb in err.get("matched_pairs", []):
            rows.append({
                "filename": name, "status": "matched",
                "pred_s": pb, "gt_s": gb, "error_ms": abs(pb - gb) * 1000.0,
            })
        for b in err.get("unmatched_pred", []):
            rows.append({"filename": name, "status": "unmatched_pred", "pred_s": b, "gt_s": None, "error_ms": None})
        for b in err.get("unmatched_gt", []):
            rows.append({"filename": name, "status": "unmatched_gt", "pred_s": None, "gt_s": b, "error_ms": None})
    path = os.path.join(output_dir, filename)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("  Saved: %s", path)
