"""
io_utils.py – Module đọc/ghi âm thanh, đọc/kiểm tra tính hợp lệ của file nhãn .lab và lưu trữ các kết quả thực nghiệm.

Cung cấp các hàm I/O:
- validate_lab_segments: Kiểm tra tính nhất quán hình thức và logic của các phân đoạn ground truth.
- parse_lab_file: Đọc file .lab thành danh sách cấu trúc dữ liệu LabSegment.
- load_ground_truth: Chuyển đổi các nhãn sil/v/uv sang 2 nhãn nhị phân speech/silence.
- load_audio: Đọc file âm thanh WAV, chuyển đổi về mono và chuẩn hóa biên độ float32 [-1, 1].
- save_audio: Lưu tín hiệu âm thanh thành định dạng chuẩn PCM WAV int16.
- Các hàm lưu trữ kết quả: save_global_threshold_json, save_global_threshold_txt,
  save_threshold_summary_csv, save_binary_search_history_csv, save_boundary_errors_csv.
"""

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
    Kiểm tra tính nhất quán và tính hợp lệ của danh sách các đoạn nhãn đọc từ file .lab.

    Các tiêu chí kiểm tra:
    1. Không rỗng.
    2. Thời gian bắt đầu start >= 0 và start < end cho từng đoạn.
    3. Không bị chồng lấn (overlap) giữa các đoạn liền kề.
    4. Cảnh báo nếu tồn tại khoảng trống (gap) giữa hai đoạn liên tiếp.

    Args:
        segments (List[LabSegment]): Danh sách các đoạn nhãn từ file .lab.
        lab_path (str): Đường dẫn file .lab đang được kiểm tra.

    Returns:
        bool: True nếu toàn bộ danh sách hợp lệ, False nếu vi phạm logic thời gian.
    """
    # Bước 1: Kiểm tra danh sách rỗng
    if not segments:
        logger.warning("  [Validation] %s: Không có segment nào!", lab_path)
        return False

    is_valid = True

    # Bước 2: Duyệt từng đoạn và kiểm tra ràng buộc thời gian
    for i, seg in enumerate(segments):
        if seg.start < 0.0:
            logger.warning("  [Validation] %s [seg %d]: start < 0 (%.4f)", lab_path, i, seg.start)
            is_valid = False
        if seg.start >= seg.end:
            logger.warning("  [Validation] %s [seg %d]: start >= end (%.4f >= %.4f)", lab_path, i, seg.start, seg.end)
            is_valid = False

        # Bước 3: So sánh với đoạn trước đó để phát hiện overlap hoặc gap
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
    Đọc và phân tích cú pháp file nhãn .lab theo định dạng: start<TAB>end<TAB>label.

    Các nhãn hợp lệ trong bài toán:
    - 'sil': Khoảng lặng (Silence)
    - 'v': Hữu thanh (Voiced speech)
    - 'uv': Vô thanh (Unvoiced speech)
    Các dòng siêu dữ liệu cuối file như F0mean, F0std được tự động bỏ qua.

    Args:
        lab_path (str): Đường dẫn tuyệt đối hoặc tương đối tới file .lab.

    Returns:
        List[LabSegment]: Danh sách các đối tượng LabSegment đã qua kiểm tra hợp lệ.
    """
    VALID_LABELS = {"sil", "v", "uv"}
    METADATA_KEYS = {"f0mean", "f0std"}

    # Bước 1: Kiểm tra sự tồn tại của file trên đĩa
    if not os.path.exists(lab_path):
        raise FileNotFoundError(f"Không tìm thấy file .lab: {lab_path}")

    segments: List[LabSegment] = []

    # Bước 2: Đọc từng dòng của file văn bản
    with open(lab_path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            parts = [p.strip() for p in raw.strip().split("\t") if p.strip()]
            if not parts:
                continue

            # Bỏ qua dòng thông tin F0 phụ trợ
            if parts[0].lower() in METADATA_KEYS:
                continue
            if len(parts) < 3:
                logger.warning("%s dòng %d: thiếu cột -> '%s'", lab_path, lineno, raw.strip())
                continue

            # Bước 3: Parse số thực cho start, end và chuẩn hóa nhãn về chữ thường
            try:
                start, end, label = float(parts[0]), float(parts[1]), parts[2].lower()
            except ValueError:
                logger.warning("%s dòng %d: không parse được -> '%s'", lab_path, lineno, raw.strip())
                continue

            # Kiểm tra tính hợp lệ của nhãn và mốc thời gian
            if label not in VALID_LABELS or start >= end:
                continue

            segments.append(LabSegment(start=start, end=end, label=label))

    # Bước 4: Kiểm tra tính hợp lệ tổng thể của các đoạn
    validate_lab_segments(segments, lab_path)
    return segments


def load_ground_truth(lab_segments: List[LabSegment]) -> List[Segment]:
    """
    Chuyển đổi các nhãn chi tiết của file .lab sang phân loại nhị phân (Speech/Silence).

    Quy tắc gộp:
    - 'sil' -> 'silence'
    - 'v' và 'uv' -> 'speech'
    Đồng thời tự động ghép (merge) các đoạn liên tiếp có cùng nhãn sau khi gộp.

    Args:
        lab_segments (List[LabSegment]): Danh sách các đoạn ban đầu từ file .lab.

    Returns:
        List[Segment]: Danh sách các đoạn nhị phân chuẩn Ground Truth.
    """
    # Bước 1: Kiểm tra danh sách rỗng
    if not lab_segments:
        return []

    to_bin = lambda lbl: "silence" if lbl == "sil" else "speech"

    # Bước 2: Khởi tạo đoạn đầu tiên
    merged: List[Segment] = []
    cur_start = lab_segments[0].start
    cur_label = to_bin(lab_segments[0].label)
    cur_end   = lab_segments[0].end

    # Bước 3: Duyệt các đoạn tiếp theo và gộp nếu cùng nhãn
    for seg in lab_segments[1:]:
        lbl = to_bin(seg.label)
        if lbl == cur_label:
            cur_end = seg.end
        else:
            merged.append(Segment(cur_start, cur_end, cur_label))
            cur_start, cur_end, cur_label = seg.start, seg.end, lbl

    # Thêm đoạn cuối cùng vào danh sách
    merged.append(Segment(cur_start, cur_end, cur_label))
    return merged


def load_audio(wav_path: str) -> Tuple[np.ndarray, int]:
    """
    Đọc file âm thanh WAV và chuyển đổi về dạng mảng 1D float32 chuẩn hoá trong khoảng [-1.0, 1.0].

    Nếu file là Stereo (2 kênh), tự động tính trung bình hai kênh để chuyển về Mono.

    Args:
        wav_path (str): Đường dẫn đến file âm thanh .wav.

    Returns:
        Tuple[np.ndarray, int]:
            - signal (np.ndarray): Mảng 1D float32 biên độ âm thanh.
            - sample_rate (int): Tần số lấy mẫu của file âm thanh (Hz).
    """
    # Bước 1: Đọc dữ liệu từ file WAV bằng scipy.io.wavfile (hoặc fallback sang soundfile)
    try:
        sr, data = wavfile.read(wav_path)
    except Exception as exc:
        if HAS_SOUNDFILE:
            data, sr = sf.read(wav_path, always_2d=False)
            data = data.astype(np.float32)
        else:
            raise RuntimeError(f"Không đọc được WAV: {wav_path}") from exc

    # Bước 2: Chuẩn hoá các kiểu số nguyên int16, int32, uint8 về dải float [-1.0, 1.0]
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

    # Bước 3: Chuyển đổi từ Stereo nhiều kênh thành Mono 1 kênh bằng trung bình cộng
    if data.ndim == 2:
        logger.info("  Chuyển đổi Stereo sang Mono (%d kênh)", data.shape[1])
        data = data.mean(axis=1)

    return data, int(sr)


def save_audio(wav_path: str, signal: np.ndarray, sample_rate: int) -> None:
    """
    Lưu mảng tín hiệu âm thanh float [-1.0, 1.0] thành file WAV PCM 16-bit int16 chuẩn.

    Args:
        wav_path (str): Đường dẫn file WAV cần ghi.
        signal (np.ndarray): Mảng 1D float chứa dữ liệu biên độ âm thanh.
        sample_rate (int): Tần số lấy mẫu (Hz).

    Returns:
        None
    """
    # Bước 1: Tạo thư mục cha nếu chưa tồn tại
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)

    # Bước 2: Cắt giới hạn giá trị biên độ trong khoảng [-1.0, 1.0] để tránh tràn số
    clipped = np.clip(signal, -1.0, 1.0)
    int16_data = (clipped * 32767.0).astype(np.int16)

    # Bước 3: Ghi dữ liệu ra đĩa
    wavfile.write(wav_path, sample_rate, int16_data)


def save_global_threshold_json(
    threshold:        float,
    output_dir:       str,
    feature_type:     FeatureType = FEATURE_TYPE,
    training_data:    str = "TinHieuHuanLuyen",
    threshold_method: str = "binary_search",
    noise_condition:  str = "original",
    filename:         str = "global_threshold.json",
) -> None:
    """
    Lưu cấu hình ngưỡng tối ưu toàn cục và các siêu tham số liên quan vào file JSON.

    Args:
        threshold (float): Giá trị ngưỡng tối ưu toàn cục tìm được.
        output_dir (str): Thư mục lưu trữ file kết quả.
        feature_type (FeatureType, optional): Kiểu đặc trưng đã sử dụng.
        training_data (str, optional): Tên bộ dữ liệu huấn luyện.
        threshold_method (str, optional): Phương pháp xác định ngưỡng.
        noise_condition (str, optional): Điều kiện nhiễu môi trường.
        filename (str, optional): Tên file JSON cần lưu. Mặc định 'global_threshold.json'.

    Returns:
        None
    """
    # Bước 1: Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    # Bước 2: Xuất dữ liệu cấu hình có cấu trúc ra file JSON
    payload = {
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
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("  Đã lưu cấu hình ngưỡng JSON: %s", path)


def save_global_threshold_txt(
    threshold: float,
    output_dir: str,
    filename: str = "global_threshold.txt",
) -> None:
    """
    Lưu báo cáo tóm tắt thông số ngưỡng tối ưu toàn cục dưới định dạng văn bản thô (.txt).

    Args:
        threshold (float): Giá trị ngưỡng tối ưu.
        output_dir (str): Thư mục lưu file text.
        filename (str, optional): Tên file văn bản. Mặc định 'global_threshold.txt'.

    Returns:
        None
    """
    # Bước 1: Tạo thư mục nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    # Bước 2: Ghi tóm tắt thông số cấu hình
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Feature      : {FEATURE_TYPE.value}\n"
                f"Frame length : {FRAME_LENGTH_MS} ms\n"
                f"Frame shift  : {FRAME_SHIFT_MS} ms\n"
                f"Min silence  : {MIN_SILENCE_DURATION_MS} ms\n"
                f"Threshold T  : {threshold:.10f}\n")

    logger.info("  Đã lưu tóm tắt ngưỡng TXT: %s", path)


def save_threshold_summary_csv(
    file_results:     List[Dict[str, Any]],
    global_threshold: float,
    output_dir:       str,
    eval_results:     Optional[List[Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lưu hai bảng CSV tổng hợp kết quả đánh giá: per-file threshold và global threshold.

    Bao gồm các cột: số khung speech/silence, giá trị threshold, MAE, RMSE, Precision, Recall, F1.

    Args:
        file_results (List[Dict[str, Any]]): Danh sách kết quả xử lý từng file huấn luyện.
        global_threshold (float): Ngưỡng toàn cục dùng chung.
        output_dir (str): Thư mục lưu trữ các file CSV.
        eval_results (Optional[List[Dict[str, Any]]], optional): Kết quả đánh giá ngưỡng toàn cục.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrame của bảng per-file và DataFrame của bảng global.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Bước 1: Xây dựng bảng kết quả ngưỡng theo từng file riêng biệt (Per-file)
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

    # Bước 2: Xây dựng bảng kết quả đánh giá ngưỡng chung toàn cục (Global Threshold)
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

    return df_pf, df_gl


def save_binary_search_history_csv(
    file_results:   List[Dict[str, Any]],
    global_history: List[BinarySearchState],
    output_dir:     str,
    filename:       str = "binary_search_history.csv",
) -> None:
    """
    Lưu lại toàn bộ lịch sử từng bước lặp của thuật toán Binary Search ra file CSV.

    Bao gồm các giá trị: Tmin, Tmax, T, Confusion Speech, Confusion Silence, Difference.

    Args:
        file_results (List[Dict[str, Any]]): Lịch sử tìm kiếm từng file riêng lẻ.
        global_history (List[BinarySearchState]): Lịch sử tìm kiếm ngưỡng toàn cục.
        output_dir (str): Thư mục lưu file CSV.
        filename (str, optional): Tên file CSV. Mặc định 'binary_search_history.csv'.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    # Bước 1: Ghi nhận lịch sử tối ưu cho từng file
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

    # Bước 2: Ghi nhận lịch sử tối ưu cho tập gộp Global
    for s in global_history:
        rows.append({
            "source": "GLOBAL", "iteration": s.iteration,
            "tmin": s.tmin, "tmax": s.tmax, "threshold": s.threshold,
            "confusion_sil": s.confusion_sil, "confusion_speech": s.confusion_speech,
            "difference": s.difference,
            "speech_below_T": s.speech_below_T, "silence_above_T": s.silence_above_T,
        })

    # Bước 3: Xuất dữ liệu ra file CSV
    path = os.path.join(output_dir, filename)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("  Đã lưu lịch sử Binary Search: %s", path)


def save_boundary_errors_csv(
    eval_results: List[Dict[str, Any]],
    output_dir:   str,
    filename:     str = "boundary_errors.csv",
) -> None:
    """
    Lưu chi tiết sai số ranh giới cho từng cặp ranh giới đã khớp (matched) và các ranh giới thừa/thiếu.

    Args:
        eval_results (List[Dict[str, Any]]): Danh sách kết quả đánh giá phân đoạn.
        output_dir (str): Thư mục lưu file CSV.
        filename (str, optional): Tên file CSV. Mặc định 'boundary_errors.csv'.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    # Bước 1: Quét từng file và lấy chi tiết từng mốc biên
    for er in eval_results or []:
        if er is None:
            continue
        err = er.get("error_metrics", {})
        name = er.get("basename", "?")

        # Cặp đã khớp thành công
        for pb, gb in err.get("matched_pairs", []):
            rows.append({
                "filename": name, "status": "matched",
                "pred_s": pb, "gt_s": gb, "error_ms": abs(pb - gb) * 1000.0,
            })
        # Mốc dự đoán thừa (không khớp GT)
        for b in err.get("unmatched_pred", []):
            rows.append({"filename": name, "status": "unmatched_pred", "pred_s": b, "gt_s": None, "error_ms": None})
        # Mốc GT bị bỏ sót
        for b in err.get("unmatched_gt", []):
            rows.append({"filename": name, "status": "unmatched_gt", "pred_s": None, "gt_s": b, "error_ms": None})

    # Bước 2: Lưu dữ liệu ra file CSV
    path = os.path.join(output_dir, filename)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("  Đã lưu chi tiết sai số ranh giới: %s", path)
