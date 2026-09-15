"""
segmentation.py – Module phân loại khung, tạo phân đoạn, lọc khoảng lặng ngắn và đánh giá sai số ranh giới.

Cung cấp các hàm xử lý hậu kỳ và đánh giá:
- classify_frames: Phân loại nhị phân từng khung dựa trên ngưỡng T.
- frames_to_segments: Ghép các khung cùng nhãn thành danh sách các đoạn thời gian (Segment).
- remove_short_silence_segments: Loại bỏ khoảng lặng ngắn (< 300 ms) theo quy định của bài toán.
- extract_boundaries: Trích xuất các mốc thời gian chuyển tiếp giữa các đoạn.
- match_boundaries: So khớp ranh giới dự đoán với ranh giới chuẩn Ground Truth.
- compute_boundary_errors: Đánh giá định lượng sai số MAE, RMSE (ms), Precision, Recall và F1-Score.
"""

import logging
from typing import Any, Dict, List, Tuple

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
    Phân loại nhị phân từng khung thời gian thành Speech (1) hoặc Silence (0) dựa trên ngưỡng T.

    Quy tắc:
    - Nếu speech_is_above=True (mặc định với MA, logMA, STE):
      Giá trị đặc trưng >= T được phân loại là Speech (1), ngược lại là Silence (0).
    - Nếu speech_is_above=False:
      Giá trị đặc trưng <= T được phân loại là Speech (1), ngược lại là Silence (0).

    Args:
        feature_values (np.ndarray): Mảng 1D float chứa giá trị đặc trưng của từng khung.
        threshold (float): Ngưỡng phân loại T.
        speech_is_above (bool, optional): Hướng so sánh với ngưỡng. Mặc định là True.

    Returns:
        np.ndarray: Mảng 1D int32 chứa nhãn phân loại (1 cho Speech, 0 cho Silence).
    """
    # So sánh giá trị đặc trưng của từng khung với ngưỡng T
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
    Chuyển đổi chuỗi nhãn cấp độ khung (frame-level) thành danh sách các đoạn liên tục (Segment).

    Ranh giới chuyển tiếp (boundary) giữa hai đoạn được tính bằng trung điểm thời gian
    giữa hai tâm khung kề nhau: boundary = (center[i-1] + center[i]) / 2.0.

    Args:
        labels (np.ndarray): Mảng 1D nhãn phân loại của các khung (1=speech, 0=silence).
        frame_centers (np.ndarray): Mảng 1D thời gian tâm của từng khung (giây).
        frame_shift_ms (int, optional): Độ dịch khung tính theo mili-giây. Mặc định 10ms.
        signal_duration_s (float, optional): Tổng thời lượng của file âm thanh (giây).

    Returns:
        List[Segment]: Danh sách các đoạn phân loại với thời gian bắt đầu, kết thúc và nhãn.
    """
    # Bước 1: Kiểm tra mảng nhãn đầu vào rỗng
    if len(labels) == 0:
        return []

    # Bước 2: Khởi tạo biến theo dõi đoạn hiện tại
    segs: List[Segment] = []
    cur_label = labels[0]
    cur_start = 0.0

    # Bước 3: Duyệt qua các khung để phát hiện thời điểm chuyển đổi nhãn
    for i in range(1, len(labels)):
        if labels[i] != cur_label:
            # Điểm phân tách là trung điểm giữa hai tâm khung kề nhau
            boundary = (frame_centers[i - 1] + frame_centers[i]) / 2.0
            if cur_label != -1:
                lbl_str = "speech" if cur_label == 1 else "silence"
                segs.append(Segment(cur_start, boundary, lbl_str))
            cur_label = labels[i]
            cur_start = boundary

    # Bước 4: Xử lý đoạn cuối cùng đến thời điểm kết thúc tín hiệu
    last_end = signal_duration_s if signal_duration_s > 0.0 else frame_centers[-1] + frame_shift_ms / 2000.0
    if cur_label != -1:
        lbl_str = "speech" if cur_label == 1 else "silence"
        segs.append(Segment(cur_start, last_end, lbl_str))

    return segs


def remove_short_silence_segments(
    segments:                List[Segment],
    min_silence_duration_ms: float = MIN_SILENCE_DURATION_MS,
) -> List[Segment]:
    """
    Loại bỏ các khoảng lặng ngắn hơn ngưỡng tối thiểu (mặc định 300 ms) bằng kỹ thuật ghép đoạn (merging).

    Quy tắc xử lý:
    1. Khoảng lặng ngắn nằm giữa hai đoạn tiếng nói -> Ghép cả 3 thành một đoạn tiếng nói duy nhất.
    2. Khoảng lặng ngắn nằm ở đầu tín hiệu -> Ghép vào đoạn kế tiếp.
    3. Khoảng lặng ngắn nằm ở cuối tín hiệu -> Ghép vào đoạn liền trước.
    4. Lặp lại quá trình cho đến khi không còn khoảng lặng nào ngắn hơn 300 ms.

    Args:
        segments (List[Segment]): Danh sách các đoạn phân loại ban đầu.
        min_silence_duration_ms (float, optional): Thời lượng tối thiểu của khoảng lặng (ms). Mặc định 300ms.

    Returns:
        List[Segment]: Danh sách các đoạn sau khi đã lọc sạch khoảng lặng ảo.
    """
    # Bước 1: Kiểm tra danh sách rỗng
    if not segments:
        return []

    min_s = min_silence_duration_ms / 1000.0
    changed = True
    result = list(segments)

    # Bước 2: Vòng lặp lọc cho đến khi trạng thái danh sách hội tụ ổn định
    while changed:
        changed = False
        out: List[Segment] = []
        i = 0
        while i < len(result):
            seg = result[i]
            # Kiểm tra xem đoạn hiện tại có phải khoảng lặng quá ngắn hay không
            if seg.label == "silence" and seg.duration_s < min_s:
                changed  = True
                has_prev = i > 0
                has_next = i < len(result) - 1

                # Trường hợp 1: Nằm giữa hai đoạn tiếng nói -> Merge thành speech dài
                if has_prev and has_next:
                    prev = out.pop()
                    merged = Segment(prev.start, result[i + 1].end, "speech")
                    out.append(merged)
                    i += 2
                # Trường hợp 2: Nằm ở cuối -> Ghép vào đoạn trước
                elif has_prev:
                    prev = out.pop()
                    out.append(Segment(prev.start, seg.end, prev.label))
                    i += 1
                # Trường hợp 3: Nằm ở đầu -> Ghép vào đoạn kế tiếp
                elif has_next:
                    nxt = result[i + 1]
                    out.append(Segment(seg.start, nxt.end, nxt.label))
                    i += 2
                else:
                    out.append(seg)
                    i += 1
            else:
                out.append(seg)
                i += 1
        result = out

    # Bước 3: Ghép lại các đoạn liền kề nếu chúng có cùng nhãn sau quá trình xóa
    merged: List[Segment] = []
    for seg in result:
        if merged and merged[-1].label == seg.label:
            merged[-1] = Segment(merged[-1].start, seg.end, seg.label)
        else:
            merged.append(Segment(seg.start, seg.end, seg.label))

    return merged


def extract_boundaries(segments: List[Segment]) -> List[float]:
    """
    Trích xuất danh sách các mốc thời gian ranh giới nội bộ từ các phân đoạn.

    Bỏ qua mốc 0.0 (đầu file) và mốc kết thúc tín hiệu (cuối file).

    Args:
        segments (List[Segment]): Danh sách các phân đoạn.

    Returns:
        List[float]: Danh sách thời điểm các biên ranh giới (giây).
    """
    return [s.end for s in segments[:-1]] if segments else []


def match_boundaries(
    pred: List[float],
    gt:   List[float],
    tolerance_s: float = BOUNDARY_MATCH_TOLERANCE_MS / 1000.0,
) -> Tuple[List[Tuple[float, float]], List[float], List[float]]:
    """
    So khớp ranh giới dự đoán và ranh giới chuẩn ground truth theo thuật toán láng giềng gần nhất (Nearest Neighbor).

    Một ranh giới Ground Truth được coi là khớp (matched) với ranh giới dự đoán gần nhất
    nếu khoảng cách thời gian giữa chúng <= tolerance_s (mặc định 100 ms).

    Args:
        pred (List[float]): Danh sách mốc ranh giới do thuật toán dự đoán (giây).
        gt (List[float]): Danh sách mốc ranh giới chuẩn từ file .lab (giây).
        tolerance_s (float, optional): Ngưỡng sai số cho phép để coi là khớp (giây). Mặc định 0.1s (100ms).

    Returns:
        Tuple[List[Tuple[float, float]], List[float], List[float]]:
            - matched_pairs: Danh sách các cặp (pred_boundary, gt_boundary) đã khớp thành công.
            - unmatched_pred: Danh sách các ranh giới dự đoán thừa (không khớp với mốc GT nào).
            - unmatched_gt: Danh sách các ranh giới GT bị bỏ sót (không có ranh giới dự đoán nào kề bên).
    """
    # Bước 1: Khởi tạo mảng đánh dấu các ranh giới dự đoán đã được sử dụng
    used = [False] * len(pred)
    matched: List[Tuple[float, float]] = []
    unmatched_gt: List[float] = []

    # Bước 2: Quét từng mốc ranh giới Ground Truth để tìm mốc dự đoán gần nhất
    for g in gt:
        best_i, best_d = -1, float("inf")
        for j, p in enumerate(pred):
            if not used[j] and abs(p - g) < best_d:
                best_d, best_i = abs(p - g), j

        # Kiểm tra khoảng cách có nằm trong dung sai sai số cho phép hay không
        if best_i >= 0 and best_d <= tolerance_s:
            matched.append((pred[best_i], g))
            used[best_i] = True
        else:
            unmatched_gt.append(g)

    # Bước 3: Thu thập các ranh giới dự đoán chưa được ghép cặp (dự đoán thừa)
    unmatched_pred = [pred[j] for j in range(len(pred)) if not used[j]]

    return matched, unmatched_pred, unmatched_gt


def compute_boundary_errors(
    pred_segments: List[Segment],
    gt_segments:   List[Segment],
    tolerance_s:   float = BOUNDARY_MATCH_TOLERANCE_MS / 1000.0,
) -> Dict[str, Any]:
    """
    Đánh giá định lượng độ chính xác phân đoạn ranh giới so với Ground Truth chuẩn.

    Các chỉ số tính toán bao gồm:
    - MAE (Mean Absolute Error): Sai số tuyệt đối trung bình giữa các biên đã khớp (ms).
    - RMSE (Root Mean Squared Error): Căn bậc hai sai số toàn phương trung bình (ms).
    - Boundary Precision: Tỷ lệ ranh giới tìm được là đúng = N_matched / N_pred.
    - Boundary Recall: Tỷ lệ ranh giới chuẩn được tìm thấy = N_matched / N_gt.
    - Boundary F1-Score: Trung bình điều hoà giữa Precision và Recall.

    Args:
        pred_segments (List[Segment]): Danh sách phân đoạn do thuật toán xác định.
        gt_segments (List[Segment]): Danh sách phân đoạn chuẩn Ground Truth từ .lab.
        tolerance_s (float, optional): Dung sai chấp nhận khớp biên (giây). Mặc định 0.1s.

    Returns:
        Dict[str, Any]: Từ điển chứa đầy đủ các chỉ số định lượng đánh giá ranh giới.
    """
    # Bước 1: Trích xuất các mốc biên nội bộ từ danh sách segment
    pred_bounds = extract_boundaries(pred_segments)
    gt_bounds   = extract_boundaries(gt_segments)

    # Bước 2: So khớp các cặp ranh giới trong khoảng sai số cho phép
    matched, unmatched_pred, unmatched_gt = match_boundaries(pred_bounds, gt_bounds, tolerance_s)

    n_pred = len(pred_bounds)
    n_gt   = len(gt_bounds)
    n_m    = len(matched)

    # Bước 3: Tính toán các chỉ số phân loại Precision, Recall và F1-Score
    precision = (n_m / n_pred) if n_pred > 0 else (1.0 if n_gt == 0 else 0.0)
    recall    = (n_m / n_gt)   if n_gt > 0   else (1.0 if n_pred == 0 else 0.0)
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

    # Bước 4: Xử lý trường hợp không khớp được cặp ranh giới nào
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

    # Bước 5: Tính sai số MAE và RMSE (đơn vị: mili-giây) trên các cặp ranh giới đã khớp
    errors_ms = [abs(p - g) * 1000.0 for p, g in matched]
    arr = np.array(errors_ms, dtype=np.float64)

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
