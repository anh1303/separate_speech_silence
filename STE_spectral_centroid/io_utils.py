"""
io_utils.py
Các hàm đọc dữ liệu đầu vào: file tín hiệu .wav và file nhãn chuẩn .lab.
KHONG chứa logic xử lý tín hiệu -> chỉ đọc/parse dữ liệu thô, nên được phép
dùng scipy.io.wavfile (không vi phạm luật "tự code hàm xử lý tín hiệu").
"""

import numpy as np
from scipy.io import wavfile


def read_wav(filepath):
    """
    Đọc file .wav và trả về tín hiệu dạng mảng số thực đã chuẩn hoá về [-1, 1].

    Input:
        filepath (str): đường dẫn tới file .wav

    Output:
        fs (int): tần số lấy mẫu (Hz)
        signal (np.ndarray, shape (N,)): tín hiệu âm thanh, mono, kiểu float64
    """
    fs, signal = wavfile.read(filepath)

    # Nếu tín hiệu là stereo (2 kênh), lấy trung bình 2 kênh -> mono
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    # Chuẩn hoá biên độ về [-1, 1] tuỳ theo kiểu dữ liệu gốc (int16, int32, float...)
    if np.issubdtype(signal.dtype, np.integer):
        max_val = np.iinfo(signal.dtype).max
        signal = signal.astype(np.float64) / max_val
    else:
        signal = signal.astype(np.float64)

    return fs, signal


def read_lab_file(filepath):
    """
    Đọc file .lab (định dạng mô tả trong README.txt) chứa các đoạn nhãn
    (sil/v/uv) và giá trị F0mean, F0std.

    Input:
        filepath (str): đường dẫn tới file .lab tương ứng file .wav

    Output:
        segments (list of tuple): danh sách (bien_trai, bien_phai, nhan),
            đơn vị thời gian: giây, nhan la 'sil'/'v'/'uv'
        f0_mean (float): trung bình F0 (Hz), None nếu không có trong file
        f0_std (float): độ lệch chuẩn F0 (Hz), None nếu không có trong file
    """
    segments = []
    f0_mean, f0_std = None, None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue

            # 2 dong cuoi la F0mean / F0std, con lai la cac dong bien thoi gian
            if parts[0] == "F0mean":
                f0_mean = float(parts[1])
            elif parts[0] == "F0std":
                f0_std = float(parts[1])
            elif len(parts) == 3:
                left, right, label = float(parts[0]), float(parts[1]), parts[2]
                segments.append((left, right, label))

    return segments, f0_mean, f0_std


def get_speech_boundaries_from_lab(segments):
    """
    Trích các biên thời gian phân tách khoảng lặng (sil) và tiếng nói (v/uv)
    từ danh sách segments đọc được từ file .lab. Dùng để so sánh với biên
    do thuật toán tự tìm ra (tính RMSE/MAE).

    Input:
        segments (list of tuple): output của read_lab_file(), dạng
            (bien_trai, bien_phai, nhan)

    Output:
        boundaries (list of float): các mốc thời gian (giây) tại điểm
            chuyển tiếp giữa 'sil' và 'v'/'uv' (không lấy biên đầu/cuối file)
    """
    boundaries = []
    for i in range(1, len(segments)):
        prev_label = segments[i - 1][2]
        curr_label = segments[i][2]
        is_transition = (prev_label == "sil") != (curr_label == "sil")
        if is_transition:
            boundaries.append(segments[i][0])  # bien_trai cua doan hien tai
    return boundaries
