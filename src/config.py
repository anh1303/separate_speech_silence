"""
config.py – Module định nghĩa tham số cấu hình toàn cục và các cấu trúc dữ liệu của dự án.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class FeatureType(Enum):
    """Kiểu đặc trưng biên độ ngắn hạn được hỗ trợ."""
    MA      = "MA"
    LOG_MA  = "logMA"
    STE     = "STE"
    LOG_STE = "logSTE"


# ── Frame parameters ───────────────────────────────────────────────────────────
FRAME_LENGTH_MS: int = 20
FRAME_SHIFT_MS:  int = 10

# ── Feature parameters ─────────────────────────────────────────────────────────
FEATURE_TYPE: FeatureType = FeatureType.LOG_MA
LOG_EPSILON:  float       = 1e-10
SPEECH_IS_ABOVE_THRESHOLD: bool = True

# ── Binary search parameters ───────────────────────────────────────────────────
MAX_ITERATIONS:      int   = 100
THRESHOLD_TOLERANCE: float = 1e-9

# ── Post-processing parameters ─────────────────────────────────────────────────
MIN_SILENCE_DURATION_MS: float = 300.0

# ── Boundary matching parameters ───────────────────────────────────────────────
BOUNDARY_MATCH_TOLERANCE_MS: float = 100.0

# ── Paths ──────────────────────────────────────────────────────────────────────
ORIGINAL_TRAINING_DIR: str = "./TinHieuHuanLuyen"
TEST_DIR:              str = "./TinHieuKiemThu"
OUTPUT_DIR:            str = "./output"

# Alias để tương thích ngược
TRAINING_DIR:          str = ORIGINAL_TRAINING_DIR


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class LabSegment:
    """Cấu trúc biểu diễn một đoạn nhãn đọc từ file .lab."""
    start: float
    end:   float
    label: str   # "sil" | "v" | "uv"

    @property
    def duration(self) -> float:
        """
        Tính thời lượng của đoạn nhãn theo đơn vị giây.

        Args:
            None (sử dụng thuộc tính đối tượng self).

        Returns:
            float: Thời lượng của đoạn (end - start).
        """
        return self.end - self.start

    def is_silence(self) -> bool:
        """
        Kiểm tra xem đoạn nhãn có phải là khoảng lặng hay không.

        Args:
            None (sử dụng thuộc tính đối tượng self).

        Returns:
            bool: True nếu nhãn là 'sil', ngược lại False.
        """
        return self.label == "sil"

    def is_speech(self) -> bool:
        """
        Kiểm tra xem đoạn nhãn có phải là tiếng nói (hữu thanh hoặc vô thanh) hay không.

        Args:
            None (sử dụng thuộc tính đối tượng self).

        Returns:
            bool: True nếu nhãn là 'v' hoặc 'uv', ngược lại False.
        """
        return self.label in ("v", "uv")


@dataclass
class Segment:
    """Cấu trúc biểu diễn một đoạn phân loại nhị phân Speech hoặc Silence."""
    start: float
    end:   float
    label: str   # "speech" | "silence"

    @property
    def duration_ms(self) -> float:
        """
        Tính thời lượng đoạn theo đơn vị mili-giây.

        Args:
            None (sử dụng thuộc tính đối tượng self).

        Returns:
            float: Thời lượng tính bằng mili-giây (ms).
        """
        return (self.end - self.start) * 1000.0

    @property
    def duration_s(self) -> float:
        """
        Tính thời lượng đoạn theo đơn vị giây.

        Args:
            None (sử dụng thuộc tính đối tượng self).

        Returns:
            float: Thời lượng tính bằng giây (s).
        """
        return self.end - self.start


@dataclass
class BinarySearchState:
    """Lưu lại trạng thái chi tiết của từng bước lặp trong thuật toán Binary Search."""
    iteration:        int
    tmin:             float
    tmax:             float
    threshold:        float
    confusion_sil:    float
    confusion_speech: float
    difference:       float
    speech_below_T:   int
    silence_above_T:  int
