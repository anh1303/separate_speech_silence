"""config.py – Tham số cấu hình toàn cục và các cấu trúc dữ liệu."""

from dataclasses import dataclass
from enum import Enum
from typing import List


class FeatureType(Enum):
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

# Alias để tương thích
TRAINING_DIR:          str = ORIGINAL_TRAINING_DIR


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class LabSegment:
    start: float
    end:   float
    label: str   # "sil" | "v" | "uv"

    @property
    def duration(self) -> float:
        return self.end - self.start

    def is_silence(self) -> bool:
        return self.label == "sil"

    def is_speech(self) -> bool:
        return self.label in ("v", "uv")


@dataclass
class Segment:
    start: float
    end:   float
    label: str   # "speech" | "silence"

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000.0

    @property
    def duration_s(self) -> float:
        return self.end - self.start


@dataclass
class BinarySearchState:
    iteration:        int
    tmin:             float
    tmax:             float
    threshold:        float
    confusion_sil:    float
    confusion_speech: float
    difference:       float
    speech_below_T:   int
    silence_above_T:  int
