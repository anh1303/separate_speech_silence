#!/usr/bin/env python3
"""
run_training.py – File thực thi chính để huấn luyện ngưỡng tối ưu trên tập dữ liệu gốc.
Sử dụng:
    python run_training.py
"""

import sys
from pathlib import Path

# Thêm root vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train.train_threshold import main

if __name__ == "__main__":
    main()
