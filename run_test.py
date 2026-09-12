#!/usr/bin/env python3
"""
run_test.py – File thực thi chính để đánh giá trên bộ dữ liệu kiểm thử (TinHieuKiemThu).
Sử dụng trong buổi chấm thi của Giáo viên:
    python run_test.py
    python run_test.py --demo   # Xem trước 4 figures và bảng điểm trên dữ liệu mẫu
"""

import sys
from pathlib import Path

# Thêm root vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test.evaluate_test import main

if __name__ == "__main__":
    main()
