# BT1 - Phân đoạn tín hiệu thành tiếng nói và khoảng lặng

Cài đặt thuật toán histogram-based (Giannakopoulos, 2014).

## Cài môi trường

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Cấu trúc thư mục

```
BT1_SilenceRemoval/
├── main.py              # điểm chạy duy nhất, xuất 4 figure cho 4 file test
├── io_utils.py           # đọc .wav và .lab
├── segmentation.py       # lõi thuật toán (framing, STE, centroid, ngưỡng, hậu xử lý)
├── evaluation.py         # RMSE/MAE giữa biên groundtruth và biên tự động
├── visualization.py      # vẽ hình theo đúng yêu cầu đề bài
├── data/
│   ├── TinHieuHuanLuyen/  # file huấn luyện
│   └── TinHieuKiemThu/    # file kiểm thử 
└── requirements.txt
```

