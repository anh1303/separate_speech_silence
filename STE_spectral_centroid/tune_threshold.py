"""
tune_threshold.py
Script rieng (khong phai main.py) dung de DO gia tri toi uu cho tham so W
(trong so cong thuc nguong T = (W*M1+M2)/(W+1)) bang tap TinHieuHuanLuyen.

Cach dung: sua W_CANDIDATES ben duoi neu muon, roi chay:
    python tune_threshold.py

Script se:
    1. Doc tat ca cap file (.wav, .lab) trong data/TinHieuHuanLuyen/
    2. Voi moi gia tri W trong W_CANDIDATES: chay pipeline tren TAT CA file
       huan luyen, tinh RMSE/MAE trung binh (bo qua file nao khong ghep
       duoc cap bien nao -- xem canh bao in ra)
    3. In bang ket qua, ve do thi RMSE/MAE theo W, va goi y W tot nhat
       (RMSE trung binh nho nhat)

LUU Y: day chi la vi du DO MOT MINH tham so W. Cac tham so khac
(num_bins, smooth_window, min_silence_ms, extend_frames) trong CONFIG hien
dang giu co dinh theo main.py -- neu muon do dong thoi nhieu tham so, co
the mo rong vong lap thanh nhieu lop (grid search) theo cung logic ben duoi.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from main import process_one_file, CONFIG

TRAIN_DIR = os.path.join("data", "TinHieuHuanLuyen")

# Danh sach cac gia tri W can thu nghiem -- co the doi khoang gia tri nay
# tuy theo ket qua ban dau (vd neu W=20 van la tot nhat, nen mo rong len 30, 40...)
W_CANDIDATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 ,0.8, 0.9 ,1, 2, 3, 5, 7, 10, 15, 20]


def load_training_pairs(train_dir):
    """
    Tim tat ca cap file (.wav, .lab) hop le trong thu muc huan luyen.

    Input:
        train_dir (str): duong dan thu muc chua file huan luyen

    Output:
        pairs (list of tuple): danh sach (wav_path, lab_path), chi gom cac
            cap ma CA HAI file deu ton tai
    """
    wav_files = sorted(glob.glob(os.path.join(train_dir, "*.wav")))
    pairs = []
    for wav_path in wav_files:
        lab_path = os.path.splitext(wav_path)[0] + ".lab"
        if os.path.exists(lab_path):
            pairs.append((wav_path, lab_path))
        else:
            print(f"  [Cảnh báo] Bỏ qua {wav_path}: không tìm thấy file .lab tương ứng")
    return pairs


def evaluate_W(W, training_pairs, base_config):
    """
    Chay pipeline tren toan bo tap huan luyen voi 1 gia tri W cu the, tra ve
    RMSE/MAE trung binh tren cac file (bo qua file bi NaN vi khong ghep
    duoc cap bien nao).

    Input:
        W (float): gia tri W dang can thu nghiem
        training_pairs (list of tuple): output cua load_training_pairs()
        base_config (dict): CONFIG goc trong main.py, se bi ghi de gia tri "W"

    Output:
        mean_rmse_ms (float): RMSE trung binh (ms) tren cac file hop le
        mean_mae_ms (float): MAE trung binh (ms) tren cac file hop le
        per_file_results (list of dict): chi tiet RMSE/MAE tung file, de in
            bang debug neu can
    """
    config = dict(base_config)  # copy de khong lam thay doi CONFIG goc
    config["W"] = W

    rmse_list, mae_list, per_file_results = [], [], []
    for wav_path, lab_path in training_pairs:
        result = process_one_file(wav_path, lab_path, config)
        filename = os.path.basename(wav_path)
        per_file_results.append({
            "filename": filename,
            "rmse_ms": result["rmse_ms"],
            "mae_ms": result["mae_ms"],
        })
        # Bo qua file NaN (truong hop khong ghep duoc cap bien nao, vd
        # thuat toan khong tim ra bien nao ca) khi tinh trung binh
        if not np.isnan(result["rmse_ms"]):
            rmse_list.append(result["rmse_ms"])
            mae_list.append(result["mae_ms"])

    mean_rmse_ms = np.mean(rmse_list) if rmse_list else float("nan")
    mean_mae_ms = np.mean(mae_list) if mae_list else float("nan")
    return mean_rmse_ms, mean_mae_ms, per_file_results


def main():
    training_pairs = load_training_pairs(TRAIN_DIR)
    if len(training_pairs) == 0:
        print(f"Không tìm thấy cặp file .wav/.lab nào trong {TRAIN_DIR}. "
              f"Hãy copy dữ liệu huấn luyện vào thư mục này trước khi chạy.")
        return

    print(f"Đã tìm thấy {len(training_pairs)} file huấn luyện. Bắt đầu dò W...\n")

    results = []  # list of (W, mean_rmse_ms, mean_mae_ms)
    for W in W_CANDIDATES:
        mean_rmse_ms, mean_mae_ms, per_file = evaluate_W(W, training_pairs, CONFIG)
        results.append((W, mean_rmse_ms, mean_mae_ms))
        print(f"W = {W:>4} | RMSE trung bình = {mean_rmse_ms:7.1f} ms "
              f"| MAE trung bình = {mean_mae_ms:7.1f} ms")

    # --- Chon W tot nhat: RMSE trung binh nho nhat (bo qua NaN) ---
    valid_results = [r for r in results if not np.isnan(r[1])]
    if not valid_results:
        print("\nKhông có giá trị W nào cho kết quả hợp lệ -- kiểm tra lại pipeline.")
        return

    best_W, best_rmse, best_mae = min(valid_results, key=lambda r: r[1])
    print(f"\n>>> W tốt nhất (RMSE nhỏ nhất): W = {best_W} "
          f"(RMSE = {best_rmse:.1f} ms, MAE = {best_mae:.1f} ms)")
    print(f">>> Hãy cập nhật CONFIG['W'] = {best_W} trong main.py")

    # --- Ve do thi RMSE/MAE theo W de quan sat truc quan xu huong ---
    W_values = [r[0] for r in results]
    rmse_values = [r[1] for r in results]
    mae_values = [r[2] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(W_values, rmse_values, marker="o", label="RMSE trung bình")
    ax.plot(W_values, mae_values, marker="s", label="MAE trung bình")
    ax.axvline(best_W, color="red", linestyle="--", alpha=0.5, label=f"W tốt nhất = {best_W}")
    ax.set_xlabel("W")
    ax.set_ylabel("Sai số biên (ms)")
    ax.set_title("Sai số trung bình trên tập huấn luyện theo giá trị W")
    ax.legend()
    fig.tight_layout()
    fig.savefig("output/tune_W_result.png", dpi=120)
    print("\nĐã lưu đồ thị RMSE/MAE theo W vào output/tune_W_result.png")
    plt.show()


if __name__ == "__main__":
    main()
