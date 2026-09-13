"""
main.py
Diem khoi chay duy nhat cua chuong trinh (theo dung yeu cau de bai: CT chay
1 lan tu main.py, duyet 4 file tin hieu kiem thu, xuat 4 figure).

Quy trinh:
    1. Nap tham so da "huan luyen" duoc (W, num_bins, min_silence_ms, ...)
       tren tap TinHieuHuanLuyen -- xem tune_threshold.py de tim lai cac
       gia tri nay, roi dien ket qua vao CONFIG ben duoi.
    2. Voi moi file trong TinHieuKiemThu: doc wav + lab -> tinh dac trung ->
       uoc luong nguong -> phan doan -> hau xu ly -> tinh RMSE/MAE -> ve hinh.
    3. Sap xep 4 cua so figure vao 4 goc man hinh.
"""

import os
import glob
import matplotlib.pyplot as plt

from io_utils import read_wav, read_lab_file, get_speech_boundaries_from_lab
from segmentation import (
    frame_signal,
    compute_short_time_energy,
    compute_spectral_centroid,
    estimate_threshold,
    detect_speech_frames,
    postprocess_segments,
    get_boundaries_from_segments,
)
from evaluation import match_boundaries, compute_rmse_mae
from visualization import plot_result_for_one_file


# ---------------------------------------------------------------------------
# CONFIG: cac sieu tham so, se duoc dien gia tri toi uu sau khi thu nghiem
# tren tap TinHieuHuanLuyen (xem cau hoi truoc ve y nghia "tin hieu huan luyen")
# ---------------------------------------------------------------------------
CONFIG = {
    "frame_length_ms": 30,      # theo paper
    "num_bins": 50,             # so bin histogram, TODO: tinh chinh
    "W": 1,                     # trong so cong thuc nguong, TODO: tinh chinh
    "smooth_window": 5,         # do rong cua so lam muot histogram, TODO
    "min_silence_ms": 300,      # yeu cau rieng cua de bai (loai khoang lang ao)
    "extend_frames": 3,         # noi dai doan tieng noi (~250ms, theo paper)
}

TEST_DIR = os.path.join("data", "TinHieuHuanLuyen")
OUTPUT_DIR = "output"


def process_one_file(wav_path, lab_path, config):
    """
    Chay toan bo pipeline phan doan cho 1 cap file (.wav, .lab).

    Input:
        wav_path (str): duong dan file tin hieu .wav
        lab_path (str): duong dan file nhan groundtruth .lab tuong ung
        config (dict): cac sieu tham so trong CONFIG

    Output:
        result (dict): tap hop toan bo ket qua trung gian va cuoi cung
            (dung de ve hinh va bao cao), gom: signal, fs, energy, centroid,
            energy_threshold, centroid_threshold, energy_debug_info,
            centroid_debug_info, pred_boundaries, gt_boundaries, rmse_ms, mae_ms
    """
    # --- Doc du lieu ---
    fs, signal = read_wav(wav_path)
    lab_segments, f0_mean, f0_std = read_lab_file(lab_path)
    gt_boundaries = get_speech_boundaries_from_lab(lab_segments)

    # --- Trich dac trung ---
    frames, frame_len_samples = frame_signal(signal, fs, config["frame_length_ms"])
    energy = compute_short_time_energy(frames)
    centroid = compute_spectral_centroid(frames, fs)

    # --- Uoc luong nguong: TINH RIENG cho file dang xet (moi file co
    #     histogram/M1/M2/T cua rieng no, vi muc nhieu nen SNR khac nhau
    #     giua cac file). W dung chung, duoc do toi uu tren tap huan luyen
    #     (xem tune_threshold.py). debug_info duoc giu lai de ve histogram. ---
    energy_threshold, energy_debug_info = estimate_threshold(
        energy, config["num_bins"], config["W"], config["smooth_window"]
    )
    centroid_threshold, centroid_debug_info = estimate_threshold(
        centroid, config["num_bins"], config["W"], config["smooth_window"]
    )

    # --- Gan nhan + hau xu ly ---
    is_speech = detect_speech_frames(energy, centroid, energy_threshold, centroid_threshold)
    segments = postprocess_segments(
        is_speech, config["frame_length_ms"], config["min_silence_ms"], config["extend_frames"]
    )
    pred_boundaries = get_boundaries_from_segments(segments)

    # --- Danh gia dinh luong ---
    pairs = match_boundaries(gt_boundaries, pred_boundaries)
    rmse_ms, mae_ms = compute_rmse_mae(pairs)

    return {
        "signal": signal,
        "fs": fs,
        "energy": energy,
        "centroid": centroid,
        "energy_threshold": energy_threshold,
        "centroid_threshold": centroid_threshold,
        "energy_debug_info": energy_debug_info,
        "centroid_debug_info": centroid_debug_info,
        "pred_boundaries": pred_boundaries,
        "gt_boundaries": gt_boundaries,
        "rmse_ms": rmse_ms,
        "mae_ms": mae_ms,
    }


def arrange_figure_at_corner(fig, corner_index, num_cols=2):
    """
    Dat vi tri cua so figure vao 1 trong 4 goc man hinh (chi hoat dong voi
    mot so backend GUI cua matplotlib, vd TkAgg/Qt5Agg; bo qua neu backend
    khong ho tro di chuyen cua so).

    Input:
        fig (matplotlib.figure.Figure): figure can dat vi tri
        corner_index (int): chi so 0..3 (0=tren-trai,1=tren-phai,
            2=duoi-trai,3=duoi-phai)
        num_cols (int): so cot luoi de tinh vi tri (mac dinh 2x2)

    Output:
        None (thay doi vi tri cua so figure "in-place")
    """
    try:
        mgr = fig.canvas.manager
        # TODO: tuy backend (TkAgg/Qt5Agg/WXAgg) ma cach dat vi tri khac nhau,
        #       tham khao doc matplotlib de goi dung ham (vd mgr.window.wm_geometry
        #       cho TkAgg). Day chi la buoc "lam dep" khi demo, khong anh huong
        #       ket qua thuat toan.
        pass
    except Exception:
        pass  # bo qua neu backend khong ho tro


def main():
    wav_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.wav")))

    if len(wav_files) == 0:
        print(f"Khong tim thay file .wav nao trong {TEST_DIR}. "
              f"Hay copy 4 file tin hieu kiem thu vao thu muc nay truoc khi chay.")
        return

    # Dam bao thu muc luu anh ket qua ton tai truoc khi ghi file
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    figures = []
    for idx, wav_path in enumerate(wav_files):
        lab_path = os.path.splitext(wav_path)[0] + ".lab"
        filename = os.path.basename(wav_path)

        print(f"[{idx + 1}/{len(wav_files)}] Dang xu ly {filename} ...")
        result = process_one_file(wav_path, lab_path, CONFIG)

        fig = plot_result_for_one_file(
            signal=result["signal"],
            fs=result["fs"],
            energy=result["energy"],
            centroid=result["centroid"],
            energy_threshold=result["energy_threshold"],
            centroid_threshold=result["centroid_threshold"],
            energy_debug_info=result["energy_debug_info"],
            centroid_debug_info=result["centroid_debug_info"],
            pred_boundaries=result["pred_boundaries"],
            gt_boundaries=result["gt_boundaries"],
            filename=filename,
            rmse_ms=result["rmse_ms"],
            mae_ms=result["mae_ms"],
        )
        arrange_figure_at_corner(fig, idx)
        figures.append(fig)

        # Luu anh ra output/<ten_file_khong_duoi>.png (dpi cao de xem ro
        # khi chen vao slide bao cao)
        out_path = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + ".png")
        fig.savefig(out_path, dpi=150)
        print(f"    Đã lưu hình: {out_path}")

        print(f"    RMSE = {result['rmse_ms']:.1f} ms | MAE = {result['mae_ms']:.1f} ms")

    plt.show()  # giu tat ca 4 cua so mo cung luc de GV quan sat



if __name__ == "__main__":
    main()
