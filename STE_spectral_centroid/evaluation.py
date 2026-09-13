"""
evaluation.py
Danh gia dinh luong do chinh xac cua thuat toan phan doan: so sanh bien
groundtruth (tu file .lab) voi bien tu dong tim duoc, bang RMSE/MAE (ms).
"""

import numpy as np


def match_boundaries(gt_boundaries, pred_boundaries):
    """
    Ghep cap moi bien groundtruth voi bien du doan GAN NHAT (khoang cach
    thoi gian nho nhat). Dung truoc khi tinh RMSE/MAE vi 2 danh sach bien
    co the khac so luong phan tu.

    Input:
        gt_boundaries (list of float): bien chuan, don vi giay
        pred_boundaries (list of float): bien thuat toan tu tim, don vi giay

    Output:
        pairs (list of tuple): danh sach (gt_time, pred_time) da ghep cap,
            do dai = min(len(gt_boundaries), len(pred_boundaries)) neu ghep
            1-1 don gian, hoac it hon neu loai bo cap trung lap
    """
    pairs = []
    used_pred = set()

    # Voi moi bien groundtruth, tim bien du doan gan nhat CHUA duoc dung.
    # Cach lam tham (greedy) nay don gian, du dung cho so luong bien nho
    # (vai chuc bien moi file) nhu trong bai toan nay.
    for gt_time in gt_boundaries:
        best_idx, best_dist = None, None
        for idx, pred_time in enumerate(pred_boundaries):
            if idx in used_pred:
                continue
            dist = abs(pred_time - gt_time)
            if best_dist is None or dist < best_dist:
                best_dist, best_idx = dist, idx

        if best_idx is not None:
            pairs.append((gt_time, pred_boundaries[best_idx]))
            used_pred.add(best_idx)

    return pairs

def compute_rmse_mae(pairs):
    """
    Tinh RMSE va MAE (don vi: mili-giay) giua cac cap bien da ghep.

    Input:
        pairs (list of tuple): danh sach (gt_time_s, pred_time_s), don vi giay

    Output:
        rmse_ms (float): Root Mean Squared Error, don vi ms
        mae_ms (float): Mean Absolute Error, don vi ms
    """
    if len(pairs) == 0:
        return float("nan"), float("nan")

    # Doi tung cap sai so tu giay sang ms, roi ap dung dung cong thuc
    # RMSE = sqrt(mean(error^2)) va MAE = mean(|error|)
    errors_ms = np.array([(pred - gt) * 1000.0 for gt, pred in pairs])
    rmse_ms = np.sqrt(np.mean(errors_ms ** 2))
    mae_ms = np.mean(np.abs(errors_ms))
    return rmse_ms, mae_ms
