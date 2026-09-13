"""
segmentation.py
Cai dat thuat toan phan doan tieng noi/khoang lang theo phuong phap histogram
(Giannakopoulos, 2014): dung dac trung Short-Time Energy (STE) va Spectral
Centroid, uoc luong nguong bang histogram + tim cuc dai cuc bo.

Chi dung cac ham built-in co ban cua numpy (sum, mean, abs, sqrt, fft...),
KHONG dung cac ham toolbox xu ly tin hieu/am thanh lam san (vd scipy.signal).
"""

import numpy as np


# ---------------------------------------------------------------------------
# BUOC 1: Chia khung (framing)
# ---------------------------------------------------------------------------
def frame_signal(signal, fs, frame_length_ms=50, overlap_ratio=0.0):
    """
    Chia tin hieu thanh cac khung ngan han (frames), khong chong lan theo
    mac dinh cua paper (frame 50ms, overlap_ratio=0).

    Input:
        signal (np.ndarray): tin hieu am thanh (mono, da chuan hoa)
        fs (int): tan so lay mau (Hz)
        frame_length_ms (float): do dai 1 khung, don vi ms (mac dinh 50ms)
        overlap_ratio (float): ty le chong lan giua 2 khung lien tiep (0..1)

    Output:
        frames (np.ndarray, shape (num_frames, frame_length_samples)):
            ma tran cac khung tin hieu
        frame_length_samples (int): so mau trong 1 khung
    """
    # Quy doi do dai khung tu ms sang so mau, va tinh buoc nhay (hop) giua
    # 2 khung lien tiep dua tren ty le chong lan mong muon
    frame_length_samples = int(round(frame_length_ms / 1000.0 * fs))
    hop_size = int(round(frame_length_samples * (1.0 - overlap_ratio)))
    hop_size = max(hop_size, 1)  # tranh vong lap vo han neu overlap_ratio=1

    # So khung toi da co the cat duoc ma khong vuot qua do dai tin hieu;
    # phan du cuoi cung (khong du 1 khung tron) se bi bo qua
    num_frames = 1 + (len(signal) - frame_length_samples) // hop_size
    num_frames = max(num_frames, 0)

    frames = np.zeros((num_frames, frame_length_samples))
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_length_samples
        frames[i, :] = signal[start:end]

    return frames, frame_length_samples


def get_frame_times(num_frames, frame_length_ms, overlap_ratio=0.0):
    """
    Tinh moc thoi gian (giay) tai TAM cua moi khung, dung de ve truc x khi
    hien thi day dac trung (energy/centroid) chong len tin hieu goc.

    Input:
        num_frames (int): tong so khung
        frame_length_ms (float): do dai 1 khung (ms)
        overlap_ratio (float): ty le chong lan da dung khi framing

    Output:
        times (np.ndarray, shape (num_frames,)): moc thoi gian (giay) cua
            tam moi khung
    """
    frame_length_s = frame_length_ms / 1000.0
    hop_s = frame_length_s * (1.0 - overlap_ratio)
    # Tam khung thu i nam o: i*hop + frame_length/2
    times = np.arange(num_frames) * hop_s + frame_length_s / 2.0
    return times


# ---------------------------------------------------------------------------
# BUOC 2: Trich dac trung - Short-Time Energy
# ---------------------------------------------------------------------------
def compute_short_time_energy(frames):
    """
    Tinh nang luong ngan han (STE) cho moi khung, theo cong thuc:
        E(i) = (1/N) * sum(|x_i(n)|^2), n = 1..N

    Input:
        frames (np.ndarray, shape (num_frames, N)): cac khung tin hieu

    Output:
        energy (np.ndarray, shape (num_frames,)): gia tri STE cua tung khung
    """
    # np.abs va **2 la cac phep toan phan tu (element-wise) co ban; mean()
    # doc theo truc thoi gian trong 1 khung (axis=1) tra ve 1 gia tri/khung
    energy = np.mean(np.abs(frames) ** 2, axis=1)
    return energy


# ---------------------------------------------------------------------------
# BUOC 2: Trich dac trung - Spectral Centroid
# ---------------------------------------------------------------------------
def compute_spectral_centroid(frames, fs):
    """
    Tinh spectral centroid cho moi khung: C_i = sum((k+1)*|X_i(k)|) / sum(|X_i(k)|)
    voi X_i la he so DFT cua khung i (dung numpy.fft.fft), k = 0..N-1.
    Cong thuc bam sat nguyen ban paper (dung toan bo pho N diem, khong cat
    rieng nua pho duoi Nyquist) -- day la mot lua chon don gian hoa cua tac
    gia, ban co the thu nghiem cat con N/2+1 diem (bo phan doi xung) va so
    sanh ket qua tren tin hieu huan luyen neu muon.

    Input:
        frames (np.ndarray, shape (num_frames, N)): cac khung tin hieu
        fs (int): tan so lay mau (Hz), khong dung truc tiep trong cong thuc
            nay nhung giu lai de tien mo rong (vd quy doi centroid ra Hz)

    Output:
        centroid (np.ndarray, shape (num_frames,)): gia tri spectral centroid
            cua tung khung (don vi: chi so k, khong phai Hz)
    """
    num_frames, N = frames.shape
    centroid = np.zeros(num_frames)

    # Trong so k+1 (k=0..N-1) giong nhau cho moi khung -> tinh 1 lan ben ngoai
    # vong lap de tranh lap lai phep tinh khong can thiet
    weights = np.arange(1, N + 1)

    for i in range(num_frames):
        spectrum = np.abs(np.fft.fft(frames[i, :]))  # |X_i(k)|, k=0..N-1
        denom = np.sum(spectrum)
        if denom == 0:
            centroid[i] = 0.0  # khung toan so 0 (vd im lang tuyet doi)
        else:
            centroid[i] = np.sum(weights * spectrum) / denom

    return centroid


# ---------------------------------------------------------------------------
# BUOC 3: Uoc luong nguong bang histogram
# ---------------------------------------------------------------------------
def smooth_histogram(hist_values, window_size=5):
    """
    Lam muot histogram bang bo loc trung binh truot (moving average) tu
    viet tay (khong dung scipy.signal / np.convolve) de giam nhieu truoc
    khi tim cuc dai cuc bo.

    Input:
        hist_values (np.ndarray): gia tri histogram (so lan xuat hien) theo bin
        window_size (int): do rong cua so loc trung binh truot (nen la so le)

    Output:
        smoothed (np.ndarray): histogram sau khi lam muot, cung do dai voi
            hist_values
    """
    n = len(hist_values)
    half = window_size // 2
    smoothed = np.zeros(n)

    # Voi moi vi tri i, lay trung binh cac gia tri trong cua so [i-half, i+half],
    # tu cat bot cua so o 2 dau mang (bien) de khong bi loi index am/vuot qua
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed[i] = np.mean(hist_values[lo:hi])

    return smoothed


def find_local_maxima(values):
    """
    Tim vi tri (index) cac cuc dai cuc bo trong mot day gia tri 1 chieu.
    Da bao gom kiem tra truong hop cuc dai nam o 2 bien (index 0 va cuoi mang).
    """
    maxima_indices = []
    n = len(values)
    
    if n == 0:
        return maxima_indices
    if n == 1:
        return [0]

    # 1. Kiem tra bien trai (index 0) - Dành cho đỉnh khoảng lặng
    if values[0] > values[1]:
        maxima_indices.append(0)

    # 2. Kiem tra cac diem o giua
    for i in range(1, n - 1):
        if values[i - 1] < values[i] > values[i + 1]:
            maxima_indices.append(i)

    # 3. Kiem tra bien phai (index n - 1)
    if values[-1] > values[-2]:
        maxima_indices.append(n - 1)

    return maxima_indices


def estimate_threshold(feature_values, num_bins=10, W=5, smooth_window=5):
    """
    Uoc luong nguong T cho MOT day dac trung (energy hoac spectral centroid),
    theo dung quy trinh trong paper (muc II.C):
        1. Tinh histogram cua feature_values
        2. Lam muot histogram
        3. Tim cac cuc dai cuc bo M1, M2 (2 dinh dau tien theo vi tri, khong
           phai theo do cao)
        4. T = (W * M1 + M2) / (W + 1)

    Input:
        feature_values (np.ndarray): day gia tri dac trung (energy hoac
            spectral centroid) cua toan bo tin hieu
        num_bins (int): so bin dung khi tinh histogram
        W (float): tham so trong so W trong cong thuc tinh nguong (sieu tham
            so, do tim gia tri toi uu bang tin hieu huan luyen)
        smooth_window (int): do rong cua so lam muot histogram

    Output:
        threshold (float): gia tri nguong T uoc luong duoc
        debug_info (dict): chua histogram goc, histogram da lam muot, tam bin
            va vi tri M1, M2 -- dung de ve hinh minh hoa qua trinh tim nguong
    """
    # np.histogram la ham thong ke co ban (tuong tu mean/std), duoc phep dung
    hist_values, bin_edges = np.histogram(feature_values, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    smoothed = smooth_histogram(hist_values.astype(float), smooth_window)
    maxima_idx = find_local_maxima(smoothed)

    # Truong hop suy bien: khong du 2 cuc dai cuc bo (thuong xay ra khi
    # smooth_window qua lon hoac tin hieu qua "phang"). Fallback: dung bin
    # dau tien va bin co gia tri cao nhat lam M1, M2 de thuat toan khong bi
    # vo bang loi -- nen kiem tra debug_info["num_maxima_found"] khi ve hinh
    # de biet co dang roi vao truong hop nay hay khong.
    if len(maxima_idx) >= 2:
        m1_idx, m2_idx = maxima_idx[0], maxima_idx[1]
    elif len(maxima_idx) == 1:
        m1_idx = maxima_idx[0]
        m2_idx = int(np.argmax(smoothed))
    else:
        m1_idx = 0
        m2_idx = int(np.argmax(smoothed))

    M1, M2 = bin_centers[m1_idx], bin_centers[m2_idx]
    threshold = (W * M1 + M2) / (W + 1)

    debug_info = {
        "hist_values": hist_values,
        "smoothed": smoothed,
        "bin_centers": bin_centers,
        "M1": M1,
        "M2": M2,
        "num_maxima_found": len(maxima_idx),
    }
    return threshold, debug_info


# ---------------------------------------------------------------------------
# BUOC 4: Gan nhan khung + hau xu ly
# ---------------------------------------------------------------------------
def detect_speech_frames(energy, centroid, energy_threshold, centroid_threshold):
    """
    Gan nhan tieng noi/khoang lang cho tung khung: mot khung duoc coi la
    tieng noi (True) neu CA HAI dac trung deu vuot nguong tuong ung.

    Input:
        energy (np.ndarray, shape (num_frames,)): STE cua tung khung
        centroid (np.ndarray, shape (num_frames,)): spectral centroid tung khung
        energy_threshold (float): nguong T1 (tu estimate_threshold tren energy)
        centroid_threshold (float): nguong T2 (tu estimate_threshold tren centroid)

    Output:
        is_speech (np.ndarray, dtype=bool, shape (num_frames,)): True = khung
            duoc phan loai la tieng noi
    """
    is_speech = (energy > energy_threshold) & (centroid > centroid_threshold)
    return is_speech


def postprocess_segments(is_speech, frame_length_ms, min_silence_ms=300, extend_frames=5):
    """
    Hau xu ly chuoi nhan khung is_speech:
        1. Noi dai moi doan tieng noi ve 2 phia (extend_frames khung, ~250ms
           theo paper)
        2. Gop cac doan tieng noi lien tiep/chong lan sau khi noi dai
        3. Loai bo cac khoang lang "ao" co do dai < min_silence_ms (yeu cau
           rieng cua de bai) bang cach gop no vao doan tieng noi ben canh

    Input:
        is_speech (np.ndarray, dtype=bool): nhan tho tung khung (True=tieng noi)
        frame_length_ms (float): do dai 1 khung (ms), dung de quy doi so
            khung <-> thoi gian (gia dinh khong overlap khi framing)
        min_silence_ms (float): do dai toi thieu cua 1 khoang lang hop le (ms)
        extend_frames (int): so khung noi dai ve moi phia doan tieng noi

    Output:
        segments (list of tuple): danh sach (start_time_s, end_time_s, label)
            voi label in {'sil', 'speech'}, da sap xep theo thoi gian, KHONG
            con khoang lang nao ngan hon min_silence_ms (tru khoang lang o
            dau/cuoi file, khong bi gioi han boi dieu kien nay)
    """
    n = len(is_speech)
    extended = np.zeros(n, dtype=bool)

    # --- Buoc 1+2: noi dai + gop doan (thuc hien truc tiep tren mang bool:
    #     "phinh to" moi khung True ra ca 2 phia, cac doan chong lan sau khi
    #     phinh se tu dong duoc coi la 1 doan lien tuc khi duyet o buoc sau) ---
    speech_indices = np.where(is_speech)[0]
    for idx in speech_indices:
        lo = max(0, idx - extend_frames)
        hi = min(n, idx + extend_frames + 1)
        extended[lo:hi] = True

    # --- Chuyen chuoi bool (theo khung) thanh danh sach doan (theo thoi gian) ---
    frame_length_s = frame_length_ms / 1000.0
    raw_segments = []
    if n > 0:
        current_label = "speech" if extended[0] else "sil"
        seg_start_frame = 0
        for i in range(1, n):
            label_i = "speech" if extended[i] else "sil"
            if label_i != current_label:
                raw_segments.append((seg_start_frame, i, current_label))
                seg_start_frame = i
                current_label = label_i
        raw_segments.append((seg_start_frame, n, current_label))

    # --- Buoc 3: loai khoang lang "ao" ngan hon min_silence_ms bang cach
    #     gop no vao doan lien ke (uu tien gop vao doan phia truoc, tru khi
    #     day la doan dau tien thi gop vao doan phia sau) ---
    cleaned = []
    for seg in raw_segments:
        start_f, end_f, label = seg
        duration_ms = (end_f - start_f) * frame_length_s * 1000.0
        if label == "sil" and duration_ms < min_silence_ms and len(cleaned) > 0:
            # Gop doan sil ngan nay vao doan ngay truoc do (mo rong end_f)
            prev_start, prev_end, prev_label = cleaned[-1]
            cleaned[-1] = (prev_start, end_f, prev_label)
        else:
            cleaned.append((start_f, end_f, label))

    # Gop cac doan lien tiep CUNG nhan (co the sinh ra sau buoc gop o tren)
    merged = []
    for seg in cleaned:
        if merged and merged[-1][2] == seg[2]:
            prev_start, prev_end, prev_label = merged[-1]
            merged[-1] = (prev_start, seg[1], prev_label)
        else:
            merged.append(seg)

    # Doi chi so khung -> thoi gian (giay) o buoc cuoi cung
    segments = [
        (start_f * frame_length_s, end_f * frame_length_s, label)
        for (start_f, end_f, label) in merged
    ]
    return segments


def get_boundaries_from_segments(segments):
    """
    Trich cac moc thoi gian bien (giay) tu danh sach segments da phan doan,
    dung de so sanh voi bien groundtruth (tinh RMSE/MAE).

    Input:
        segments (list of tuple): (start_time_s, end_time_s, label), output
            cua postprocess_segments()

    Output:
        boundaries (list of float): cac moc thoi gian chuyen tiep sil<->speech
            (khong lay bien dau/cuoi file)
    """
    # Bien chuyen tiep giua 2 doan lien tiep chinh la end_time_s cua doan
    # truoc (= start_time_s cua doan sau, vi 2 doan lien tiep khong co khoang
    # ho). Bo qua diem cuoi cung (khong phai bien chuyen tiep, ma la ket thuc file).
    boundaries = [seg[1] for seg in segments[:-1]]
    return boundaries
