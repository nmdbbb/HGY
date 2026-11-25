"""Đặc trưng nét vẽ rút từ ảnh bìa.

Dùng thống kê mực và tông thủ công thay vì embedding của mạng tổng quát, vì
ba lý do: không cần nhãn, chạy được trên CPU, và mỗi chiều đều dịch được thành
một câu giải thích cho người đọc ("nét dày, nhiều screentone").

Embedding kiểu CLIP trộn nội dung vào phong cách — hai truyện cùng vẽ robot sẽ
nằm gần nhau dù nét vẽ khác hẳn — mà phần "vẽ cái gì" đã do mặt theme lo.
"""

import io
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.preprocessing import StandardScaler

USER_AGENT = "truyen-reco-academic/0.1 (IT4613 course project)"
TARGET_SIZE = 320

FEATURE_NAMES = [
    "ink_ratio", "edge_density", "stroke_mean", "stroke_std",
    "screentone_energy", "high_freq_energy", "contrast", "entropy",
    "saturation_mean", "saturation_std", "value_mean", "value_std",
    "colorfulness", "color_count", "warm_ratio",
]


def download_one(args):
    media_id, url, cache_dir = args
    path = Path(cache_dir) / f"{media_id}.jpg"
    if path.exists():
        return media_id, True
    if not url:
        return media_id, False
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read()
        Image.open(io.BytesIO(raw)).convert("RGB").save(path, quality=90)
        return media_id, True
    except (urllib.error.URLError, OSError, ValueError):
        return media_id, False


def select_for_covers(media, max_items):
    """Chia ngân sách ảnh bìa theo xuất xứ, không xếp theo độ phổ biến toàn cục.

    Xếp theo độ phổ biến toàn cục thì 20.000 suất đầu gần như toàn manga Nhật,
    vì AniList có hàng chục năm manga với lượng người đọc tích luỹ lớn. Kết quả
    là mặt `art` phủ dày manga và phủ mỏng manhwa/manhua — tức là hệ chỉ biết
    so nét vẽ trong một nửa danh mục, đúng nửa không cần. Chia theo tỉ lệ của
    từng nước trong tập lõi rồi mới xếp theo độ phổ biến *trong nước đó*.
    """
    # Bỏ trước những dòng không có URL dùng được: trên AniList 100% truyện có
    # ảnh nên không đổi gì, nhưng trên Goodreads 37,4% tác phẩm chỉ có ảnh
    # placeholder (đã quy về None ở tầng ingest) và nếu để lại thì chúng ăn mất
    # 37% ngân sách tải ảnh mà không đem về đặc trưng nào.
    if "cover_url" in media.columns:
        media = media[media["cover_url"].notna()]
    if len(media) <= max_items or "country" not in media.columns:
        return media
    shares = media["country"].fillna("??").value_counts(normalize=True)
    picked = []
    for country, share in shares.items():
        quota = max(1, int(round(max_items * share)))
        block = media[media["country"].fillna("??").eq(country)]
        picked.append(block.sort_values("popularity", ascending=False).head(quota))
    return pd.concat(picked).head(max_items)


def download_covers(media, cache_dir, max_items=20_000, workers=8):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ordered = select_for_covers(media, max_items)
    jobs = [(row.media_id, row.cover_url, cache_dir)
            for row in ordered.itertuples(index=False)]

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _, ok in pool.map(download_one, jobs):
            done += int(ok)
    return done


def extract_features(path):
    """Mười lăm thống kê về cách vẽ, không phải về nội dung vẽ gì."""
    try:
        image = Image.open(path).convert("RGB")
    except (OSError, ValueError):
        return None

    image.thumbnail((TARGET_SIZE, TARGET_SIZE))
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    if rgb.ndim != 3 or min(rgb.shape[:2]) < 32:
        return None

    grey = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    # Mực: ngưỡng hoá theo phân vị để không phụ thuộc độ sáng bản scan
    threshold = np.quantile(grey, 0.25)
    ink = grey < threshold
    ink_ratio = float(ink.mean())

    # Biên và độ dày nét
    gy, gx = np.gradient(grey)
    gradient = np.hypot(gx, gy)
    edge_density = float((gradient > np.quantile(gradient, 0.9)).mean())
    if ink.any():
        distance = ndimage.distance_transform_edt(ink)
        strokes = distance[distance > 0]
        stroke_mean = float(strokes.mean()) if strokes.size else 0.0
        stroke_std = float(strokes.std()) if strokes.size else 0.0
    else:
        stroke_mean = stroke_std = 0.0

    # Screentone và hatching tạo năng lượng tuần hoàn ở dải tần trung
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(grey - grey.mean())))
    height, width = spectrum.shape
    yy, xx = np.ogrid[:height, :width]
    radius = np.hypot(yy - height / 2, xx - width / 2)
    limit = min(height, width) / 2
    total_energy = spectrum.sum() + 1e-9
    mid_band = (radius > limit * 0.25) & (radius <= limit * 0.6)
    high_band = radius > limit * 0.6
    screentone_energy = float(spectrum[mid_band].sum() / total_energy)
    high_freq_energy = float(spectrum[high_band].sum() / total_energy)

    contrast = float(grey.std())
    histogram, _ = np.histogram(grey, bins=32, range=(0, 1), density=True)
    histogram = histogram / (histogram.sum() + 1e-9)
    entropy = float(-(histogram * np.log2(histogram + 1e-9)).sum())

    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = np.where(maximum > 0, (maximum - minimum) / (maximum + 1e-9), 0.0)

    # Colorfulness theo Hasler & Suesstrunk
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = red - green
    yb = 0.5 * (red + green) - blue
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                         + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    quantized = (rgb * 5).astype(np.int16)
    color_count = float(len(np.unique(quantized.reshape(-1, 3), axis=0)) / 216.0)
    warm_ratio = float((red > blue).mean())

    return [
        ink_ratio, edge_density, stroke_mean, stroke_std,
        screentone_energy, high_freq_energy, contrast, entropy,
        float(saturation.mean()), float(saturation.std()),
        float(maximum.mean()), float(maximum.std()),
        colorfulness, color_count, warm_ratio,
    ]


def build_art(media, cache_dir, workers=8):
    """Trả về (ma trận đặc trưng, mảng media_id có ảnh, bảng thô chưa chuẩn hoá)."""
    cache_dir = Path(cache_dir)
    paths = [(row.media_id, cache_dir / f"{row.media_id}.jpg")
             for row in media.itertuples(index=False)]
    paths = [(media_id, path) for media_id, path in paths if path.exists()]
    if not paths:
        return np.zeros((0, len(FEATURE_NAMES))), np.array([]), np.zeros((0, len(FEATURE_NAMES)))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda item: extract_features(item[1]), paths))

    ids, rows = [], []
    for (media_id, _), features in zip(paths, results):
        if features is not None:
            ids.append(media_id)
            rows.append(features)

    raw = np.asarray(rows, dtype=np.float32)
    scaled = StandardScaler().fit_transform(np.nan_to_num(raw))
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    return scaled / np.maximum(norms, 1e-9), np.asarray(ids), raw
