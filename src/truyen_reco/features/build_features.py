"""Dựng các vector đặc trưng theo từng mặt (facet).

Giữ riêng từng mặt thay vì nối thành một vector dài, vì hai lý do:
  - phần giải thích cần biết mặt nào làm hai truyện giống nhau;
  - trọng số từng mặt học riêng được cho mỗi người đọc.

Mọi đặc trưng suy từ hành vi chỉ được tính trên tập train.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler

CURRENT_YEAR = 2026


def _svd(matrix, dim, seed=42):
    dim = min(dim, max(1, min(matrix.shape) - 1))
    model = TruncatedSVD(n_components=dim, random_state=seed)
    return model.fit_transform(matrix), model


def _normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-9)


def build_theme(media, dim=128):
    """Thể loại và tag cộng đồng. Tag AniList có rank phần trăm nên dùng luôn
    làm trọng số thay vì nhị phân hoá."""
    vocab = {}
    rows, cols, vals = [], [], []
    for row_index, record in enumerate(media.itertuples(index=False)):
        for name, rank in zip(record.tag_names, record.tag_ranks):
            key = f"tag:{name}"
            col = vocab.setdefault(key, len(vocab))
            rows.append(row_index)
            cols.append(col)
            vals.append(max(float(rank), 1.0) / 100.0)
        for genre in record.genres:
            key = f"genre:{genre}"
            col = vocab.setdefault(key, len(vocab))
            rows.append(row_index)
            cols.append(col)
            vals.append(1.0)

    matrix = sp.csr_matrix((vals, (rows, cols)), shape=(len(media), len(vocab)))
    weighted = TfidfTransformer().fit_transform(matrix)
    vectors, _ = _svd(weighted, dim)
    return _normalize(vectors), vocab


def build_creator(media, dim=32):
    """Hai không gian tách biệt: người viết truyện và người vẽ.

    AniList tách vai Story/Art ở phần lớn tác phẩm, nên trục tác giả và trục
    hoạ sĩ dựng được riêng — đúng với giả thuyết tác giả chi phối nhịp kể còn
    hoạ sĩ chi phối nét vẽ.
    """
    blocks = []
    for field in ("staff_story_ids", "staff_art_ids"):
        vocab = {}
        rows, cols = [], []
        for row_index, ids in enumerate(media[field]):
            for staff_id in ids:
                col = vocab.setdefault(staff_id, len(vocab))
                rows.append(row_index)
                cols.append(col)
        if not vocab:
            blocks.append(np.zeros((len(media), 1)))
            continue
        matrix = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                               shape=(len(media), len(vocab)))
        vectors, _ = _svd(matrix, dim)
        blocks.append(vectors)
    return _normalize(np.hstack(blocks))


PACE_COLUMNS = [
    "log_chapters", "log_volumes", "duration_years", "is_finished",
    "chapters_per_year", "age_years", "log_popularity",
    "drop_rate", "pause_rate", "completion_rate", "mean_drop_position",
    "mean_score", "score_spread",
]


def build_pace(media, train):
    """Nhịp truyện: cấu trúc xuất bản cộng dấu vết hành vi người đọc.

    Trong đó mean_drop_position là thứ Goodreads không có: biết người đọc bỏ
    ở chương bao nhiêu trên tổng số chương.
    """
    frame = pd.DataFrame({"media_id": media["media_id"].values})
    chapters = pd.to_numeric(media["chapters"], errors="coerce").values
    volumes = pd.to_numeric(media["volumes"], errors="coerce").values
    start = pd.to_numeric(media["start_year"], errors="coerce").values
    end = pd.to_numeric(media["end_year"], errors="coerce").values

    frame["log_chapters"] = np.log1p(np.nan_to_num(chapters, nan=0.0))
    frame["log_volumes"] = np.log1p(np.nan_to_num(volumes, nan=0.0))
    duration = np.where(np.isnan(end), CURRENT_YEAR, end) - start
    frame["duration_years"] = np.nan_to_num(duration, nan=0.0).clip(0, 60)
    frame["is_finished"] = (media["pub_status"] == "FINISHED").astype(float).values
    frame["chapters_per_year"] = np.nan_to_num(
        chapters / np.clip(duration, 1, None), nan=0.0, posinf=0.0).clip(0, 500)
    frame["age_years"] = np.nan_to_num(CURRENT_YEAR - start, nan=0.0).clip(0, 80)
    frame["log_popularity"] = np.log1p(media["popularity"].fillna(0).values)

    # Dấu vết hành vi, chỉ lấy từ train
    stats = train.copy()
    chapter_lookup = dict(zip(media["media_id"], chapters))
    stats["total_chapters"] = stats["media_id"].map(chapter_lookup)

    grouped = stats.groupby("media_id")
    counts = grouped["status"].value_counts().unstack(fill_value=0)
    for column in ("DROPPED", "PAUSED", "COMPLETED", "CURRENT"):
        if column not in counts:
            counts[column] = 0
    engaged = counts[["DROPPED", "PAUSED", "COMPLETED", "CURRENT"]].sum(axis=1)
    engaged = engaged.replace(0, np.nan)

    behaviour = pd.DataFrame({
        "drop_rate": counts["DROPPED"] / engaged,
        "pause_rate": counts["PAUSED"] / engaged,
        "completion_rate": counts["COMPLETED"] / engaged,
    })

    dropped = stats[(stats["status"] == "DROPPED")
                    & stats["total_chapters"].notna()
                    & (stats["total_chapters"] > 0)].copy()
    dropped["position"] = (dropped["progress"] / dropped["total_chapters"]).clip(0, 1)
    behaviour["mean_drop_position"] = dropped.groupby("media_id")["position"].mean()

    scores = stats[stats["score"].notna()].groupby("media_id")["score"]
    behaviour["mean_score"] = scores.mean()
    behaviour["score_spread"] = scores.std()

    frame = frame.merge(behaviour, left_on="media_id", right_index=True, how="left")
    for column in PACE_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
        median = frame[column].median()
        frame[column] = frame[column].fillna(0.0 if pd.isna(median) else median)

    values = StandardScaler().fit_transform(frame[PACE_COLUMNS].values)
    return _normalize(np.nan_to_num(values)), frame[["media_id"] + PACE_COLUMNS]


def build_all(media, train, art_vectors=None, dims=None):
    dims = dims or {}
    media = media.reset_index(drop=True)
    index = {media_id: i for i, media_id in enumerate(media["media_id"])}

    facets = {
        "theme": build_theme(media, dims.get("theme", 128))[0],
        "creator": build_creator(media, dims.get("creator", 32)),
    }
    pace_vectors, pace_table = build_pace(media, train)
    facets["pace"] = pace_vectors

    if art_vectors is not None:
        facets["art"] = art_vectors

    return {"index": index, "media": media, "facets": facets, "pace_table": pace_table}
