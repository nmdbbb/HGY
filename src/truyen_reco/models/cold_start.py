"""Người đọc mới và truyện mới.

Ý chính: cho người mới chọn vài ảnh bìa thay vì chọn thể loại. Người ta đánh
giá nét vẽ trong một giây, nên vài cú nhấp đã dựng được hồ sơ sở thích về mặt
nét vẽ — thứ mà "chọn 3 thể loại" không lấy được.
"""

import numpy as np
from sklearn.cluster import KMeans


def onboarding_grid(facet_model, media, index, n_options=12, pool=600, seed=42):
    """Chọn lưới ảnh bìa trải đều các cụm nét vẽ.

    Lấy từ nhóm phổ biến để người mới nhận ra truyện, nhưng trải khắp không
    gian nét vẽ để câu trả lời của họ phân biệt được nhiều gu khác nhau.
    """
    if "art" not in facet_model.facets:
        popular = media.sort_values("popularity", ascending=False).head(n_options)
        return [index[m] for m in popular["media_id"] if m in index]

    vectors = facet_model.facets["art"]
    available = facet_model.available["art"]
    popular = media.sort_values("popularity", ascending=False)
    positions = [index[m] for m in popular["media_id"]
                 if m in index and available[index[m]]][:pool]
    if len(positions) <= n_options:
        return positions

    block = vectors[positions]
    labels = KMeans(n_clusters=n_options, n_init=5,
                    random_state=seed).fit_predict(block)
    picks = []
    for cluster in range(n_options):
        members = [positions[i] for i in np.where(labels == cluster)[0]]
        if members:
            picks.append(members[0])  # đã sắp theo độ phổ biến
    return picks


def profile_from_picks(facet_model, picked, pace_preference=None, pace_table=None):
    """Hồ sơ từ vài lựa chọn ban đầu.

    pace_preference: 'short' thích bộ ngắn gọn đã hoàn, 'long' thích bộ dài.
    """
    profile = facet_model.profile(picked)
    if pace_preference and pace_table is not None and "pace" in facet_model.facets:
        columns = list(pace_table.columns)
        if "log_chapters" in columns:
            axis = np.zeros(facet_model.facets["pace"].shape[1])
            # Hồ sơ nhịp dựng thẳng từ lựa chọn, không cần lịch sử đọc
            direction = 1.0 if pace_preference == "long" else -1.0
            axis[0] = direction
            norm = np.linalg.norm(axis)
            if norm > 1e-9:
                existing = profile.get("pace")
                blended = axis / norm if existing is None else 0.5 * (existing + axis / norm)
                blended_norm = np.linalg.norm(blended)
                profile["pace"] = blended / blended_norm if blended_norm > 1e-9 else None
    return profile


def item_cold_start_vector(facet_model, media_row, index, creator_lookup):
    """Truyện mới chưa ai đọc vẫn gợi ý được: nét vẽ lấy từ ảnh bìa có ngay
    ngày đầu, và nếu hoạ sĩ đã có tác phẩm khác thì mượn luôn vector trung
    bình của hoạ sĩ đó."""
    if "art" not in facet_model.facets:
        return None
    position = index.get(media_row.media_id)
    if position is not None and facet_model.available["art"][position]:
        return facet_model.facets["art"][position]

    siblings = []
    for staff_id in set(media_row.staff_art_ids):
        siblings.extend(creator_lookup.get(staff_id, []))
    siblings = [p for p in siblings if facet_model.available["art"][p]]
    if not siblings:
        return None
    centre = facet_model.facets["art"][siblings].mean(axis=0)
    norm = np.linalg.norm(centre)
    return centre / norm if norm > 1e-9 else None


def build_creator_lookup(media, index):
    lookup = {}
    for row in media.itertuples(index=False):
        position = index.get(row.media_id)
        if position is None:
            continue
        for staff_id in row.staff_art_ids:
            lookup.setdefault(staff_id, []).append(position)
    return lookup
