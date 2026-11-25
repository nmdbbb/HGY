"""Sinh lý do cho từng gợi ý.

Lý do rút thẳng từ mặt nào đóng góp nhiều nhất vào điểm số, nên nó mô tả đúng
cơ chế đã chọn truyện đó, không phải một câu dán thêm cho đẹp.
"""

import numpy as np

FACET_LABEL = {
    "art": "nét vẽ",
    "pace": "nhịp truyện",
    "theme": "chủ đề",
    "creator": "người sáng tác",
}


def describe_art(raw_row, names):
    """Dịch thống kê nét vẽ thành chữ người đọc hiểu được."""
    values = dict(zip(names, raw_row))
    parts = []
    if values.get("stroke_mean", 0) > 2.2:
        parts.append("nét dày")
    elif values.get("stroke_mean", 0) < 1.4:
        parts.append("nét mảnh")
    if values.get("screentone_energy", 0) > 0.32:
        parts.append("nhiều screentone")
    if values.get("saturation_mean", 0) > 0.45:
        parts.append("màu rực")
    elif values.get("saturation_mean", 0) < 0.15:
        parts.append("gần như đen trắng")
    if values.get("ink_ratio", 0) > 0.3:
        parts.append("mảng tối lớn")
    return ", ".join(parts)


def describe_pace(media_row):
    parts = []
    chapters = media_row.get("chapters")
    if chapters and chapters == chapters:
        if chapters >= 200:
            parts.append(f"bộ dài {int(chapters)} chương")
        elif chapters <= 60:
            parts.append(f"bộ ngắn {int(chapters)} chương")
        else:
            parts.append(f"{int(chapters)} chương")
    parts.append("đã hoàn" if media_row.get("pub_status") == "FINISHED" else "đang ra")
    return ", ".join(parts)


class Explainer:
    def __init__(self, facet_model, media, index, art_raw=None, art_ids=None,
                 art_names=None):
        self.facets = facet_model
        self.index = index
        self.reverse = {v: k for k, v in index.items()}
        self.media = media.set_index("media_id")
        self.art_names = art_names or []
        self.art_lookup = {}
        if art_raw is not None and art_ids is not None:
            self.art_lookup = {int(i): art_raw[n] for n, i in enumerate(art_ids)}

    def _title(self, media_id):
        if media_id in self.media.index:
            title = self.media.loc[media_id, "title"]
            if isinstance(title, str):
                return title
        return f"#{media_id}"

    def _nearest_in_history(self, facet, item_position, history_positions):
        vectors = self.facets.facets.get(facet)
        if vectors is None or not history_positions:
            return None
        usable = [p for p in history_positions if self.facets.available[facet][p]]
        if not usable or not self.facets.available[facet][item_position]:
            return None
        scores = vectors[usable] @ vectors[item_position]
        return usable[int(np.argmax(scores))]

    def explain(self, media_id, history_ids, facet_scores, weights):
        """facet_scores: {mặt: điểm cosine của chính truyện này}."""
        item_position = self.index.get(media_id)
        if item_position is None:
            return "Gợi ý theo độ phổ biến."

        contributions = {
            name: weights.get(name, 0.0) * value
            for name, value in facet_scores.items()
            if value is not None and np.isfinite(value)
        }
        history_positions = [self.index[m] for m in history_ids if m in self.index]

        if not contributions:
            row = self.media.loc[media_id] if media_id in self.media.index else {}
            return f"Phổ biến trong nhóm bạn đang đọc ({describe_pace(row)})."

        facet = max(contributions, key=contributions.get)
        anchor = self._nearest_in_history(facet, item_position, history_positions)
        anchor_title = self._title(self.reverse[anchor]) if anchor is not None else None
        row = self.media.loc[media_id] if media_id in self.media.index else {}

        if facet == "creator" and anchor_title:
            return f"Cùng ê-kíp sáng tác với {anchor_title} mà bạn đã đọc."

        if facet == "art":
            style = describe_art(self.art_lookup.get(media_id, []), self.art_names)
            detail = f" ({style})" if style else ""
            if anchor_title:
                return f"Nét vẽ gần với {anchor_title}{detail}."
            return f"Nét vẽ hợp gu bạn đã chọn{detail}."

        if facet == "pace":
            base = describe_pace(row)
            if anchor_title:
                return f"Nhịp tương tự {anchor_title} — {base}."
            return f"Nhịp hợp gu bạn chọn — {base}."

        shared = []
        if anchor is not None and media_id in self.media.index:
            anchor_id = self.reverse[anchor]
            if anchor_id in self.media.index:
                tags_a = set(self.media.loc[media_id, "tag_names"][:12])
                tags_b = set(self.media.loc[anchor_id, "tag_names"][:12])
                shared = sorted(tags_a & tags_b)[:3]
        if shared and anchor_title:
            return f"Cùng {', '.join(shared)} như {anchor_title}."
        if anchor_title:
            return f"Chủ đề gần với {anchor_title}."
        return "Chủ đề khớp với sở thích của bạn."

    def facet_breakdown(self, facet_scores, weights):
        """Dòng phụ cho giao diện: mỗi mặt giống bao nhiêu phần trăm."""
        parts = []
        for name, value in facet_scores.items():
            if value is None or not np.isfinite(value) or weights.get(name, 0) <= 0:
                continue
            parts.append(f"{FACET_LABEL.get(name, name)} {max(0.0, value) * 100:.0f}%")
        return " · ".join(parts)
