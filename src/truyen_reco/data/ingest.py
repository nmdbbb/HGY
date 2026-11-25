"""Đọc JSONL thô từ crawler AniList, chuẩn hoá thành Parquet.

Hai bảng đầu ra:
  interactions.parquet  một dòng mỗi (user, truyện)
  media.parquet         một dòng mỗi truyện
"""

import json
from pathlib import Path

import pandas as pd

# Vai trò staff không liên quan tới sáng tác, loại khỏi v_creator
NON_CREATIVE = (
    "translat", "letter", "edit", "assist", "design", "typeset",
    "proofread", "publish", "produc", "adapt",
)


def classify_role(role):
    """Phân vai staff thành 'story', 'art', cả hai, hoặc không tính.

    AniList ghi vai kèm khoảng chương, ví dụ 'Art (chs 1-179)',
    'Story (chs 93-201)', nên phải so khớp theo chuỗi con.
    """
    if not role:
        return ()
    text = role.lower()
    if any(word in text for word in NON_CREATIVE):
        return ()
    has_story = "story" in text
    has_art = "art" in text
    if has_story and has_art:
        return ("story", "art")
    if has_art:
        return ("art",)
    if has_story:
        return ("story",)
    return ()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _date(block):
    # Ngày do người dùng tự nhập: gặp năm -1, 0, 9999, ngày 31 của tháng 2...
    # Ngày không hợp lệ thì coi như thiếu, không làm sập cả lần đọc.
    if not block or not block.get("year"):
        return None
    year = block["year"]
    if not (1990 <= year <= 2030):
        return None
    try:
        return pd.Timestamp(year=year, month=block.get("month") or 1,
                            day=block.get("day") or 1)
    except (ValueError, TypeError):
        return None


def load_interactions(raw_dir):
    rows = []
    for record in read_jsonl(Path(raw_dir) / "lists.jsonl"):
        user = record["user"]
        source = record.get("source")
        for entry in record.get("entries", []):
            rows.append({
                "user": user,
                "media_id": entry["mediaId"],
                "status": entry.get("status"),
                "score": entry.get("score") or None,
                "progress": entry.get("progress") or 0,
                "progress_volumes": entry.get("progressVolumes") or 0,
                "repeat": entry.get("repeat") or 0,
                "started_at": _date(entry.get("startedAt")),
                "completed_at": _date(entry.get("completedAt")),
                "created_at": entry.get("createdAt") or 0,
                "updated_at": entry.get("updatedAt") or 0,
                # Quốc gia của truyện đã dẫn tới người đọc này, tức tầng lấy
                # mẫu — KHÔNG phải quốc gia của truyện ở dòng này. Quốc gia
                # truyện nằm ở media.parquet (`country`).
                "recruit_country": source,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    return dedupe(frame)


def dedupe(frame, max_items_per_user=3000):
    """Gộp mục lặp và bỏ tài khoản đánh dấu hàng loạt.

    Một truyện nằm trong nhiều danh sách tuỳ chỉnh thì API trả về nhiều lần
    (đo được ~5% dòng, có người lặp 2.400 lần). Lặp giống hệt thì vô hại với
    nhãn nhưng làm ma trận tương tác và số đếm k-core sai; lặp khác tiến độ thì
    giữ bản cập nhật gần nhất. Tài khoản >3.000 mục (đo được tới 21.307) là đánh
    dấu cả catalogue, không phản ánh gu: một người như vậy kéo độ phổ biến và
    ALS lệch về phía mình.
    """
    frame = (frame.sort_values("updated_at")
             .drop_duplicates(subset=["user", "media_id"], keep="last"))
    sizes = frame.groupby("user")["media_id"].transform("size")
    return frame[sizes <= max_items_per_user].reset_index(drop=True)


def load_media(raw_dir):
    rows = []
    for media in read_jsonl(Path(raw_dir) / "media.jsonl"):
        title = media.get("title") or {}
        start = media.get("startDate") or {}
        end = media.get("endDate") or {}
        tags = [t for t in (media.get("tags") or []) if not t.get("isMediaSpoiler")]

        story_ids, art_ids, story_names, art_names = [], [], [], []
        for edge in (media.get("staff") or {}).get("edges", []):
            node = edge.get("node") or {}
            staff_id = node.get("id")
            name = (node.get("name") or {}).get("full")
            if staff_id is None:
                continue
            kinds = classify_role(edge.get("role"))
            if "story" in kinds:
                story_ids.append(staff_id)
                story_names.append(name)
            if "art" in kinds:
                art_ids.append(staff_id)
                art_names.append(name)

        rows.append({
            "media_id": media["id"],
            "title": title.get("english") or title.get("romaji") or title.get("native"),
            "title_romaji": title.get("romaji"),
            "country": media.get("countryOfOrigin"),
            "format": media.get("format"),
            "pub_status": media.get("status"),
            "chapters": media.get("chapters"),
            "volumes": media.get("volumes"),
            "start_year": start.get("year"),
            "end_year": end.get("year"),
            "genres": media.get("genres") or [],
            "tag_names": [t["name"] for t in tags],
            "tag_ranks": [t.get("rank") or 0 for t in tags],
            "cover_url": (media.get("coverImage") or {}).get("extraLarge"),
            "staff_story_ids": story_ids,
            "staff_art_ids": art_ids,
            "staff_story_names": story_names,
            "staff_art_names": art_names,
            "description": media.get("description") or "",
            "average_score": media.get("averageScore"),
            "popularity": media.get("popularity") or 0,
            "favourites": media.get("favourites") or 0,
            "is_adult": bool(media.get("isAdult")),
        })
    return pd.DataFrame(rows).drop_duplicates(subset="media_id")


def run(raw_dir, out_dir):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    interactions = load_interactions(raw_dir)
    media = load_media(raw_dir)

    interactions.to_parquet(out_dir / "interactions.parquet", index=False)
    media.to_parquet(out_dir / "media.parquet", index=False)
    return interactions, media
