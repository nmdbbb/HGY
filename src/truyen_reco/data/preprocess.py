"""Gán mức tín hiệu, lọc lõi, và chia train/test theo thời gian.

Chia theo thời gian chứ không ngẫu nhiên: chia ngẫu nhiên cho mô hình nhìn thấy
tương lai và làm lát đánh giá cold-start mất ý nghĩa.
"""

import numpy as np
import pandas as pd

# PLANNING là "định đọc", không phải đã đọc. Gộp nó vào tương tác dương là
# lặp lại đúng sai lầm của is_read=false bên Goodreads.
POSITIVE = {"COMPLETED", "CURRENT", "REPEATING"}
NEGATIVE = {"DROPPED"}
WISHLIST = {"PLANNING"}
WEAK = {"PAUSED"}

CONFIDENCE = {"COMPLETED": 3.0, "REPEATING": 3.5, "CURRENT": 2.0, "PAUSED": 1.0}

# Nhãn COMPLETED không thể với tới ở truyện đang ra. Người đọc yêu một bộ
# manhwa đang ra thì nằm ở CURRENT vĩnh viễn — họ đã đọc hết phần có và đang
# đợi chương mới. Chấm CURRENT = 2.0 trong khi COMPLETED = 3.0 nghĩa là hạ
# điểm mọi truyện đang ra, mà đang ra gần như trùng với manhwa/manhua, còn
# manga Nhật có hàng chục năm tác phẩm đã hoàn. Tức là bộ trọng số theo nhãn
# tự nó dìm đúng hai nhóm cần gợi ý. Thay bằng tiến độ so với phần đã có:
# đọc tới đầu sóng của một bộ đang ra là tín hiệu mạnh ngang đọc hết một bộ
# đã hoàn.
CATCHUP_RANGE = {"CURRENT": (1.0, 2.0), "PAUSED": (0.5, 1.0)}


def chapter_frontier(frame, media=None, quantile=0.95):
    """Số chương thực có của mỗi truyện, để quy tiến độ về tỉ lệ bắt kịp.

    Truyện đã hoàn thì lấy `chapters`. Truyện đang ra thường để trống trường
    đó, nên suy ra từ tiến độ xa nhất mà người đọc đạt được. Lấy phân vị 95
    chứ không lấy max: một người nhập tay "chương 9999" sẽ kéo lệch cả thang.

    Đây là thuộc tính của bộ truyện (nó dài bao nhiêu), không phải tín hiệu sở
    thích, nên suy ra trong mẫu không làm rò rỉ nhãn đánh giá.
    """
    observed = (frame[frame["progress"] > 0]
                .groupby("media_id")["progress"]
                .quantile(quantile))
    frontier = observed.to_dict()
    if media is not None and "chapters" in media.columns:
        finished = media["pub_status"].eq("FINISHED") if "pub_status" in media.columns \
            else pd.Series(True, index=media.index)
        for row in media[finished & media["chapters"].notna()].itertuples(index=False):
            if row.chapters and row.chapters > 0:
                frontier[row.media_id] = float(row.chapters)
    return {k: v for k, v in frontier.items() if v and v > 0}


def label(interactions, media=None, frontier=None):
    frame = interactions.copy()
    frame["signal"] = np.select(
        [frame["status"].isin(POSITIVE),
         frame["status"].isin(NEGATIVE),
         frame["status"].isin(WISHLIST),
         frame["status"].isin(WEAK)],
        ["positive", "negative", "wishlist", "weak"],
        default="other",
    )
    frame["confidence"] = frame["status"].map(CONFIDENCE).fillna(0.0)

    if frontier is None:
        frontier = chapter_frontier(frame, media)
    size = frame["media_id"].map(frontier)
    catchup = (frame["progress"] / size).clip(upper=1.0)
    frame["catchup"] = catchup.where(frame["progress"] > 0)

    # Chỉ các nhãn chưa kết thúc mới quy theo tiến độ; COMPLETED và REPEATING
    # đã tự nói lên là đọc hết. Thiếu tiến độ thì giữ nguyên mức theo nhãn.
    for status, (floor, span) in CATCHUP_RANGE.items():
        rows = frame["status"].eq(status) & frame["catchup"].notna()
        frame.loc[rows, "confidence"] = floor + span * frame.loc[rows, "catchup"]

    # Điểm số có mặt thì nâng/hạ độ tin cậy quanh mức trung lập 70/100
    # Lưu ý: AniList ghi score = 0 cho "chưa chấm", không phải 0 điểm; ingest
    # đã quy 0 về None. Chỉ 20% số mục có điểm tường minh (đo trên dữ liệu thật).
    scored = frame["score"].notna()
    if scored.any():
        frame.loc[scored, "confidence"] += (frame.loc[scored, "score"] - 70.0) / 30.0
    frame["confidence"] = frame["confidence"].clip(lower=0.1)

    # created_at là lúc thêm vào danh sách — mốc thời gian đáng tin nhất
    event = frame["created_at"].where(frame["created_at"] > 0, frame["updated_at"])
    frame["event_time"] = pd.to_datetime(event, unit="s", errors="coerce")
    return frame


def filter_core(frame, min_user=5, min_item=5, max_rounds=12):
    """Lọc k-core lặp: bỏ user/item quá thưa cho tới khi ổn định."""
    kept = frame
    for _ in range(max_rounds):
        user_counts = kept["user"].value_counts()
        item_counts = kept["media_id"].value_counts()
        good_users = user_counts[user_counts >= min_user].index
        good_items = item_counts[item_counts >= min_item].index
        filtered = kept[kept["user"].isin(good_users) & kept["media_id"].isin(good_items)]
        if len(filtered) == len(kept):
            break
        kept = filtered
    return kept


def temporal_split(frame, test_fraction=0.2):
    """Cắt theo mốc thời gian: phần đuôi gần nhất làm test."""
    valid = frame[frame["event_time"].notna()]
    cutoff = valid["event_time"].quantile(1 - test_fraction)
    train = frame[frame["event_time"] <= cutoff]
    test = frame[frame["event_time"] > cutoff]
    return train, test, cutoff


def holdout_users(frame, fraction=0.1, min_items=6, seed=42):
    """Giữ riêng một nhóm người đọc, không cho vào train.

    Cắt theo thời gian không tạo ra người đọc mới: gần như ai cũng có vài mục
    trước mốc cắt, nên lát cold-start sẽ rỗng và đóng góp của đặc trưng nội
    dung thành vô hình. Nhóm giữ riêng này mới là người đọc mới thật.
    """
    counts = frame["user"].value_counts()
    eligible = counts[counts >= min_items].index.to_numpy()
    if len(eligible) == 0:
        return frame, frame.iloc[0:0]
    rng = np.random.default_rng(seed)
    n_hold = max(1, int(len(eligible) * fraction))
    held = set(rng.choice(eligible, size=min(n_hold, len(eligible)), replace=False))
    return frame[~frame["user"].isin(held)], frame[frame["user"].isin(held)]


def evaluation_slices(train, test, sparse_threshold=10):
    """Các lát đánh giá — đóng góp của đặc trưng nội dung nằm ở đây.

    Chỉ số tổng thể sẽ che mất đóng góp của nét vẽ và nhịp truyện, vì phần lớn
    lượt đánh giá rơi vào user đã có nhiều lịch sử, nơi CF tự nó đã đủ tốt.
    """
    train_users = set(train["user"])
    train_items = set(train["media_id"])
    counts = train["user"].value_counts()
    sparse = set(counts[counts < sparse_threshold].index)

    return {
        "all": test,
        "new_user": test[~test["user"].isin(train_users)],
        "new_item": test[~test["media_id"].isin(train_items)],
        "sparse_user": test[test["user"].isin(sparse)],
        "warm": test[test["user"].isin(train_users - sparse)
                     & test["media_id"].isin(train_items)],
    }


def run(interactions, media=None, min_user=5, min_item=5, test_fraction=0.2,
        cold_user_fraction=0.1):
    labelled = label(interactions, media=media)
    positives = labelled[labelled["signal"].isin({"positive", "weak"})]
    core = filter_core(positives, min_user=min_user, min_item=min_item)

    warm, cold_users = holdout_users(core, fraction=cold_user_fraction)
    train, test, cutoff = temporal_split(warm, test_fraction=test_fraction)

    slices = evaluation_slices(train, test)
    slices["new_user"] = cold_users
    return {
        "labelled": labelled,
        "core": core,
        "train": train,
        "test": test,
        "cold_users": cold_users,
        "cutoff": cutoff,
        "slices": slices,
    }
