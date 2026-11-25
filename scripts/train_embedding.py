"""Huấn luyện và đánh giá mô hình nhúng hai tháp.

    python scripts/train_embedding.py                  # mục tiêu "sắp đọc gì"
    python scripts/train_embedding.py --split taste    # mục tiêu "gu"

Hai mục tiêu là hai tuyên bố khác nhau, nên báo riêng (xem mục cuối
`docs/data-diagnosis.md`):

  temporal  10 tuần cuối làm test. Đúng với cách người ta đọc truyện nhiều kỳ,
            nhưng trong cửa sổ đó thời điểm phát hành lấn át gu.
  taste     giữ ngẫu nhiên 20% mục của mỗi người. Thời gian bị trung hoà, nên
            đây mới là chỗ đo được đặc trưng nội dung có nắm được gu không.

Mọi cấu hình, kể cả ALS, đều chấm toàn danh mục. Bản cũ chỉ xếp lại top-400
ứng viên của ALS, mà chỉ 35,6% truyện trong test lọt vào đó — trần ấy chặn mọi
cải tiến và làm sáu dòng ablation phẳng như nhau.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(min(4, torch.get_num_threads()))

from truyen_reco.evaluation import metrics  # noqa: E402
from truyen_reco.models.two_tower import (  # noqa: E402
    ServedTwoTower, TwoTower, build_history_tensors, log_q, pack_for_serving,
    pool_history, unit)

TIME_COLUMNS = ["log_pop_all", "log_pop_60d", "is_releasing", "age_years",
                "is_finished"]

# Hai biến thể được lưu ra để phục vụ, vì chúng là hai sản phẩm khác nhau chứ
# không phải một tốt một kém (xem mục 6 `docs/research-notes.md`):
#   feed      gu + thời điểm — chính xác nhất (+48% NDCG so với ALS trên taste)
#             nhưng chấm truyện mới bằng 0 vì đầu thời điểm dìm truyện ít đọc.
#   discover  không hiệu chỉnh logQ — kém hơn ở đầu phân phối, nhưng là cấu
#             hình duy nhất chấm được truyện mới (new_item 0,0173) và phủ
#             catalog 0,158, tỉ lệ manhua trong top-10 lên 0,11.
SAVE_AS = {
    "hai tháp (gu + thời điểm)": "feed",
    "gu, không hiệu chỉnh logQ": "discover",
    # Lưu riêng tên khác, KHÔNG ghi đè `feed`: bản có cổng thắng ở warm
    # (0,1039 -> 0,1099) nhưng mất ở người đọc mới (0,3771 -> 0,2499), nên chưa
    # được phép thay bản đang phục vụ trước khi giải thích xong chỗ mất đó.
    "gu + phổ biến, cổng hai phần + bỏ ngẫu nhiên": "gated",
}

VARIANTS = {
    "gu": ("hai tháp (chỉ gu)", dict(use_time=False)),
    "feed": ("hai tháp (gu + thời điểm)", dict(use_time=True)),
    # Ablation cho hai quyết định đã sửa, để biết sửa có đúng không. `halflife`
    # đã đo xong (chênh 3%, không phải nguyên nhân) nên không nằm trong mặc
    # định nữa — chạy lại chỉ để tái lập.
    "halflife": ("gu, bán rã 180 ngày", dict(use_time=False, half_life_days=180.0)),
    "discover": ("gu, không hiệu chỉnh logQ", dict(use_time=False, use_logq=False)),
    # Chẩn đoán nước (`scripts/diagnose_country.py`): `feed` cho manhua 1,0%
    # trong top-10 trong khi người ta đọc 6,3% — dìm 6,3 lần, còn manhwa thì
    # được đẩy lên (đọc 36,2% nhận 48,2%). Nghi can là `log_pop_all` trong đầu
    # thời điểm: nó là độ phổ biến *tích luỹ cả đời*, mà danh mục CN nhỏ và mới
    # hơn nên luôn thua. Biến thể này bỏ đúng cột đó, giữ lại phần "đang ra,
    # đang nổi 60 ngày" — tức là phần mà chẩn đoán đo được là mạnh (AUC 0,664),
    # chứ không phải phần phổ biến toàn thời.
    "feed_recency": ("gu + thời điểm, bỏ phổ biến toàn thời",
                     dict(use_time=True,
                          time_columns=("log_pop_60d", "is_releasing",
                                        "age_years", "is_finished"))),
    # Giả thuyết `log_pop_all` đã bị BÁC BỎ: bỏ cột đó làm CN 0,010 -> 0,000,
    # KR 0,55 -> 0,65, phủ catalog 0,0336 -> 0,0091, NDCG 0,1169 -> 0,1072.
    # Tức cột đẩy KR là `log_pop_60d`, và bỏ `log_pop_all` chỉ làm mô hình dựa
    # vào nó nhiều hơn. Nhưng đo tiếp thì dữ liệu *không* thiếu manhua gần đây:
    # CN giữ 6,5% lượt đọc trong 60 ngày cuối, đúng bằng tỉ lệ cả tập.
    # Đính chính cách giải thích: đo trọng số học được thì đầu này 94% là
    # `log_pop_all` (+0,891 so với +0,054), tức nó là **đầu độ phổ biến cả
    # đời**, không phải đầu "đang nổi". Bỏ `log_pop_all` không làm mô hình dựa
    # vào cột khác, nó xoá gần hết cái đầu đó. Và chỗ dìm manhua hiện ra ngay:
    # top-100 theo đầu này có 0,0% manhua, top-1000 có 5,6%.
    #
    # Cổng theo người đọc: đo được là *đúng một nửa*. Warm 0,1039 -> 0,1099
    # (+5,8%) và tương quan hạng giữa g_u với độ nổi những bộ họ đã đọc là
    # +0,845, nên cơ chế đúng. Nhưng người đọc mới rơi 0,3771 -> 0,2499 vì
    # cổng *lấy mất* tín hiệu phổ biến của đường gộp-lịch-sử — đường duy nhất
    # người mới có. Hai biến thể dưới tách hai thay đổi để quy công đúng chỗ:
    #
    #   feed_gated_nodrop  cổng tách `base(log(1+n)) + delta_u`, KHÔNG bỏ ngẫu
    #                      nhiên. Chỉ đo công của việc đưa quan hệ với độ dài
    #                      lịch sử vào thẳng cấu trúc.
    #   feed_gated         thêm bỏ ngẫu nhiên `delta_u` (DropoutNet, áp cho
    #                      cổng). Dự đoán ghi trước: `new_user` phải hồi về
    #                      gần 0,3771 mà `warm` giữ được phần lớn +5,8%. Nếu
    #                      `new_user` vẫn ~0,27 thì cơ chế "cổng lấy mất tín
    #                      hiệu của đường lịch sử" là SAI, ghi là bác bỏ.
    "feed_gated_nodrop": ("gu + phổ biến, cổng hai phần (không bỏ ngẫu nhiên)",
                          dict(use_time=True, trend_gate=True,
                               trend_dropout=0.0)),
    "feed_gated": ("gu + phổ biến, cổng hai phần + bỏ ngẫu nhiên",
                   dict(use_time=True, trend_gate=True, trend_dropout=0.3)),
}


# ---------------------------------------------------------------- dữ liệu
def load_state(out_dir):
    out_dir = Path(out_dir)
    index = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    item_index = {int(k): v for k, v in index["item_index"].items()}
    facets = np.load(out_dir / "facets.npz")
    return {
        "item_index": item_index,
        "facets": {k[len("facet_"):]: facets[k] for k in facets.files
                   if k.startswith("facet_")},
        "als_user": facets["user_factors"],
        "als_item": facets["item_factors"],
        "media": pd.read_parquet(out_dir / "media_core.parquet"),
        "train": pd.read_parquet(out_dir / "train.parquet"),
        "test": pd.read_parquet(out_dir / "test.parquet"),
        "cold_users": pd.read_parquet(out_dir / "cold_users.parquet"),
        "cutoff": pd.Timestamp(index["cutoff"]),
        "als_user_index": index["user_index"],
    }


def content_matrix(facets, names=("theme", "creator", "pace", "art")):
    """Nối các mặt, mỗi mặt chuẩn hoá L2 riêng để cùng thang.

    Truyện thiếu một mặt (ví dụ 57,8% có ảnh bìa) để vector 0 ở khối đó; phép
    chiếu tuyến tính cộng 0, tức là tự động chấm trên các mặt nó thực có.
    """
    blocks = []
    for name in names:
        if name not in facets:
            continue
        block = facets[name].astype(np.float32)
        norm = np.linalg.norm(block, axis=1, keepdims=True)
        blocks.append(block / np.maximum(norm, 1e-9))
    return np.hstack(blocks)


def time_features(media, train, item_index, cutoff, window_days=60):
    """Đặc trưng thời điểm, tính chỉ từ train.

    Chẩn đoán: trong top-400 ứng viên, độ phổ biến 60 ngày là tín hiệu mạnh
    nhất (AUC 0,664), hơn cả ALS (0,543). Nên nó phải là một trục riêng, đo
    được và tắt được, chứ không nằm lẫn trong vector nhúng.
    """
    n_items = len(item_index)
    table = np.zeros((n_items, len(TIME_COLUMNS)), dtype=np.float32)

    positions = train["media_id"].map(item_index)
    valid = positions.notna()
    counts = np.bincount(positions[valid].astype(int), minlength=n_items)

    recent = train[train["event_time"] > cutoff - pd.Timedelta(days=window_days)]
    recent_positions = recent["media_id"].map(item_index).dropna().astype(int)
    recent_counts = np.bincount(recent_positions, minlength=n_items)

    table[:, 0] = np.log1p(counts)
    table[:, 1] = np.log1p(recent_counts)
    info = media.set_index("media_id")
    for media_id, position in item_index.items():
        if media_id not in info.index:
            continue
        row = info.loc[media_id]
        table[position, 2] = float(row["pub_status"] == "RELEASING")
        year = row["start_year"] if pd.notna(row["start_year"]) else cutoff.year
        table[position, 3] = np.clip(cutoff.year - float(year), 0, 60) / 10.0
        table[position, 4] = float(row["pub_status"] == "FINISHED")

    mean, std = table.mean(axis=0), table.std(axis=0)
    return (table - mean) / np.maximum(std, 1e-6), counts


def taste_split(frame, test_fraction=0.2, seed=42):
    """Giữ ngẫu nhiên một phần mục của mỗi người, không cắt theo thời gian."""
    rng = np.random.default_rng(seed)
    held = np.zeros(len(frame), dtype=bool)
    for _, positions in frame.groupby("user").indices.items():
        if len(positions) < 5:
            continue
        pick = rng.choice(positions,
                          size=max(1, int(len(positions) * test_fraction)),
                          replace=False)
        held[pick] = True
    return frame[~held], frame[held]


# ------------------------------------------------------------- huấn luyện
def train_model(train, item_index, content, times, cutoff, train_counts,
                dim=64, epochs=8, batch_size=2048, n_negatives=1024,
                learning_rate=5e-3, use_time=True, max_history=200,
                half_life_days=None, use_logq=True, id_dropout=0.2,
                trend_gate=False, trend_dropout=0.3, seed=0, log=print):
    """Gộp lịch sử bằng cách tách tuyến tính, không gom từng mục một lượt.

    `v_i = e_id[i] + P c_i` và phép gộp là trung bình có trọng số, nên
    `gộp(v) = gộp(e_id) + P · gộp(c)`. Vế `gộp(c)` không phụ thuộc tham số nên
    tính một lần; vế `gộp(e_id)` là một phép nhân ma trận thưa trên toàn bộ
    người đọc, khoảng 20 MFLOP mỗi bước. Gom từng mục cho cả lô thì phải chạm
    4096 × 100 hàng đặc trưng mỗi bước, chậm hơn hai bậc.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    users = sorted(train["user"].unique())
    user_index = {user: i for i, user in enumerate(users)}
    n_items, n_users = len(item_index), len(users)

    frame = train[train["media_id"].isin(item_index)].copy()
    frame["user_position"] = frame["user"].map(user_index).astype(int)
    frame["item_position"] = frame["media_id"].map(item_index).astype(int)
    frame["timestamp"] = frame["event_time"].astype("int64") // 10**9
    frame = frame.sort_values("timestamp")

    history_items, history_times = [], []
    blocks = dict(tuple(frame.groupby("user_position")))
    for position in range(n_users):
        block = blocks.get(position)
        history_items.append(block["item_position"].to_numpy() if block is not None
                             else np.array([], dtype=np.int64))
        history_times.append(block["timestamp"].to_numpy() if block is not None
                             else np.array([], dtype=np.int64))

    cutoff_seconds = int(cutoff.timestamp())
    hist_index, hist_weight = build_history_tensors(
        history_items, history_times, cutoff_seconds, n_users,
        max_len=max_history, half_life_days=half_life_days)

    # Độ dài lịch sử mỗi người: phần chung của cổng độ phổ biến là hàm của nó,
    # nên nó phải có mặt ngay trong lúc huấn luyện, không phải khớp lại sau.
    hist_len = (hist_index >= 0).sum(dim=1)

    # Ma trận lịch sử thưa (trọng số chưa chuẩn hoá) + tổng trọng số mỗi người
    rows, cols, values = [], [], []
    for position in range(n_users):
        mask = hist_index[position] >= 0
        cols.append(hist_index[position][mask].numpy())
        values.append(hist_weight[position][mask].numpy())
        rows.append(np.full(int(mask.sum()), position))
    sparse_history = torch.sparse_coo_tensor(
        np.vstack([np.concatenate(rows), np.concatenate(cols)]),
        np.concatenate(values).astype(np.float32),
        (n_users, n_items)).coalesce()
    total_weight = torch.from_numpy(
        np.array([v.sum() for v in values], dtype=np.float32))

    content_tensor = torch.from_numpy(content)
    time_tensor = torch.from_numpy(times) if use_time else None
    pooled_content = torch.sparse.mm(sparse_history, content_tensor)

    # Trọng số của chính mục đang dự đoán, để trừ nó ra khỏi hồ sơ: nếu không
    # mô hình đọc được đáp án qua hồ sơ của chính nó.
    #
    # Tính thẳng bằng vector, không tra bảng băm. Bản trước dựng một dict khoá
    # (user, item) với 6.535 x 300 = 1,96 triệu phần tử, tốn ~320 MB và là
    # nguyên nhân chính làm đầy bộ nhớ. `hist_index` lấy đúng `max_history`
    # mục cuối của mỗi người, nên "mục này có trong lịch sử không" chỉ là
    # "nó nằm trong bao nhiêu mục cuối" — đếm lùi là ra, không cần bảng.
    from_end = frame.groupby("user_position").cumcount(ascending=False)
    within = (from_end < max_history).to_numpy()
    if half_life_days:
        age = np.maximum(
            (cutoff_seconds - frame["timestamp"].to_numpy()) / 86400.0, 0.0)
        row_weight = np.where(within, 0.5 ** (age / half_life_days),
                              0.0).astype(np.float32)
    else:
        row_weight = within.astype(np.float32)

    model = TwoTower(n_users, n_items, content.shape[1],
                     time_dim=times.shape[1] if use_time else 0, dim=dim,
                     id_dropout=id_dropout, trend_gate=trend_gate,
                     trend_dropout=trend_dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_steps = epochs * max(1, (n_rows_hint := len(frame)) // batch_size + 1)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    correction = (log_q(train_counts, batch_size, n_negatives, n_items)
                  if use_logq else None)

    user_positions = frame["user_position"].to_numpy()
    item_positions = frame["item_position"].to_numpy()
    confidence = frame["confidence"].to_numpy().astype(np.float32)
    n_rows = len(frame)

    for epoch in range(epochs):
        order = rng.permutation(n_rows)
        total, steps = 0.0, 0
        pooled_identity = None
        for start in range(0, n_rows, batch_size):
            rows_batch = order[start:start + batch_size]
            batch_users = torch.from_numpy(user_positions[rows_batch])
            batch_items = torch.from_numpy(item_positions[rows_batch])
            batch_conf = torch.from_numpy(confidence[rows_batch])
            batch_w = torch.from_numpy(row_weight[rows_batch]).unsqueeze(1)
            negatives = torch.from_numpy(
                rng.integers(0, n_items, size=n_negatives))

            pooled_identity = torch.sparse.mm(sparse_history, model.item_id.weight)

            all_items = torch.cat([batch_items, negatives])
            raw = model.item_vectors(
                content_tensor[all_items], all_items, drop_id=True)
            vectors = unit(raw)          # vế chấm điểm: cosine
            positive_vectors = vectors[:len(batch_items)]

            denominator = (total_weight[batch_users].unsqueeze(1)
                           - batch_w).clamp(min=1e-3)
            identity_sum = (pooled_identity[batch_users]
                            - batch_w * model.item_id(batch_items))
            content_sum = (pooled_content[batch_users]
                           - batch_w * content_tensor[batch_items])
            pooled = (identity_sum + model.content(content_sum)) / denominator
            user_vectors = model.user_vectors(pooled, batch_users, drop_id=True)

            logits_positive = (user_vectors * positive_vectors).sum(-1, keepdim=True)
            logits_negative = user_vectors @ vectors.T
            # Mẫu âm trùng mục dương của chính dòng đó là âm giả, phải chặn.
            same = all_items.unsqueeze(0) == batch_items.unsqueeze(1)
            logits_negative = logits_negative.masked_fill(same, -1e4)

            if use_time:
                bias = model.time_bias(time_tensor[all_items])
                # Cổng theo người đọc: `gate` là (B,1) nên nó nhân đúng theo
                # hàng, tức mỗi người có mức thích-truyện-phổ-biến riêng. Không
                # cổng thì `gate` là 1.0 và hai dòng dưới trùng bản cũ từng chữ.
                #
                # `drop_delta=True` bỏ ngẫu nhiên phần riêng của từng người.
                # Những bước đó mô hình chỉ còn phần chung — đúng cảnh người đọc
                # mới lúc phục vụ — nên đường gộp-lịch-sử buộc phải tự gánh
                # thông tin phổ biến. Bản không có bước nào như vậy đã đo được
                # là làm rỗng đường gu của người đọc mới: top-10 của họ chỉ
                # trùng 0,082 với top-10 chấm bằng riêng gu, và tắt cổng thì
                # NDCG rơi về 0,0264 (mục 9.7 `docs/research-notes.md`).
                gate = (model.trend_weight(batch_users,
                                           n_history=hist_len[batch_users],
                                           drop_delta=True)
                        if model.trend is not None else 1.0)
                logits_positive = logits_positive + gate * bias[:len(batch_items)].unsqueeze(1)
                logits_negative = logits_negative + gate * bias.unsqueeze(0)

            if correction is not None:
                # Trừ log q khỏi vế mẫu âm: bù việc truyện phổ biến hay bị rút
                # làm mẫu âm. Vế mẫu dương không trừ, vì nó được đưa vào xác
                # định chứ không phải rút theo q.
                logits_negative = logits_negative - correction[all_items].unsqueeze(0)

            logits = torch.cat([logits_positive, logits_negative], dim=1) / 0.07
            target = torch.zeros(len(logits), dtype=torch.long)
            losses = torch.nn.functional.cross_entropy(
                logits, target, reduction="none")
            loss = (losses * batch_conf).sum() / batch_conf.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            total += float(loss.detach())
            steps += 1
        log(f"  epoch {epoch + 1}/{epochs}  loss {total / max(steps, 1):.4f}")

    return {
        "model": model,
        "user_index": user_index,
        "hist_index": hist_index,
        "hist_weight": hist_weight,
        "content": content_tensor,
        "times": time_tensor,
    }


# --------------------------------------------------------------- suy luận
@torch.no_grad()
def item_table(state, train_counts, dim=64):
    """Vector của toàn bộ truyện. Truyện chưa ai đọc bị tắt vector ID.

    Vector ID của một truyện chưa xuất hiện trong train chưa được học, nó chỉ
    là nhiễu khởi tạo. Để nguyên là bơm nhiễu vào đúng lát `new_item`. Tắt đi
    mới khớp với cách đã huấn luyện (bỏ ngẫu nhiên vector ID).
    """
    model = state["model"]
    n_items = len(state["content"])
    ids = torch.arange(n_items)
    vectors = model.item_vectors(state["content"], ids, drop_id=False)
    cold = torch.from_numpy((train_counts == 0).astype(np.float32)).unsqueeze(1)
    vectors = vectors - cold * model.item_id(ids)
    bias = model.time_bias(state["times"]) if state["times"] is not None else None
    # raw để gộp lịch sử (tuyến tính), unit để chấm điểm (cosine)
    return vectors, unit(vectors), bias


def pack(state, raw_vectors, unit_vectors, bias, use_time):
    """Gói mô hình thành blob phục vụ được — cũng chính là blob đem đánh giá."""
    return pack_for_serving(
        state["model"], state["user_index"], state["hist_index"],
        state["hist_weight"], raw_vectors, unit_vectors, bias, use_time)


def make_recommender(served, reverse, k=10):
    """Bộ gợi ý đánh giá, đi qua đúng lớp mà API dùng.

    Trước đây hàm này chấm điểm bằng một bản sao logic riêng trong script. Khi
    bản sao và bản phục vụ lệch nhau thì bảng kết quả không nói gì về sản phẩm,
    nên giờ chỉ còn một đường: `ServedTwoTower`.
    """
    def recommend(user, history):
        top = served.top_k(user, list(history), k=k)
        return [reverse[int(i)] for i in top]
    return recommend


def static_recommender(scores, reverse, k=10):
    def recommend(_user, history):
        values = scores.astype(float).copy()
        if len(history):
            values[np.asarray(history, dtype=int)] = -np.inf
        top = np.argpartition(-values, k)[:k]
        return [reverse[int(i)] for i in top[np.argsort(-values[top])]]
    return recommend


def als_recommender(user_factors, item_factors, als_index, popularity, reverse, k=10):
    def recommend(user, history):
        position = als_index.get(user)
        if position is None:
            values = np.log1p(popularity).astype(float).copy()
        else:
            values = (item_factors @ user_factors[position]).astype(float)
        if len(history):
            values[np.asarray(history, dtype=int)] = -np.inf
        top = np.argpartition(-values, k)[:k]
        return [reverse[int(i)] for i in top[np.argsort(-values[top])]]
    return recommend


# ------------------------------------------------------------------ chạy
def build_histories(train, cold_users, item_index, seed_items=3):
    history = (train.sort_values("event_time")
                    .groupby("user")["media_id"].apply(list).to_dict())
    history = {u: [item_index[m] for m in items if m in item_index]
               for u, items in history.items()}
    seed, holdout = {}, {}
    for user, frame in cold_users.groupby("user"):
        ordered = frame.sort_values("event_time")["media_id"].tolist()
        if len(ordered) <= seed_items:
            continue
        seed[user] = [item_index[m] for m in ordered[:seed_items] if m in item_index]
        holdout[user] = set(ordered[seed_items:])
    return history, seed, holdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/processed", type=Path)
    parser.add_argument("--split", choices=("temporal", "taste"), default="temporal")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=1500)
    parser.add_argument("--variants", nargs="+", default=["gu", "feed", "discover"],
                        choices=list(VARIANTS),
                        help="biến thể cần huấn luyện")
    parser.add_argument("--extra-variants", action="store_true",
                        help="chạy cả bốn biến thể, kể cả ablation bán rã")
    parser.add_argument("--save", action="store_true",
                        help="lưu trọng số hai biến thể cho API phục vụ")
    parser.add_argument("--out-name", default=None,
                        help="tên tệp kết quả (mặc định embedding_<split>.json). "
                             "Mỗi lần chạy GHI ĐÈ tệp mặc định, nên khi chạy một "
                             "tập biến thể khác trên cùng thư mục thì phải đặt "
                             "tên riêng, nếu không kết quả cũ biến mất lặng lẽ.")
    parser.add_argument("--art-check", type=int, default=0, metavar="N",
                        help="lặp N seed cho hai bậc cuối của thang (không art "
                             "/ có art) để biết mức tăng có lớn hơn nhiễu seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        metavar="S",
                        help="lặp mọi biến thể qua cùng danh sách seed này rồi "
                             "in phép kiểm ghép cặp theo seed (biến thể đầu làm "
                             "mốc). Bắt buộc dùng khi chênh lệch cần kết luận "
                             "nhỏ hơn nhiễu khởi tạo.")
    parser.add_argument("--art-seeds", nargs="+", type=int, default=None,
                        metavar="S",
                        help="chạy đúng những seed này thay vì 0..N-1. Mỗi "
                             "lần huấn luyện mất ~43 phút, nên khi đã có seed "
                             "0–2 thì chỉ cần thêm 3 4 là đủ df=4 (ngưỡng "
                             "t=2,78) mà không chạy lại ba seed cũ — chúng "
                             "tái lập được vì seed cố định cả khởi tạo và mẫu âm")
    parser.add_argument("--facet-ablation", action="store_true",
                        help="thang 4 bậc theme -> +creator -> +pace -> +art, "
                             "cấu hình tắt logQ (xem SAVE_AS)")
    args = parser.parse_args()

    started = time.monotonic()
    state = load_state(args.out_dir)
    item_index = state["item_index"]
    reverse = {v: k for k, v in item_index.items()}
    media, cutoff = state["media"], state["cutoff"]

    # `state` bị ghi đè trong vòng lặp biến thể, nên giữ lại các mặt từ đây
    state_facets = state["facets"]
    content = content_matrix(state_facets)
    print(f"nội dung: {content.shape[1]} chiều "
          f"({' + '.join(f'{k} {v.shape[1]}' for k, v in state_facets.items())})")

    # Thang ablation từng mặt: cùng một cấu hình, chỉ đổi khối đặc trưng đưa
    # vào. Chạy ở cấu hình `discover` (tắt logQ, không dùng đầu thời điểm) vì
    # lát `new_item` là chỗ duy nhất tách được công của nội dung — truyện chưa
    # ai đọc có vector ID bằng 0 nên điểm của nó hoàn toàn do `P·c_i` quyết
    # định. Bật logQ hay bật đầu thời điểm thì lát đó về 0 và không đo được gì.
    # Bậc thứ tư (đủ bốn mặt) chính là biến thể `discover`, không huấn luyện lại.
    facet_ladder = [
        ("chỉ theme", ("theme",)),
        ("+ creator", ("theme", "creator")),
        ("+ pace", ("theme", "creator", "pace")),
    ]

    if args.split == "temporal":
        train, test = state["train"], state["test"]
        cold_users = state["cold_users"]
    else:
        pooled = pd.concat([state["train"], state["test"]])
        train, test = taste_split(pooled)
        cold_users = state["cold_users"]
        print("chia theo gu: giữ ngẫu nhiên 20% mục mỗi người, thời gian trung hoà")

    times, train_counts = time_features(media, train, item_index, cutoff)
    history, seed, holdout = build_histories(train, cold_users, item_index)
    combined = dict(history)
    combined.update(seed)

    slices = dict(preprocess_slices(train, test))
    cold = cold_users[[m in holdout.get(u, set())
                       for u, m in zip(cold_users["user"], cold_users["media_id"])]]
    slices["new_user"] = cold
    print(f"người đọc train {len(history):,} | người đọc mới {len(seed):,}")
    for name, frame in slices.items():
        print(f"  lát {name:<12} {len(frame):>8,} tương tác | "
              f"{frame['user'].nunique():>6,} người")

    creator_map, country_map = {}, {}
    for row in media.itertuples(index=False):
        key = tuple(sorted(set(row.staff_art_ids) | set(row.staff_story_ids)))
        creator_map[row.media_id] = key or None
        country_map[row.media_id] = row.country

    recent = times[:, TIME_COLUMNS.index("log_pop_60d")]
    recommenders = {
        "phổ biến": static_recommender(np.log1p(train_counts), reverse, args.k),
        "đang nổi (60 ngày)": static_recommender(recent, reverse, args.k),
        "ALS": als_recommender(state["als_user"], state["als_item"],
                               state["als_user_index"], train_counts,
                               reverse, args.k),
    }

    chosen = list(VARIANTS) if args.extra_variants else args.variants
    variants = [VARIANTS[key] for key in chosen]

    results = {}

    def run_slices(label, fn):
        results[label] = {}
        line = f"{label:<28}"
        for name, frame in slices.items():
            scored = metrics.evaluate(
                fn, frame, combined, k=args.k, max_users=args.max_users,
                creator_map=creator_map, country_map=country_map,
                catalog_size=len(item_index))
            results[label][name] = scored
            line += f"{scored['ndcg']:>12.4f}"
        print(line, flush=True)

    header = f"{'cấu hình':<28}" + "".join(f"{s:>12}" for s in slices)
    print("\n" + "=" * 88)
    print(f"NDCG@{args.k} — chia theo {args.split}, chấm toàn danh mục")
    print("=" * 88)
    print(header)
    print("-" * len(header))
    for label, fn in recommenders.items():
        run_slices(label, fn)

    # Mỗi biến thể: huấn luyện, đánh giá ngay, rồi giải phóng. Giữ cả bốn mô
    # hình trong một dict tốn ~310 MB mà không dùng lại lần nào sau đó.
    #
    # `--seeds` lặp *cùng* một danh sách seed cho mọi biến thể, nên phép so giữa
    # hai biến thể thành ghép cặp: hai nhánh cùng khởi tạo, phần dao động do seed
    # triệt tiêu trong hiệu số. Không có nó thì hai biến thể chạy ở hai seed khác
    # nhau và mọi chênh lệch nhỏ hơn nhiễu seed đều vô nghĩa — đúng cái bẫy đã
    # làm mặt `art` bị kết luận sai một lần (xem docs/research-notes.md mục 10).
    seed_runs = {}
    for label, options in variants:
        for seed in (args.seeds or [0]):
            tag = label if not args.seeds else f"{label} [seed {seed}]"
            print(f"\nhuấn luyện {tag}  {options}", flush=True)
            opts = dict(options)
            columns = opts.pop("time_columns", None)
            block = (times[:, [TIME_COLUMNS.index(c) for c in columns]]
                     if columns else times)
            state = train_model(
                train, item_index, content, block, cutoff, train_counts,
                dim=args.dim, epochs=args.epochs, seed=seed, **opts)
            raw, units, bias = item_table(state, train_counts, args.dim)
            blob = pack(state, raw, units, bias, opts.get("use_time", True))
            run_slices(tag, make_recommender(ServedTwoTower(blob), reverse, args.k))
            if args.seeds:
                seed_runs.setdefault(label, {})[seed] = {
                    s: results[tag][s]["ndcg"] for s in slices}
            # Chỉ lưu trọng số ở seed đầu: quét seed là để đo nhiễu, không phải
            # để chọn seed may mắn rồi phát bản đó.
            if args.save and label in SAVE_AS and seed == (args.seeds or [0])[0]:
                path = args.out_dir / f"two_tower_{SAVE_AS[label]}.pt"
                torch.save(blob, path)
                print(f"  -> {path} ({path.stat().st_size / 1e6:.1f} MB)")
            del state, raw, units, bias, blob
            gc.collect()
            # Ghi ngay sau mỗi biến thể, không đợi hết vòng. Lý do là một lần
            # mất bài thật: hai biến thể ở quy mô đầy đủ chạy hơn bốn giờ và
            # chỉ ghi kết quả ở cuối, nên khi phiên bị khởi động lại ở epoch
            # 12/15 của biến thể đầu thì mất sạch, kể cả ba dòng baseline đã
            # đánh giá xong từ hai giờ trước. Tệp dở còn đọc được; không có
            # tệp thì không.
            checkpoint(args, results, started, done=True)

    if len(seed_runs) >= 2:
        names = list(seed_runs)
        base = names[0]
        shared = sorted(set.intersection(*(set(v) for v in seed_runs.values())))
        print("\n" + "=" * 88)
        print(f"GHÉP CẶP THEO SEED — mốc so: {base} | {len(shared)} seed dùng chung")
        print("=" * 88)
        for name in names[1:]:
            print(f"\n  {name}  −  {base}")
            print(f"  {'lát':<14}{'mốc':>9}{'biến thể':>10}{'chênh TB':>11}"
                  f"{'sd':>9}{'t':>8}{'df':>4}{'dương':>8}")
            for slice_name in slices:
                mine = np.array([seed_runs[name][s][slice_name] for s in shared])
                theirs = np.array([seed_runs[base][s][slice_name] for s in shared])
                pairs = mine - theirs
                if not np.any(pairs):
                    continue
                gap, spread = float(pairs.mean()), float(pairs.std(ddof=1))
                t = gap / (spread / np.sqrt(len(pairs))) if spread else float("inf")
                print(f"  {slice_name:<14}{theirs.mean():>9.4f}{mine.mean():>10.4f}"
                      f"{gap:>+11.4f}{spread:>9.4f}{t:>8.2f}{len(pairs) - 1:>4}"
                      f"{int((pairs > 0).sum()):>5}/{len(pairs)}")
        print("ngưỡng p=0,05 hai phía: df=2 → 4,30 | df=3 → 3,18 | df=4 → 2,78")
        results["_seed_runs"] = {k: {str(s): v for s, v in runs.items()}
                                 for k, runs in seed_runs.items()}

    if args.facet_ablation:
        print("\n" + "=" * 88)
        print("THANG ABLATION TỪNG MẶT — cấu hình discover (tắt logQ, không "
              "đầu thời điểm)")
        print("=" * 88)
        print(header)
        print("-" * len(header))
        for label, names in facet_ladder:
            block = content_matrix(state_facets, names=names)
            trained = train_model(
                train, item_index, block, times, cutoff, train_counts,
                dim=args.dim, epochs=args.epochs, use_time=False,
                use_logq=False, log=lambda *_: None)
            raw, units, bias = item_table(trained, train_counts, args.dim)
            blob = pack(trained, raw, units, bias, False)
            run_slices(f"mặt: {label} ({block.shape[1]}d)",
                       make_recommender(ServedTwoTower(blob), reverse, args.k))
            del trained, raw, units, bias, blob
            gc.collect()
        full = VARIANTS["discover"][0]
        if full in results:
            print(f"{'mặt: + art (đủ bốn mặt, 220d)':<28}" + "".join(
                f"{results[full][s]['ndcg']:>12.4f}" for s in slices)
                + "   = biến thể discover ở trên")

    if args.art_check:
        # Mặt `art` chỉ thêm +0,0016 NDCG ở lát new_item (+11% tương đối). Mức
        # đó đủ nhỏ để nhiễu khởi tạo giải thích được, nên trước khi nói "ảnh
        # bìa có ích" phải biết hai bậc cuối cách nhau bao xa so với độ dao
        # động giữa các seed. Chỉ lặp hai bậc cuối: các bậc dưới cách nhau xa
        # hơn nhiều nên không cần.
        print("\n" + "=" * 88)
        print(f"NHIỄU SEED — hai bậc cuối, {args.art_check} seed, lát new_item")
        print("=" * 88)
        runs = {}
        for name, names in (("không art (205d)", ("theme", "creator", "pace")),
                            ("có art (220d)", ("theme", "creator", "pace", "art"))):
            block = content_matrix(state_facets, names=names)
            values = []
            for seed in (args.art_seeds or list(range(args.art_check))):
                trained = train_model(
                    train, item_index, block, times, cutoff, train_counts,
                    dim=args.dim, epochs=args.epochs, use_time=False,
                    use_logq=False, seed=seed, log=lambda *_: None)
                raw, units, bias = item_table(trained, train_counts, args.dim)
                blob = pack(trained, raw, units, bias, False)
                scored = metrics.evaluate(
                    make_recommender(ServedTwoTower(blob), reverse, args.k),
                    slices["new_item"], combined, k=args.k,
                    max_users=args.max_users, catalog_size=len(item_index))
                values.append(scored["ndcg"])
                print(f"  {name:<18} seed {seed}  {scored['ndcg']:.4f}", flush=True)
                del trained, raw, units, bias, blob
                gc.collect()
            runs[name] = values
        for name, values in runs.items():
            print(f"{name:<18} trung bình {np.mean(values):.4f}  "
                  f"lệch chuẩn {np.std(values):.4f}  "
                  f"khoảng [{min(values):.4f}, {max(values):.4f}]")

        # So theo cặp, không so hai trung bình độc lập: hai cấu hình dùng *cùng*
        # seed nên cùng khởi tạo, và phần dao động do seed triệt tiêu trong hiệu
        # số. So độc lập là bỏ mất chính thiết kế ghép cặp đó và kết luận "không
        # tách được khỏi nhiễu" chỉ vì dùng sai phép kiểm — lỗi tôi đã mắc ở
        # bản đầu của hàm này.
        pairs = np.array(runs["có art (220d)"]) - np.array(runs["không art (205d)"])
        gap, spread = float(pairs.mean()), float(pairs.std(ddof=1))
        positive = int((pairs > 0).sum())
        print(f"\nghép cặp theo seed: chênh lệch {gap:+.4f} ± {spread:.4f} "
              f"| {positive}/{len(pairs)} seed dương")
        if spread > 0:
            t = gap / (spread / np.sqrt(len(pairs)))
            print(f"  t = {t:.2f} với df = {len(pairs) - 1} "
                  f"(ngưỡng p=0,05 hai phía: df=2 → 4,30; df=4 → 2,78)")
        print("  so độc lập (phép kiểm SAI ở đây, giữ để đối chiếu): "
              f"nhiễu lớn nhất {max(np.std(v) for v in runs.values()):.4f}")
        results["_art_check"] = {k: list(map(float, v)) for k, v in runs.items()}

    print("\n" + "=" * 88)
    print(f"Recall@{args.k}")
    print("=" * 88)
    print(header)
    print("-" * len(header))
    for label in [k for k in results if not k.startswith('_')]:
        print(f"{label:<28}" + "".join(
            f"{results[label][s]['recall']:>12.4f}" for s in slices))

    print("\n" + "=" * 88)
    print("NGOÀI ĐỘ CHÍNH XÁC (lát 'all')")
    print("=" * 88)
    print(f"{'cấu hình':<28}{'phủ catalog':>13}{'trùng tác giả':>15}"
          f"{'KR':>7}{'CN':>7}")
    for label in [k for k in results if not k.startswith('_')]:
        row = results[label]["all"]
        print(f"{label:<28}{row['coverage']:>13.4f}{row['creator_conc']:>15.3f}"
              f"{row.get('mix_KR', float('nan')):>7.2f}{row.get('mix_CN', float('nan')):>7.2f}")

    checkpoint(args, results, started)


def checkpoint(args, results, started, done=False):
    """Ghi kết quả ra đĩa; gọi được nhiều lần, lần sau ghi đè lần trước."""
    name = args.out_name or f"embedding_{args.split}.json"
    path = args.out_dir / name
    path.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    mark = "  [điểm kiểm tra]" if done else ""
    print(f"\n-> {path}   ({(time.monotonic() - started) / 60:.1f} phút)"
          f"{mark}", flush=True)


def preprocess_slices(train, test):
    from truyen_reco.data import preprocess
    return preprocess.evaluation_slices(train, test)


if __name__ == "__main__":
    main()
