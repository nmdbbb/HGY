"""Phân tích sâu, nối tiếp C–G trong scripts/diagnose_data.py.

    python scripts/analyze_deeper.py

H. Truy hồi cùng hoạ sĩ 11,8x: bao nhiêu phần là cùng-bộ-truyện (franchise)?
I. Gu trôi theo thời gian nhanh cỡ nào — chu kỳ bán rã 180 ngày có căn cứ không?
J. ALS hiện tại phục vụ người đọc manhwa/manhua thế nào so với người đọc manga?
K. Tín hiệu DROPPED có đáng làm mẫu âm không?

Kết quả và cách đọc: docs/research-notes.md
"""

import json
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

D = "data/processed/"
K = 10
rng = np.random.default_rng(0)

z = np.load(D + "facets.npz")
index = json.load(open(D + "index.json", encoding="utf-8"))
item_index = {int(k): v for k, v in index["item_index"].items()}
media = pd.read_parquet(D + "media_core.parquet")
train = pd.read_parquet(D + "train.parquet")
test = pd.read_parquet(D + "test.parquet")
n_items = len(item_index)


def normalized(block):
    norms = np.linalg.norm(block, axis=1)
    ok = norms > 1e-8
    out = block.copy()
    out[ok] = out[ok] / norms[ok, None]
    return out, ok


# ------------------------------------------------------------------ H
print("=== H. 11,8x của nét vẽ: kiểu vẽ hay cùng bộ truyện? ===")
art, art_ok = normalized(z["facet_art"].astype(np.float32))


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


title_tokens, artist_sets = {}, {}
for row in media.itertuples(index=False):
    position = item_index.get(row.media_id)
    if position is None:
        continue
    title_tokens[position] = tokens(row.title_romaji) or tokens(row.title)
    artist_sets[position] = set(row.staff_art_ids)

by_artist = defaultdict(list)
for position, artists in artist_sets.items():
    if not art_ok[position]:
        continue
    for artist in artists:
        by_artist[artist].append(position)
prolific = {a for a, items in by_artist.items() if len(items) >= 3}


def same_franchise(a, b):
    """Đại diện thô cho 'cùng bộ': tên chung quá nửa số từ.

    Bắt được phần tiếp theo, ngoại truyện, spin-off đặt tên theo bộ gốc.
    KHÔNG bắt được franchise đổi hẳn tên — nên số 'cùng bộ' dưới đây là cận
    dưới, và phần 'khác bộ' là cận trên.
    """
    t1, t2 = title_tokens.get(a, set()), title_tokens.get(b, set())
    if not t1 or not t2:
        return False
    return len(t1 & t2) / min(len(t1), len(t2)) >= 0.5


queries = [p for p, artists in artist_sets.items()
           if art_ok[p] and artists & prolific]
queries = rng.choice(queries, min(1500, len(queries)), replace=False)
covered = np.where(art_ok)[0]
art_covered = art[covered]

hit_franchise = hit_cross = 0
baseline_cross_rates = []
for q in queries:
    scores = art_covered @ art[q]
    top = np.argpartition(-scores, K + 1)[:K + 1]
    neighbours = [int(covered[i]) for i in top[np.argsort(-scores[top])]
                  if covered[i] != q][:K]
    query_artists = artist_sets[q]
    same_artist_pool = {p for a in query_artists & prolific
                        for p in by_artist[a] if p != q}
    cross_pool = {p for p in same_artist_pool if not same_franchise(p, q)}
    baseline_cross_rates.append(len(cross_pool) / len(covered))
    for p in neighbours:
        if artist_sets[p] & query_artists:
            if same_franchise(p, q):
                hit_franchise += 1
            else:
                hit_cross += 1

total = len(queries) * K
precision_all = (hit_franchise + hit_cross) / total
precision_cross = hit_cross / total
baseline_cross = float(np.mean(baseline_cross_rates))
print(f"  precision@{K} cùng hoạ sĩ (mọi loại): {precision_all:.4f}")
print(f"  ... trong đó CÙNG BỘ truyện        : {hit_franchise/total:.4f} "
      f"({hit_franchise/(hit_franchise+hit_cross):.0%} số lần trúng)")
print(f"  ... KHÁC BỘ truyện (mới là kiểu vẽ): {precision_cross:.4f}")
print(f"  ngẫu nhiên khác bộ                 : {baseline_cross:.5f}")
print(f"  -> lift khác bộ: {precision_cross/max(baseline_cross,1e-9):.1f}x")

# ------------------------------------------------------------------ I
print("\n=== I. Gu trôi theo thời gian (cosine theme giữa các mục cách nhau g tháng) ===")
theme, theme_ok = normalized(z["facet_theme"].astype(np.float32))
valid = train[train["event_time"] > pd.Timestamp("2015-01-01")].copy()
valid["position"] = valid["media_id"].map(item_index)
valid["month"] = (valid["event_time"].astype("int64") // 10**9) // (86400 * 30)
spans = valid.groupby("user")["month"].agg(["min", "max", "count"])
eligible = spans[(spans["count"] >= 40) & (spans["max"] - spans["min"] >= 24)].index
sample_users = rng.choice(eligible, min(400, len(eligible)), replace=False)

gaps = {0: [], 3: [], 6: [], 12: [], 24: []}
for user in sample_users:
    block = valid[valid["user"] == user].dropna(subset=["position"])
    if len(block) > 200:
        block = block.sample(200, random_state=0)
    positions = block["position"].astype(int).to_numpy()
    months = block["month"].to_numpy()
    keep = theme_ok[positions]
    positions, months = positions[keep], months[keep]
    if len(positions) < 10:
        continue
    sim = theme[positions] @ theme[positions].T
    diff = np.abs(months[:, None] - months[None, :])
    upper = np.triu_indices(len(positions), k=1)
    sim, diff = sim[upper], diff[upper]
    for gap in gaps:
        mask = np.abs(diff - gap) <= 1
        if mask.sum() >= 5:
            gaps[gap].append(sim[mask].mean())

base = np.mean(gaps[0])
print(f"  {'cách nhau':<12}{'cosine':>8}{'so với cùng tháng':>20}")
for gap, values in gaps.items():
    mean = np.mean(values)
    print(f"  {gap:>2} tháng     {mean:>8.4f}{mean/base:>19.0%} ({len(values)} người)")

# ------------------------------------------------------------------ J
print("\n=== J. ALS hiện tại theo nhóm người đọc (NDCG@10, chấm toàn danh mục) ===")
user_factors = z["user_factors"]
item_factors = z["item_factors"]
user_index = index["user_index"]
country = media.set_index("media_id")["country"]
country_of = np.array([country.get(mid, "??")
                       for mid, p in sorted(item_index.items(), key=lambda kv: kv[1])])
train_positions = train.copy()
train_positions["position"] = train_positions["media_id"].map(item_index)
seen_by_user = train_positions.groupby("user")["position"].apply(
    lambda s: s.dropna().astype(int).to_numpy())
test_sets = test.groupby("user")["media_id"].apply(set)


def ndcg(recommended, relevant, k=10):
    gains = sum(1.0 / np.log2(i + 2) for i, m in enumerate(recommended[:k])
                if m in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return gains / ideal if ideal else np.nan


reverse = {v: k for k, v in item_index.items()}
groups = defaultdict(list)
users = [u for u in test_sets.index if u in user_index and u in seen_by_user.index]
for user in rng.choice(users, min(2500, len(users)), replace=False):
    seen = seen_by_user[user]
    shares = pd.Series(country_of[seen]).value_counts(normalize=True)
    dominant = shares.index[0] if shares.iloc[0] >= 0.6 else "hỗn hợp"
    scores = item_factors @ user_factors[user_index[user]]
    scores[seen] = -np.inf
    top = np.argpartition(-scores, K)[:K]
    top = top[np.argsort(-scores[top])]
    groups[dominant].append(ndcg([reverse[int(t)] for t in top], test_sets[user]))

for name, values in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    print(f"  {name:<10} {np.nanmean(values):.4f}  ({len(values)} người)")

# ------------------------------------------------------------------ K
print("\n=== K. DROPPED có đáng làm mẫu âm không ===")
interactions = pd.read_parquet(D + "interactions.parquet")
drops = interactions[interactions["status"] == "DROPPED"].copy()
drops["position"] = drops["media_id"].map(item_index)
drops = drops.dropna(subset=["position"])
drop_sets = drops.groupby("user")["position"].apply(
    lambda s: s.astype(int).to_numpy())

rows = []
candidates = [u for u in test_sets.index
              if u in drop_sets.index and len(drop_sets[u]) >= 5
              and u in seen_by_user.index and len(seen_by_user[u]) >= 10]
for user in rng.choice(candidates, min(600, len(candidates)), replace=False):
    dropped = drop_sets[user]
    dropped = dropped[theme_ok[dropped]]
    history = seen_by_user[user]
    history = history[theme_ok[history]]
    future = np.array([item_index[m] for m in test_sets[user] if m in item_index])
    future = future[theme_ok[future]] if len(future) else future
    if len(dropped) < 3 or len(history) < 5 or len(future) < 3:
        continue
    history_sample = rng.choice(history, min(len(dropped), len(history)),
                                replace=False)
    to_drop = (theme[future] @ theme[dropped].T).max(axis=1).mean()
    to_history = (theme[future] @ theme[history_sample].T).max(axis=1).mean()
    drop_vs_history = (theme[dropped] @ theme[history].T).max(axis=1).mean()
    random_items = rng.choice(np.where(theme_ok)[0], len(dropped), replace=False)
    random_vs_history = (theme[random_items] @ theme[history].T).max(axis=1).mean()
    rows.append((to_drop, to_history, drop_vs_history, random_vs_history))

rows = np.array(rows)
print(f"  ({len(rows)} người có >=5 truyện bỏ dở)")
print(f"  truyện SẮP đọc giống truyện ĐÃ BỎ    : {rows[:,0].mean():.4f}")
print(f"  truyện SẮP đọc giống truyện ĐÃ THÍCH : {rows[:,1].mean():.4f}")
print(f"  truyện ĐÃ BỎ giống lịch sử           : {rows[:,2].mean():.4f}")
print(f"  truyện NGẪU NHIÊN giống lịch sử      : {rows[:,3].mean():.4f}")
print("  -> nếu dòng 1 ~ dòng 2: bỏ dở không phải vì chủ đề, mẫu âm theo")
print("     theme sẽ phạt oan; nếu dòng 1 < dòng 2 rõ: DROPPED dùng được.")
