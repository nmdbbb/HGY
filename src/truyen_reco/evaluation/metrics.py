"""Chỉ số đánh giá, đo theo từng lát dữ liệu.

Đo tổng thể sẽ che mất đóng góp của nét vẽ và nhịp truyện: phần lớn lượt đánh
giá rơi vào người đọc đã có nhiều lịch sử, nơi lọc cộng tác tự nó đã đủ tốt.
Đóng góp của đặc trưng nội dung nằm ở lát người mới và truyện mới.
"""

import numpy as np


def precision_at_k(recommended, relevant, k):
    if not recommended:
        return 0.0
    top = recommended[:k]
    return len(set(top) & relevant) / len(top)


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return np.nan
    return len(set(recommended[:k]) & relevant) / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    if not relevant:
        return np.nan
    gains = [1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k])
             if item in relevant]
    dcg = sum(gains)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else np.nan


def average_precision(recommended, relevant, k):
    if not relevant:
        return np.nan
    hits, total = 0, 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            total += hits / (i + 1)
    return total / min(len(relevant), k)


def intra_list_diversity(recommended, facet_vectors, index):
    positions = [index[item] for item in recommended if item in index]
    if len(positions) < 2:
        return np.nan
    block = facet_vectors[positions]
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    normalized = block / np.maximum(norms, 1e-9)
    similarity = normalized @ normalized.T
    upper = similarity[np.triu_indices(len(positions), k=1)]
    return float(1.0 - upper.mean())


def creator_concentration(recommended, creator_map):
    """Tỉ lệ chỗ trong danh sách bị chiếm bởi người sáng tác lặp lại.

    Gợi ý toàn tác phẩm của cùng một hoạ sĩ thì đúng một cách tầm thường mà vô
    dụng, nên phải đo chứ không chỉ chặn.
    """
    groups = [creator_map.get(item) for item in recommended]
    groups = [g for g in groups if g]
    if not groups:
        return np.nan
    return 1.0 - len(set(groups)) / len(groups)


def country_mix(recommended, country_map):
    counts = {}
    for item in recommended:
        country = country_map.get(item)
        if country:
            counts[country] = counts.get(country, 0) + 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def mrr_at_k(recommended, relevant, k):
    """Nghịch đảo hạng của cú đúng ĐẦU TIÊN — 0 nếu không có cú nào trong top-k."""
    for position, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def hit_rate_at_k(recommended, relevant, k):
    """Có hay không ít nhất một cú đúng; khác recall ở chỗ không chia theo cỡ."""
    return 1.0 if set(recommended[:k]) & relevant else 0.0


def novelty(recommended, popularity, n_users):
    """Độ mới lạ = trung bình -log2(tỉ lệ người đã đọc mục đó).

    Mục càng ít người đọc thì càng "mới lạ". Mục không có trong bảng độ phổ
    biến bị bỏ qua chứ không tính là 0: 0 lượt sẽ cho -log2(0) = vô cực, và một
    giá trị vô cực lẫn vào trung bình thì cả cột thành vô nghĩa.
    """
    values = []
    for item in recommended:
        count = popularity.get(item, 0)
        if count > 0 and n_users > 0:
            values.append(-np.log2(count / n_users))
    return float(np.mean(values)) if values else np.nan


def average_popularity_rank(recommended, pop_percentile):
    """ARP: phân vị độ phổ biến trung bình của danh sách, 1,0 = đầu danh mục."""
    values = [pop_percentile[item] for item in recommended
              if item in pop_percentile]
    return float(np.mean(values)) if values else np.nan


def long_tail_share(recommended, head_items):
    """Tỉ lệ mục trong danh sách KHÔNG thuộc đầu danh mục."""
    if not recommended:
        return np.nan
    return 1.0 - len([i for i in recommended if i in head_items]) / len(recommended)


def evaluate(recommend_fn, test_frame, train_by_user, k=10, max_users=2000,
             facet_vectors=None, index=None, creator_map=None,
             country_map=None, catalog_size=None, seed=42,
             popularity=None, n_users=None, pop_percentile=None,
             head_items=None):
    """Chạy đánh giá trên một lát. recommend_fn(user, history) -> list media_id."""
    grouped = test_frame.groupby("user")["media_id"].apply(set)
    users = list(grouped.index)
    if len(users) > max_users:
        rng = np.random.default_rng(seed)
        users = list(rng.choice(users, max_users, replace=False))

    # `country_map` có khoá cho mọi truyện nhưng giá trị có thể None hết (bộ
    # Goodreads không có trường xuất xứ). Dict đó vẫn "truthy", nên nếu chỉ
    # kiểm tra `if country_map` thì tỉ lệ manhua sẽ ra 0,00 và bảng kết quả
    # tuyên bố một con số về mô hình trong khi nguyên nhân là thiếu dữ liệu.
    # Thiếu dữ liệu phải ra nan, không ra 0.
    has_country = bool(country_map) and any(
        v is not None for v in country_map.values())

    rows = {m: [] for m in ("precision", "recall", "ndcg", "map",
                            "diversity", "creator_conc",
                            "mrr", "hit_rate", "novelty", "arp", "tail")}
    recommended_items = set()
    mixes = []

    for user in users:
        relevant = grouped[user]
        history = train_by_user.get(user, [])
        recommended = recommend_fn(user, history)
        if not recommended:
            continue
        recommended_items.update(recommended)

        rows["precision"].append(precision_at_k(recommended, relevant, k))
        rows["recall"].append(recall_at_k(recommended, relevant, k))
        rows["ndcg"].append(ndcg_at_k(recommended, relevant, k))
        rows["map"].append(average_precision(recommended, relevant, k))
        rows["mrr"].append(mrr_at_k(recommended, relevant, k))
        rows["hit_rate"].append(hit_rate_at_k(recommended, relevant, k))
        # Ba chỉ số ngoài độ chính xác chỉ tính khi người gọi đưa bảng độ phổ
        # biến; không có bảng thì để trống để chúng ra nan, không ra 0.
        if popularity is not None and n_users:
            rows["novelty"].append(novelty(recommended, popularity, n_users))
        if pop_percentile is not None:
            rows["arp"].append(average_popularity_rank(recommended, pop_percentile))
        if head_items is not None:
            rows["tail"].append(long_tail_share(recommended, head_items))
        if facet_vectors is not None and index is not None:
            rows["diversity"].append(
                intra_list_diversity(recommended, facet_vectors, index))
        if creator_map:
            rows["creator_conc"].append(creator_concentration(recommended, creator_map))
        if has_country:
            mixes.append(country_mix(recommended, country_map))

    result = {name: (float(np.nanmean(values)) if values else np.nan)
              for name, values in rows.items()}
    result["users_evaluated"] = len(rows["precision"])
    result["coverage"] = (len(recommended_items) / catalog_size
                          if catalog_size else np.nan)
    if mixes:
        for country in ("JP", "KR", "CN"):
            result[f"mix_{country}"] = float(
                np.mean([m.get(country, 0.0) for m in mixes]))
    elif country_map is not None:
        # Ghi nan một cách tường minh thay vì bỏ trống khoá: bên in bảng dùng
        # .get() nên khoá trống sẽ thành số mặc định, còn nan thì in ra "nan".
        for country in ("JP", "KR", "CN"):
            result[f"mix_{country}"] = float("nan")
    return result
