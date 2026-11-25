"""Đánh giá: ablation từng mặt, đo theo từng lát, cộng phép kiểm nét vẽ độc lập.

Với người đọc mới, dùng giao thức onboarding: lấy vài tương tác sớm nhất của
họ làm hồ sơ ban đầu rồi dự đoán phần còn lại. Không làm vậy thì lát cold-start
không đo được gì ngoài độ phổ biến, và đóng góp của nét vẽ sẽ vô hình.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from truyen_reco.data import preprocess  # noqa: E402
from truyen_reco.evaluation import metrics  # noqa: E402
from truyen_reco.models.collaborative import ALS, PopularityModel  # noqa: E402
from truyen_reco.models.content_based import FacetModel  # noqa: E402
from truyen_reco.models.hybrid import HybridRecommender  # noqa: E402

SEED_ITEMS = 3


def load(out_dir):
    blob = np.load(out_dir / "facets.npz", allow_pickle=False)
    meta = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    facets = {k[len("facet_"):]: blob[k] for k in blob.files if k.startswith("facet_")}

    als = ALS()
    als.user_factors = blob["user_factors"]
    als.item_factors = blob["item_factors"]
    popularity = PopularityModel()
    popularity.scores = blob["popularity"]

    return {
        "facets": facets,
        "als": als,
        "popularity": popularity,
        "user_index": meta["user_index"],
        "item_index": {int(k): v for k, v in meta["item_index"].items()},
        "media": pd.read_parquet(out_dir / "media_core.parquet"),
        "train": pd.read_parquet(out_dir / "train.parquet"),
        "test": pd.read_parquet(out_dir / "test.parquet"),
        "cold_users": pd.read_parquet(out_dir / "cold_users.parquet"),
    }


def artist_retrieval_check(facets, media, index, top_k=10, min_works=3):
    """Láng giềng gần nhất trong không gian nét vẽ có phải cùng hoạ sĩ không.

    Nhãn miễn phí, không cần hệ gợi ý, nên chạy được trước mọi thứ khác. Nếu
    phép kiểm này thất bại thì đặc trưng nét vẽ không mang thông tin về nét vẽ.
    """
    if "art" not in facets:
        return None
    vectors = facets["art"]
    available = np.linalg.norm(vectors, axis=1) > 1e-8

    by_artist = defaultdict(list)
    for row in media.itertuples(index=False):
        position = index.get(row.media_id)
        if position is None or not available[position]:
            continue
        for staff_id in set(row.staff_art_ids):
            by_artist[staff_id].append(position)

    targets = {a: p for a, p in by_artist.items() if len(p) >= min_works}
    if not targets:
        return None

    positions = np.where(available)[0]
    block = vectors[positions]
    lookup = {p: i for i, p in enumerate(positions)}

    artist_of = defaultdict(set)
    for artist, works in by_artist.items():
        for position in works:
            artist_of[position].add(artist)

    hits, total, random_rate = [], 0, []
    for artist, works in list(targets.items())[:500]:
        for position in works[:5]:
            scores = block @ vectors[position]
            scores[lookup[position]] = -np.inf
            best = np.argpartition(-scores, top_k)[:top_k]
            neighbours = positions[best]
            hit = sum(1 for n in neighbours if artist in artist_of[n])
            hits.append(hit / top_k)
            random_rate.append((len(works) - 1) / max(len(positions) - 1, 1))
            total += 1

    return {
        "precision_at_k": float(np.mean(hits)) if hits else np.nan,
        "random_baseline": float(np.mean(random_rate)) if random_rate else np.nan,
        "queries": total,
        "artists": len(targets),
    }


def build_histories(train, cold_users, item_index):
    """Lịch sử train cho người đọc đã biết, cộng hồ sơ mồi cho người đọc mới.

    Người mới được đưa SEED_ITEMS truyện sớm nhất của họ làm hồ sơ onboarding —
    tương đương màn chọn vài ảnh bìa lúc đăng ký — rồi dự đoán phần còn lại.
    """
    history = (train.sort_values("event_time")
                    .groupby("user")["media_id"].apply(list).to_dict())
    history = {u: [item_index[m] for m in items if m in item_index]
               for u, items in history.items()}

    seed, holdout = {}, {}
    for user, frame in cold_users.groupby("user"):
        ordered = frame.sort_values("event_time")["media_id"].tolist()
        if len(ordered) <= SEED_ITEMS:
            continue
        seed[user] = [item_index[m] for m in ordered[:SEED_ITEMS] if m in item_index]
        holdout[user] = set(ordered[SEED_ITEMS:])
    return history, seed, holdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/processed", type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=1500)
    args = parser.parse_args()

    state = load(args.out_dir)
    media, train, test = state["media"], state["train"], state["test"]
    item_index = state["item_index"]
    facet_model = FacetModel(state["facets"], item_index)

    print("=" * 72)
    print("KIỂM NÉT VẼ ĐỘC LẬP (láng giềng có cùng hoạ sĩ không)")
    print("=" * 72)
    check = artist_retrieval_check(state["facets"], media, item_index, top_k=args.k)
    if check:
        lift = check["precision_at_k"] / max(check["random_baseline"], 1e-9)
        print(f"  precision@{args.k}: {check['precision_at_k']:.4f}")
        print(f"  ngẫu nhiên:      {check['random_baseline']:.4f}")
        print(f"  cao gấp:         {lift:.1f} lần  "
              f"({check['queries']} truy vấn, {check['artists']} hoạ sĩ)")
        print("  -> gấp dưới 2 lần nghĩa là đặc trưng nét vẽ gần như vô dụng")
    else:
        print("  không có đặc trưng nét vẽ")

    history, seed, holdout = build_histories(train, state["cold_users"], item_index)
    combined = dict(history)
    combined.update(seed)
    print(f"\nNgười đọc đã biết: {len(history):,} | người đọc mới (giữ riêng): "
          f"{len(seed):,}")

    creator_map = {}
    country_map = {}
    for row in media.itertuples(index=False):
        key = tuple(sorted(set(row.staff_art_ids) | set(row.staff_story_ids)))
        creator_map[row.media_id] = key or None
        country_map[row.media_id] = row.country

    slices = preprocess.evaluation_slices(train, test)
    cold = state["cold_users"]
    cold = cold[[m in holdout.get(u, set())
                 for u, m in zip(cold["user"], cold["media_id"])]]
    slices["new_user"] = cold

    configurations = {
        "popularity": {"cf": 0.0, "facets": {}},
        "CF": {"cf": 1.0, "facets": {}},
        "CF+theme": {"cf": 1.0, "facets": {"theme": 1.0}},
        "CF+theme+creator": {"cf": 1.0, "facets": {"theme": 1.0, "creator": 0.8}},
        "CF+theme+creator+pace": {"cf": 1.0, "facets": {"theme": 1.0, "creator": 0.8,
                                                        "pace": 0.6}},
        "tất cả mặt": {"cf": 1.0, "facets": {"theme": 1.0, "creator": 0.8,
                                             "pace": 0.6, "art": 0.7}},
    }
    if "art" not in state["facets"]:
        configurations.pop("tất cả mặt")

    results = {}
    for name, config in configurations.items():
        model = HybridRecommender(
            state["als"], facet_model, state["popularity"],
            state["user_index"], item_index,
            cf_weight=config["cf"], facet_weights=config["facets"],
            personalize=bool(config["facets"]),
        ).set_creator_groups(media, item_index)
        model.fit_weights(combined)

        def recommend(user, _history, model=model):
            return model.recommend(user, k=args.k,
                                   history=combined.get(user, []))

        results[name] = {}
        for slice_name in ("all", "warm", "sparse_user", "new_user", "new_item"):
            frame = slices.get(slice_name)
            if frame is None or frame.empty:
                continue
            results[name][slice_name] = metrics.evaluate(
                recommend, frame, combined, k=args.k,
                max_users=args.max_users,
                facet_vectors=state["facets"].get("theme"), index=item_index,
                creator_map=creator_map, country_map=country_map,
                catalog_size=len(item_index),
            )
        print(f"\n[{name}] xong")

    print("\n" + "=" * 72)
    print(f"ABLATION — NDCG@{args.k} theo từng lát")
    print("=" * 72)
    slice_names = ["all", "warm", "sparse_user", "new_user", "new_item"]
    header = f"{'cấu hình':<24}" + "".join(f"{s:>13}" for s in slice_names)
    print(header)
    print("-" * len(header))
    for name, per_slice in results.items():
        line = f"{name:<24}"
        for slice_name in slice_names:
            value = per_slice.get(slice_name, {}).get("ndcg", float("nan"))
            line += f"{value:>13.4f}" if value == value else f"{'--':>13}"
        print(line)

    print("\n" + "=" * 72)
    print("CHỈ SỐ NGOÀI ĐỘ CHÍNH XÁC (lát 'all')")
    print("=" * 72)
    print(f"{'cấu hình':<24}{'phủ catalog':>13}{'đa dạng':>11}"
          f"{'trùng tác giả':>15}{'KR':>7}{'CN':>7}")
    for name, per_slice in results.items():
        row = per_slice.get("all", {})
        print(f"{name:<24}{row.get('coverage', float('nan')):>13.4f}"
              f"{row.get('diversity', float('nan')):>11.3f}"
              f"{row.get('creator_conc', float('nan')):>15.3f}"
              f"{row.get('mix_KR', float('nan')):>7.2f}"
              f"{row.get('mix_CN', float('nan')):>7.2f}")

    payload = {"ablation": results, "artist_check": check}
    (args.out_dir / "evaluation.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\n-> {args.out_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
