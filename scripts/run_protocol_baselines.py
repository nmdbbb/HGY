"""Bậc thang mốc của nhánh "hướng ban đầu" — L2 của đề xuất, trên giao thức đề xuất.

    python scripts/run_protocol_baselines.py --dir data/processed_hgn_none
    python scripts/run_protocol_baselines.py --dir data/processed_hgn_volume

Năm cấu hình chạy trên CÙNG một bộ ứng viên và cùng phép loại trừ, để chênh lệch
giữa chúng là chênh lệch của mô hình chứ không của giao thức:

    ngẫu nhiên          mốc dưới, để biết một con số "có vẻ nhỏ" thật ra lớn bao nhiêu
    phổ biến            đếm số người đọc trong train
    điểm Bayes          điểm trung bình làm trơn theo số lượt (m = trung vị số lượt)
    Item-KNN            cosine giữa cột item, có hạ trọng số cặp ít người chung
    User-KNN            cosine giữa dòng user, chỉ tính cho người được đánh giá
    ALS                 lọc cộng tác ẩn (Hu, Koren, Volinsky 2008), dùng lại models/collaborative

Hai phép loại trừ, cả hai đều theo đúng mục 7.1 của đề xuất:

1. bỏ mọi mục người đọc đã có trong **train và val** (val không được rò sang test);
2. bỏ mọi mục **cùng bộ truyện** với một mục họ đã đọc — nếu không làm, mô hình
   ghi điểm cao bằng cách gợi "tập kế tiếp", đúng cái mà đề xuất gọi là "chỉ số
   cao nhưng gợi ý vô dụng". Với bản đã gộp bộ thì phép này gần như không còn
   việc; với bản chưa gộp thì nó cắt đi phần lớn điểm dễ.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from truyen_reco.data.protocol_peruser import (  # noqa: E402
    excluded_candidates, reachable_test)
from truyen_reco.evaluation import metrics  # noqa: E402
from truyen_reco.models.knn import (  # noqa: E402
    top_k_neighbours, user_item_matrix)
from truyen_reco.models.bpr import BPRMF  # noqa: E402
from truyen_reco.models.content import (  # noqa: E402
    ContentModel, build_item_features, user_profiles)
from truyen_reco.models.collaborative import ALS  # noqa: E402


def load(directory):
    directory = Path(directory)
    data = {name: pd.read_parquet(directory / f"{name}.parquet")
            for name in ("train", "val", "test")}
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    data["user_index"] = {k: v for k, v in index["user_index"].items()}
    data["item_index"] = {int(k): v for k, v in index["item_index"].items()}
    series = json.loads((directory / "series_of.json").read_text(encoding="utf-8"))
    data["series_of"] = {int(k): v for k, v in series.items()}
    merge_path = directory / "merge_map.json"
    data["merged_of"] = ({int(k): v for k, v in json.loads(
        merge_path.read_text(encoding="utf-8")).items()}
        if merge_path.exists() else {})
    data["report"] = json.loads(
        (directory / "protocol_report.json").read_text(encoding="utf-8"))
    return data


def ranker(score_fn, user_index, reverse, excluded, k):
    """Biến hàm điểm thành recommend_fn(user, history) -> list media_id."""
    def recommend(user, _history):
        row = user_index.get(user)
        if row is None:
            return []
        scores = score_fn(row)
        if scores is None:
            return []
        scores = scores.astype(np.float32, copy=True)
        drop = excluded.get(user)
        if drop is not None and len(drop):
            scores[drop] = -np.inf
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [reverse[int(i)] for i in top if np.isfinite(scores[int(i)])]
    return recommend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/processed_hgn_none", type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--neighbours", type=int, default=200)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--models", nargs="+",
                        default=["random", "popular", "bayes", "itemknn",
                                 "userknn", "als", "bpr", "content",
                                 "sessionknn"])
    parser.add_argument("--session-length", type=int, default=5,
                        help="L mục gần nhất coi là 'phiên' cho Session-KNN")
    parser.add_argument("--media", default="data/processed_gr/media.parquet",
                        type=Path, help="metadata truyện cho mô hình nội dung")
    parser.add_argument("--min-df", type=int, default=20,
                        help="kệ phải xuất hiện trên >= N truyện mới vào từ vựng")
    parser.add_argument("--tail-head-pct", type=float, default=20.0,
                        help="bao nhiêu %% truyện đầu bảng tính là 'đầu danh mục' "
                             "khi đo riêng trên mục test ít tương tác")
    parser.add_argument("--bpr-epochs", type=int, default=20)
    parser.add_argument("--bpr-bias", action="store_true",
                        help="BẬT số hạng chệch theo item của BPR. Mặc định tắt, "
                             "vì bật thì mô hình sập về bảng phổ biến "
                             "(docs/protocol-arm.md mục 9)")
    parser.add_argument("--label-suffix", default="",
                        help="thêm hậu tố vào tên cấu hình, để hai lần chạy "
                             "khác tham số không ghi đè nhau trong JSON")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = load(args.dir)
    train, val, test = data["train"], data["val"], data["test"]
    user_index, item_index = data["user_index"], data["item_index"]
    reverse = {v: k for k, v in item_index.items()}
    print(f"{args.dir}: train {len(train):,} | val {len(val):,} | test "
          f"{len(test):,} | {len(user_index):,} người | {len(item_index):,} mục")

    matrix = user_item_matrix(train, user_index, item_index)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    n_users = matrix.shape[0]
    popularity = {reverse[i]: int(counts[i]) for i in range(len(counts))}
    order = np.argsort(-counts)
    percentile = np.empty(len(counts))
    percentile[order] = 1.0 - np.arange(len(counts)) / max(len(counts) - 1, 1)
    pop_percentile = {reverse[i]: float(percentile[i]) for i in range(len(counts))}
    head_items = {reverse[int(i)] for i in order[:max(1, len(counts) // 100)]}

    print("  dựng tập loại trừ (đã đọc + cùng bộ)...", flush=True)
    started = time.monotonic()
    excluded = excluded_candidates(train, val, user_index, item_index,
                                   data["series_of"])
    sizes = np.array([len(v) for v in excluded.values()])
    print(f"    {time.monotonic() - started:.0f}s | loại trung vị "
          f"{np.median(sizes):.0f} mục/người (tối đa {sizes.max():,})")

    # Đáp án phải nằm trong rổ ứng viên, nếu không thì mẫu số của Recall/NDCG
    # chứa những mục không mô hình nào với tới được (xem reachable_test).
    test, reach = reachable_test(test, excluded, item_index)
    print(f"    đáp án với tới được: {reach['test_rows_kept']:,}/"
          f"{reach['test_rows_in']:,} dòng test "
          f"({100 * reach['test_rows_kept'] / max(reach['test_rows_in'], 1):.1f}%)"
          f" | bỏ vì cùng bộ/đã đọc {reach['dropped_same_series_or_seen']:,}"
          f" | bỏ vì mục không có trong train {reach['dropped_item_not_in_train']:,}")

    cut = max(1, int(len(counts) * args.tail_head_pct / 100.0))
    head_by_pct = {reverse[int(i)] for i in order[:cut]}
    test_tail = test[~test["media_id"].isin(head_by_pct)]
    print(f"    lát mục ít tương tác: {len(test_tail):,}/{len(test):,} dòng test "
          f"nằm ngoài {args.tail_head_pct:.0f}% truyện đầu bảng "
          f"({test_tail['user'].nunique():,} người)")

    train_by_user = train.groupby("user")["media_id"].apply(list).to_dict()
    results = {}

    def run(name, score_fn):
        name = name + args.label_suffix
        started = time.monotonic()
        recommend = ranker(score_fn, user_index, reverse, excluded, args.k)
        out = metrics.evaluate(
            recommend, test, train_by_user, k=args.k,
            max_users=args.max_users, catalog_size=len(item_index),
            seed=args.seed, popularity=popularity, n_users=n_users,
            pop_percentile=pop_percentile, head_items=head_items)
        # Lát thứ hai: CHỈ những dòng test có mục nằm ngoài đầu danh mục. Đây là
        # phép so "nội dung có cứu truyện ít tương tác không" của đề xuất; đo
        # trên cùng một recommend_fn nên khác biệt duy nhất là tập đáp án.
        tail = metrics.evaluate(
            recommend, test_tail, train_by_user, k=args.k,
            max_users=args.max_users, catalog_size=len(item_index),
            seed=args.seed)
        out["ndcg_tail"] = tail["ndcg"]
        out["recall_tail"] = tail["recall"]
        out["users_tail"] = tail["users_evaluated"]
        out["seconds"] = round(time.monotonic() - started, 1)
        results[name] = out
        print(f"  {name:<12} NDCG@{args.k} {out['ndcg']:.4f} | Recall "
              f"{out['recall']:.4f} | HR {out['hit_rate']:.4f} | "
              f"{out['seconds']:.0f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    if "random" in args.models:
        run("ngẫu nhiên", lambda row: rng.random(len(counts)).astype(np.float32))
    if "popular" in args.models:
        run("phổ biến", lambda row: counts)
    if "bayes" in args.models:
        sums = np.asarray(matrix.sum(axis=0)).ravel()      # train là nhị phân
        rated = train.dropna(subset=["score"])
        mean_by_item = (rated.groupby("media_id")["score"].mean()
                        / 20.0)                            # về thang 1–5
        count_by_item = rated.groupby("media_id")["score"].size()
        prior = float(mean_by_item.mean())
        m = float(np.median(count_by_item.values)) if len(count_by_item) else 1.0
        bayes = np.full(len(counts), prior, dtype=np.float32)
        for media_id, mean in mean_by_item.items():
            position = item_index.get(media_id)
            if position is None:
                continue
            n = float(count_by_item.get(media_id, 0))
            bayes[position] = (n * mean + m * prior) / (n + m)
        del sums
        run("điểm Bayes", lambda row: bayes)
    if "itemknn" in args.models:
        print(f"  dựng Item-KNN (k={args.neighbours})...", flush=True)
        started = time.monotonic()
        similarity = top_k_neighbours(matrix, k=args.neighbours)
        print(f"    {time.monotonic() - started:.0f}s | "
              f"{similarity.nnz:,} cặp láng giềng", flush=True)
        scores_all = None
        def item_knn(row):
            profile = matrix[row]
            if profile.nnz == 0:
                return None
            return np.asarray((profile @ similarity).todense()).ravel()
        run("Item-KNN", item_knn)
    if "userknn" in args.models:
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        def user_knn(row, k_users=300):
            profile = matrix[row]
            if profile.nnz == 0:
                return None
            similarity = np.asarray((matrix @ profile.T).todense()).ravel()
            similarity /= norms * norms[row]
            similarity[row] = 0.0
            picked = np.argpartition(-similarity, k_users)[:k_users]
            weights = similarity[picked]
            keep = weights > 0
            if not keep.any():
                return None
            return np.asarray(
                (sparse.csr_matrix(weights[keep]) @ matrix[picked[keep]]).todense()
            ).ravel()
        run("User-KNN", user_knn)
    if "als" in args.models:
        print(f"  huấn luyện ALS ({args.factors} nhân tố, "
              f"{args.iterations} vòng)...", flush=True)
        started = time.monotonic()
        model = ALS(factors=args.factors, iterations=args.iterations)
        model.fit(matrix)
        print(f"    {time.monotonic() - started:.0f}s", flush=True)
        run("ALS", lambda row: model.item_factors @ model.user_factors[row])

    if "bpr" in args.models:
        print(f"  huấn luyện BPR-MF ({args.factors} nhân tố, "
              f"{args.bpr_epochs} epoch)...", flush=True)
        started = time.monotonic()
        bpr = BPRMF(factors=args.factors, iterations=args.bpr_epochs,
                    seed=args.seed, use_item_bias=args.bpr_bias)
        bpr.fit(matrix, log=lambda line: print(line, flush=True))
        print(f"    {time.monotonic() - started:.0f}s", flush=True)
        run("BPR-MF" + (" (có chệch)" if args.bpr_bias else ""),
            bpr.scores_for)

    # Gộp vào kết quả cũ thay vì ghi đè: chạy lại một mô hình lẻ không được
    # xoá số của những mô hình khác, nếu không thì bảng so sánh phải chạy lại
    # toàn bộ mỗi lần muốn thêm một cấu hình.
    if "sessionknn" in args.models:
        # Session-KNN của đề xuất (L6) = Item-KNN nhưng hồ sơ chỉ gồm L mục
        # GẦN NHẤT. Đặt cạnh Item-KNN (toàn bộ hồ sơ) thì chênh lệch giữa hai
        # cái trả lời đúng câu "thứ tự đọc có giá trị?" mà không phải huấn luyện
        # mô hình chuỗi nào — và kết quả đó quyết định có nên bỏ vài giờ cho
        # GRU4Rec/SASRec hay không.
        if "similarity" not in dir():
            print(f"  dựng Item-KNN cho Session-KNN (k={args.neighbours})...",
                  flush=True)
            similarity = top_k_neighbours(matrix, k=args.neighbours)
        ordered = train.sort_values(["user", "created_at"], kind="mergesort")
        tail_rows = ordered.groupby("user", sort=False).tail(args.session_length)
        session = {}
        for user, group in tail_rows.groupby("user", sort=False)["media_id"]:
            row = user_index.get(user)
            if row is None:
                continue
            positions = [item_index[m] for m in group if m in item_index]
            if positions:
                session[row] = np.asarray(positions, dtype=np.int32)

        def session_knn(row, decay=None):
            positions = session.get(row)
            if positions is None:
                return None
            if decay is None:
                weights = np.ones(len(positions), dtype=np.float32)
            else:
                # Mục cuối trọng số 1, lùi về trước nhân `decay` mỗi bậc.
                weights = np.power(decay, np.arange(len(positions) - 1, -1, -1)
                                   ).astype(np.float32)
            profile = sparse.csr_matrix(
                (weights, (np.zeros(len(positions), dtype=np.int32), positions)),
                shape=(1, matrix.shape[1]))
            return np.asarray((profile @ similarity).todense()).ravel()

        run(f"Session-KNN (L={args.session_length})", session_knn)
        run(f"Session-KNN (L={args.session_length}, giảm 0,7)",
            lambda row: session_knn(row, decay=0.7))

    if "content" in args.models:
        if not args.media.exists():
            print(f"  [bỏ qua nội dung] thiếu {args.media}")
        else:
            print(f"  dựng đặc trưng nội dung (min_df={args.min_df})...",
                  flush=True)
            started = time.monotonic()
            media = pd.read_parquet(args.media)
            item_matrix, vocabulary = build_item_features(
                media, item_index, merged_of=data["merged_of"],
                min_df=args.min_df)
            profiles = user_profiles(train, item_matrix, user_index, item_index)
            print(f"    {time.monotonic() - started:.0f}s | từ vựng "
                  f"{len(vocabulary):,} | {item_matrix.nnz:,} ô khác 0",
                  flush=True)
            run("nội dung", ContentModel(item_matrix, profiles).scores_for)

    out = args.dir / "baseline_results.json"
    if out.exists():
        previous = json.loads(out.read_text(encoding="utf-8"))
        merged = dict(previous.get("results", {}))
        merged.update(results)
        results = merged
    out.write_text(json.dumps({"dir": str(args.dir), "k": args.k,
                               "max_users": args.max_users,
                               "protocol": data["report"],
                               "reachable": reach,
                               "results": results},
                              indent=2, ensure_ascii=False), encoding="utf-8")

    columns = [("ndcg", f"NDCG@{args.k}"), ("ndcg_tail", "NDCG ít-lượt"),
               ("recall", f"Recall@{args.k}"), ("hit_rate", "HR@10"),
               ("mrr", "MRR"), ("map", "MAP"), ("coverage", "phủ"),
               ("novelty", "mới lạ"), ("arp", "ARP"), ("tail", "long-tail")]
    header = f"  {'cấu hình':<14}" + "".join(f"{label:>11}" for _, label in columns)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, row in results.items():
        line = f"  {name:<14}"
        for key, _ in columns:
            value = row.get(key, float("nan"))
            line += f"{value:>11.4f}" if isinstance(value, float) else f"{value:>11}"
        print(line)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
