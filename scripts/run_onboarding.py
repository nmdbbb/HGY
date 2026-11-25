"""Onboarding đo được: đường cong "k lựa chọn đầu tiên" so với lịch sử đầy đủ.

    python scripts/run_onboarding.py --picks 1 3 5 10 20

`cold_start.py` dựng được hồ sơ từ vài lựa chọn ban đầu, nhưng chưa bao giờ
được **chấm như một bộ gợi ý**, nên nó là dòng duy nhất còn mang trạng thái
"một phần" trong bảng. Script này đóng chỗ đó.

Thiết kế chỉ đổi **một** thứ so với các dòng khác trong bảng: đầu vào của hàm
điểm là `k` mục gần nhất thay vì toàn bộ lịch sử. Rổ ứng viên, tập loại trừ,
tập test, số người chấm, hạt giống — giữ nguyên hệt. Nhờ vậy con số đặt thẳng
cạnh Item-KNN lịch sử đầy đủ (0,1018) được, và hiệu giữa hai số đúng là *giá
trị của việc biết thêm lịch sử*, không lẫn thứ gì khác.

Giữ tập loại trừ nguyên vẹn là chỗ dễ làm sai nhất. Nếu mô phỏng người mới cho
đúng tinh thần thì chỉ nên loại `k` mục đã lộ, vì người mới chưa đọc phần còn
lại. Nhưng làm thế thì rổ ứng viên **to hơn** rổ của mọi dòng khác, và con số
rơi xuống vì lý do chẳng liên quan gì tới onboarding. Chúng tôi chọn so được
với bảng, và ghi rõ đây là "bộ gợi ý bị bịt mắt bớt lịch sử", không phải "người
đọc mới hoàn toàn".
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from truyen_reco.data.protocol_peruser import (  # noqa: E402
    excluded_candidates, reachable_test)
from truyen_reco.evaluation import metrics  # noqa: E402
from truyen_reco.models.content import build_item_features  # noqa: E402
from truyen_reco.models.knn import (  # noqa: E402
    top_k_neighbours, user_item_matrix)
from run_protocol_baselines import load, ranker  # noqa: E402


def recent_items(train, user_index, item_index):
    """Danh sách cột theo thứ tự thời gian GIẢM dần cho từng hàng người."""
    frame = train[["user", "media_id", "created_at"]].copy()
    frame["row"] = frame["user"].map(user_index)
    frame["col"] = frame["media_id"].map(item_index)
    frame = frame.dropna(subset=["row", "col"])
    frame = frame.sort_values(["row", "created_at"], ascending=[True, False],
                              kind="mergesort")
    grouped = frame.groupby("row", sort=False)["col"].apply(
        lambda s: s.to_numpy(np.int32))
    return grouped.to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/processed_hgn_none", type=Path)
    parser.add_argument("--out-name", default="onboarding.json")
    parser.add_argument("--picks", type=int, nargs="+",
                        default=[1, 2, 3, 5, 10, 20])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--neighbours", type=int, default=500)
    parser.add_argument("--shrink", type=float, default=20.0)
    parser.add_argument("--media", default="data/processed_gr/media.parquet",
                        type=Path)
    parser.add_argument("--min-df", type=int, default=20)
    parser.add_argument("--mechanism", action="store_true",
                        help="thêm các ô tách nguyên nhân của đường cong")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    started = time.monotonic()
    data = load(args.dir)
    train, val, test = data["train"], data["val"], data["test"]
    user_index, item_index = data["user_index"], data["item_index"]
    reverse = {v: k for k, v in item_index.items()}
    matrix = user_item_matrix(train, user_index, item_index)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    order = np.argsort(-counts)
    percentile = np.empty(len(counts))
    percentile[order] = 1.0 - np.arange(len(counts)) / max(len(counts) - 1, 1)
    extra = {
        "popularity": {reverse[i]: int(counts[i]) for i in range(len(counts))},
        "n_users": matrix.shape[0],
        "pop_percentile": {reverse[i]: float(percentile[i])
                           for i in range(len(counts))},
        "head_items": {reverse[int(i)]
                       for i in order[:max(1, len(counts) // 100)]},
    }
    train_by_user = train.groupby("user")["media_id"].apply(list).to_dict()
    excluded_all = excluded_candidates(train, val, user_index, item_index,
                                       data["series_of"])
    test_frame, _ = reachable_test(test, excluded_all, item_index)
    history = recent_items(train, user_index, item_index)
    print(f"{args.dir}: {len(item_index):,} mục | {len(user_index):,} người | "
          f"test {len(test_frame):,} dòng  ({time.monotonic() - started:.0f}s)",
          flush=True)

    similarity = top_k_neighbours(matrix, k=args.neighbours,
                                  shrink=args.shrink).tocsr()
    print(f"  láng giềng mục: {similarity.nnz:,} cặp "
          f"({time.monotonic() - started:.0f}s)", flush=True)

    media = pd.read_parquet(args.media)
    item_matrix, _ = build_item_features(media, item_index,
                                         merged_of=data["merged_of"],
                                         min_df=args.min_df)
    item_matrix = sparse.csr_matrix(item_matrix, dtype=np.float32)
    norms = np.sqrt(item_matrix.multiply(item_matrix).sum(axis=1))
    norms = np.asarray(norms).ravel()
    norms[norms == 0] = 1.0
    item_matrix = sparse.diags(1.0 / norms) @ item_matrix
    print(f"  nội dung: {item_matrix.shape} "
          f"({time.monotonic() - started:.0f}s)", flush=True)

    rng = np.random.default_rng(args.seed)

    def picks_of(row, n_picks, mode="recent"):
        """`mode` tách hai cách giải thích đường cong khỏi nhau.

        Đường cong đo được giảm đơn điệu: 2 mục gần nhất cho NDCG 0,1240 còn
        lịch sử đầy đủ chỉ 0,0989. Hai cách giải thích:

        * **gần đây** — mục mới nhất gần test nhất về cả thời gian lẫn gu;
        * **số lượng** — cộng tương đồng không chuẩn hoá qua N mục là một bộ
          khuếch đại độ phổ biến, vì truyện phổ biến là láng giềng của mọi
          người.

        Chúng cho cùng một đường cong nhưng khác nhau ở phép đo này: nếu chỉ
        số lượng quyết định thì `recent`, `random` và `oldest` cho cùng con số.
        """
        seen = history.get(row)
        if seen is None or not len(seen):
            return None
        if n_picks is None:
            return seen
        if mode == "oldest":
            return seen[-n_picks:]
        if mode == "random" and len(seen) > n_picks:
            return seen[rng.choice(len(seen), n_picks, replace=False)]
        return seen[:n_picks]

    def knn_scores(n_picks, mode="recent", normalize=False):
        def score(row):
            picked = picks_of(row, n_picks, mode)
            if picked is None:
                return None
            block = similarity[picked]
            out = np.asarray(block.sum(axis=0)).ravel()
            return out / len(picked) if normalize else out
        return score

    def content_scores(n_picks, mode="recent"):
        def score(row):
            picked = picks_of(row, n_picks, mode)
            if picked is None:
                return None
            profile = np.asarray(item_matrix[picked].mean(axis=0)).ravel()
            return item_matrix @ profile
        return score

    popular = counts.astype(np.float32)

    results = {}

    def run(label, score_fn):
        got = metrics.evaluate(
            ranker(score_fn, user_index, reverse, excluded_all, args.k),
            test_frame, train_by_user, k=args.k, max_users=args.max_users,
            catalog_size=len(item_index), seed=args.seed, **extra)
        results[label] = got
        print(f"  {label:<34} NDCG {got['ndcg']:.4f} | Recall "
              f"{got['recall']:.4f} | phủ {got['coverage']:.4f} | "
              f"long-tail {got['tail']:.4f}  "
              f"({time.monotonic() - started:.0f}s)", flush=True)
        path = args.dir / args.out_name
        path.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    run("phổ biến (0 lựa chọn)", lambda row: popular)
    for n_picks in args.picks:
        run(f"Item-KNN sau {n_picks} lựa chọn", knn_scores(n_picks))
    run("Item-KNN lịch sử đầy đủ", knn_scores(None))
    for n_picks in args.picks:
        run(f"Nội dung sau {n_picks} lựa chọn", content_scores(n_picks))
    run("Nội dung lịch sử đầy đủ", content_scores(None))

    if args.mechanism:
        # Phép đo tách nguyên nhân. `normalize` chia cho số mục: nó KHÔNG đổi
        # thứ hạng trong một lần chấm (chia cho hằng số), nên nếu con số đổi
        # thì có lỗi ở đâu đó — đây là một ô kiểm tra chính mình.
        for n_picks in (2, 5):
            for mode in ("recent", "random", "oldest"):
                run(f"cơ chế: Item-KNN {n_picks} mục {mode}",
                    knn_scores(n_picks, mode))
        run("cơ chế: Item-KNN 2 mục recent, chia N",
            knn_scores(2, "recent", normalize=True))
        run("cơ chế: Nội dung 2 mục random", content_scores(2, "random"))
        run("cơ chế: Nội dung 2 mục oldest", content_scores(2, "oldest"))

    print(f"\nxong ({(time.monotonic() - started) / 60:.1f} phút) -> "
          f"{args.dir / args.out_name}")


if __name__ == "__main__":
    main()
