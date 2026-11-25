"""Chọn siêu tham số trên VALIDATION, rồi mới đọc test đúng một lần.

    python scripts/tune_protocol.py --dir data/processed_hgn_none

Vì sao phải có script này, và nó sửa lỗi của chính tôi: phép quét `L` của
Session-KNN ở mục 11 `docs/protocol-arm.md` được chạy **trên tập test**, rồi tôi
báo con số tốt nhất (0,1276 ở L=10, giảm 0,7). Dù kết luận định tính không đổi,
con số đó là **số đã được chọn bằng test** — đúng loại lỗi mà đồ án này bắt ở
chỗ khác ("chọn quy mô sau khi nhìn kết quả"). Mục 7.1 của đề xuất cũng ghi rõ:
mỗi mô hình phải được tìm siêu tham số trên *cùng* tập validation với ngân sách
tương đương.

Quy trình ở đây, và ba chỗ phải cẩn thận:

1. **Chọn trên val, đọc test một lần.** Mọi cấu hình được chấm trên val; chỉ
   cấu hình thắng được chấm trên test. Script in cả hai để thấy khoảng cách
   val–test, nhưng việc chọn không bao giờ nhìn test.
2. **Rổ ứng viên khi chấm val phải loại theo TRAIN thôi.** Lúc chấm val, các mục
   trong val chính là đáp án, nên nếu loại chúng (như khi chấm test) thì đáp án
   rỗng. Đây là một lỗi im lặng: nó không báo gì, chỉ cho NDCG bằng 0 đều ở mọi
   cấu hình và mọi cấu hình "ngang nhau".
3. **Hồ sơ mô hình vẫn chỉ dựng từ train**, kể cả khi chấm test. Dùng val để
   dựng hồ sơ thì cấu hình thắng trên val sẽ được lợi hai lần.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from truyen_reco.models.knn import (  # noqa: E402
    top_k_neighbours, user_item_matrix)
from run_protocol_baselines import load, ranker  # noqa: E402


def ndcg_on(frame, excluded, score_fn, user_index, reverse, train_by_user,
            k, max_users, seed, catalog_size):
    recommend = ranker(score_fn, user_index, reverse, excluded, k)
    out = metrics.evaluate(recommend, frame, train_by_user, k=k,
                           max_users=max_users, catalog_size=catalog_size,
                           seed=seed)
    return out["ndcg"], out["users_evaluated"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/processed_hgn_none", type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--neighbours", nargs="+", type=int,
                        default=[50, 200, 500])
    parser.add_argument("--shrink", nargs="+", type=float, default=[0.0, 20.0])
    parser.add_argument("--lengths", nargs="+", type=int,
                        default=[1, 2, 3, 5, 10, 20])
    parser.add_argument("--decays", nargs="+", type=float,
                        default=[1.0, 0.9, 0.7, 0.5])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = load(args.dir)
    train, val, test = data["train"], data["val"], data["test"]
    user_index, item_index = data["user_index"], data["item_index"]
    reverse = {v: k for k, v in item_index.items()}
    matrix = user_item_matrix(train, user_index, item_index)
    train_by_user = train.groupby("user")["media_id"].apply(list).to_dict()
    empty = val.iloc[0:0]

    # Hai tập loại trừ khác nhau cho hai lần chấm (xem chỗ cẩn thận số 2).
    excluded_train = excluded_candidates(train, empty, user_index, item_index,
                                         data["series_of"])
    excluded_all = excluded_candidates(train, val, user_index, item_index,
                                       data["series_of"])
    val_frame, val_report = reachable_test(val, excluded_train, item_index)
    test_frame, test_report = reachable_test(test, excluded_all, item_index)
    print(f"{args.dir}")
    print(f"  val  : {val_report['test_rows_kept']:,}/"
          f"{val_report['test_rows_in']:,} dòng với tới được")
    print(f"  test : {test_report['test_rows_kept']:,}/"
          f"{test_report['test_rows_in']:,} dòng với tới được\n")

    ordered = train.sort_values(["user", "created_at"], kind="mergesort")
    results = {}

    def session_scorer(similarity, length, decay):
        tail_rows = ordered.groupby("user", sort=False).tail(length)
        session = {}
        for user, group in tail_rows.groupby("user", sort=False)["media_id"]:
            row = user_index.get(user)
            if row is None:
                continue
            positions = [item_index[m] for m in group if m in item_index]
            if positions:
                session[row] = np.asarray(positions, dtype=np.int32)

        def score(row):
            positions = session.get(row)
            if positions is None:
                return None
            weights = np.power(decay, np.arange(len(positions) - 1, -1, -1))
            profile = sparse.csr_matrix(
                (weights.astype(np.float32),
                 (np.zeros(len(positions), dtype=np.int32), positions)),
                shape=(1, matrix.shape[1]))
            return np.asarray((profile @ similarity).todense()).ravel()
        return score

    print("=" * 78)
    print("Item-KNN: chọn (k láng giềng, hạ trọng số) trên VAL")
    print("=" * 78)
    best = None
    similarities = {}
    for neighbours in args.neighbours:
        for shrink in args.shrink:
            started = time.monotonic()
            similarity = top_k_neighbours(matrix, k=neighbours, shrink=shrink)
            similarities[(neighbours, shrink)] = similarity
            score = lambda row, s=similarity: (
                None if matrix[row].nnz == 0
                else np.asarray((matrix[row] @ s).todense()).ravel())
            value, n = ndcg_on(val_frame, excluded_train, score, user_index,
                               reverse, train_by_user, args.k, args.max_users,
                               args.seed, len(item_index))
            print(f"  k={neighbours:<4} shrink={shrink:<5} val NDCG {value:.4f}"
                  f"  ({n} người, {time.monotonic() - started:.0f}s)", flush=True)
            if best is None or value > best[0]:
                best = (value, neighbours, shrink)
    results["itemknn"] = {"val_ndcg": best[0], "neighbours": best[1],
                          "shrink": best[2]}
    print(f"  -> chọn k={best[1]}, shrink={best[2]}")

    item_similarity = similarities[(best[1], best[2])]
    value, _ = ndcg_on(test_frame, excluded_all,
                       lambda row: (None if matrix[row].nnz == 0 else
                                    np.asarray((matrix[row] @ item_similarity)
                                               .todense()).ravel()),
                       user_index, reverse, train_by_user, args.k,
                       args.max_users, args.seed, len(item_index))
    results["itemknn"]["test_ndcg"] = value
    print(f"  test NDCG của cấu hình đã chọn: {value:.4f}\n")

    print("=" * 78)
    print("Session-KNN: chọn (L, hệ số giảm) trên VAL — dùng ma trận đã chọn")
    print("=" * 78)
    best = None
    for length in args.lengths:
        for decay in args.decays:
            value, n = ndcg_on(val_frame, excluded_train,
                               session_scorer(item_similarity, length, decay),
                               user_index, reverse, train_by_user, args.k,
                               args.max_users, args.seed, len(item_index))
            print(f"  L={length:<4} giảm={decay:<4} val NDCG {value:.4f}",
                  flush=True)
            if best is None or value > best[0]:
                best = (value, length, decay)
    results["sessionknn"] = {"val_ndcg": best[0], "length": best[1],
                             "decay": best[2]}
    print(f"  -> chọn L={best[1]}, giảm={best[2]}")
    value, _ = ndcg_on(test_frame, excluded_all,
                       session_scorer(item_similarity, best[1], best[2]),
                       user_index, reverse, train_by_user, args.k,
                       args.max_users, args.seed, len(item_index))
    results["sessionknn"]["test_ndcg"] = value
    print(f"  test NDCG của cấu hình đã chọn: {value:.4f}")

    out = args.dir / "tuning.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n-> {out}")
    print("Đọc đúng: con số test ở trên là của cấu hình do VAL chọn, đọc đúng")
    print("một lần. Nó không phải số lớn nhất trên test, và không được thay")
    print("bằng số lớn nhất trên test.")


if __name__ == "__main__":
    main()
