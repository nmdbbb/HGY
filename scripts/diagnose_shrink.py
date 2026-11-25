"""Vì sao hạ trọng số cặp mỏng (significance weighting) làm XẤU đi trên bộ này?

    python scripts/diagnose_shrink.py --dir data/processed_hgn_none

Lời giải thích tôi viết lần đầu ở `docs/protocol-arm.md` mục 12 là: sau lọc
k-core, số người đọc chung của các cặp đã đủ lớn nên hệ số `n/(n + 20)` chỉ còn
là một độ chệch gần đều. **Phép đo phân bố `n_chung` bác bỏ chính lời đó**:
trên 512 mục mẫu, 98,8% cặp đồng xuất hiện có `n_chung < 20`, trung vị
`n_chung = 1`, nên hệ số trung vị là `1/21 = 0,048`. Nó không gần đều; nó dập
gần như mọi cặp xuống 1/20.

Giả thuyết thay thế, ghi TRƯỚC khi chạy script này: hạ trọng số không phải một
độ chệch mà là một **bộ lọc theo độ phổ biến** đặt lên tập láng giềng. Vì
`top_k` xếp theo tương đồng *đã hạ*, cặp nào có `n_chung` lớn mới trụ lại, mà
`n_chung` lớn gần như đồng nghĩa với "cả hai mục đều nhiều người đọc". Mục 10 đã
đo rằng trên bộ này độ chính xác đến từ việc **đi ra khỏi đầu danh mục**, nên một
bộ lọc đẩy láng giềng về phía đầu bảng phải làm giảm NDCG.

Điều kiện bác bỏ giả thuyết này: `shrink = 20` cho **ARP thấp hơn hoặc bằng** và
**tỉ lệ long-tail cao hơn hoặc bằng** `shrink = 0`. Lúc đó chỗ mất không phải do
lệch về đầu bảng và phải tìm nguyên nhân khác.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from truyen_reco.data.protocol_peruser import (  # noqa: E402
    excluded_candidates, reachable_test)
from truyen_reco.evaluation import metrics  # noqa: E402
from truyen_reco.models.knn import (  # noqa: E402
    top_k_neighbours, user_item_matrix)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_protocol_baselines import load, ranker  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/processed_hgn_none", type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--neighbours", type=int, default=500)
    parser.add_argument("--shrink", nargs="+", type=float, default=[0.0, 20.0])
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = load(args.dir)
    train, val = data["train"], data["val"]
    user_index, item_index = data["user_index"], data["item_index"]
    reverse = {v: k for k, v in item_index.items()}
    matrix = user_item_matrix(train, user_index, item_index)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    order = np.argsort(-counts)
    percentile = np.empty(len(counts))
    percentile[order] = 1.0 - np.arange(len(counts)) / max(len(counts) - 1, 1)
    popularity = {reverse[i]: int(counts[i]) for i in range(len(counts))}
    pop_percentile = {reverse[i]: float(percentile[i]) for i in range(len(counts))}
    head_items = {reverse[int(i)] for i in order[:max(1, len(counts) // 100)]}

    train_by_user = train.groupby("user")["media_id"].apply(list).to_dict()
    excluded = excluded_candidates(train, val.iloc[0:0], user_index, item_index,
                                   data["series_of"])
    frame, _ = reachable_test(val, excluded, item_index)

    print(f"{args.dir}  k={args.neighbours}  {len(item_index):,} mục")
    out = {}
    for shrink in args.shrink:
        similarity = top_k_neighbours(matrix, k=args.neighbours, shrink=shrink)
        # Bao nhiêu láng giềng thật sự còn lại, và chúng nằm ở đâu trên bảng
        # phổ biến: đây là chỗ giả thuyết "bộ lọc theo độ phổ biến" sống hay chết.
        per_item = np.diff(similarity.indptr)
        neighbour_pop = percentile[similarity.indices] if similarity.nnz else np.array([0.0])

        def score(row, s=similarity):
            return (None if matrix[row].nnz == 0
                    else np.asarray((matrix[row] @ s).todense()).ravel())

        got = metrics.evaluate(
            ranker(score, user_index, reverse, excluded, args.k),
            frame, train_by_user, k=args.k, max_users=args.max_users,
            catalog_size=len(item_index), seed=args.seed,
            popularity=popularity, n_users=matrix.shape[0],
            pop_percentile=pop_percentile, head_items=head_items)
        out[shrink] = {
            "ndcg": got["ndcg"], "coverage": got["coverage"],
            "arp": got.get("arp"), "long_tail": got.get("tail"),
            "novelty": got.get("novelty"),
            "neighbours_median": float(np.median(per_item)),
            "neighbours_mean": float(per_item.mean()),
            "neighbour_pop_median": float(np.median(neighbour_pop)),
        }
        row = out[shrink]
        print(f"  shrink={shrink:<5} NDCG {row['ndcg']:.4f} | phủ "
              f"{row['coverage']:.4f} | ARP {row['arp']:.4f} | long-tail "
              f"{row['long_tail']:.4f} | mới lạ {row['novelty']:.2f}")
        print(f"              láng giềng/mục trung vị {row['neighbours_median']:.0f}"
              f" (trung bình {row['neighbours_mean']:.1f}) | phân vị phổ biến"
              f" của láng giềng, trung vị {row['neighbour_pop_median']:.4f}")

    path = args.dir / "shrink_diagnosis.json"
    path.write_text(json.dumps({str(k): v for k, v in out.items()}, indent=2),
                    encoding="utf-8")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
