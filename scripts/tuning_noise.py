"""Đo ĐỘ PHÂN GIẢI của lưới tinh chỉnh: chênh lệch nhỏ cỡ nào thì vẫn là nhiễu?

    python scripts/tuning_noise.py --dir data/processed_hgn_volume

Vì sao cần script này. `tune_protocol.py` chọn cấu hình bằng argmax trên val, và
trên bản gộp tập nó chọn L=20/giảm 0,5 với val 0,2661 — hơn L=10/giảm 0,5 đúng
**0,0001** và hơn L=5/giảm 0,5 đúng **0,0001**. Mục 7.1 của đề xuất chỉ nói
"tinh chỉnh trên validation với ngân sách tương đương"; nó không nói phải làm gì
khi mặt val **bẹt** và argmax rơi vào một trong nhiều điểm không phân biệt được.
Lấy argmax ở đó là chọn theo nhiễu, rồi báo "L=20 là tham số tốt nhất" là một
phát biểu không có dữ liệu đỡ.

Phép đo: val được chấm trên một **mẫu người đọc** (`--max-users`), nên đổi mẫu
thì số đổi. Chạy cùng một lưới ở nhiều `seed` mẫu khác nhau cho hai thứ:

1. độ lệch của *từng* cấu hình giữa các mẫu — tức sai số của một con số val;
2. độ lệch của *hiệu số* giữa hai cấu hình, **ghép cặp theo mẫu** — tức thứ duy
   nhất nói được "cấu hình A hơn B" có nghĩa hay không. Ghép cặp vì hai cấu hình
   dùng *cùng* mẫu người đọc, nên phần dao động do mẫu triệt tiêu trong hiệu số.

Script này **không bao giờ đọc tập test**. Nó chỉ nói về độ phân giải của phép
chọn, và phép chọn chỉ được nhìn val.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

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
    parser.add_argument("--dir", default="data/processed_hgn_volume", type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--neighbours", type=int, default=500)
    parser.add_argument("--shrink", type=float, default=0.0)
    parser.add_argument("--lengths", nargs="+", type=int, default=[2, 5, 10, 20])
    parser.add_argument("--decays", nargs="+", type=float, default=[0.5, 0.7])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    data = load(args.dir)
    train, val = data["train"], data["val"]
    user_index, item_index = data["user_index"], data["item_index"]
    reverse = {v: k for k, v in item_index.items()}
    matrix = user_item_matrix(train, user_index, item_index)
    train_by_user = train.groupby("user")["media_id"].apply(list).to_dict()
    # Loại theo TRAIN thôi: lúc chấm val thì val chính là đáp án.
    excluded = excluded_candidates(train, val.iloc[0:0], user_index, item_index,
                                  data["series_of"])
    frame, report = reachable_test(val, excluded, item_index)
    print(f"{args.dir}  val {report['test_rows_kept']:,}/"
          f"{report['test_rows_in']:,} dòng với tới được")
    similarity = top_k_neighbours(matrix, k=args.neighbours, shrink=args.shrink)
    ordered = train.sort_values(["user", "created_at"], kind="mergesort")

    def scorer(length, decay):
        tail = ordered.groupby("user", sort=False).tail(length)
        session = {}
        for user, group in tail.groupby("user", sort=False)["media_id"]:
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

    grid = list(itertools.product(args.lengths, args.decays))
    table = {}
    for length, decay in grid:
        score = scorer(length, decay)
        values = []
        for seed in args.seeds:
            out = metrics.evaluate(
                ranker(score, user_index, reverse, excluded, args.k),
                frame, train_by_user, k=args.k, max_users=args.max_users,
                catalog_size=len(item_index), seed=seed)
            values.append(out["ndcg"])
        table[(length, decay)] = values
        print(f"  L={length:<3} giảm={decay:<4} "
              f"val {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}  "
              f"[{min(values):.4f}, {max(values):.4f}]", flush=True)

    print("\nHiệu số GHÉP CẶP theo mẫu (cái duy nhất nói được A hơn B):")
    best = max(table, key=lambda key: float(np.mean(table[key])))
    print(f"  mốc = L={best[0]}, giảm={best[1]} "
          f"(trung bình cao nhất {np.mean(table[best]):.4f})")
    for key in grid:
        if key == best:
            continue
        pairs = np.array(table[best]) - np.array(table[key])
        gap, spread = float(pairs.mean()), float(pairs.std(ddof=1))
        verdict = "không phân biệt được"
        if spread > 0:
            t = gap / (spread / np.sqrt(len(pairs)))
            limit = {2: 4.30, 3: 3.18, 4: 2.78}.get(len(pairs) - 1, 2.78)
            if abs(t) > limit:
                verdict = "KHÁC BIỆT THẬT"
            print(f"  so L={key[0]:<3} giảm={key[1]:<4} {gap:+.4f} ± {spread:.4f}"
                  f"  t={t:+6.2f}  {verdict}")
        else:
            print(f"  so L={key[0]:<3} giảm={key[1]:<4} {gap:+.4f} ± 0  y hệt")

    out = args.dir / "tuning_noise.json"
    out.write_text(json.dumps(
        {f"L{length}_d{decay}": vals for (length, decay), vals in table.items()},
        indent=2), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
