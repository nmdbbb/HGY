"""Vì sao 31% loại truyện một người đang đọc không xuất hiện trong top-10.

    python scripts/diagnose_country.py --users 400

Bậc 4 đo được `feed` chỉ phủ 0,685 số nước trong lịch sử mỗi người. Trước khi
thêm ràng buộc đa dạng — việc mà tiêu chí đã chốt sẵn làm fallback — phải biết
thiếu hụt đó từ đâu. Ba giả thuyết loại trừ nhau, mỗi cái dẫn tới một việc
khác:

  A. Người đọc lệch sẵn. Ai đọc 95% manga mà nhận 10 manga là *đúng*; "phủ
     nước" thấp chỉ phản ánh chính phân phối đọc. Việc phải làm: đổi phép đo,
     không đổi mô hình.
  B. Mô hình lệch hơn người đọc. Tỉ lệ nước trong top-10 lệch khỏi tỉ lệ nước
     trong lịch sử theo một hướng cố định. Việc phải làm: sửa chấm điểm.
  C. Truyện của nước thiểu số mà người đó đọc nằm ở đuôi dài. Lúc đó không phải
     lỗi đa dạng mà là lỗi đuôi dài, và ràng buộc theo nước sẽ che mất nguyên
     nhân thật.

Phép đo là hiệu chỉnh (calibration) theo nghĩa Steck 2018: so phân phối nước
trong gợi ý với phân phối nước trong lịch sử của *chính người đó*, chứ không so
với một quy tắc "ít nhất 2 nước" áp cho mọi người.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from truyen_reco.api.service import RecommenderService  # noqa: E402

COUNTRIES = ("JP", "KR", "CN")


def shares(counter):
    total = sum(counter.get(c, 0) for c in COUNTRIES)
    if not total:
        return None
    return {c: counter.get(c, 0) / total for c in COUNTRIES}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/processed", type=Path)
    parser.add_argument("--users", type=int, default=400)
    parser.add_argument("--engine", default="feed")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    service = RecommenderService(args.out_dir)
    test = pd.read_parquet(args.out_dir / "test.parquet")
    popularity = service.model.popularity.scores
    rank_of = np.empty(len(popularity), dtype=np.int64)
    rank_of[np.argsort(-popularity)] = np.arange(len(popularity))

    test_by_user = test.groupby("user")["media_id"].apply(list).to_dict()
    rng = np.random.default_rng(args.seed)
    pool = [u for u in service.history
            if len(service.history[u]) >= 10 and u in test_by_user]
    users = rng.choice(pool, size=min(args.users, len(pool)), replace=False)

    rows = []
    for user in users:
        history = service.history[user]
        read = Counter(service.countries.get(service.reverse_item[p])
                       for p in history)
        read_share = shares(read)
        if read_share is None:
            continue

        result = service.recommend(user_id=str(user), top_k=10,
                                   engine=args.engine)
        got = Counter(card.get("country") for card in result["items"])
        got_share = shares(got) or {c: 0.0 for c in COUNTRIES}

        truth = Counter(service.countries.get(m) for m in test_by_user[user])
        truth_share = shares(truth)

        # Truyện của nước thiểu số mà người này thực sự đọc trong test nằm ở
        # hạng phổ biến nào — kiểm giả thuyết C
        minority = min((c for c in COUNTRIES if read_share[c] > 0),
                       key=lambda c: read_share[c])
        tail_ranks = [int(rank_of[service.item_index[m]])
                      for m in test_by_user[user]
                      if m in service.item_index
                      and service.countries.get(m) == minority]

        rows.append({
            "user": str(user),
            "n_countries_read": sum(1 for c in COUNTRIES if read_share[c] > 0),
            "read": read_share, "got": got_share, "truth": truth_share,
            "minority": minority,
            "minority_read_share": read_share[minority],
            "minority_got_share": got_share.get(minority, 0.0),
            "minority_tail_median": (float(np.median(tail_ranks))
                                     if tail_ranks else None),
        })

    multi = [r for r in rows if r["n_countries_read"] >= 2]
    print(f"{len(rows)} người đọc | {len(multi)} người đọc ≥2 nước "
          f"({len(multi) / max(len(rows), 1) * 100:.1f}%)")

    print("\n" + "=" * 84)
    print("A. Phân phối nước: lịch sử vs test thật vs top-10")
    print("=" * 84)
    print(f"{'nước':<8}{'lịch sử':>12}{'test thật':>12}{'top-10':>12}"
          f"{'lệch top-10 − lịch sử':>24}")
    for country in COUNTRIES:
        in_read = np.mean([r["read"][country] for r in rows])
        in_truth = np.mean([r["truth"][country] for r in rows if r["truth"]])
        in_got = np.mean([r["got"][country] for r in rows])
        print(f"{country:<8}{in_read:>12.3f}{in_truth:>12.3f}{in_got:>12.3f}"
              f"{in_got - in_read:>+24.3f}")

    print("\n" + "=" * 84)
    print("B. Nước thiểu số của mỗi người: đọc bao nhiêu, nhận lại bao nhiêu")
    print("=" * 84)
    bands = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 1.01)]
    print(f"{'tỉ lệ đọc nước thiểu số':<26}{'số người':>10}"
          f"{'tỉ lệ trong top-10':>20}{'có mặt ≥1':>12}{'hạng đuôi':>12}")
    for low, high in bands:
        block = [r for r in multi if low <= r["minority_read_share"] < high]
        if not block:
            continue
        present = np.mean([r["minority_got_share"] > 0 for r in block])
        tails = [r["minority_tail_median"] for r in block
                 if r["minority_tail_median"] is not None]
        print(f"{f'{low:.0%} – {high:.0%}':<26}{len(block):>10}"
              f"{np.mean([r['minority_got_share'] for r in block]):>20.3f}"
              f"{present:>12.2f}"
              f"{(f'{np.median(tails):,.0f}' if tails else '—'):>12}")

    print("\n" + "=" * 84)
    print("C. Hiệu chỉnh: khoảng cách giữa phân phối gợi ý và phân phối đọc")
    print("=" * 84)

    def l1(a, b):
        return sum(abs(a[c] - b[c]) for c in COUNTRIES) / 2

    model_gap = np.mean([l1(r["got"], r["read"]) for r in rows])
    truth_gap = np.mean([l1(r["truth"], r["read"]) for r in rows if r["truth"]])
    print(f"  top-10 vs lịch sử        {model_gap:.3f}")
    print(f"  test THẬT vs lịch sử     {truth_gap:.3f}   <- mức sàn không thể "
          "vượt: chính người đọc cũng không đọc đúng tỉ lệ cũ")
    verdict = ("B — mô hình lệch hơn người đọc" if model_gap > truth_gap + 0.05
               else "A — thiếu hụt nằm trong mức người đọc tự lệch")
    print(f"\n  kết luận: {verdict}")

    path = args.out_dir / "country_diagnosis.json"
    path.write_text(json.dumps(
        {"rows": rows, "model_gap": float(model_gap),
         "truth_gap": float(truth_gap)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
