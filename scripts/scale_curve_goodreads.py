"""Lấy mẫu người đọc bao nhiêu thì ra quy mô nào — đo, không ngoại suy.

`probe_goodreads_shape.py` từng đề xuất `--sample-users 42562` để khớp số tương
tác của AniList, bằng cách chia số tương tác cho độ dài lịch sử trung bình. Phép
chia đó **sai về bản chất**, vì nó giả định lọc lõi không đổi gì khi mẫu mỏng
đi. Lần chạy thử 3.000/342.378 người đọc cho thấy ngược lại:

    58.865 tương tác thô  ->  18.765 dương sau lọc lõi  (32%)
    3.000 người đọc       ->  578 người còn trong train (19%)
    62.944 tác phẩm       ->  1.598 tác phẩm còn lại    (2,5%)

Nguyên nhân là lọc lõi `min_item = 5`: mỗi tác phẩm cần 5 người đọc mới trụ
được. Cắt mỏng số người thì gần hết danh mục rụng, rồi người đọc mất mục nên
rụng tiếp theo `min_user = 5`. Sụt giảm là **cấp số nhân**, không phải tuyến
tính — nên con số để khớp quy mô phải đo bằng cách chạy thật lọc lõi ở vài mức.

    python scripts/scale_curve_goodreads.py --sizes 20000 40000 80000 160000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from truyen_reco.data import preprocess  # noqa: E402


def shape_at(interactions, n_users, seed, min_user, min_item):
    """Hình dạng sau lọc lõi + chia theo thời gian khi chỉ giữ n_users người."""
    frame = interactions
    pool = frame["user"].unique()
    if n_users is not None:
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(pool, min(n_users, len(pool)),
                              replace=False).tolist())
        frame = frame[frame["user"].isin(keep)]

    prepared = preprocess.run(frame, min_user=min_user, min_item=min_item)
    train, test = prepared["train"], prepared["test"]
    lengths = train.groupby("user")["media_id"].size()
    slices = prepared["slices"]
    sparse = slices.get("sparse_user")
    all_slice = slices.get("all")
    return {
        "users_sampled": len(pool) if n_users is None else min(n_users, len(pool)),
        "rows_raw": int(len(frame)),
        "core_rows": int(len(prepared["core"])),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_users": int(train["user"].nunique()),
        "train_items": int(train["media_id"].nunique()),
        "median_history": float(lengths.median()),
        "pct_under_10": round(100.0 * float((lengths < 10).mean()), 1),
        # Tỉ lệ người đọc thuộc lát thưa: đây là tính chất mà P5 dựa vào, nên
        # nếu lấy mẫu làm méo nó thì không được lấy mẫu.
        "sparse_user_pct": (None if not (sparse is not None and all_slice is not None
                                         and all_slice["user"].nunique())
                            else round(100.0 * sparse["user"].nunique()
                                       / all_slice["user"].nunique(), 1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gr-dir", default="data/processed_gr", type=Path)
    parser.add_argument("--anilist-dir", default="data/processed", type=Path)
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[20_000, 40_000, 80_000, 160_000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-user", type=int, default=5)
    parser.add_argument("--min-item", type=int, default=5)
    args = parser.parse_args()

    interactions = pd.read_parquet(args.gr_dir / "interactions.parquet")
    print(f"đọc {len(interactions):,} tương tác thô "
          f"| {interactions['user'].nunique():,} người đọc\n", flush=True)

    target = None
    if (args.anilist_dir / "train.parquet").exists():
        a_train = pd.read_parquet(args.anilist_dir / "train.parquet",
                                  columns=["user", "media_id"])
        a_len = a_train.groupby("user")["media_id"].size()
        target = {
            "train_rows": int(len(a_train)),
            "train_users": int(a_train["user"].nunique()),
            "train_items": int(a_train["media_id"].nunique()),
            "median_history": float(a_len.median()),
            "pct_under_10": round(100.0 * float((a_len < 10).mean()), 1),
        }
        print(f"mốc AniList: train {target['train_rows']:,} tương tác"
              f" | {target['train_users']:,} người | {target['train_items']:,} truyện"
              f" | lịch sử tv {target['median_history']:.0f}"
              f" | <10 mục {target['pct_under_10']}%\n")

    header = (f"{'mẫu người':>11}{'train':>12}{'người':>9}{'truyện':>9}"
              f"{'lịch sử tv':>12}{'<10 mục':>9}{'lát thưa':>10}")
    print(header)
    print("-" * len(header))

    rows = []
    for size in list(args.sizes) + [None]:
        t0 = time.monotonic()
        row = shape_at(interactions, size, args.seed, args.min_user, args.min_item)
        row["seconds"] = round(time.monotonic() - t0, 1)
        rows.append(row)
        label = "tất cả" if size is None else f"{size:,}"
        print(f"{label:>11}{row['train_rows']:>12,}{row['train_users']:>9,}"
              f"{row['train_items']:>9,}{row['median_history']:>12.0f}"
              f"{row['pct_under_10']:>8.1f}%"
              f"{(row['sparse_user_pct'] or 0):>9.1f}%", flush=True)

    out = {"target_anilist": target, "curve": rows,
           "seed": args.seed, "min_user": args.min_user,
           "min_item": args.min_item}
    path = args.gr_dir / "scale_curve.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {path}")

    if target:
        # Mức nào cho số tương tác gần AniList nhất — và quan trọng hơn, mức đó
        # có còn giữ được hình dạng lịch sử (trung vị, % người thưa) hay không.
        best = min(rows, key=lambda r: abs(r["train_rows"] - target["train_rows"]))
        print(f"\ngần số tương tác AniList nhất: mẫu "
              f"{best['users_sampled']:,} người -> {best['train_rows']:,} tương tác "
              f"({best['train_rows'] / target['train_rows']:.2f}x)")
        print(f"  ở mức đó: lịch sử tv {best['median_history']:.0f}"
              f" | <10 mục {best['pct_under_10']}%"
              f" | truyện {best['train_items']:,}")
        print("  so với quy mô đầy đủ: nếu hai số sau lệch nhiều thì lấy mẫu đã "
              "đổi chính tính chất đang muốn so, và khớp quy mô là lựa chọn sai.")


if __name__ == "__main__":
    main()
