"""Đọc + lọc lõi + chia tập, KHÔNG dựng đặc trưng và KHÔNG huấn luyện.

Chạy cái này trước `train.py` để biết bộ Goodreads có hình dạng gì, rồi mới
quyết định có cần `--sample-users` hay không. Quyết định ngược lại — huấn luyện
trước rồi mới xem dữ liệu — là cách để lỡ mất một tiếng rưỡi vì OOM, hoặc tệ
hơn, so hai nhánh ở hai quy mô khác nhau mà không biết.

    python scripts/probe_goodreads_shape.py
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

from truyen_reco.data import ingest_goodreads, preprocess  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/external/goodreads", type=Path)
    parser.add_argument("--out-dir", default="data/processed_gr", type=Path)
    parser.add_argument("--min-user", type=int, default=5)
    parser.add_argument("--min-item", type=int, default=5)
    parser.add_argument("--anilist-dir", default="data/processed", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    print("đọc dữ liệu thô...", flush=True)
    interactions, media = ingest_goodreads.run(args.raw_dir, args.out_dir)
    report = json.loads(
        (args.out_dir / "ingest_report.json").read_text(encoding="utf-8"))
    print(f"  {time.monotonic() - t0:.0f}s")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    print("\nlọc lõi và chia tập...", flush=True)
    prepared = preprocess.run(interactions, media=media,
                              min_user=args.min_user, min_item=args.min_item)
    train, test = prepared["train"], prepared["test"]
    core = prepared["core"]

    print(f"\n  trạng thái: {interactions['status'].value_counts().to_dict()}")
    print(f"  tương tác dương sau lọc lõi: {len(core):,}"
          f"  ({len(core) / max(len(interactions), 1) * 100:.1f}% số dòng thô)")
    print(f"  người đọc còn lại: {core['user'].nunique():,}"
          f" | tác phẩm còn lại: {core['media_id'].nunique():,}")
    print(f"  mốc cắt: {prepared['cutoff']}")
    print(f"  train {len(train):,} | test {len(test):,}")
    for name, frame in prepared["slices"].items():
        print(f"    lát {name:<12} {len(frame):>9,} tương tác"
              f" | {frame['user'].nunique():>7,} người")

    lengths = train.groupby("user")["media_id"].size()
    print(f"\n  độ dài lịch sử train: trung vị {lengths.median():.0f}"
          f" | p75 {np.percentile(lengths, 75):.0f}"
          f" | p90 {np.percentile(lengths, 90):.0f}"
          f" | p99 {np.percentile(lengths, 99):.0f}"
          f" | tối đa {lengths.max():,}")
    print(f"  % người đọc có < 10 mục: {100 * (lengths < 10).mean():.1f}%")

    # Chi phí bộ nhớ của bảng lịch sử trong hai tháp, tính trước thay vì chờ
    # bị kill: hist_index là [n_users x 300] int64 cộng hist_weight float32.
    n_users = train["user"].nunique()
    bytes_hist = n_users * 300 * (8 + 4)
    print(f"\n  dự tính bảng lịch sử hai tháp: {n_users:,} người x 300 mục"
          f" = {bytes_hist / 2**30:.2f} GB")
    if bytes_hist > 1.0 * 2**30:
        print("  -> VƯỢT 1 GB: phải dùng --sample-users, nếu không sẽ bị kill")

    if args.anilist_dir.exists() and (args.anilist_dir / "train.parquet").exists():
        a_train = pd.read_parquet(args.anilist_dir / "train.parquet",
                                  columns=["user", "media_id"])
        a_users = a_train["user"].nunique()
        a_len = a_train.groupby("user")["media_id"].size()
        print("\n  so với nhánh AniList:")
        print(f"    người đọc   {a_users:>9,}  ->  {n_users:>9,}"
              f"   ({n_users / a_users:.1f}x)")
        print(f"    tương tác   {len(a_train):>9,}  ->  {len(train):>9,}"
              f"   ({len(train) / len(a_train):.1f}x)")
        print(f"    lịch sử tv  {a_len.median():>9.0f}  ->  {lengths.median():>9.0f}")
        # Khớp quy mô phải khớp theo SỐ TƯƠNG TÁC, không theo số người: hai
        # bộ có độ dài lịch sử rất khác nhau nên khớp số người sẽ cho hai
        # bộ lượng tín hiệu lệch hẳn.
        #
        # ĐÍNH CHÍNH: phép chia dưới đây SAI, giữ lại để thấy nó sai ở
        # đâu. Nó giả định lọc lõi không đổi gì khi mẫu mỏng đi, trong khi
        # `min_item=5` làm gần hết danh mục rụng rồi kéo người đọc rụng
        # theo — sụt theo cấp số nhân, không tuyến tính. Đo thật bằng
        # `scripts/scale_curve_goodreads.py` cho 160.000 người, tức gấp
        # 3,8 lần con số ngoại suy này.
        want = int(len(a_train) / max(lengths.mean(), 1))
        print(f"\n  ngoại suy thô (KHÔNG dùng): --sample-users {want:,}")
        print("  con số đúng phải đo bằng scale_curve_goodreads.py: "
              "lọc lõi sụt theo cấp số nhân")


if __name__ == "__main__":
    main()
