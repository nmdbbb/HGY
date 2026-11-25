"""Dựng AniList theo giao thức B (chia 70/10/20 theo thời gian của từng người).

    python scripts/prepare_anilist_protocol.py

Vì sao cần: AniList trước đây chỉ đi qua giao thức A (một mốc thời gian chung,
không có tập validation), còn thang 24 phương pháp chạy bằng các script cần
cấu trúc giao thức B (`train/val/test` + `index.json` + `series_of.json`). Kết
quả là AniList chỉ có 9 phương pháp, và hình xếp hạng của hai bộ đặt cạnh nhau
thực ra là **hai giao thức khác nhau**. Script này đóng khoảng trống đó.

Dùng lại đúng những gì nhánh A đã chốt, để chỉ khác nhau ở phép chia:

* dương = `signal` thuộc {positive, weak} của `preprocess.label`;
* thời điểm = `created_at`, và `updated_at` khi `created_at` = 0 (12,6% dòng
  AniList có `created_at` = 1970, dấu thời gian không hợp lệ). Đã đo: dòng dùng
  dự phòng chiếm 14,9% train / 9,7% val / 10,5% test, nên không dồn vào test.

Khác nhánh A, theo giao thức B: lọc lõi 10/5 (A dùng 5/5), không tách nhóm
người đọc lạnh riêng, chia theo từng người thay vì một mốc chung.
AniList không có "tập nhiều quyển" nên không có phép gộp bộ: mỗi mục là một
bộ truyện, `series_of` rỗng.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from truyen_reco.data import preprocess, protocol_peruser  # noqa: E402
from prepare_protocol import save  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/processed", type=Path)
    parser.add_argument("--out", default="data/processed_anilist_b", type=Path)
    parser.add_argument("--min-user", type=int, default=10)
    parser.add_argument("--min-item", type=int, default=5)
    args = parser.parse_args()

    interactions = pd.read_parquet(args.source_dir / "interactions.parquet")
    media = pd.read_parquet(args.source_dir / "media.parquet")
    print(f"đọc {len(interactions):,} tương tác thô | "
          f"{interactions['user'].nunique():,} người | "
          f"{interactions['media_id'].nunique():,} truyện")

    frame = preprocess.label(interactions, media=media)
    frame["event_ts"] = frame["created_at"].where(frame["created_at"] > 0,
                                                  frame["updated_at"])
    bundle = protocol_peruser.run(
        frame, rule="signal", merge="none", groups={},
        min_user=args.min_user, min_item=args.min_item,
        time_column="event_ts")
    save(bundle, args.out)
    report = bundle["report"]
    for key in ("rows_positive", "users_positive", "items_positive",
                "rows_core", "users_core", "items_core", "density_pct",
                "train_rows", "val_rows", "test_rows", "train_users",
                "test_users", "train_items", "hist_median", "hist_mean"):
        value = report.get(key)
        print(f"  {key:<16} "
              f"{format(value, ',') if isinstance(value, int) else value}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
