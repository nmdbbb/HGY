"""Dựng nhánh "hướng ban đầu": giao thức của đề xuất đồ án, cả hai biến thể gộp.

    python scripts/prepare_protocol.py                 # cả hai biến thể
    python scripts/prepare_protocol.py --merge none    # chỉ bản không gộp

Mỗi biến thể ghi ra một thư mục riêng, vì chúng là **hai bộ dữ liệu khác nhau**
(số mục khác nhau, nên mọi chỉ số xếp hạng cũng không cùng thang):

    data/processed_hgn_none/      tập = từng tác phẩm (như HGN làm)
    data/processed_hgn_volume/    tập = một bộ truyện được đánh số (One Piece…)

So với số HGN công bố (2.411.314 tương tác / 34.445 người / 33.121 truyện,
mật độ 0,211%) để biết mình đang đứng ở đâu — nhưng **không** coi đó là đích
phải trùng: tầng đọc dữ liệu ở đây gộp ấn bản theo `work_id` trước (89.411
book_id -> 62.944 work_id), còn HGN làm ở mức book_id. Khác nhau ở bước đó thì
quy mô sau lọc không thể trùng, và ép cho trùng bằng cách đổi ngưỡng lọc chính
là chọn tiền xử lý theo con số mình muốn.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from truyen_reco.data import protocol_peruser  # noqa: E402

HGN = {"rows": 2_411_314, "users": 34_445, "items": 33_121, "density_pct": 0.211,
       "recall10": 0.1743, "ndcg10": 0.1927}


def save(bundle, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        bundle[name].to_parquet(out_dir / f"{name}.parquet", index=False)
    users = sorted(bundle["train"]["user"].unique().tolist())
    items = sorted(bundle["train"]["media_id"].unique().tolist())
    index = {"user_index": {str(u): i for i, u in enumerate(users)},
             "item_index": {str(m): i for i, m in enumerate(items)}}
    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8")
    (out_dir / "series_of.json").write_text(
        json.dumps({str(k): v for k, v in bundle["series_of"].items()},
                   ensure_ascii=False), encoding="utf-8")
    (out_dir / "merge_map.json").write_text(
        json.dumps({str(k): v for k, v in bundle["merged_of"].items()},
                   ensure_ascii=False), encoding="utf-8")
    (out_dir / "protocol_report.json").write_text(
        json.dumps(bundle["report"], indent=2, ensure_ascii=False),
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/processed_gr", type=Path,
                        help="nơi có interactions.parquet đã đọc từ dữ liệu thô")
    parser.add_argument("--raw-dir", default="data/external/goodreads", type=Path)
    parser.add_argument("--out-prefix", default="data/processed_hgn")
    parser.add_argument("--merge", nargs="+", default=["none", "volume"],
                        choices=("none", "volume", "franchise"))
    parser.add_argument("--rule", default="rating",
                        choices=protocol_peruser.POSITIVE_RULES)
    parser.add_argument("--min-user", type=int, default=10)
    parser.add_argument("--min-item", type=int, default=5)
    args = parser.parse_args()

    path = args.source_dir / "interactions.parquet"
    if not path.exists():
        sys.exit(f"thiếu {path} — chạy train.py --source goodreads trước")
    interactions = pd.read_parquet(path)
    print(f"đọc {len(interactions):,} tương tác thô "
          f"| {interactions['user'].nunique():,} người "
          f"| {interactions['media_id'].nunique():,} tác phẩm\n")

    reports = {}
    for merge in args.merge:
        started = time.monotonic()
        bundle = protocol_peruser.run(
            interactions, raw_dir=args.raw_dir, rule=args.rule, merge=merge,
            min_user=args.min_user, min_item=args.min_item)
        out_dir = Path(f"{args.out_prefix}_{merge}")
        save(bundle, out_dir)
        report = bundle["report"]
        report["seconds"] = round(time.monotonic() - started, 1)
        reports[merge] = report
        print(f"[{merge}] {report['seconds']:.0f}s -> {out_dir}")

    labels = [("rows_positive", "dòng dương (rating>=4)"),
              ("items_after_merge", "mục sau khi gộp bộ"),
              ("merge_items_merged", "số bộ được gộp"),
              ("merge_rows_collapsed", "dòng bị dồn khi gộp"),
              ("rows_core", "dòng sau k-core 10/5"),
              ("users_core", "người đọc"),
              ("items_core", "mục"),
              ("density_pct", "mật độ (%)"),
              ("train_rows", "train"),
              ("val_rows", "val"),
              ("test_rows", "test"),
              ("hist_median", "lịch sử train trung vị"),
              ("hist_mean", "lịch sử train trung bình")]
    width = 26
    header = "  " + "chỉ số".ljust(width) + "".join(
        f"{m:>18}" for m in args.merge) + f"{'HGN công bố':>18}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    hgn_for = {"rows_core": HGN["rows"], "users_core": HGN["users"],
               "items_core": HGN["items"], "density_pct": HGN["density_pct"]}
    for key, label in labels:
        cells = ""
        for merge in args.merge:
            value = reports[merge].get(key)
            cells += f"{'—' if value is None else format(value, ',') if isinstance(value, int) else value:>18}"
        reference = hgn_for.get(key)
        cells += f"{'—' if reference is None else format(reference, ',') if isinstance(reference, int) else reference:>18}"
        print("  " + label.ljust(width) + cells)
    print("\n  Mốc HGN để đối chiếu chỉ số sau này: "
          f"Recall@10 {HGN['recall10']}, NDCG@10 {HGN['ndcg10']}")
    print("  (chỉ so trực tiếp được với biến thể 'none'; bản gộp bộ có số mục")
    print("   khác nên không cùng thang — đúng như đề xuất đã ghi.)")

    out = Path(f"{args.out_prefix}_summary.json")
    out.write_text(json.dumps({"hgn_reference": HGN, "variants": reports},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
