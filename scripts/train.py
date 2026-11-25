"""Chạy toàn bộ pipeline: dữ liệu thô -> đặc trưng -> mô hình -> artifact.

    python scripts/train.py --skip-covers      # bỏ bước tải ảnh bìa
    python scripts/train.py --max-covers 20000

Nhánh đối chứng Goodreads dùng CÙNG script này, chỉ đổi nguồn đọc dữ liệu:

    python scripts/train.py --source goodreads \
           --raw-dir data/external/goodreads --out-dir data/processed_gr \
           --cover-dir data/raw/covers_gr

Một script cho cả hai bộ là có chủ ý: nếu mỗi bộ có một pipeline riêng thì
chênh lệch kết quả không còn quy được về dữ liệu, vì nó cũng có thể do code
khác nhau. Mọi khác biệt phải nằm gọn trong bộ chuyển đổi ở tầng ingest.
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

from truyen_reco.data import ingest, ingest_goodreads, preprocess  # noqa: E402
from truyen_reco.features import art as art_features  # noqa: E402
from truyen_reco.features import build_features  # noqa: E402
from truyen_reco.models.collaborative import ALS, PopularityModel, build_matrix  # noqa: E402


def step(message):
    print(f"\n{'=' * 68}\n{message}\n{'=' * 68}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("anilist", "goodreads"),
                        default="anilist",
                        help="nguồn dữ liệu; chỉ đổi tầng ingest, phần sau giữ nguyên")
    parser.add_argument("--raw-dir", default=None, type=Path)
    parser.add_argument("--out-dir", default="data/processed", type=Path)
    parser.add_argument("--cover-dir", default="data/raw/covers", type=Path)
    parser.add_argument("--max-covers", type=int, default=20_000)
    parser.add_argument("--skip-covers", action="store_true")
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--min-user", type=int, default=5)
    parser.add_argument("--min-item", type=int, default=5)
    parser.add_argument("--sample-users", type=int, default=0, metavar="N",
                        help="lấy mẫu N người đọc ngay sau khi đọc dữ liệu thô, "
                             "để hai nhánh chạy ở quy mô so sánh được")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--reuse-ingest", action="store_true",
                        help="dùng lại interactions/media.parquet đã đọc trước đó "
                             "thay vì đọc lại dữ liệu thô (~25 phút trên Goodreads); "
                             "tự từ chối nếu tầng ingest đã sửa sau khi cache được ghi")
    args = parser.parse_args()
    if args.raw_dir is None:
        args.raw_dir = Path("data/raw/anilist" if args.source == "anilist"
                            else "data/external/goodreads")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    step(f"1/6  Đọc dữ liệu thô ({args.source})")
    reader = ingest if args.source == "anilist" else ingest_goodreads
    interactions = media = None
    if args.reuse_ingest:
        # Dùng lại cache là an toàn CHỈ KHI tầng ingest chưa đổi kể từ lúc ghi.
        # So mốc sửa file thay vì tin vào ký ức: đọc lại một cache cũ sau khi
        # sửa ingest là cách im lặng nhất để làm bẩn cả nhánh đối chứng, vì
        # mọi con số vẫn ra bình thường, chỉ là của một bản dữ liệu khác.
        cached = [args.out_dir / "interactions.parquet", args.out_dir / "media.parquet"]
        if all(p.exists() for p in cached):
            cache_time = min(p.stat().st_mtime for p in cached)
            code_time = Path(reader.__file__).stat().st_mtime
            if code_time > cache_time:
                print("  cache có nhưng tầng ingest đã sửa sau đó -> đọc lại dữ liệu thô")
            else:
                interactions = pd.read_parquet(cached[0])
                media = pd.read_parquet(cached[1])
                print(f"  dùng lại cache {cached[0]} (ghi lúc "
                      f"{time.strftime('%H:%M %d/%m', time.localtime(cache_time))})")
    if interactions is None:
        interactions, media = reader.run(args.raw_dir, args.out_dir)
    print(f"  {len(interactions):,} tương tác | {len(media):,} truyện "
          f"| {interactions['user'].nunique():,} người đọc")
    origins = media["country"].value_counts().head(5).to_dict()
    print("  xuất xứ truyện:", origins if origins else
          "KHÔNG CÓ TRƯỜNG XUẤT XỨ — mọi chỉ số theo nước là N/A trên nhánh này")

    if args.sample_users and interactions["user"].nunique() > args.sample_users:
        # Lấy mẫu theo NGƯỜI ĐỌC, không theo dòng: cắt ngẫu nhiên theo dòng sẽ
        # làm mỏng lịch sử của mọi người cùng lúc, tức là đổi luôn tính chất
        # đang muốn so sánh. Lấy mẫu người đọc giữ nguyên hình dạng lịch sử của
        # từng người, chỉ giảm số người.
        rng = np.random.default_rng(args.sample_seed)
        pool = interactions["user"].unique()
        keep = set(rng.choice(pool, args.sample_users, replace=False).tolist())
        before = len(interactions)
        interactions = interactions[interactions["user"].isin(keep)]
        print(f"  lấy mẫu {args.sample_users:,}/{len(pool):,} người đọc "
              f"(seed {args.sample_seed}): {before:,} -> {len(interactions):,} tương tác")
        used_media = set(interactions["media_id"])
        media = media[media["media_id"].isin(used_media)].reset_index(drop=True)

    step("2/6  Gán mức tín hiệu, lọc lõi, chia theo thời gian")
    prepared = preprocess.run(interactions, media=media, min_user=args.min_user,
                              min_item=args.min_item)
    train, test = prepared["train"], prepared["test"]
    labelled = prepared["labelled"]
    print("  phân bố trạng thái:",
          labelled["signal"].value_counts().to_dict())
    print(f"  sau lọc lõi: {len(prepared['core']):,} tương tác dương")
    print(f"  mốc cắt: {prepared['cutoff']}")
    print(f"  train {len(train):,} | test {len(test):,}")
    for name, frame in prepared["slices"].items():
        print(f"    lát {name:<12} {len(frame):>8,} tương tác"
              f" | {frame['user'].nunique():>6,} người")

    # Chỉ giữ metadata của truyện thực sự xuất hiện trong tập lõi
    used = set(prepared["core"]["media_id"])
    media = media[media["media_id"].isin(used)].reset_index(drop=True)
    print(f"  truyện trong tập lõi: {len(media):,}")

    art_vectors = art_ids = art_raw = None
    if not args.skip_covers:
        step("3/6  Tải ảnh bìa và rút đặc trưng nét vẽ")
        downloaded = art_features.download_covers(
            media, args.cover_dir, max_items=args.max_covers)
        print(f"  đã có {downloaded:,} ảnh bìa")
        matrix, art_ids, art_raw = art_features.build_art(media, args.cover_dir)
        print(f"  rút đặc trưng cho {len(art_ids):,} truyện "
              f"({matrix.shape[1]} chiều)")
        if len(art_ids):
            position = {m: i for i, m in enumerate(media["media_id"])}
            art_vectors = np.zeros((len(media), matrix.shape[1]), dtype=np.float32)
            for row, media_id in enumerate(art_ids):
                if media_id in position:
                    art_vectors[position[media_id]] = matrix[row]
    else:
        step("3/6  Bỏ qua ảnh bìa")

    step("4/6  Dựng vector theo từng mặt")
    built = build_features.build_all(media, train, art_vectors=art_vectors)
    index = built["index"]
    for name, vectors in built["facets"].items():
        covered = (np.linalg.norm(vectors, axis=1) > 1e-8).mean()
        print(f"  {name:<9} {vectors.shape[1]:>4} chiều | phủ {covered * 100:5.1f}% truyện")

    step("5/6  Huấn luyện lọc cộng tác (ALS)")
    users = sorted(train["user"].unique())
    user_index = {user: i for i, user in enumerate(users)}
    item_index = index
    matrix = build_matrix(train, user_index, item_index)
    print(f"  ma trận {matrix.shape[0]:,} x {matrix.shape[1]:,} "
          f"| {matrix.nnz:,} ô khác 0 "
          f"| độ thưa {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.6f}")

    als = ALS(factors=args.factors, iterations=args.iterations).fit(matrix)
    popularity = PopularityModel().fit(matrix)
    print(f"  xong {args.factors} nhân tố, {args.iterations} vòng lặp")

    step("6/6  Lưu artifact")
    np.savez_compressed(
        args.out_dir / "facets.npz",
        **{f"facet_{k}": v for k, v in built["facets"].items()},
        user_factors=als.user_factors, item_factors=als.item_factors,
        popularity=popularity.scores,
        art_raw=art_raw if art_raw is not None else np.zeros((0, 0)),
        art_ids=art_ids if art_ids is not None else np.zeros(0),
    )
    (args.out_dir / "index.json").write_text(json.dumps({
        "user_index": user_index,
        "item_index": {str(k): v for k, v in item_index.items()},
        "art_feature_names": art_features.FEATURE_NAMES,
        "cutoff": str(prepared["cutoff"]),
    }), encoding="utf-8")
    media.to_parquet(args.out_dir / "media_core.parquet", index=False)
    train.to_parquet(args.out_dir / "train.parquet", index=False)
    test.to_parquet(args.out_dir / "test.parquet", index=False)
    prepared["cold_users"].to_parquet(args.out_dir / "cold_users.parquet", index=False)
    built["pace_table"].to_parquet(args.out_dir / "pace_table.parquet", index=False)

    print(f"  -> {args.out_dir}")
    print(f"\nTổng thời gian: {(time.monotonic() - started) / 60:.1f} phút")


if __name__ == "__main__":
    main()
