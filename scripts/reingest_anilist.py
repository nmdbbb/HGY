"""Đọc lại dữ liệu thô AniList sau khi crawl thêm, ghi interactions/media.parquet.

Chỉ làm bước 1 của scripts/train.py (tầng ingest), không chạy tiếp ALS/feature/
cover — các bước đó không cần để dựng lại processed_anilist_b.

    python scripts/reingest_anilist.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from truyen_reco.data import ingest  # noqa: E402

RAW_DIR = Path("data/raw/anilist")
OUT_DIR = Path("data/processed")


def main():
    started = time.monotonic()
    interactions, media = ingest.run(RAW_DIR, OUT_DIR)
    print(f"{len(interactions):,} tương tác | {len(media):,} truyện | "
          f"{interactions['user'].nunique():,} người đọc")
    origins = media["country"].value_counts().head(5).to_dict()
    print("xuất xứ truyện:", origins)
    print(f"-> {OUT_DIR}/interactions.parquet, {OUT_DIR}/media.parquet "
          f"({(time.monotonic() - started) / 60:.1f} phút)")


if __name__ == "__main__":
    main()
