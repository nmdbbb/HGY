"""Xuất phần của HƯỚNG CŨ mà thang L2-L10 của đề xuất không có ô nào.

    python scripts/export_oldarm.py --out <thư mục>

Vì sao cần script riêng. `export_matrix.py` xếp mọi thứ vào thang L2-L10 và chỉ
lấy một con số cho mỗi ô: NDCG@10 ở lát `all`. Làm thế thì hướng cũ (hai tháp,
bốn mặt nội dung, đầu thời điểm, cổng độ phổ biến, định tuyến) hiện ra **đúng ở
cột mà nó thua**, vì `all` bị chi phối bởi người đọc đã có lịch sử. Toàn bộ lý do
tồn tại của hướng cũ là các lát **cold-start**, và lát đó không có trong bảng.

Ba tệp:

- `slices.csv`   NDCG@10 theo từng lát đánh giá, cho cả hai họ mô hình. Đây là
                 chỗ thứ tự đảo ngược lần nữa: họ láng giềng thắng ở `all` trên
                 Goodreads, nhưng ở `new_item` nó được **đúng 0**.
- `ablation.csv` Thang ablation của chính hướng cũ: mặt nội dung, hiệu chỉnh
                 logQ, cổng độ phổ biến, định tuyến. Thang L2-L10 không có bậc
                 nào cho những thứ này.
- `extras.csv`   Những hạng mục hướng cũ đo mà đề xuất không liệt kê: độ trễ,
                 tỉ lệ có lý do giải thích, hỗn hợp quốc gia, onboarding.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SLICES = ["all", "new_user", "new_item", "sparse_user", "warm"]

LEGS = [
    ("anilist", "AniList · A"),
    ("gr_matched", "Goodreads khớp · A"),
    ("gr_full", "Goodreads đầy đủ · A"),
]

# (chân, nhãn mô hình, họ, tệp, khoá trong tệp)
SLICE_ROWS = [
    ("anilist", "phổ biến", "mốc", "anilist_emb", "phổ biến"),
    ("anilist", "đang nổi (60 ngày)", "mốc", "anilist_emb", "đang nổi (60 ngày)"),
    ("anilist", "ALS", "nhân tố ẩn", "anilist_emb", "ALS"),
    ("anilist", "hệ hai tầng (đủ mặt)", "lai ghép", "anilist_eval", "tất cả mặt"),
    ("anilist", "hai tháp (feed)", "nhân tố ẩn", "anilist_emb",
     "hai tháp (gu + thời điểm) [seed 0]"),
    ("anilist", "hai tháp + cổng", "nhân tố ẩn", "anilist_emb",
     "gu + phổ biến, cổng hai phần + bỏ ngẫu nhiên [seed 0]"),
    ("anilist", "định tuyến feed → cổng", "lai ghép", "anilist_saved",
     "ĐỊNH TUYẾN: feed → gated"),
    ("anilist", "Item-KNN", "láng giềng", "anilist_cross", "Item-KNN (k=500)"),
    ("anilist", "Session-KNN", "láng giềng", "anilist_cross",
     "Session-KNN (L=10, giảm 0.7)"),

    ("gr_matched", "phổ biến", "mốc", "gr_emb", "phổ biến"),
    ("gr_matched", "đang nổi (60 ngày)", "mốc", "gr_emb", "đang nổi (60 ngày)"),
    ("gr_matched", "ALS", "nhân tố ẩn", "gr_emb", "ALS"),
    ("gr_matched", "hệ hai tầng (đủ mặt)", "lai ghép", "gr_eval", "tất cả mặt"),
    ("gr_matched", "hai tháp (feed)", "nhân tố ẩn", "gr_emb",
     "hai tháp (gu + thời điểm)"),
    ("gr_matched", "hai tháp + cổng", "nhân tố ẩn", "gr_gate",
     "gu + phổ biến, cổng hai phần + bỏ ngẫu nhiên"),
    ("gr_matched", "Item-KNN", "láng giềng", "gr_cross", "Item-KNN (k=500)"),
    ("gr_matched", "Session-KNN", "láng giềng", "gr_cross",
     "Session-KNN (L=5, giảm 0.7)"),

    ("gr_full", "phổ biến", "mốc", "gr_full_emb", "phổ biến"),
    ("gr_full", "đang nổi (60 ngày)", "mốc", "gr_full_emb", "đang nổi (60 ngày)"),
    ("gr_full", "ALS", "nhân tố ẩn", "gr_full_emb", "ALS"),
    ("gr_full", "hệ hai tầng (đủ mặt)", "lai ghép", "gr_full_eval", "tất cả mặt"),
    ("gr_full", "hai tháp (feed)", "nhân tố ẩn", "gr_full_emb",
     "hai tháp (gu + thời điểm)"),
    ("gr_full", "hai tháp + cổng", "nhân tố ẩn", "gr_full_emb",
     "gu + phổ biến, cổng hai phần + bỏ ngẫu nhiên"),
    ("gr_full", "Item-KNN", "láng giềng", "gr_full_cross", "Item-KNN (k=500)"),
    ("gr_full", "Session-KNN", "láng giềng", "gr_full_cross",
     "Session-KNN (L=10, giảm 0.7)"),
]

# (nhóm, thứ tự bậc, chân, nhãn bậc, tệp, khoá)
ABLATION_ROWS = [
    ("Mặt nội dung trong hệ hai tầng", 1, "anilist", "chỉ độ phổ biến",
     "anilist_eval", "popularity"),
    ("Mặt nội dung trong hệ hai tầng", 2, "anilist", "+ CF",
     "anilist_eval", "CF"),
    ("Mặt nội dung trong hệ hai tầng", 3, "anilist", "+ theme",
     "anilist_eval", "CF+theme"),
    ("Mặt nội dung trong hệ hai tầng", 4, "anilist", "+ creator",
     "anilist_eval", "CF+theme+creator"),
    ("Mặt nội dung trong hệ hai tầng", 5, "anilist", "+ pace",
     "anilist_eval", "CF+theme+creator+pace"),
    ("Mặt nội dung trong hệ hai tầng", 6, "anilist", "+ art (đủ bốn)",
     "anilist_eval", "tất cả mặt"),
    ("Mặt nội dung trong hệ hai tầng", 1, "gr_matched", "chỉ độ phổ biến",
     "gr_eval", "popularity"),
    ("Mặt nội dung trong hệ hai tầng", 2, "gr_matched", "+ CF", "gr_eval", "CF"),
    ("Mặt nội dung trong hệ hai tầng", 3, "gr_matched", "+ theme",
     "gr_eval", "CF+theme"),
    ("Mặt nội dung trong hệ hai tầng", 4, "gr_matched", "+ creator",
     "gr_eval", "CF+theme+creator"),
    ("Mặt nội dung trong hệ hai tầng", 5, "gr_matched", "+ pace",
     "gr_eval", "CF+theme+creator+pace"),
    ("Mặt nội dung trong hệ hai tầng", 6, "gr_matched", "+ art (đủ bốn)",
     "gr_eval", "tất cả mặt"),

    ("Mặt nội dung trong hai tháp", 1, "gr_matched", "chỉ theme (128 chiều)",
     "gr_ladder", "mặt: chỉ theme (128d)"),
    ("Mặt nội dung trong hai tháp", 2, "gr_matched", "+ creator (192)",
     "gr_ladder", "mặt: + creator (192d)"),
    ("Mặt nội dung trong hai tháp", 3, "gr_matched", "+ pace (205)",
     "gr_ladder", "mặt: + pace (205d)"),
    ("Mặt nội dung trong hai tháp", 4, "gr_matched", "+ art (220, đủ bốn)",
     "gr_ladder", "gu, không hiệu chỉnh logQ"),

    ("Cổng độ phổ biến", 1, "gr_matched", "không cổng (feed)",
     "gr_emb", "hai tháp (gu + thời điểm)"),
    ("Cổng độ phổ biến", 2, "gr_matched", "cổng, không bỏ ngẫu nhiên",
     "gr_gate", "gu + phổ biến, cổng hai phần (không bỏ ngẫu nhiên)"),
    ("Cổng độ phổ biến", 3, "gr_matched", "cổng + DropoutNet",
     "gr_gate", "gu + phổ biến, cổng hai phần + bỏ ngẫu nhiên"),
    ("Cổng độ phổ biến", 1, "anilist", "không cổng (feed)",
     "anilist_saved", "two_tower_feed"),
    ("Cổng độ phổ biến", 3, "anilist", "cổng + DropoutNet",
     "anilist_saved", "two_tower_gated (cổng hai phần)"),

    ("Định tuyến theo có/không lịch sử", 1, "anilist", "chỉ feed",
     "anilist_saved", "two_tower_feed"),
    ("Định tuyến theo có/không lịch sử", 2, "anilist", "chỉ cổng",
     "anilist_saved", "two_tower_gated (cổng hai phần)"),
    ("Định tuyến theo có/không lịch sử", 3, "anilist", "định tuyến feed → cổng",
     "anilist_saved", "ĐỊNH TUYẾN: feed → gated"),
]


def load(root):
    def read(path):
        path = Path(path)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    data = {
        "anilist_emb": read(root / "processed/embedding_temporal.json"),
        "anilist_eval": read(root / "processed/evaluation.json").get("ablation", {}),
        "anilist_cross": read(root / "processed/crossarm_knn.json"),
        "anilist_saved": read(root / "processed/eval_saved.json"),
        "anilist_product": read(root / "processed/product_metrics.json"),
        "anilist_country": read(root / "processed/country_threshold.json"),
        "anilist_index": read(root / "processed/index.json"),
        "anilist_taste": read(root / "processed/embedding_taste.json"),
        "anilist_artist": read(root / "processed/evaluation.json").get("artist_check", {}),
        "gr_emb": read(root / "processed_gr/embedding_feed_seed0.json"),
        "gr_gate": read(root / "processed_gr/embedding_gate_seed0.json"),
        "gr_ladder": read(root / "processed_gr/embedding_ladder_seed0.json"),
        "gr_eval": read(root / "processed_gr/evaluation_matched.json").get("ablation", {}),
        "gr_cross": read(root / "processed_gr/crossarm_knn.json"),
        "gr_full_emb": read(root / "processed_gr_full/embedding_temporal_p3p5.json"),
        "gr_full_eval": read(root / "processed_gr_full/evaluation.json").get("ablation", {}),
        "gr_full_cross": read(root / "processed_gr_full/crossarm_knn.json"),
    }
    return data


def ndcg(data, source, key, slice_name):
    row = data.get(source, {}).get(key)
    if not isinstance(row, dict):
        return None
    cell = row.get(slice_name)
    if not isinstance(cell, dict):
        return None
    value = cell.get("ndcg")
    return None if value is None else round(float(value), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load(args.root)
    label = dict(LEGS)

    rows, missing = [], []
    for leg, model, family, source, key in SLICE_ROWS:
        found = False
        for slice_name in SLICES:
            value = ndcg(data, source, key, slice_name)
            if value is None:
                continue
            found = True
            rows.append({"chan": label[leg], "chan_id": leg, "mo_hinh": model,
                         "ho": family, "lat": slice_name, "ndcg": value})
        if not found:
            missing.append(f"{leg}/{model} ({source} -> {key})")
    with (args.out / "slices.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["chan", "chan_id", "mo_hinh", "ho", "lat", "ndcg"])
        writer.writeheader()
        writer.writerows(rows)

    ladder = []
    for group, order, leg, step, source, key in ABLATION_ROWS:
        for slice_name in SLICES:
            value = ndcg(data, source, key, slice_name)
            if value is None:
                continue
            ladder.append({"nhom": group, "thu_tu": order, "chan": label[leg],
                           "chan_id": leg, "bac": step, "lat": slice_name,
                           "ndcg": value})
        if not any(r["nhom"] == group and r["bac"] == step and r["chan_id"] == leg
                   for r in ladder):
            missing.append(f"{group}/{leg}/{step} ({source} -> {key})")
    with (args.out / "ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "nhom", "thu_tu", "chan", "chan_id", "bac", "lat", "ndcg"])
        writer.writeheader()
        writer.writerows(ladder)

    # Nhánh GU: phép chia theo gu (`--split taste`) giữ ngẫu nhiên 20% mục mỗi
    # người và trung hoà thời gian, nên nó đo đúng câu hỏi ban đầu của đồ án —
    # "người này thích kiểu truyện nào" — chứ không phải "sắp đọc gì".
    taste_rows = []
    for key, row in data["anilist_taste"].items():
        if key.startswith("_") or not isinstance(row, dict):
            continue
        for slice_name in SLICES:
            cell = row.get(slice_name)
            if not isinstance(cell, dict) or cell.get("ndcg") is None:
                continue
            taste_rows.append({"mo_hinh": key, "lat": slice_name,
                               "ndcg": round(float(cell["ndcg"]), 4)})
    with (args.out / "taste.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mo_hinh", "lat", "ndcg"])
        writer.writeheader()
        writer.writerows(taste_rows)

    # Bốn mặt: tên đặc trưng thật, không phải nhãn. `art_feature_names` đọc từ
    # index.json nên nếu tầng đặc trưng đổi thì bảng này đổi theo.
    art_names = data["anilist_index"].get("art_feature_names", [])
    pace_names = ["log_chapters", "log_volumes", "duration_years", "is_finished",
                  "chapters_per_year", "age_years", "log_popularity", "drop_rate",
                  "pause_rate", "completion_rate", "mean_drop_position",
                  "mean_score", "score_spread"]
    facets = [
        ("theme", 128, "thẻ thể loại / mô-típ, nén xuống 128 chiều",
         "nền của mọi biến thể; một mình nó cho 0,0224 ở lát new_item (Goodreads)"),
        ("creator", 64, "hoạ sĩ và tác giả, tách vai vẽ / vai viết",
         "mặt mạnh nhất trên Goodreads (+43,3% ở new_item)"),
        ("pace", len(pace_names), "; ".join(pace_names),
         "mặt mạnh nhất trên AniList (+104%); trên Goodreads đóng góp −3,1% vì "
         "drop_rate, pause_rate, mean_drop_position đều bằng 0 — đó là dự đoán P4"),
        ("art", len(art_names), "; ".join(art_names),
         "đo trực tiếp từ ảnh bìa: tỉ lệ mực, mật độ nét, bề dày nét, năng lượng "
         "screentone, độ tương phản, màu"),
    ]
    with (args.out / "facets.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mat", "so_chieu", "dac_trung", "bang_chung"])
        writer.writerows(facets)

    product = data["anilist_product"]
    artist = data["anilist_artist"]
    engines = {row["engine"]: row for row in product.get("engines", [])}
    auto = engines.get("auto", {})
    onboarding = product.get("onboarding", {}).get("auto", {})
    index = data["anilist_index"]
    extras = [
        ("Lát đánh giá tách riêng", 5, "lát",
         "preprocess.evaluation_slices",
         "all / new_user / new_item / sparse_user / warm — thang L2-L10 chỉ có một lát"),
        ("Mặt nội dung", 4, "mặt", "content_based.FacetModel",
         "theme, creator, pace, art; mỗi mặt một khối đặc trưng riêng"),
        ("Chiều vectơ nội dung", 220, "chiều", "train_embedding.content_matrix",
         "theme 128 + creator 64 + pace 13 + art 15"),
        ("Cột đầu thời điểm", 5, "cột", "train_embedding.TIME_COLUMNS",
         "2 cột bẹt thành 0 trên Goodreads vì mọi quyển đều FINISHED"),
        ("Danh mục chấm điểm (AniList)", len(index.get("item_index", {})), "mục",
         "processed/index.json", "chấm toàn danh mục, không lấy mẫu âm"),
        ("Độ trễ mỗi lần gợi ý (p95)", round(auto.get("p95_ms", 0), 2), "ms",
         "product_metrics.json", f"trung vị {round(auto.get('median_ms', 0), 2)} ms, 800 người"),
        ("Tỉ lệ gợi ý có lý do giải thích", auto.get("reason_rate"), "tỉ lệ",
         "product_metrics.json", "mọi gợi ý đều kèm được một lý do đọc được"),
        ("Số nước trong top-10", round(auto.get("countries", 0), 2), "nước",
         "product_metrics.json", "manga JP / manhwa KR / manhua CN"),
        ("Thu hồi theo nước ưa thích", round(auto.get("country_recall", 0), 3),
         "tỉ lệ", "product_metrics.json", "so với mốc 1,94 nước của chính người đọc"),
        ("Hồ sơ dịch được sau 1 lượt onboarding",
         round(onboarding.get("1", {}).get("shift", 0), 3), "cosine",
         "product_metrics.json", "chọn 1 truyện đã đổi hồ sơ 0,80"),
        ("Tìm lại đúng hoạ sĩ chỉ từ nét vẽ",
         round(artist.get("precision_at_k", 0) / max(artist.get("random_baseline", 1), 1e-12), 1),
         "lần ngẫu nhiên", "evaluation.json → artist_check",
         f"precision@10 {artist.get('precision_at_k', 0):.5f} so với ngẫu nhiên "
         f"{artist.get('random_baseline', 0):.5f}; {artist.get('queries', 0):,} truy vấn, "
         f"{artist.get('artists', 0):,} hoạ sĩ"),
        ("Dự đoán ghi trước rồi mới chạy", 11, "dự đoán",
         "docs/goodreads-arm.md, docs/protocol-arm.md",
         "P1-P5 cho nhánh dữ liệu, Q1-Q6 cho nhánh giao thức"),
    ]
    with (args.out / "extras.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["hang_muc", "gia_tri", "don_vi", "nguon", "ghi_chu"])
        for item in extras:
            writer.writerow(["" if v is None else v for v in item])

    print(f"slices.csv   {len(rows)} dòng")
    print(f"ablation.csv {len(ladder)} dòng")
    print(f"extras.csv   {len(extras)} dòng")
    if missing:
        print("\nKHÔNG TÌM THẤY (phải sửa khoá hoặc chạy lại phép đo):")
        for item in missing:
            print("  -", item)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
