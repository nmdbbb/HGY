"""Xuất bảng "dữ liệu x phương pháp" ra CSV để dựng trang kết quả.

    python scripts/export_matrix.py --out <thư mục>

Mọi con số đọc từ tệp kết quả trên đĩa, không gõ tay — gõ tay một bảng 60 ô là
cách chắc chắn nhất để có một ô sai mà không ai phát hiện. Phần *biên tập* duy
nhất ở đây là việc gán mỗi phương pháp vào chương nào của đề xuất (L2-L10) và
đánh dấu ô nào còn trống; phần đó nằm trong `LADDER` bên dưới và đọc được.

Năm "chân" (một chân = một bộ dữ liệu + một giao thức):

- `anilist`      AniList, giao thức nhánh dữ liệu (cắt theo thời điểm toàn cục)
- `gr_matched`   Goodreads khớp quy mô, cùng giao thức đó
- `gr_full`      Goodreads quy mô đầy đủ, cùng giao thức đó
- `hgn_none`     Goodreads, giao thức đề xuất (70/10/20 theo người), không gộp bộ
- `hgn_volume`   Goodreads, giao thức đề xuất, gộp tập thành bộ
- `anilist_b`    AniList, giao thức đề xuất (70/10/20 theo người). Suy ra TỰ ĐỘNG
                 từ các ô `hgn_none` — cùng tên mô hình, nguồn đổi sang AniList
                 — để hai bộ chắc chắn đo cùng một danh sách phương pháp.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

LEGS = ["anilist", "gr_matched", "gr_full", "hgn_none", "hgn_volume",
        "anilist_b"]

# Mỗi dòng: (chương, phương pháp, {chân: (nguồn, khoá)}, ghi chú)
# nguồn: tên tệp JSON đã nạp; khoá: đường dẫn trong tệp đó.
LADDER = [
    ("5.1 Baseline", "Random", {
        "hgn_none": ("hgn_none_base", "ngẫu nhiên"),
        "hgn_volume": ("hgn_volume_base", "ngẫu nhiên")}, ""),
    ("5.1 Baseline", "MostPopular", {
        "anilist": ("anilist_emb", "phổ biến"),
        "gr_matched": ("gr_emb", "phổ biến"),
        "gr_full": ("gr_full_emb", "phổ biến"),
        "hgn_none": ("hgn_none_base", "phổ biến"),
        "hgn_volume": ("hgn_volume_base", "phổ biến")}, ""),
    ("5.1 Baseline", "Top-rated Bayesian", {
        "hgn_none": ("hgn_none_base", "điểm Bayes"),
        "hgn_volume": ("hgn_volume_base", "điểm Bayes")}, ""),
    ("5.1 Baseline", "Đang nổi (60 ngày) — thêm ngoài đề xuất", {
        "anilist": ("anilist_emb", "đang nổi (60 ngày)"),
        "gr_matched": ("gr_emb", "đang nổi (60 ngày)"),
        "gr_full": ("gr_full_emb", "đang nổi (60 ngày)")},
     "mốc mạnh nhất trên Goodreads ở giao thức A"),

    ("5.2 L2 — CF cơ bản", "User-KNN (cosine)", {
        "hgn_none": ("hgn_none_base", "User-KNN"),
        "hgn_volume": ("hgn_volume_base", "User-KNN")}, ""),
    ("5.2 L2 — CF cơ bản", "Item-KNN (cosine, significance weighting)", {
        "anilist": ("anilist_cross", "Item-KNN (k=500)"),
        "gr_matched": ("gr_cross", "Item-KNN (k=500)"),
        "gr_full": ("gr_full_cross", "Item-KNN (k=500)"),
        "hgn_none": ("hgn_none_tune", "itemknn"),
        "hgn_volume": ("hgn_volume_tune", "itemknn")},
     "k và shrink chọn trên validation"),
    ("5.2 L2 — CF cơ bản", "SVD++", {
        "rating": ("svdpp", "SVD++")},
     "số hạng ẩn trùng khít tập train nên không thêm tín hiệu"),
    ("5.2 L2 — CF cơ bản", "Item-KNN Pearson", {
        "hgn_none": ("more", "Item-KNN Pearson")},
     "tương quan phi tính chính xác từ số đồng xuất hiện"),
    ("5.2 L2 — CF cơ bản", "MF cho rating hiện (FunkSVD có chệch) — MAE/RMSE", {
        "rating": ("ladder", "A")},
     "đo trên tiền xử lý thứ ba: toàn bộ dòng có chấm điểm"),

    ("5.3 L3 — CF nâng cao", "BPR-MF", {
        "hgn_none": ("hgn_none_base", "BPR-MF"),
        "hgn_volume": ("hgn_volume_base", "BPR-MF")},
     "bản có chệch theo item sập về bảng phổ biến (0,0344)"),
    ("5.3 L3 — CF nâng cao", "iALS / ALS", {
        "anilist": ("anilist_emb", "ALS"),
        "gr_matched": ("gr_emb", "ALS"),
        "gr_full": ("gr_full_emb", "ALS"),
        "hgn_none": ("hgn_none_base", "ALS"),
        "hgn_volume": ("hgn_volume_base", "ALS")},
     "mốc chung của cả hai nhánh"),
    ("5.3 L3 — CF nâng cao", "NeuMF", {
        "hgn_none": ("neural", "NeuMF")},
     "ngân sách 120 epoch, val chọn 64 — đã bẹt"),
    ("5.3 L3 — CF nâng cao", "LightGCN", {
        "hgn_none": ("neural", "LightGCN")},
     "3 tầng; ngân sách 100 epoch, val chọn 90 — cao nhất trong sáu mô hình nơ-ron"),

    ("5.4 L4 — nội dung", "TF-IDF trên shelves và tác giả", {
        "hgn_none": ("hgn_none_base", "nội dung"),
        "hgn_volume": ("hgn_volume_base", "nội dung")},
     "ở giao thức A nội dung nằm trong hai tháp, không chạy riêng"),
    ("5.4 L4 — nội dung", "Nội dung LSA chiều thấp", {
        "hgn_none": ("more", "Nội dung LSA (thay sentence embedding)")},
     "SVD 128 chiều của TF-IDF; THAY cho sentence embedding, không phải nó"),

    ("5.5 L5 — tri thức", "Lọc theo ràng buộc", {
        "hgn_none": ("more", "Lọc ràng buộc trên bảng phổ biến")},
     "4 ràng buộc SUY RA từ lịch sử, không phải khai báo"),
    ("5.5 L5 — tri thức", "Case-based", {
        "hgn_none": ("more", "Case-based (case = mục gần nhất)")},
     "case = mục đọc gần nhất; chưa có vòng lặp phê bình"),
    ("5.5 L5 — tri thức", "Onboarding 2 lựa chọn", {
        "hgn_none": ("onboard", "Item-KNN sau 2 lựa chọn")},
     "hỏi 2 mục rồi chấm; HƠN cả lịch sử đầy đủ (0,0989)"),

    ("5.6 L6 — phiên & chuỗi", "Session-KNN", {
        "anilist": ("anilist_cross", "Session-KNN (L=10, giảm 0.7)"),
        "gr_matched": ("gr_cross", "Session-KNN (L=5, giảm 0.7)"),
        "gr_full": ("gr_full_cross", "Session-KNN (L=10, giảm 0.7)"),
        "hgn_none": ("hgn_none_tune", "sessionknn"),
        "hgn_volume": ("hgn_volume_tune", "sessionknn")},
     "L và hệ số giảm chọn trên validation"),
    ("5.6 L6 — phiên & chuỗi", "Luật kết hợp (support, confidence)", {
        "hgn_none": ("ladder", "B")}, "9,5 triệu luật có co >= 5"),
    ("5.6 L6 — phiên & chuỗi", "GRU4Rec", {
        "hgn_none": ("neural", "GRU4Rec")},
     "chuỗi 50; ngân sách 400 epoch, val chọn 320 — ở ngân sách 8 chỉ được 0,0200"),
    ("5.6 L6 — phiên & chuỗi", "SASRec", {
        "hgn_none": ("neural", "SASRec")},
     "2 khối, mất mát ở mọi vị trí; ngân sách 120, val chọn 20 — đã bẹt thật"),

    ("5.7 L9 — kiểu gợi ý tin tức", "Hai tháp (bộ mã nội dung, tháp người đọc)", {
        "anilist": ("anilist_emb", "hai tháp (gu + thời điểm) [seed 0]"),
        "gr_matched": ("gr_emb", "hai tháp (gu + thời điểm)"),
        "gr_full": ("gr_full_emb", "hai tháp (gu + thời điểm)")},
     "cùng họ với NAML/NRMS (bộ mã nội dung + sampled softmax + hiệu chỉnh logQ), KHÔNG có attention"),
    ("5.7 L9 — kiểu gợi ý tin tức", "NRMS-style", {
        "hgn_none": ("neural", "NRMS-style")},
     "attention ở tầng người đọc; KHÔNG có attention mức từ; ngân sách 120, val chọn 52"),

    ("5.8 L10 — học tương phản", "DropoutNet (bỏ ngẫu nhiên đặc trưng phổ biến)", {
        "anilist": ("anilist_emb", "gu + phổ biến, cổng hai phần + bỏ ngẫu nhiên [seed 0]"),
        "gr_matched": ("gr_gate", "gu + phổ biến, cổng hai phần + bỏ ngẫu nhiên"),
        "gr_full": ("gr_full_emb", "gu + phổ biến, cổng hai phần + bỏ ngẫu nhiên")},
     "tăng cường bằng cách bỏ đặc trưng, không phải contrastive loss"),
    ("5.8 L10 — học tương phản", "CL4SRec", {
        "hgn_none": ("neural", "CL4SRec")},
     "SASRec + NT-Xent, lambda 0,1; ngân sách 120, val chọn 72 — THẤP HƠN SASRec trần"),

    ("5.9 L7 — lai ghép", "Switching (định tuyến theo có/không lịch sử)", {
        "anilist": ("anilist_emb", "gu + phổ biến, cổng hai phần + bỏ ngẫu nhiên [seed 0]")},
     "RoutedTwoTower; lập được bằng phép kiểm ghép cặp 4 seed trên AniList"),
    ("5.9 L7 — lai ghép", "Song song có trọng số (grid search trên val)", {
        "hgn_none": ("ladder", "C_blend")}, ""),
    ("5.9 L7 — lai ghép", "Đa dạng hoá MMR", {
        "hgn_none": ("ladder", "C_mmr")}, ""),
    ("5.9 L7 — lai ghép", "Content-boosted CF", {
        "hgn_none": ("more", "Content-boosted CF")},
     "bơm 10 pseudo-rating mỗi người rồi chạy ALS"),
    ("5.9 L7 — lai ghép", "Pipeline sinh ứng viên rồi xếp lại", {
        "anilist": ("anilist_eval", "tất cả mặt"),
        "gr_matched": ("gr_eval", "tất cả mặt"),
        "gr_full": ("gr_full_eval", "tất cả mặt")},
     "xếp lại top-400 của ALS; loại cùng bộ chỉ có ở giao thức B"),
]

# nguồn của Goodreads-B -> nguồn tương ứng của AniList-B. `svdpp` không có ở đây
# vì nó đo bài dự đoán điểm, mà chỉ 21,6% dòng AniList có điểm.
ANILIST_B_SOURCE = {
    "hgn_none_base": "anl_base", "hgn_none_tune": "anl_tune",
    "ladder": "anl_ladder", "more": "anl_more",
    "onboard": "anl_onboard", "neural": "anl_neural",
}
for _chapter, _method, _cells, _note in LADDER:
    _cell = _cells.get("hgn_none")
    if _cell and _cell[0] in ANILIST_B_SOURCE:
        _cells["anilist_b"] = (ANILIST_B_SOURCE[_cell[0]], _cell[1])


def load_all(root):
    def read(path):
        path = Path(path)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    data = {
        "anilist_emb": read(root / "processed/embedding_temporal.json"),
        "anilist_eval": read(root / "processed/evaluation.json"),
        "anilist_cross": read(root / "processed/crossarm_knn.json"),
        "gr_emb": read(root / "processed_gr/embedding_feed_seed0.json"),
        "gr_gate": read(root / "processed_gr/embedding_gate_seed0.json"),
        "gr_eval": read(root / "processed_gr/evaluation_matched.json"),
        "gr_cross": read(root / "processed_gr/crossarm_knn.json"),
        "gr_full_emb": read(root / "processed_gr_full/embedding_temporal_p3p5.json"),
        "gr_full_eval": read(root / "processed_gr_full/evaluation.json"),
        "gr_full_cross": read(root / "processed_gr_full/crossarm_knn.json"),
        "hgn_none_base": read(root / "processed_hgn_none/baseline_results.json").get("results", {}),
        "hgn_volume_base": read(root / "processed_hgn_volume/baseline_results.json").get("results", {}),
        "hgn_none_tune": read(root / "processed_hgn_none/tuning.json"),
        "hgn_volume_tune": read(root / "processed_hgn_volume/tuning.json"),
        "ladder": read(root / "processed_hgn_none/ladder_gaps.json"),
        "more": read(root / "processed_hgn_none/more_methods.json"),
        "svdpp": read(root / "processed_hgn_none/svdpp.json"),
        "onboard": read(root / "processed_hgn_none/onboarding.json"),
        "neural": read(root / "processed_hgn_none/neural_results.json"),
        # Hai lượt chạy lại ngân sách lớn ghi tệp riêng để không tranh nhau
        # cùng một JSON; chúng ĐÈ lên tệp chung vì chúng mới hơn và đọc ở
        # ngân sách lớn hơn.
        "neural_long": {
            **read(root / "processed_hgn_none/neural_gru400.json"),
            **read(root / "processed_hgn_none/neural_cl4srec.json"),
            **read(root / "processed_hgn_none/neural_lightgcn.json")},
    }
    data["neural"] = {**data["neural"], **data.pop("neural_long")}

    anl = root / "processed_anilist_b"
    data["anl_base"] = read(anl / "baseline_results.json").get("results", {})
    data["anl_tune"] = read(anl / "tuning.json")
    data["anl_ladder"] = read(anl / "ladder_gaps.json")
    data["anl_more"] = read(anl / "more_methods.json")
    data["anl_onboard"] = read(anl / "onboarding.json")
    # Mỗi mô hình nơ-ron ghi một tệp riêng để các lượt chạy không tranh nhau.
    data["anl_neural"] = {}
    for part in sorted(anl.glob("neural_*.json")) if anl.exists() else []:
        data["anl_neural"].update(read(part))
    for key in ("anilist_eval", "gr_eval", "gr_full_eval"):
        data[key] = data[key].get("ablation", {})
    return data


def pick(data, source, key):
    """Lấy NDCG@10 của một ô. Trả về None nếu chưa có — None in ra ô trống."""
    blob = data.get(source, {})
    if source.endswith("ladder"):
        if key == "A":
            return None  # MAE/RMSE đi bảng riêng, không có NDCG
        if key == "B":
            rows = blob.get("B_association_rules", {}).get("results", {})
            row = rows.get("luật kết hợp (max confidence)")
            return row["ndcg"] if row else None
        rows = blob.get("C_hybrid", {}).get("results", {})
        for name, row in rows.items():
            if key == "C_mmr" and "MMR" in name:
                return row.get("ndcg")
            if key == "C_blend" and "MMR" not in name:
                return row.get("ndcg")
        return None
    if source.endswith("_tune"):
        row = blob.get(key)
        return row.get("test_ndcg") if row else None
    row = blob.get(key)
    if row is None:
        return None
    if "all" in row and isinstance(row["all"], dict):
        return row["all"].get("ndcg")
    return row.get("ndcg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_all(args.root)

    rows = []
    for chapter, method, cells, note in LADDER:
        row = {"chuong": chapter, "phuong_phap": method, "ghi_chu": note}
        found = 0
        for leg in LEGS:
            value = pick(data, *cells[leg]) if leg in cells else None
            row[leg] = "" if value is None else round(float(value), 4)
            found += 1 if value is not None else 0
        if "rating" in cells:
            row["trang_thai"] = "đã chạy"
        elif found:
            row["trang_thai"] = "đã chạy"
        elif note.startswith("CHƯA"):
            row["trang_thai"] = "chưa chạy"
        else:
            row["trang_thai"] = "một phần"
        rows.append(row)

    fields = ["chuong", "phuong_phap"] + LEGS + ["trang_thai", "ghi_chu"]
    with (args.out / "coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Bảng dài cho đồ thị: một dòng cho mỗi (chân, mô hình) có số.
    long_rows = []
    leg_label = {
        "anilist": "AniList · giao thức A",
        "gr_matched": "Goodreads khớp · A",
        "gr_full": "Goodreads đầy đủ · A",
        "hgn_none": "Goodreads · B không gộp",
        "hgn_volume": "Goodreads · B gộp bộ",
        "anilist_b": "AniList · B",
    }
    family = {
        "MostPopular": "mốc", "Random": "mốc", "Top-rated Bayesian": "mốc",
        "Đang nổi (60 ngày) — thêm ngoài đề xuất": "mốc",
        "User-KNN (cosine)": "láng giềng",
        "Item-KNN (cosine, significance weighting)": "láng giềng",
        "Session-KNN": "láng giềng",
        "Luật kết hợp (support, confidence)": "láng giềng",
        "iALS / ALS": "nhân tố ẩn", "BPR-MF": "nhân tố ẩn",
        "Hai tháp (bộ mã nội dung, tháp người đọc)": "nhân tố ẩn",
        "DropoutNet (bỏ ngẫu nhiên đặc trưng phổ biến)": "nhân tố ẩn",
        "Switching (định tuyến theo có/không lịch sử)": "lai ghép",
        "Song song có trọng số (grid search trên val)": "lai ghép",
        "Đa dạng hoá MMR": "lai ghép",
        "Pipeline sinh ứng viên rồi xếp lại": "lai ghép",
        "TF-IDF trên shelves và tác giả": "nội dung",
        "Nội dung LSA chiều thấp": "nội dung",
        "Item-KNN Pearson": "láng giềng", "SVD++": "nhân tố ẩn",
        "GRU4Rec": "chuỗi", "SASRec": "chuỗi", "CL4SRec": "chuỗi",
        "NeuMF": "nhân tố ẩn", "LightGCN": "nhân tố ẩn",
        "NRMS-style": "nhân tố ẩn",
        "Content-boosted CF": "lai ghép",
        "Lọc theo ràng buộc": "tri thức", "Case-based": "tri thức",
        "Onboarding 2 lựa chọn": "tri thức",
    }
    for row in rows:
        for leg in LEGS:
            if row[leg] == "":
                continue
            long_rows.append({
                "chan": leg_label[leg], "chan_id": leg,
                "mo_hinh": row["phuong_phap"], "chuong": row["chuong"],
                "ho": family.get(row["phuong_phap"], "khác"),
                "ndcg": row[leg]})
    with (args.out / "models.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["chan", "chan_id", "mo_hinh", "chuong", "ho", "ndcg"])
        writer.writeheader()
        writer.writerows(long_rows)

    # Bảng ngoài-độ-chính-xác của nhánh giao thức: đánh đổi chính xác vs long-tail.
    tradeoff = []
    for source, label in (("hgn_none_base", "B không gộp"),
                          ("hgn_volume_base", "B gộp bộ"),
                          ("anl_base", "AniList B")):
        for name, row in data[source].items():
            if not isinstance(row, dict) or "ndcg" not in row:
                continue
            tradeoff.append({
                "chan": label, "mo_hinh": name,
                "ndcg": round(row["ndcg"], 4),
                "phu": round(row.get("coverage", 0.0), 4),
                "long_tail": round(row.get("tail") or 0.0, 4),
                "arp": round(row.get("arp") or 0.0, 4),
                "ndcg_it_luot": round(row.get("ndcg_tail") or 0.0, 4)})
    ladder = data["ladder"]
    for chan_label, blob in (("B không gộp", ladder),
                             ("AniList B", data["anl_ladder"])):
        for part in ("B_association_rules", "C_hybrid"):
            for name, row in blob.get(part, {}).get("results", {}).items():
                if row.get("ndcg") is None:
                    continue
                tradeoff.append({
                    "chan": chan_label, "mo_hinh": name,
                    "ndcg": round(row["ndcg"], 4),
                    "phu": round(row.get("coverage") or 0.0, 4),
                    "long_tail": round(row.get("tail") or 0.0, 4),
                    "arp": round(row.get("arp") or 0.0, 4),
                    "ndcg_it_luot": 0.0})
    # Sáu mô hình nơ-ron cũng đo trên cùng cấu hình `hgn_none`, nên chúng
    # thuộc bảng này. Thiếu chúng thì chỗ đánh đổi thấy nhất bị khuyết:
    # CL4SRec = SASRec + tương phản cho NDCG thấp hơn 10% nhưng long-tail
    # 0,675 -> 0,945. Một ô rỗng là lượt chạy cũ chưa được truyền bảng
    # độ phổ biến; để rỗng chứ không ghi 0, vì 0 là một giá trị có nghĩa.
    def clean(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        return round(number, 4) if number == number else ""

    neural_rows = [("B không gộp", name, row)
                   for name, row in data["neural"].items()]
    neural_rows += [("AniList B", name, row)
                    for name, row in data["anl_neural"].items()]
    for chan_label, name, row in neural_rows:
        if not isinstance(row, dict) or "ndcg" not in row:
            continue
        tradeoff.append({
            "chan": chan_label, "mo_hinh": name,
            "ndcg": clean(row["ndcg"]), "phu": clean(row.get("coverage")),
            "long_tail": clean(row.get("tail")), "arp": clean(row.get("arp")),
            "ndcg_it_luot": ""})

    with (args.out / "tradeoff.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "chan", "mo_hinh", "ndcg", "phu", "long_tail", "arp", "ndcg_it_luot"])
        writer.writeheader()
        writer.writerows(tradeoff)

    # Sáu cấu hình có cần sáu không? Câu trả lời phải ĐO, không phải khẳng
    # định: với mỗi cặp cấu hình, lấy các mô hình chạy ở CẢ HAI rồi xem thứ
    # hạng có đổi không (Spearman) và mức lệch lớn nhất là bao nhiêu. Cặp nào
    # cho thứ hạng y hệt thì cấu hình thứ hai là ĐỐI CHỨNG đã trả lời xong,
    # không phải một kết quả song song phải mang theo khắp mọi bảng.
    def spearman(xs, ys):
        order_x = sorted(range(len(xs)), key=lambda i: -xs[i])
        order_y = sorted(range(len(ys)), key=lambda i: -ys[i])
        rank_x = {idx: pos for pos, idx in enumerate(order_x)}
        rank_y = {idx: pos for pos, idx in enumerate(order_y)}
        n = len(xs)
        if n < 3:
            return None
        gap = sum((rank_x[i] - rank_y[i]) ** 2 for i in range(n))
        return 1.0 - 6.0 * gap / (n * (n * n - 1))

    by_chan = {}
    for row in long_rows:
        by_chan.setdefault(row["chan_id"], {})[row["mo_hinh"]] = row["ndcg"]
    agree = []
    for first, second in (("gr_matched", "gr_full"),
                          ("hgn_none", "hgn_volume"),
                          ("anilist", "gr_matched"),
                          ("anilist", "gr_full"),
                          ("hgn_none", "gr_matched")):
        left, right = by_chan.get(first, {}), by_chan.get(second, {})
        shared = sorted(set(left) & set(right))
        if len(shared) < 3:
            continue
        xs = [left[name] for name in shared]
        ys = [right[name] for name in shared]
        # Bo cac moc gan 0 ra khoi phep tinh do lech: Random di tu 0,0001 sang
        # 0,0002 la "+100%" nhung khong noi gi ve cau hinh — ty le tren mot so
        # chia gan 0 chi do nhieu, khong do hieu ung.
        deltas = sorted(abs(y - x) / x for x, y in zip(xs, ys) if x >= 0.01)
        agree.append({
            "a": first, "b": second, "n_chung": len(shared),
            "spearman": round(spearman(xs, ys), 3),
            "lech_lon_nhat": round(100 * deltas[-1], 1) if deltas else "",
            "lech_trung_vi": (round(100 * deltas[len(deltas) // 2], 1)
                              if deltas else ""),
            "dau_bang_a": max(shared, key=lambda name: left[name]),
            "dau_bang_b": max(shared, key=lambda name: right[name])})
    with (args.out / "agree.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "a", "b", "n_chung", "spearman", "lech_lon_nhat",
            "lech_trung_vi", "dau_bang_a", "dau_bang_b"])
        writer.writeheader()
        writer.writerows(agree)

    # MAE/RMSE
    rating = []
    for name, row in ladder.get("A_rating_prediction", {}).get("results", {}).items():
        mae, rmse = row.get("mae"), row.get("rmse")
        if mae is None or mae != mae:
            continue
        rating.append({"mo_hinh": name, "mae": round(mae, 4),
                       "rmse": round(rmse, 4)})
    # SVD++ chạy bằng script riêng vì nó cần cùng phép chia mà không phải chạy
    # lại cả lưới MF; tệp riêng, cùng bảng.
    for name, row in data.get("svdpp", {}).items():
        rating.append({"mo_hinh": f"{name} (f={row['factors']}, "
                                  f"reg={row['reg']})",
                       "mae": round(row["mae"], 4),
                       "rmse": round(row["rmse"], 4)})
    onboarding = []
    # Hai bộ, cùng thí nghiệm. Thiếu AniList thì đường cong Goodreads trông
    # như một quy luật chung, trong khi nó đi NGƯỢC chiều trên bộ kia.
    for bo, source in (("Goodreads", "onboard"), ("AniList", "anl_onboard")):
      for name, row in data.get(source, {}).items():
        if name.startswith("cơ chế"):
            kind, picks = "cơ chế", None
            tail = name.split(":", 1)[1].strip()
        elif "lựa chọn" in name and "phổ biến" not in name:
            kind = "Item-KNN" if name.startswith("Item-KNN") else "Nội dung"
            picks = int(name.split("sau")[1].split("lựa")[0])
            tail = name
        else:
            kind, picks, tail = "mốc", None, name
        onboarding.append({
            "bo": bo,
            "nhom": kind, "so_lua_chon": "" if picks is None else picks,
            "ten": tail, "ndcg": round(row["ndcg"], 4),
            "recall": round(row["recall"], 4),
            "phu": round(row["coverage"], 4),
            "long_tail": (round(row["tail"], 4)
                          if row.get("tail") == row.get("tail") else "")})
    with (args.out / "onboarding.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "bo", "nhom", "so_lua_chon", "ten", "ndcg", "recall", "phu",
            "long_tail"])
        writer.writeheader()
        writer.writerows(onboarding)

    with (args.out / "rating.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mo_hinh", "mae", "rmse"])
        writer.writeheader()
        writer.writerows(rating)

    done = sum(1 for r in rows if r["trang_thai"] == "đã chạy")
    partial = sum(1 for r in rows if r["trang_thai"] == "một phần")
    print(f"{len(rows)} phương pháp: {done} đã chạy, {partial} một phần, "
          f"{len(rows) - done - partial} chưa chạy")
    print(f"{len(long_rows)} ô có số | {len(agree)} cặp cấu hình | {len(tradeoff)} dòng đánh đổi | "
          f"{len(rating)} dòng MAE/RMSE | "
          f"{len(onboarding)} dòng onboarding")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
