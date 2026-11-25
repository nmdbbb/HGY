"""Hai bảng nền cho trang kết quả: DỮ LIỆU và MỤC TIÊU.

    python scripts/export_standard.py --out <thư mục>

Hai script kia (`export_matrix.py`, `export_oldarm.py`) xuất *kết quả*. Script
này xuất phần đứng trước kết quả, và nó cố ý **không** chia phương pháp thành
"cũ" và "mới": bảng phương pháp xếp theo **họ mô hình**, mỗi dòng ghi thêm nó
thuộc chương nào của đề xuất, nhưng chương không còn là trục tổ chức. Xếp theo
chương thì hai tháp, cổng độ phổ biến và định tuyến bị nhét vào ba chương không
sinh ra cho chúng, còn xếp theo họ thì mọi phương pháp đứng ngang nhau.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

OBJECTIVES = [
    # Hai mục tiêu mang báo cáo. Hai mục còn lại là chặng đường
    # cũ: từ khi AniList có bản giao thức B, giao thức A không còn là
    # cách duy nhất để so hai bộ nữa, nên nó xuống phụ lục.
    ("Mục kế tiếp", "Trong 20% cuối lịch sử của chính người này, "
     "mục nào tới",
     "70/10/20 theo thời gian của TẪNG người",
     "NDCG@10, Recall@10, HitRate@10",
     "mục đã đọc VÀ mục cùng bộ truyện",
     "CẢ HAI bộ", "chính"),
    ("Dự đoán điểm", "Người này chấm truyện này mấy sao",
     "70/10/20 theo từng người, trên các dòng CÓ chấm điểm",
     "MAE, RMSE (thang 1–5)", "không loại gì",
     "Goodreads (78% dòng AniList không có điểm)", "chính"),
    ("Sắp đọc gì tiếp (giao thức A)",
     "Kỳ tới người này sẽ đọc gì",
     "Cắt theo MỘT mốc thời điểm chung cho mọi người",
     "NDCG@10, Recall@10", "mục đã đọc",
     "AniList, Goodreads khớp quy mô, Goodreads đầy đủ", "phụ lục"),
    ("Gu", "Người này thích KIỂU truyện nào",
     "Giữ ngẫu nhiên 20% mục của mỗi người, trung hoà thời gian",
     "NDCG@10, Recall@10", "mục đã đọc",
     "AniList — chỉ một bộ, nên không so ngang được", "phụ lục"),
]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    def read(path):
        path = args.root / path
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    curve = read("processed_gr/scale_curve.json")
    target = curve.get("target_anilist", {})
    rows = {row["users_sampled"]: row for row in curve.get("curve", [])}
    matched = rows.get(160000, {})
    full = max(rows) if rows else None
    full_row = rows.get(full, {})
    none = read("processed_hgn_none/protocol_report.json")
    volume = read("processed_hgn_volume/protocol_report.json")
    rating = read("processed_hgn_none/ladder_gaps.json").get(
        "A_rating_prediction", {})
    anl = read("processed_anilist_b/protocol_report.json")

    data_rows = [
        ("AniList", "giao thức A — mốc thời điểm chung",
         target.get("train_users"), target.get("train_items"),
         target.get("train_rows"), target.get("median_history"),
         target.get("pct_under_10"),
         "chính",
         "Danh sách đọc công khai, crawl qua GraphQL; tên người đọc băm trước "
         "khi lưu. Có trạng thái đọc/bỏ dở/tạm dừng và số chương đã đọc."),
        ("Goodreads", "giao thức A — khớp quy mô AniList",
         matched.get("train_users"), matched.get("train_items"),
         matched.get("train_rows"), matched.get("median_history"),
         matched.get("pct_under_10"),
         "chính",
         "UCSD Comics & Graphic. Lấy mẫu 160.000 người đọc — con số ĐO được "
         "để khớp AniList cả số tương tác (0,96×) lẫn cỡ danh mục (0,99×)."),
        ("Goodreads", "giao thức A — quy mô đầy đủ",
         full_row.get("train_users"), full_row.get("train_items"),
         full_row.get("train_rows"), full_row.get("median_history"),
         full_row.get("pct_under_10"),
         "đối chứng — quy mô",
         "Gấp 2,15× tổng dữ liệu nhưng giữ nguyên hình dạng từng người "
         "(35,7 so với 35,8 quan sát mỗi người)."),
        ("Goodreads", "giao thức B — không gộp bộ",
         none.get("train_users"), none.get("train_items"),
         none.get("train_rows"), none.get("hist_median"), None,
         "chính",
         "Dương là rating ≥ 4, lọc lõi 10/5. Mỗi quyển in là một mục."),
        ("Goodreads", "giao thức B — gộp tập thành bộ",
         volume.get("train_users"), volume.get("train_items"),
         volume.get("train_rows"), volume.get("hist_median"), None,
         "đối chứng — gộp bộ",
         "Gộp 3.190 bộ truyện, dồn 810.643 dòng. Đây là cấu hình đề xuất yêu "
         "cầu (\"Naruto Vol.1…72 thành một item\")."),
        ("AniList", "giao thức B — chia 70/10/20 theo từng người",
         anl.get("train_users"), anl.get("train_items"),
         anl.get("train_rows"), anl.get("hist_median"), None,
         "chính",
         "Cùng phép chia và cùng các script với giao "
         "thức B của Goodreads, nên hai bộ so được "
         "trực tiếp. Dương = nhãn đọc, không "
         "phải rating ≥ 4 (78% dòng AniList không có điểm)."),
        ("Goodreads", "dự đoán điểm — toàn bộ dòng có điểm",
         rating.get("users"), rating.get("items"), rating.get("rows_rated"),
         None, None,
         "bài toán khác",
         "Không lọc ngưỡng rating, vì lọc rating ≥ 4 làm cột điểm chỉ còn hai "
         "giá trị và bài toán dự đoán điểm biến mất."),
    ]
    with (args.out / "data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bo", "cau_hinh", "nguoi_doc", "truyen", "tuong_tac",
                         "lich_su_trung_vi", "pct_duoi_10", "vai_tro",
                         "ghi_chu"])
        for row in data_rows:
            writer.writerow(["" if value is None else value for value in row])

    # Hai bo du lieu, mo ta thang. Sau cau hinh o tren la BIEN THE tien xu ly
    # cua hai bo nay, khong phai sau nguon du lieu: bang do dong y
    # (`agree.csv`) cho thay bien the khong doi thu hang, nen bao cao chinh
    # chi dung hai dong duoi day.
    bo_rows = [
        ("AniList",
         "Danh sách đọc công khai, API GraphQL. Tìm người đọc từ bảng hoạt "
         "động của từng truyện — vẫn là mẫu người đọc TÍCH CỰC hơn ngẫu "
         "nhiên (7,6% có dưới 10 mục, so với ~40% ở Goodreads), nhưng độ "
         "lệch đã giảm so với lần crawl đầu (từng 3,6%) sau khi quét rộng "
         "ra 5.000 truyện/nước (từ 2.000) và sâu tới 8 trang hoạt động mỗi "
         "nhóm truyện — quét sâu hơn vẫn ra người mới nên 8 trang là trần "
         "đo được, không phải con số tự chọn. 37.282 người thô, còn "
         "33.754 sau lọc lõi.",
         "một bộ truyện (manga/manhwa/manhua)",
         anl.get("train_users"), anl.get("train_items"),
         anl.get("train_rows"), anl.get("hist_median"),
         "trạng thái đọc / đang đọc / bỏ "
         "dở / tạm dừng, số chương đã "
         "đọc, điểm 1–10 (tùy chọn)",
         "Tên người đọc băm trước khi "
         "lưu; không phát tán danh sách thô; "
         "giới hạn 30 yêu cầu/phút."),
        ("Goodreads",
         "UCSD Book Graph — phần Comics & Graphic (McAuley Lab). Bộ dựng "
         "sẵn, không phải tự crawl. Ở quy mô giao thức B (lọc lõi 10/5), "
         "lớn hơn AniList 1,7 lần về số người — nhưng ÍT hơn 2,7 lần về số "
         "tương tác, vì lịch sử mỗi người ở đây mỏng hơn nhiều (16 so với "
         "79 mục).",
         "một quyển đã in (Naruto Vol. 7 là một "
         "mục riêng)",
         none.get("train_users"), none.get("train_items"),
         none.get("train_rows"), none.get("hist_median"),
         "điểm 1–5 sao, kệ sách (shelves) người "
         "đọc tự đặt, ngày thêm",
         "Chỉ dùng học thuật, không phân phối "
         "lại, không thương mại; trích dẫn Wan & "
         "McAuley RecSys'18 và Wan, Misra, Nakashole & McAuley ACL'19."),
    ]
    with (args.out / "bo.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bo", "nguon", "don_vi", "nguoi_doc", "truyen",
                         "tuong_tac", "lich_su_trung_vi", "tin_hieu",
                         "dieu_khoan"])
        for row in bo_rows:
            writer.writerow(["" if value is None else value for value in row])
    print(f"bo.csv         {len(bo_rows)} dòng")

    with (args.out / "objectives.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["muc_tieu", "cau_hoi", "phep_chia", "chi_so",
                         "loai_khoi_ro_ung_vien", "do_tren", "vai_tro"])
        writer.writerows(OBJECTIVES)

    print(f"data.csv       {len(data_rows)} dòng")
    print(f"objectives.csv {len(OBJECTIVES)} dòng")
    if any(value is None for row in data_rows for value in row[2:5]):
        print("  CẢNH BÁO: có ô quy mô rỗng — kiểm lại tệp nguồn")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
