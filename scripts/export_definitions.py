"""Hai bảng ĐỊNH NGHĨA, viết theo MỘT khuôn cho mọi dòng.

    python scripts/export_definitions.py --out <thư mục>

Đây là bảng *định nghĩa*, không phải bảng kết quả: nó nói mỗi phương pháp
dùng tín hiệu gì và chấm điểm bằng công thức nào, và mỗi chỉ số tính thế
nào, đọc thế nào. Kết quả nằm ở `export_matrix.py` và `export_oldarm.py`.

Khuôn phương pháp: tên · dùng tín hiệu gì · chấm điểm bằng công thức nào ·
tham số và cách chọn · chương đề xuất · trạng thái.
Khuôn chỉ số: tên · công thức · miền giá trị · đo thế nào · đọc thế nào.
"""
import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument("--out", required=True, type=Path)
OUT = _parser.parse_args().out
OUT.mkdir(parents=True, exist_ok=True)

# ho | phương pháp | tín hiệu | công thức chấm điểm | tham số & cách chọn
#    | chương | trạng thái
M = [
 ("Mốc", "Random", "không dùng gì",
  "s(u,i) = ngẫu nhiên đều", "không có", "5.1", "đã chạy"),
 ("Mốc", "MostPopular", "số tương tác",
  "s(u,i) = c_i (số người đọc i trong train)", "không có", "5.1", "đã chạy"),
 ("Mốc", "Top-rated Bayesian", "điểm chấm",
  "s(u,i) = (v_i·r̄_i + m·r̄) / (v_i + m), v_i = số lượt chấm",
  "m = trung vị v_i, không dò", "5.1", "đã chạy"),
 ("Mốc", "Đang nổi 60 ngày", "số tương tác có mốc thời gian",
  "s(u,i) = log(1 + c_i^(60 ngày cuối trước mốc chia))", "cửa sổ 60 ngày, cố định",
  "ngoài đề xuất", "đã chạy"),

 ("Láng giềng", "User-KNN", "ma trận người × mục nhị phân",
  "s(u,i) = Σ_{v∈N_k(u)} sim(u,v)·r_vi, sim = cosine",
  "k = 200, mặc định", "5.2", "đã chạy"),
 ("Láng giềng", "Item-KNN cosine", "ma trận người × mục nhị phân",
  "s(u,i) = Σ_{j∈N_k(i)∩I_u} sim(i,j); sim = cos(i,j)·n_ij/(n_ij+λ)",
  "k ∈ {50,200,500}, λ ∈ {0,20} — val chọn k=500, λ=0", "5.2", "đã chạy"),
 ("Láng giềng", "Item-KNN Pearson", "ma trận người × mục nhị phân",
  "sim(i,j) = (n_ij/N − p_i p_j) / √(p_i(1−p_i)·p_j(1−p_j)), p_i = c_i/N",
  "k ∈ {200,500}, bỏ cặp n_ij < 1 — val chọn", "5.2", "đã chạy"),
 ("Láng giềng", "Session-KNN", "L mục gần nhất, có thứ tự",
  "s(u,i) = Σ_{t=1..L} γ^(L−t)·sim(i_t, i), sim của Item-KNN đã chọn",
  "L ∈ {1,2,3,5,10,20}, γ ∈ {1;0,9;0,7;0,5} — val chọn", "5.6", "đã chạy"),
 ("Láng giềng", "Luật kết hợp", "ma trận người × mục nhị phân",
  "s(u,i) = max_{a∈I_u} conf(a→i), conf = n_ai/c_a, giữ n_ai ≥ 5",
  "ngưỡng support = 5 người đọc chung, cố định", "5.6", "đã chạy"),

 ("Nhân tố ẩn", "iALS / ALS", "ma trận người × mục nhị phân có trọng số tin cậy",
  "s(u,i) = p_u·q_i, giải luân phiên min Σ c_ui(r_ui − p_u·q_i)² + λ‖·‖²",
  "64 nhân tố, 15 vòng, λ=0,05, α=20 — mặc định", "5.3", "đã chạy"),
 ("Nhân tố ẩn", "BPR-MF", "cặp (mục đã đọc, mục chưa đọc)",
  "max Σ log σ(p_u·q_i − p_u·q_j); biến thể có thêm b_i − b_j",
  "64 nhân tố, 30 và 60 epoch; chệch theo item bật/tắt", "5.3", "đã chạy"),
 ("Nhân tố ẩn", "MF có chệch (rating)", "điểm chấm 1–5",
  "r̂ = μ + b_u + b_i + p_u·q_i, SGD theo lô chia theo số lần trong lô",
  "f ∈ {16,32,64}, λ ∈ {0,02;0,05}, 20 epoch — val chọn f=64, λ=0,02",
  "5.2", "đã chạy"),
 ("Nhân tố ẩn", "SVD++", "điểm chấm cộng tập mục có chấm",
  "r̂ = μ + b_u + b_i + q_i·(p_u + |N(u)|^(−1/2) Σ_{j∈N(u)} y_j)",
  "f ∈ {32,64}, reg ∈ {0,02;0,05}, 15 epoch — val chọn f=32, reg=0,05",
  "5.2", "đã chạy"),
 ("Nhân tố ẩn", "NeuMF", "ma trận người × mục nhị phân",
  "s = w·[p_u^G ⊙ q_i^G ; MLP([p_u^M, q_i^M])], hai nhánh hai embedding riêng",
  "GMF 32 chiều, MLP 64-32-16, 1 mẫu âm, BCE; ngân sách 120 epoch, val chọn 64 rồi dừng sớm",
  "5.3", "đã chạy"),
 ("Nhân tố ẩn", "LightGCN", "đồ thị hai phía người–mục",
  "E^(k+1) = D^(−1/2) A D^(−1/2) E^(k); e = mean_k E^(k); s = e_u·e_i",
  "64 chiều, 3 tầng, BPR, L2 1e-4; ngân sách 100 epoch (lô 16.384, lr 3e-3), val chọn 90 — lượt 8 epoch cũ chỉ cho 0,0473",
  "5.3", "đã chạy"),

 ("Nội dung", "TF-IDF shelves + tác giả", "thẻ cộng đồng và tên tác giả",
  "s(u,i) = cos(h_u, c_i), h_u = Σ_t 0,5^(t/20)·c_{i_t} / Σ trọng số",
  "min_df = 20 (đo từ phân bố: 47,7% thẻ chỉ xuất hiện 1 lần)", "5.4",
  "đã chạy"),
 ("Nội dung", "Nội dung LSA chiều thấp", "cùng TF-IDF, rút chiều",
  "c_i ← SVD_d(TF-IDF), s(u,i) = cos(h_u, c_i)",
  "d ∈ {64,128} — val chọn; thay cho sentence embedding, KHÔNG phải nó",
  "5.4", "đã chạy"),
 ("Nội dung", "Bốn mặt theme/creator/pace/art", "thẻ, tác giả, nhịp đọc, ảnh bìa",
  "c_i = [theme 128 ‖ creator 64 ‖ pace 13 ‖ art 15] = 220 chiều",
  "từng khối chuẩn hoá riêng rồi nối; không dò", "5.4", "đã chạy"),
 ("Nội dung", "Sentence embedding đa ngôn ngữ", "mô tả truyện",
  "c_i = encoder(description)", "chưa chạy — không tải được mô hình trong môi "
  "trường này", "5.4", "chưa chạy"),

 ("Hai tháp", "Hai tháp chỉ gu", "ID mục + nội dung bốn mặt",
  "s(u,i) = f(h_u)·g([id_i ‖ c_i]); sampled softmax có hiệu chỉnh logQ",
  "64 chiều, 15–30 epoch; logQ bật/tắt là một bậc ablation", "5.7", "đã chạy"),
 ("Hai tháp", "Hai tháp gu + đầu thời điểm", "thêm 5 cột thời điểm",
  "g([id_i ‖ c_i ‖ t_i]), t = [log pop, log pop 60 ngày, tuổi, đang ra, đã xong]",
  "cùng cấu hình; 2 cột cuối bẹt thành 0 trên Goodreads", "5.7", "đã chạy"),
 ("Hai tháp", "Cổng độ phổ biến hai phần", "thêm một cổng theo người đọc",
  "s = s_gu + g_u·s_pop, g_u = base(log(1+n_u)) + δ_u",
  "hai phần tách nền theo độ dài lịch sử khỏi phần riêng", "ngoài đề xuất",
  "đã chạy"),
 ("Hai tháp", "DropoutNet", "như trên, bỏ ngẫu nhiên đặc trưng phổ biến",
  "khi huấn luyện, với xác suất 0,3 đặt đặc trưng phổ biến = 0",
  "p = 0,3, cố định", "5.8", "đã chạy"),
 ("Hai tháp", "NRMS-style", "nội dung + attention trên lịch sử",
  "h_u = AddAttn(MHSA(c_{i_1..i_L})), s = h_u·enc(c_i)",
  "64 chiều, 2 đầu, nội dung 128 chiều; ngân sách 120 epoch, val chọn 52 rồi dừng sớm; KHÔNG có attention mức từ",
  "5.7", "đã chạy"),
 ("Hai tháp", "CL4SRec", "chuỗi mục + hai biến thể tăng cường",
  "L = L_SASRec + 0,1·NT-Xent(view1, view2); view ∈ {cắt, che, đảo}",
  "λ = 0,1, nhiệt độ 1,0; ngân sách 120 epoch, val chọn 72 rồi dừng sớm — ĐÃ BẸT",
  "5.8", "đã chạy"),

 ("Chuỗi", "GRU4Rec", "L mục gần nhất, có thứ tự",
  "h = GRU(e_{i_1..i_L}), s(u,i) = W h · e_i, mất mát BPR",
  "64 chiều, 1 tầng, dropout 0,2, L = 50; mất mát BPR ở MỌI vị trí; ngân sách 400 epoch, val chọn 320 rồi dừng sớm — ĐÃ BẸT", "5.6",
  "đã chạy"),
 ("Chuỗi", "SASRec", "L mục gần nhất, có thứ tự",
  "self-attention nhân quả 2 khối; mất mát BCE ở MỌI vị trí của chuỗi",
  "64 chiều, 2 khối, 2 đầu, L = 50; ngân sách 120 epoch, val chọn 20 rồi dừng sớm — ĐÃ BẸT",
  "5.6", "đã chạy"),

 ("Tri thức", "Lọc theo ràng buộc", "thuộc tính mục + lịch sử người đọc",
  "s(u,i) = log(1+c_i) nếu i thoả 4 ràng buộc, −∞ nếu không; tự nới khi rổ rỗng",
  "4 ràng buộc: nước, định dạng, nội dung người lớn, số tập ≤ 10",
  "5.5", "đã chạy"),
 ("Tri thức", "Case-based", "thuộc tính mục + mục đọc gần nhất",
  "s(u,i) = cos(c_i, c_case) + 0,2·[cùng nước] + 0,1·[cùng định dạng] "
  "+ 0,1·exp(−|Δ log số tập|)",
  "trọng số đặt tay, chưa dò; case = mục gần nhất", "5.5", "đã chạy"),
 ("Tri thức", "Critiquing có vòng lặp", "phê bình của người dùng",
  "cập nhật trọng số thuộc tính sau mỗi lời phê bình",
  "chưa chạy — cần người dùng thật", "5.5", "chưa chạy"),
 ("Tri thức", "Onboarding k lựa chọn", "k mục đầu người đọc chọn",
  "s(u,i) = Σ_{j∈k mục} sim(i,j) — đúng Item-KNN nhưng chỉ k mục làm đầu vào",
  "k ∈ {1,2,3,5,10,20} quét thẳng; k=2 cho đỉnh và HƠN lịch sử đầy đủ",
  "5.5", "đã chạy"),

 ("Lai ghép", "Định tuyến theo lịch sử", "hai mô hình đã huấn luyện",
  "s = s_cổng nếu |I_u| > 0, s = s_không-cổng nếu |I_u| = 0",
  "ngưỡng = có hay không có lịch sử; lập được bằng kiểm ghép cặp 4 seed",
  "5.9", "đã chạy"),
 ("Lai ghép", "Song song có trọng số", "ba mô hình đã huấn luyện",
  "s = Σ_m w_m · z(s_m), z = chuẩn hoá z-score TRÊN TỪNG NGƯỜI",
  "w trên lưới đơn hình bước 0,25 — val chọn (0,75; 0,25; 0)", "5.9",
  "đã chạy"),
 ("Lai ghép", "Hai tầng sinh ứng viên rồi xếp lại", "ALS + nội dung bốn mặt",
  "tầng 1: top-400 của ALS; tầng 2: xếp lại bằng điểm nội dung",
  "400 ứng viên, cố định", "5.9", "đã chạy"),
 ("Lai ghép", "Content-boosted CF", "nội dung sinh pseudo-rating cho CF",
  "bơm M mục điểm nội dung cao nhất mỗi người với trọng số 0,3 rồi chạy ALS",
  "M ∈ {10,20} — val chọn; trọng số 0,3 cố định", "5.9", "đã chạy"),
 ("Lai ghép", "Đa dạng hoá MMR", "danh sách của một mô hình khác",
  "chọn lần lượt argmax[λ·rel(i) − (1−λ)·max_{j∈S} sim(i,j)] trên top-50",
  "λ ∈ {1;0,9;0,7;0,5} — val chọn 0,9", "5.9", "đã chạy"),
]

# chỉ số | công thức | miền | đo thế nào | đọc thế nào
K = [
 ("Precision@10", "|top10 ∩ đáp án| / 10", "0–1",
  "Trung bình trên 2.000 người đọc lấy mẫu (1.500 ở nhánh A), chấm toàn danh mục",
  "Trong 10 gợi ý có bao nhiêu đúng. Bị chặn trên bởi số đáp án: người chỉ có 2 "
  "mục trong test thì precision tối đa là 0,2."),
 ("Recall@10", "|top10 ∩ đáp án| / |đáp án|", "0–1", "như trên",
  "Bắt được bao nhiêu phần đáp án. Người có 50 mục trong test thì 10 gợi ý tối "
  "đa bắt được 0,2 — nên recall thấp không hẳn là mô hình kém."),
 ("F1@10", "2·P·R / (P+R)", "0–1", "suy từ hai cột trên",
  "Gộp hai cột trên; trên dữ liệu có số đáp án lệch nhau nhiều thì nó kế thừa "
  "cả hai trần ở trên."),
 ("HitRate@10", "[|top10 ∩ đáp án| > 0]", "0–1", "như trên",
  "Tỉ lệ người đọc có ÍT NHẤT một gợi ý đúng. Luôn cao hơn precision, và là "
  "chỉ số gần nhất với \"người dùng có thấy gì dùng được không\"."),
 ("NDCG@10", "DCG/IDCG, DCG = Σ_t rel_t / log2(t+1)", "0–1", "như trên",
  "Giống recall nhưng tính cả VỊ TRÍ: đúng ở hạng 1 giá trị hơn đúng ở hạng 10. "
  "Đây là chỉ số chính của mọi bảng trong trang này."),
 ("MRR", "1 / hạng của mục đúng đầu tiên", "0–1", "như trên",
  "Chỉ quan tâm mục đúng đầu tiên nằm ở đâu. Dùng cho giao diện chỉ hiện vài "
  "gợi ý."),
 ("MAP", "trung bình của precision tại mỗi vị trí đúng", "0–1", "như trên",
  "Nghiêm hơn NDCG với lỗi ở đầu danh sách."),
 ("MAE", "mean |r − r̂|", "0–4 (thang 1–5)",
  "Trên 849.905 dòng test của tiền xử lý rating đầy đủ; dự đoán kẹp về [1,5]",
  "Lệch trung bình mấy sao. 0,60 nghĩa là trung bình đoán lệch hơn nửa sao."),
 ("RMSE", "√(mean (r − r̂)²)", "0–4 (thang 1–5)", "như MAE",
  "Như MAE nhưng phạt nặng lỗi lớn. RMSE luôn ≥ MAE; khoảng cách giữa hai cột "
  "cho biết lỗi có tập trung ở vài trường hợp tệ hay rải đều."),
 ("Độ phủ danh mục", "|các mục từng xuất hiện trong top-10| / |danh mục|",
  "0–1", "Gộp mọi danh sách của 2.000 người đọc được chấm",
  "Hệ chỉ gợi ý 1% danh mục thì 99% truyện không bao giờ được ai thấy. Độ chính "
  "xác cao mà độ phủ 0,001 là một bảng phổ biến đội lốt mô hình."),
 ("Độ mới lạ", "mean −log2(c_i / N)", "0–log2(N)", "như trên",
  "Trung bình độ \"ít ai đọc\" của những gì được gợi ý. Cao hơn nghĩa là đi xa "
  "khỏi những truyện ai cũng biết."),
 ("Đa dạng trong danh sách", "1 − mean cos(c_i, c_j) trên các cặp trong top-10",
  "0–1", "Cần vectơ nội dung; có ở hai tháp và ở hệ xếp lại",
  "Mười truyện giống nhau thì đúng mười lần cũng chỉ là một gợi ý."),
 ("ARP", "mean phân vị phổ biến của mục được gợi ý", "0–1", "như trên",
  "0,9995 nghĩa là gợi ý nằm ở đúng 0,05% đầu bảng — dấu hiệu mô hình đã sập "
  "về bảng phổ biến. Đây là chỉ số bắt BPR-MF có chệch sập trước khi có lý thuyết."),
 ("Tỉ lệ long-tail", "phần gợi ý nằm ngoài 1% truyện nhiều lượt nhất", "0–1",
  "như trên", "Đọc cùng ARP. Hai chỉ số này cùng nói một chuyện ở hai thang."),
 ("Độ trễ mỗi lần gợi ý", "phân vị 95 của thời gian một lần gọi", "ms",
  "800 người đọc, qua đúng lớp phục vụ mà API dùng",
  "8,74 ms p95 nghĩa là hệ chạy được thật, không chỉ chạy được trong notebook."),
]

with (OUT / "methods.csv").open("w", newline="", encoding="utf-8") as handle:
    w = csv.writer(handle)
    w.writerow(["ho", "phuong_phap", "tin_hieu", "cong_thuc", "tham_so",
                "chuong_de_xuat", "trang_thai"])
    w.writerows(M)

with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    w = csv.writer(handle)
    w.writerow(["chi_so", "cong_thuc", "mien", "do_the_nao", "doc_the_nao"])
    w.writerows(K)

print(f"methods.csv  {len(M)} dòng, 7 cột")
print(f"metrics.csv  {len(K)} dòng, 5 cột")
for status in ("đã chạy", "đang chạy", "chưa chạy", "một phần"):
    print(f"  {status:12} {sum(1 for r in M if r[6] == status)}")
