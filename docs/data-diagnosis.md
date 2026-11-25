# Chẩn đoán dữ liệu — vì sao sáu dòng ablation phẳng

Đo ngày 2026-11-06 trên dữ liệu thật. Chạy lại: `python scripts/diagnose_data.py`.
Các mục C–G lấy mẫu (800 truyện, 1.500 người đọc, seed 0), nên số dao động nhẹ.

Câu hỏi: vì sao từ CF tới đủ bốn mặt, NDCG@10 chỉ đi từ 0,1072 tới 0,1072?
Câu trả lời không phải "đặc trưng nội dung vô dụng". Dữ liệu có ba tính chất
mà thiết kế hiện tại không tính tới.

## 1. Thứ đang được đo là "sắp đọc gì trong 10 tuần tới", và nó do thời gian quyết định

Tập test là 10 tuần cuối (26/8 – 6/11/2026).

| | train | test |
|---|---|---|
| `CURRENT` (vừa bắt đầu đọc) | 60,8% | **82,4%** |
| Truyện đang ra (`RELEASING`) | 42,9% | **63,5%** |
| Truyện ra từ 2025 trở đi | 8,1% | **26,1%** |
| Lượt đọc KR | 34,3% | **48,7%** |

Truyện tranh là ấn phẩm nhiều kỳ: người ta đọc theo nhịp phát hành. Bên trong
top-400 ứng viên của ALS, tín hiệu phân biệt truyện trúng tốt nhất là tín hiệu
thời gian, không phải gu:

| Điểm (AUC bên trong ứng viên, 0,5 = vô dụng) | AUC |
|---|---|
| Độ phổ biến 60 ngày trước mốc cắt | **0,664** |
| Đang ra | 0,630 |
| Năm ra | 0,628 |
| theme | 0,600 |
| pace | 0,594 |
| Độ phổ biến toàn thời gian | 0,576 |
| art | 0,559 |
| **ALS** | **0,543** |
| creator | 0,503 |

Mọi cấu hình trong ablation đều mù thời gian: ALS học trên toàn bộ lịch sử từ
2013, còn các mặt nội dung là vector tĩnh. Chúng cùng chạm một trần nên các
dòng phẳng như nhau.

Thêm một trần cứng nữa: chỉ **35,6%** truyện trong test lọt vào top-400 ứng
viên ALS. 64% còn lại, xếp lại thế nào cũng không với tới.

## 2. Tín hiệu nội dung có thật, bổ sung cho ALS, nhưng trùng với tín hiệu thời gian

Láng giềng gần nhất trong từng mặt có được đọc cùng nhau không? Lift =
P(đọc j | đọc i) / P(đọc j):

| Mặt | lift |
|---|---|
| ALS | 32,0 |
| theme | 15,2 |
| creator | 9,3 |
| pace | 8,7 |
| art | 5,0 |
| ngẫu nhiên | 2,6 |

Vậy các mặt mang thông tin. Chúng có cộng được với ALS không? Hồi quy
logistic trên ứng viên, kiểm chéo theo người đọc:

| Tín hiệu | AUC |
|---|---|
| ALS | 0,543 |
| ALS + 4 mặt nội dung | 0,628 (**+0,085**) |
| ALS + thời gian | 0,690 |
| ALS + thời gian + 4 mặt | 0,697 (**+0,007**) |

Đây là câu trả lời cho "1+1 có bằng 2 không": không. Nội dung gần như không
tương quan với ALS (r = −0,04), nên nó *có* thêm được vào ALS. Nhưng phần nó
thêm phần lớn là thời gian đội lốt: `pace` chứa tuổi truyện và đã-hoàn-chưa,
`theme` mang các tag của từng thời kỳ. Khi đưa thời gian vào trực tiếp, nội
dung chỉ còn +0,007.

Hai điều suy ra:
- Ablation không thấy nội dung giúp được gì dù logistic thấy +0,085. Nghĩa là
  cách trộn trong `hybrid.py` (z-score rồi trung bình có trọng số cố định)
  đang làm mất tín hiệu này. AUC đo trên cả 400 ứng viên, NDCG chỉ nhìn top-10,
  nên cần đo lại bằng NDCG trước khi kết luận chắc.
- `creator`: lift 9,3 ở cấp truyện nhưng AUC 0,503 trong ứng viên. Truyện cùng
  tác giả đúng là được đọc cùng nhau, nhưng người đọc *đã đọc rồi*, và truyện
  đã đọc bị loại khỏi ứng viên. Mặt này hợp với giải thích hơn là xếp hạng.

## 3. Người đọc ăn tạp, nhưng đó không phải nguyên nhân chính

Gu chỉ chụm hơn tập ngẫu nhiên cùng độ phổ biến 1,2–1,6 lần, ở mọi mặt. Trung
vị một người có 179 mục rải trên cả ba nước, nên một vector tâm bị loãng. Chia
lịch sử thành 3 cụm thì độ chụm theme tăng từ 0,500 lên 0,607. Nhưng truyện
trong test vẫn gần *tâm* (+0,199 so với ngẫu nhiên) hơn là gần *truyện gần nhất
trong lịch sử* (+0,157). Ăn tạp là có thật, nhưng chia đa sở thích không phải
đòn bẩy lớn.

## Các tính chất dữ liệu khác đo được trên đường

- 13,5% mục có `created_at = 0`, phần lớn là mục cũ (AniList thêm trường này
  về sau). Các mục này rơi về 1970 và luôn vào train. Không làm hỏng phép
  chia, nhưng train cũ hơn thực tế.
- Năm `created_at`: 2026 có 782.633 mục, hơn cả 2024 và 2025 cộng lại. Lý do
  là người đọc được tuyển qua *hoạt động gần đây*, nên mẫu nghiêng về người
  đang đọc tích cực bây giờ.
- Chỉ 7,3% mục nằm trong lô nhập hàng loạt (≥10 mục cùng một giây), nên nhập
  danh sách hàng loạt không phải vấn đề đáng kể.

## Hệ quả cho thiết kế

Gu (nội dung, ALS) và thời điểm (đang ra, mới ra, đang nổi) là hai trục riêng.
Với truyện nhiều kỳ, trục thời điểm mạnh hơn trong cửa sổ 10 tuần. Thiết kế
hiện tại chỉ có trục gu. Việc tiếp theo là quyết định đo cái gì, rồi mới chọn
mô hình:
- Đo **"gu"**: chia theo người đọc hoặc ngẫu nhiên trong lịch sử, để thời gian
  không lấn át.
- Đo **"sắp đọc gì"** (đúng với cách người ta đọc truyện nhiều kỳ): đưa thời
  điểm vào làm một trục chính, và lấy ứng viên từ cả nguồn "đang nổi / mới ra",
  không chỉ từ ALS.
