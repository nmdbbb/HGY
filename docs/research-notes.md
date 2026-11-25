# Ghi chú nghiên cứu — đối chiếu văn liệu và phân tích sâu

Ngày 2026-11-06, trong lúc mô hình hai tháp đang huấn luyện. Phần phân tích
chạy lại bằng `python scripts/analyze_deeper.py` (mục H–K, nối tiếp C–G trong
`scripts/diagnose_data.py`). Các mục lấy mẫu (1.500 truy vấn, 400–2.500 người
đọc, seed 0) nên số dao động nhẹ.

---

## 1. Đối chiếu văn liệu: thiết kế hiện tại đứng ở đâu

**Bỏ vector ID ngẫu nhiên lúc huấn luyện** — đúng là phương pháp đã công bố:
DropoutNet ([Volkovs, Yu & Poutanen, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/file/dbd22ba3bd0df8f385bdac3e9f8be207-Paper.pdf))
coi cold-start là bài toán thiếu dữ liệu và áp dropout lên *đầu vào* để mạng
học cách chạy khi thiếu vế preference. Thiết kế của ta trùng ý tưởng này
(bỏ `e_id` 30% để `P·c_i` tự gánh), nên khi viết báo cáo phải trích dẫn nó,
không được trình bày như sáng kiến riêng.

**Mẫu âm trong lô và hiệu chỉnh logQ** — mẫu âm trong lô được rút theo phân
phối lượt đọc, chuẩn công nghiệp là hiệu chỉnh logQ bằng importance sampling
(Yi et al. 2019; dùng ở Google, Pinterest, ByteDance theo
[Khrylchenko et al., RecSys 2025](https://arxiv.org/abs/2507.09331) — bài này
còn chỉ ra chính công thức logQ chuẩn có một lỗi hệ thống ở vế mẫu dương).
Bản hiện tại của ta **chưa** hiệu chỉnh logQ; hệ quả đã biết của việc đó là
truyện phổ biến bị phạt làm mẫu âm quá tay, tức mô hình bị đẩy khỏi truyện
phổ biến. Với mục tiêu "gu" thì đó là hướng lệch chấp nhận được (độ nổi đã có
đầu thời điểm lo), nhưng là một hạn chế phải ghi trong báo cáo và là nút vặn
nếu kết quả tháp gu kém.

**Cách chia dữ liệu đánh giá** — văn liệu đánh giá gần đây xác nhận đúng vấn
đề ta gặp: chia không theo mốc thời gian toàn cục gây rò rỉ và làm baseline
phổ biến "được" dùng thông tin tương lai
([Sun, 2023](https://arxiv.org/abs/2210.04149));
[Time to Split, RecSys 2025](https://arxiv.org/abs/2507.16289) khuyến nghị
global temporal split và chỉ ra leave-one-out làm méo cả thứ hạng giữa các
mô hình; độ phổ biến phải neo vào một mốc thời gian (top tuần trước, tháng
trước — [góc nhìn task-centric](https://arxiv.org/abs/2503.21188)). Việc ta
tách hai mục tiêu temporal/taste và thêm baseline "đang nổi 60 ngày" nằm đúng
hướng văn liệu này.

**Ngành webtoon thực tế** — Naver Webtoon đổi xếp mặc định từ "phổ biến" sang
AI Curator từ 6/2023; báo cáo của Korea University (do công ty công bố — số
liệu nguồn công ty, đọc dè chừng) nói CTR +59,5%, lượt xem +67,5%, thanh toán
+62,7%, và *giảm tập trung vào truyện nổi*
([Asia Economy, 12/2024](https://www.asiae.co.kr/en/article/2024121009240191746),
[ANN](https://www.animenewsnetwork.com/news/2024-12-26/research-report-finds-naver-webtoon-ai-recommendation-system-boosts-diversity/.219310)).
Không tìm thấy bài peer-review nào mô hình hoá tiêu thụ truyện nhiều kỳ
theo từng chương trên dữ liệu Naver/Kakao — tức là phần "nhịp phát hành chi
phối cửa sổ 10 tuần" ta đo được không có số công bố để đối chiếu trực tiếp.

---

## 2. H — 11,8× của nét vẽ: kiểu vẽ hay cùng bộ truyện?

Tách số lần trúng cùng-hoạ-sĩ theo "cùng bộ" (tên chung ≥ nửa số từ) và
"khác bộ":

| | precision@10 | |
|---|---|---|
| Cùng hoạ sĩ, mọi loại | 0,0033 | 11,8× ngẫu nhiên |
| — trong đó cùng bộ truyện | 0,0011 | 35% số lần trúng |
| — khác bộ truyện | 0,0021 | **7,6×** ngẫu nhiên khác bộ (0,00028) |

**Kết luận:** một phần ba cái 11,8× đúng là nhận ra bìa cùng bộ, nhưng phần
khác-bộ vẫn 7,6× — đặc trưng nét vẽ nắm được *kiểu vẽ* thật, không chỉ
franchise. Ngưỡng bậc 3 (≥2×) vẫn đạt sau khi trừ nhiễu. Lưu ý trung thực:
đại diện "cùng bộ" bằng trùng tên là cận dưới (franchise đổi hẳn tên không
bắt được), nên 7,6× là cận trên của phần kiểu-vẽ-thuần.

## 3. I — Gu trôi rất chậm; trọng số 180 ngày đang trộn hai thứ

Cosine theme giữa các mục của cùng một người, theo khoảng cách thời gian:

| Cách nhau | cosine | so với cùng tháng |
|---|---|---|
| 0 tháng | 0,2580 | 100% |
| 3 tháng | 0,2469 | 96% |
| 12 tháng | 0,2402 | 93% |
| 24 tháng | 0,2289 | 89% |

**Kết luận:** gu chủ đề gần như đứng yên — sau 2 năm mới giảm 11%, tức chu kỳ
bán rã của *gu* tính bằng nhiều năm. Vậy trọng số bán rã 180 ngày trong tháp
người đọc không phải đang bám "gu đổi", mà đang bám "đợt phát hành gần đây" —
một dạng tín hiệu thời điểm lọt vào tháp gu. Hệ quả: nên ablate thêm biến thể
bán rã dài/∞ cho tháp gu (nhất là ở nhánh taste); ghi rõ đây là nhận định
*sau khi* thấy số, chưa phải kết quả kiểm chứng.

## 4. J — ALS hiện tại phục vụ ai tốt nhất? Ngược với lo ngại

NDCG@10 của ALS (chấm toàn danh mục), nhóm người đọc theo nước chiếm ≥60%
lịch sử:

| Nhóm | NDCG@10 | số người |
|---|---|---|
| Đọc chủ yếu KR | **0,1318** | 774 |
| Đọc chủ yếu CN | 0,1091 | **18** — quá ít, đừng kết luận |
| Hỗn hợp | 0,0882 | 485 |
| Đọc chủ yếu JP | 0,0728 | 1.223 |

**Kết luận:** người đọc manhwa được phục vụ tốt gần *gấp đôi* người đọc manga
— ngược hẳn lo ngại "hệ dìm manhwa". Lý giải hợp với chẩn đoán: danh mục KR
nhỏ hơn và tập trung hơn, test lại nghiêng KR (48,7% lượt đọc), nên hành vi
người đọc manhwa dễ đoán hơn; còn manga có 86 nghìn bộ trải 50 năm, đuôi dài
khó hơn nhiều. Việc cần nói thật trong báo cáo: nhóm thuần-manhua chỉ có 18
người trong mẫu — dữ liệu hiện tại không đủ để tuyên bố gì về chất lượng gợi ý
cho người đọc manhua thuần.

## 5. K — DROPPED không dùng được làm mẫu âm theo chủ đề

| Đo (max-cosine theme) | giá trị |
|---|---|
| Truyện *sắp* đọc giống truyện *đã bỏ* | 0,5554 |
| Truyện *sắp* đọc giống truyện *đã thích* | 0,5684 |
| Truyện *đã bỏ* giống lịch sử người đó | **0,6974** |
| Truyện ngẫu nhiên giống lịch sử | 0,5699 |

**Kết luận:** người ta bỏ dở những truyện nằm *giữa* vùng gu của mình (0,6974
— giống lịch sử hơn cả mức ngẫu nhiên rất xa), và truyện họ đọc tiếp giống
truyện đã bỏ gần bằng truyện đã thích. Tức bỏ dở không phải vì lệch chủ đề mà
vì chất lượng/nhịp kể. Hai hệ quả:
- **Không** đưa DROPPED làm mẫu âm trong tháp gu — sẽ phạt đúng vùng gu của
  người đọc.
- Giữ nguyên cách dùng hiện tại: tỉ lệ bỏ và *bỏ ở chương nào* là thuộc tính
  của truyện (mặt `pace`) — tín hiệu chất lượng, không phải tín hiệu gu. Phân
  tích này xác nhận quyết định đó bằng số.

---

## Tổng hợp hệ quả thiết kế

| Phát hiện | Việc làm |
|---|---|
| H: khác-bộ còn 7,6× | Giữ mặt `art`; chưa cần Gram matrix; cập nhật tiêu chí bậc 3 |
| I: gu trôi ~% mỗi năm | Thêm ablation bán rã ∞ cho tháp gu (sau lần chạy hiện tại) |
| J: JP khó nhất, CN n=18 | Báo cáo theo nhóm người đọc; không tuyên bố gì về manhua thuần |
| K: bỏ dở ⊄ lệch gu | Không thêm mẫu âm DROPPED; giữ drop-rate trong `pace` |
| LogQ chưa hiệu chỉnh | ~~Nút vặn đầu tiên nếu tháp gu yếu~~ — **đoán sai chiều**, xem mục 6 |
| DropoutNet, Yi 2019 | Trích dẫn trong báo cáo — phương pháp trùng văn liệu |

---

## 6. logQ: đo được thì ngược với dự đoán của mục 1

Mục 1 viết "logQ chưa hiệu chỉnh … là nút vặn đầu tiên nếu tháp gu yếu", ngầm
hiểu bật logQ thì khá hơn. Ablation ngày 2026-11-07 (4 biến thể × 2 mục tiêu)
cho thấy **bật logQ làm hỏng đúng phần cold-start**:

| Nhánh chỉ-gu, temporal | có logQ | không logQ |
|---|---|---|
| NDCG@10 `all` | **0,0530** | 0,0498 |
| NDCG@10 `new_user` | **0,2427** | 0,1565 |
| NDCG@10 `new_item` (n=16.121) | 0,0007 | **0,0173** (25×) |
| phủ catalog | 0,048 | **0,158** |
| tỉ lệ CN trong top-10 | 0,06 | **0,11** |
| trùng tác giả | 0,016 | 0,030 |

Cơ chế: logQ trừ `log q_i` khỏi logit âm; truyện phổ biến có `q_i` lớn nên bị
trừ nhiều, tức **bị phạt nhẹ hơn**, nên mô hình nghiêng về đầu phân phối. Tắt
logQ thì mô hình bị đẩy ra đuôi — và truyện mới nằm đúng ở đuôi: chúng có
`train_counts = 0` nên vector ID bị zero, điểm hoàn toàn do `P·c_i` quyết định.
Vậy 0,0173 là con số duy nhất trong đồ án đo **riêng** giá trị của bốn mặt nội
dung, không lẫn CF và không lẫn độ nổi.

Hệ quả cho báo cáo: không trình bày logQ như "đúng/sai" mà như một **nút đánh
đổi head–tail**, và nêu rõ cấu hình nào dùng cho mục tiêu nào. Mục 1 giữ nguyên
làm bằng chứng rằng suy luận cơ chế nghe hợp lý vẫn có thể sai chiều.

Mục I (bán rã) cũng bị ablation hạ cấp: bỏ bán rã 180 ngày chỉ đổi 0,0514 →
0,0530 (+3%), không phải nguyên nhân của lần chạy đầu kém, dù phép đo "tổng
trọng số hiệu dụng 0,232" là đúng — phép gộp chuẩn hoá theo tổng trọng số nên
mô hình bù lại được.

---

## 7. `pace` đổi dấu khi bỏ tầng ứng viên

Thang ablation bốn mặt ở lát `new_item` (xem `docs/success-criteria.md`):

| Bậc | NDCG@10 | so bậc trước |
|---|---|---|
| chỉ `theme` | 0,0055 | — |
| + `creator` | 0,0073 | +33% |
| + `pace` | **0,0149** | **+104%** |
| + `art` | 0,0165 | +11% |

Điều đáng ghi không phải là thang tăng đơn điệu, mà là `pace` **đổi dấu**. Ở
bản hai tầng cũ nó là mặt duy nhất làm *giảm* điểm (`sparse_user` 0,0519 →
0,0485), và tiêu chí bậc 2 đã chốt sẵn "mặt âm thì bỏ và ghi lại là đã bỏ".
Nếu lúc đó tôi thi hành đúng câu đó thì đã bỏ mất mặt mạnh nhất cho truyện mới.

Cơ chế: ứng viên lấy từ top-400 của ALS, mà muốn vào được top-400 thì phải đã
có người đọc. Trong một tập toàn truyện đã có độ phổ biến, số chương / trạng
thái phát hành / tỉ lệ bỏ dở gần như không phân biệt được gì — tín hiệu nhịp
truyện bị *tập ứng viên* triệt tiêu, không phải bản thân nó vô dụng. Chấm toàn
danh mục thì nó là mặt hữu ích nhất, và điều đó hợp lý: nhịp truyện là thứ biết
được ngay khi truyện vừa ra chương đầu, trong khi `theme` cần tag cộng đồng tích
luỹ và `creator` cần tác giả đã có tác phẩm khác.

Bài học phương pháp, đáng ghi vào báo cáo hơn cả con số: **một ablation chỉ đo
được đóng góp của đặc trưng *trong phạm vi tập ứng viên của nó*.** Kết luận "mặt
X vô dụng" rút ra từ một hệ hai tầng là kết luận về cả tầng ứng viên, không
riêng về mặt X. Đây cũng là lý do sáu dòng ablation ban đầu phẳng như nhau.

---

## 8. Manhua bị dìm 6,3 lần, và thủ phạm không phải "thiếu đa dạng"

Bậc 4 cho thấy `feed` chỉ phủ 0,685 số nước trong lịch sử mỗi người. Tiêu chí
đã chốt sẵn fallback là "thêm ràng buộc đa dạng theo nước". Chạy
`scripts/diagnose_country.py` (400 người đọc ≥10 truyện, 91,2% đọc ≥2 nước)
trước khi thêm, và nó cho một kết luận khác hẳn:

| nước | lịch sử | test thật | top-10 | lệch |
|---|---|---|---|---|
| JP | 0,575 | 0,534 | 0,508 | −0,067 |
| KR | 0,362 | 0,392 | **0,482** | **+0,120** |
| CN | 0,063 | 0,074 | **0,010** | **−0,053** |

Hiệu chỉnh tổng thể (khoảng cách L1 giữa phân phối gợi ý và phân phối đọc, theo
nghĩa Steck 2018) là **0,220**, trong khi *chính hành vi đọc thật ở tập test*
cách lịch sử 0,182. Tức mô hình chỉ lệch hơn mức sàn 0,038 — **xét tổng thể thì
không có vấn đề đa dạng**, và một ràng buộc đa dạng chung sẽ phá đi 0,038 đó
bằng cách làm méo cả JP và KR.

Vấn đề nằm riêng ở CN: đọc 6,3% nhưng chỉ nhận 1,0%, **dìm 6,3 lần**, trong khi
KR được đẩy từ 36,2% lên 48,2%. Phân rã theo tỉ lệ đọc nước thiểu số:

| tỉ lệ đọc nước thiểu số | số người | tỉ lệ trong top-10 | có mặt ≥1 | hạng phổ biến (trung vị) |
|---|---|---|---|---|
| 0–5% | 211 | 0,020 | 0,14 | 4.340 |
| 5–15% | 93 | 0,069 | 0,30 | 3.509 |
| 15–30% | 46 | 0,224 | 0,57 | 2.443 |
| ≥30% | 15 | 0,420 | 0,73 | 2.988 |

Tỉ lệ nhận lại **tăng đều theo tỉ lệ đọc** — mô hình không bỏ qua nước thiểu
số, nó phục vụ theo tỉ lệ nhưng hụt một nhịp ở mọi mức. Và hạng phổ biến trung
vị của những truyện ấy là 2.400–4.300 trên 34.608, tức **top 12%** — không phải
đuôi dài, nên giả thuyết "đuôi dài" không giải thích được.

Nghi can còn lại là đầu thời điểm, và có dấu vết: cùng bộ dữ liệu, `feed` cho CN
0,01 nhưng `discover` (không đầu thời điểm) cho 0,07 và ALS cho 0,05. Trong năm
cột thời điểm, `log_pop_all` là độ phổ biến **tích luỹ cả đời** — danh mục CN có
4.763 bộ, mới hơn và nhỏ hơn JP 86.237 bộ, nên luôn thua ở cột đó. Còn chẩn đoán
gốc đo được cột mạnh là `log_pop_60d` (AUC 0,664), không phải `log_pop_all`.

**Việc làm, và lý do không gọi là vá:** thêm biến thể `feed_recency` bỏ đúng
cột `log_pop_all`, giữ bốn cột còn lại. Đây là giả thuyết có cơ chế, kiểm được
bằng hai số cùng lúc (NDCG và tỉ lệ CN) và **bác bỏ được**: nếu NDCG giảm mà CN
không tăng thì giả thuyết sai và `log_pop_all` vô can. Ràng buộc đa dạng theo
nước vẫn là fallback, nhưng chỉ dùng nếu phép kiểm này thất bại — vì nó che
nguyên nhân thay vì sửa.


---

## 9. `log_pop_all` vô can, và cổng theo người đọc chỉ đúng một nửa

### 9.1 Phép kiểm ở §8 bác bỏ chính giả thuyết của tôi

Bỏ đúng cột `log_pop_all`, giữ bốn cột còn lại:

| | NDCG all | warm | phủ catalog | KR | CN |
|---|---|---|---|---|---|
| `feed` (đủ 5 cột) | **0,1169** | **0,1039** | **0,0336** | 0,55 | 0,01 |
| `feed_recency` (bỏ `log_pop_all`) | 0,1072 | 0,0927 | 0,0091 | **0,65** | **0,00** |

CN xuống **0,000**, KR lên 0,65, phủ catalog sụt 3,7 lần, NDCG giảm 8,3%. Ngược
dấu hoàn toàn với dự đoán. Giả thuyết **bị bác bỏ**: `log_pop_all` không dìm
manhua — nó đang *kìm* cột `log_pop_60d` lại, và bỏ nó đi thì mô hình dựa vào
độ nổi 60 ngày nhiều hơn. Baseline "đang nổi 60 ngày" có KR = **0,99**, nên cột
đẩy KR chính là cột đó.

Ghi lại nguyên văn câu sai thay vì xoá, vì đây là lần thứ hai trong dự án một
dự đoán của tôi sai **dấu**, không chỉ sai độ lớn (lần đầu: logQ, §6).

### 9.2 Dữ liệu không thiếu manhua gần đây — đo trực tiếp thay vì đoán tiếp

| cửa sổ | JP | KR | CN |
|---|---|---|---|
| toàn bộ train | 0,593 | 0,343 | 0,063 |
| 365 ngày cuối | 0,497 | 0,436 | 0,066 |
| 60 ngày cuối | 0,458 | 0,476 | **0,065** |

Manhua giữ đúng 6,5% lượt đọc trong 60 ngày cuối, bằng tỉ lệ cả tập, và 61,4%
danh mục CN có hoạt động (JP 61,3%, KR 75,9%). Vậy 1,0% trong top-10 là lỗi của
mô hình, không phải phản ánh dữ liệu.

Chỗ khác nhau nằm ở **lượt đọc trung bình mỗi bộ** trong 60 ngày: KR 13,2 — CN
7,8 — JP 5,6; và mức tập trung ở top-100 của mỗi nước: JP 11,6%, KR 22,4%, CN
51,6%. Một bảng xếp hạng theo *số lượt tuyệt đối* thì đỉnh bảng gần như toàn KR.

### 9.3 Lỗi cấu trúc: đầu thời điểm là một hằng số cộng chung cho mọi người

`điểm = <q_u, v_i> + bias_i`. Cùng một `bias_i` cộng vào điểm của **tất cả** mọi
người, nên nó đẩy mọi người về **cùng một** bảng đang nổi — bảng đó là KR. Đó
cũng là lý do phủ catalog của `feed` chỉ 0,0336, thấp hơn cả ALS (0,0826).

Mà "bám theo truyện đang nổi" là **một nét của người đọc**, không phải một hằng
số của hệ. Cổng vô hướng học riêng từng người: `điểm = <q_u, v_i> + g_u·bias_i`,
khởi tạo `g_u = 1` nên lúc chưa học nó trùng khít bản cũ.

### 9.4 Kết quả: thắng ở người có lịch sử, mất ở người đọc mới

Cùng seed, cùng mọi thứ, khác đúng một chỗ là cổng:

| lát | không cổng | có cổng | |
|---|---|---|---|
| all | 0,1169 | **0,1192** | +2,0% |
| warm | 0,1039 | **0,1099** | +5,8% |
| sparse_user | 0,1268 | **0,1277** | +0,7% |
| new_user | **0,3771** | 0,2499 | **−33,7%** |
| Recall@10 all | 0,0391 | **0,0397** | +1,5% |
| phủ catalog | **0,0336** | 0,0309 | −8% |
| CN trong top-10 | 0,01 | 0,02 | ×2 |

Đọc theo đúng ba dòng đã chốt trước khi chạy:

- **Nửa đúng.** Warm tăng 5,8% chỉ nhờ cho phép từng người có mức bám-đang-nổi
  khác nhau. Điều này *không thể* xảy ra nếu "bám đang nổi" là một hằng số —
  nên nó là bằng chứng nhận dạng được cho cơ chế, không phải một mức tăng nhờ
  thêm tham số chung.
- **Nửa sai, và là nửa tôi đi tìm.** CN chỉ 0,01 → 0,02, còn cách tỉ lệ đọc
  0,063 ba lần. **Cổng không phải chỗ chặn manhua.** Hằng số chung là một lỗi
  cấu trúc thật, nhưng không phải lỗi chi phối. Ghi là bác bỏ cho phần CN.
- **Một chỗ mất phải giải thích, không được đánh đổi suông.** `new_user` mất
  33,7%. Người đọc mới không có `g_u` nào được học nên lúc phục vụ họ nhận
  *trung bình toàn dân*. Mà người đọc mới cũng là nhóm **không có tín hiệu gu
  nào khác** — baseline phổ biến đạt 0,3167 ở lát này, cao hơn mọi lát khác —
  nên họ là nhóm phụ thuộc đầu thời điểm nhiều nhất, và cũng là nhóm bị cắt
  đầu thời điểm nếu trung bình học được nhỏ hơn 1.

`scripts/diagnose_trend_gate.py` đo ba thứ, **chỉ trên train** nên không mượn
đáp án của test: phân phối `g_u`; `g_u` theo độ dài lịch sử; và tương quan giữa
`g_u` với độ nổi trung bình của những bộ chính người đó đã đọc.

**Cách đọc chốt trước khi xem số:**

1. Nếu `g_u` **giảm** theo độ dài lịch sử (người ít lịch sử bám đang nổi nhiều
   hơn) thì *trung bình toàn dân là ước lượng chệch* cho người đọc mới, và việc
   phải làm là ước lượng `g_u` cho người mới bằng đường khớp trên train tại độ
   dài lịch sử của họ. Đây là chọn giữa hai ước lượng cho một biến ẩn, quyết
   định bằng bằng chứng train, **không phải** dò một hằng số theo điểm test.
2. Nếu `g_u` **không** phụ thuộc độ dài lịch sử thì chỗ mất 33,7% *không* do
   chọn ước lượng, và tôi phải tìm chỗ khác — không được giữ cổng rồi gọi đó là
   đánh đổi.
3. Nếu `g_u` **không** tương quan với độ nổi truyện đã đọc thì cổng không học
   cái nét tôi nói nó học, và mức tăng 5,8% ở warm phải được giải thích lại
   (khả năng cao nhất: cổng chỉ hấp thụ độ lệch phổ biến riêng từng người, một
   thứ hữu ích nhưng tầm thường hơn nhiều).

Tới khi giải thích xong, bản có cổng lưu riêng ra `two_tower_gated.pt` và
**không** được ghi đè bản đang phục vụ.

### 9.5 Đính chính: cái tôi gọi là "đầu thời điểm" là **đầu độ phổ biến**

Đo trọng số thực sự học được của đầu này (cả hai bản):

| cột | không cổng | có cổng |
|---|---|---|
| `log_pop_all` | **+0,8914** | **+0,7277** |
| `log_pop_60d` | +0,0536 | +0,0372 |
| `is_releasing` | −0,0342 | −0,0253 |
| `age_years` | −0,0053 | −0,0021 |
| `is_finished` | −0,0119 | −0,0102 |

**94% trọng số nằm ở độ phổ biến tích luỹ cả đời.** Vậy mọi chỗ tôi viết "danh
sách đang nổi", kể cả câu ở §9.1 rằng bỏ `log_pop_all` làm mô hình "dựa vào
`log_pop_60d` nhiều hơn", đều sai tên gọi: bỏ `log_pop_all` không làm mô hình
dựa vào cột khác, nó **xoá gần hết cái đầu đó**. Con số ở §9.1 vẫn đúng, cách
tôi giải thích con số thì không.

Và chỗ dìm manhua hiện ra ngay khi gọi đúng tên: xếp toàn danh mục theo đầu này,
**top-100 có 0,0% manhua**, top-1000 có 5,6%. Top-10 gợi ý lấy từ đúng vùng cao
nhất đó, nên manhua không có cửa — không phải vì manhua ít được đọc gần đây
(§9.2 đã bác bỏ), mà vì **không bộ manhua nào nằm trong nhóm phổ biến nhất mọi
thời**. Hai bản cho cùng một thứ tự (tương quan hạng +1,0000, trùng 997/1000 ở
top-1000), nên cổng không dịch được chỗ đó, đúng như §9.4 đo được.

Theo đó cổng cũng phải đổi tên: nó là **mức thích truyện phổ biến** của từng
người, không phải mức bám truyện đang nổi. Chẩn đoán xác nhận đúng nét đó, bằng
một phép đo không liên quan gì tới NDCG:

| nhóm theo độ nổi những bộ chính họ đã đọc | số người | `g_u` trung bình |
|---|---|---|
| đọc truyện ít nổi nhất | 2.157 | 1,2025 |
| ở giữa | 2.221 | 1,3147 |
| đọc truyện nổi nhất | 2.157 | 1,4529 |

tương quan hạng **+0,845**. Còn với tỉ lệ manhua trong lịch sử thì tương quan
chỉ **−0,046** (người đọc >20% manhua có `g_u` 1,3248, phần còn lại 1,3231 —
bằng nhau). Cổng **mù màu quốc tịch**, nên không thể là đường sửa manhua. Đây là
lý do cơ chế, chứ không phải chỉ là "đo thấy CN không tăng".

### 9.6 Người đọc mới: `g_u` phải đọc tại độ dài lịch sử của họ

`fit_trend_prior` khớp `g_u ≈ 1,5886 − 0,0605·log(1+n)` trên **chỉ train**.
Trung bình toàn dân 1,3233 chính là đường khớp đọc tại `n ≈ 130` (độ dài trung
bình của train) — một người vừa chọn 3 truyện không ở đó, họ ở 1,5047.

`scripts/eval_saved.py` chấm lại blob đã lưu mà không huấn luyện lại (0,2 phút
thay vì 44 phút), nên phép so là **ghép cặp hoàn hảo**: cùng trọng số, chỉ khác
đường chấm điểm. Bản `--no-fit` tái lập đúng từng chữ số con số của
`train_embedding.py` (0,1192 / 0,2499 / 0,1279 / 0,1099), nên bản thân nó cũng
là một phép kiểm cho "chỉ có một đường chấm điểm".

| lát | trung bình toàn dân | đường khớp train | |
|---|---|---|---|
| all | 0,1192 | 0,1192 | — |
| **new_user** | 0,2499 | **0,2710** | **+8,4%** |
| new_item | 0,0000 | 0,0000 | — |
| sparse_user | 0,1279 | 0,1279 | — |
| warm | 0,1099 | 0,1099 | — |

Mọi lát khác không nhúc nhích một chữ số nào, đúng như phải vậy: người đọc đã
có trong train dùng `g_u` riêng của mình, không đi qua đường khớp. Một thay đổi
chỉ chạm đúng nhóm mà nó được suy ra cho — đó là điều kiện để gọi là nhận dạng
được, không phải một mức tăng chung chung.

Nhưng +8,4% chỉ lấp **17%** khoảng cách 0,3771 → 0,2499. Phần còn lại không do
chọn ước lượng, nên theo đúng dòng 2 đã chốt ở §9.4, tôi không được gọi đó là
đánh đổi rồi dừng.

### 9.7 Quét cổng: hệ này có **một** cái núm, và nó đánh đổi đơn điệu

Quét giá trị cổng phát cho người đọc mới trên **cùng một** blob. Mục đích là trả
lời một câu có/không — "có giá trị nào lấy lại được 0,3771 không" — **không**
phải để chọn giá trị: bản phục vụ vẫn dùng đường khớp trên train.

| cổng người mới | NDCG new_user | phủ catalog | CN |
|---|---|---|---|
| 0,0 | 0,0264 | **0,1307** | **0,11** |
| 0,5 | 0,0789 | 0,1285 | 0,10 |
| 0,8937 | 0,1717 | 0,1097 | 0,09 |
| 1,0 | 0,1973 | 0,1019 | 0,08 |
| 1,3233 (trung bình) | 0,2499 | 0,0803 | 0,06 |
| **1,5047 (đường khớp)** | **0,2754** | 0,0699 | 0,06 |
| 2,0 | 0,3022 | 0,0562 | 0,05 |
| 3,0 | 0,3312 | 0,0395 | 0,04 |
| 5,0 | 0,3694 | 0,0203 | 0,02 |
| bản không cổng (để đối chiếu) | 0,3771 | 0,0599 | 0,03 |

Bản không cổng cho đúng một con số ở mọi dòng (0,3771), như phải vậy vì nó
không có cổng — một phép kiểm âm cho chính cái vòng quét.

Ba điều đọc được, và không điều nào là thứ tôi đoán trước:

1. **Đơn điệu cả hai chiều, không có đỉnh ở giữa.** NDCG tăng suốt từ 0,026 lên
   0,369 trong khi phủ catalog rơi 0,131 → 0,020 và manhua rơi 0,11 → 0,02.
   Không có điểm nào được cả hai. Vậy `feed` và `discover` không phải hai cấu
   hình tình cờ mà là **hai đầu của cùng một núm**, và đường cong giữa chúng đo
   được bằng chín điểm.
2. **Manhua là hàm của đúng cái núm đó.** Tắt đầu phổ biến cho người đọc mới thì
   CN lên **0,11**, tức *gần gấp đôi* tỉ lệ đọc thật 0,063, và phủ catalog lên
   0,1307. Khớp với `discover` (CN 0,11, phủ 0,158). Nên câu trả lời cho §8
   không phải "thêm ràng buộc đa dạng" mà là "chọn điểm nào trên núm" — một
   quyết định sản phẩm, công bố được, chứ không phải một ràng buộc che nguyên
   nhân.
3. **Chỗ mất ở `new_user` không nằm trong cổng.** Phải đẩy cổng tới 5,0 (hơn ba
   lần đường khớp) mới gần lại 0,3771. Và nó cũng không nằm ở thang: đã đo tỉ lệ
   độ lan toả thời điểm/gu cho người đọc mới là 0,478 ở bản có cổng so với 0,333
   ở bản không cổng — bản có cổng *đã* nghiêng về phổ biến **nhiều hơn 43%** mà
   vẫn kém. Hai đầu thời điểm lại cùng hướng (+1,0000). Nên phần còn lại nằm ở
   **hướng của tháp gu cho người đọc 3 lượt chọn**, chỗ duy nhất chưa loại trừ —
   ghi là câu hỏi mở, không ghi là đánh đổi.

Hệ quả cho sản phẩm, và đây là chỗ phép đo quyết định thiết kế chứ không phải
ngược lại: `two_tower_gated.pt` **không** ghi đè bản đang phục vụ. Bản có cổng
thắng ở `warm` (+5,8%) và `sparse_user`, mất ở `new_user`, nên nó là **engine
thứ ba** chứ không phải bản thay thế — và điều phải nói thẳng trong báo cáo là
mức được/mất ấy, không phải một con số `all` duy nhất che cả hai.

### 9.8 Chỗ mất ở người đọc mới: cổng **lấy mất** tín hiệu của đường lịch sử

§9.7 còn một chỗ chưa giải thích và tôi đã ghi là câu hỏi mở. Phép đo quyết
định: với người đọc mới 3 lượt chọn, top-10 của họ **đến từ đâu** (40 người giả
lập, cùng hạt giống cho cả hai mô hình):

| | trùng top-10 phổ biến | nằm trong top-200 phổ biến | trùng top-10 chấm bằng **riêng gu** |
|---|---|---|---|
| `feed` | 0,062 | 0,185 | **0,532** |
| `gated` | 0,243 | **0,720** | 0,082 |

Đọc được ngay: top-10 của người đọc mới ở bản không cổng **do gu quyết định**
(0,532 trùng với bảng chấm bằng riêng gu, và chỉ 0,185 nằm trong top-200 phổ
biến), còn ở bản có cổng thì **do độ phổ biến quyết định** (0,720 nằm trong
top-200 phổ biến, gu chỉ còn 0,082). Và §9.7 đã đo: tắt cổng về 0 thì NDCG
người đọc mới rơi xuống **0,0264** — đường gu của bản có cổng *rỗng* với người
đọc mới, trong khi đường gu của bản không cổng đạt tới 0,3771, hơn cả baseline
phổ biến 0,3167.

Cơ chế, và lần này nó là một cơ chế đã có tên: khi mỗi người có một hệ số phổ
biến riêng `g_u`, đường **gộp-lịch-sử** không còn cần mang thông tin độ phổ biến
nữa — `g_u` giải thích hết phần đó. Nhưng người đọc mới *chỉ có* đường gộp-lịch-
sử: họ không có `g_u`, không có vector ID. Cổng vô tình bòn rút đúng con đường
duy nhất họ đi. Đây chính là lập luận của **DropoutNet** (Volkovs, Yu &
Poutanen, NeurIPS 2017), thứ mà thiết kế này đã áp cho vector ID ngay từ đầu
(mục 4 docstring `two_tower.py`) — nhưng tôi thêm cổng mà **quên** áp cho nó.

Một phép đo tôi đã đọc sai, ghi lại để không lặp: tôi từng kết luận "không phải
chuyện thang" vì độ lan toả (std trên toàn danh mục) của số hạng phổ biến so với
gu là 0,478 ở bản có cổng và 0,333 ở bản không cổng. Con số đúng, nhưng **sai
thống kê**: std trên cả 34.608 truyện không nói gì về top-10, mà top-10 là thứ
duy nhất được phục vụ. Thống kê đúng là bảng ngay trên: *bao nhiêu phần của
top-10 do số hạng nào quyết định*.

**Việc làm, và lý do không gọi là vá.** Cổng tách thành hai phần:

    g_u = base(log(1 + n_u)) + delta_u

`base` là hàm của độ dài lịch sử nên dùng được cho **mọi** người kể cả người
mới — nó đưa quan hệ đã đo được (§9.6: giảm đơn điệu qua cả sáu bậc, tương quan
hạng −0,51) vào thẳng *cấu trúc* thay vì khớp lại bằng hồi quy sau khi huấn
luyện xong. Cái khác biệt không chỉ là gọn hơn: hồi quy hậu kiểm không sửa được
gì ở chỗ mất thật, vì đường gu vẫn **học** trong điều kiện luôn có `delta_u`.
`delta_u` bị bỏ ngẫu nhiên (p = 0,3) lúc huấn luyện, nên có những bước mô hình
chỉ còn `base` — đúng cảnh người đọc mới — và đường gộp-lịch-sử buộc phải tự
gánh.

Hai biến thể để **quy công đúng chỗ**, vì nếu chỉ chạy một bản thì không tách
được công của việc tái tham số hoá khỏi công của việc bỏ ngẫu nhiên:

| biến thể | cổng hai phần | bỏ ngẫu nhiên `delta_u` |
|---|---|---|
| `feed_gated_nodrop` | có | không |
| `feed_gated` | có | p = 0,3 |

**Dự đoán ghi trước khi chạy**, và cách đọc từng trường hợp:

1. `new_user` hồi về gần 0,3771 ở `feed_gated` mà `warm` giữ được phần lớn
   +5,8% → cơ chế đúng, và `gated` thành ứng viên mặc định.
2. `new_user` hồi nhưng `warm` mất hết phần tăng → cổng và đường người-mới là
   một đánh đổi thật, không phải lỗi thực thi; công bố cả hai số, không chọn hộ.
3. `new_user` vẫn ~0,27 ở **cả hai** biến thể → cơ chế "cổng lấy mất tín hiệu
   của đường lịch sử" **sai**, ghi là bác bỏ và quay lại tìm chỗ khác. Lúc đó
   nghi can tiếp theo là chính phép chia nhiệt độ 0,07 trên vế cosine so với
   thang tự do của `bias`, chỗ duy nhất còn lại chưa loại trừ.
4. `feed_gated_nodrop` một mình đã hồi → công thuộc về tái tham số hoá, không
   phải bỏ ngẫu nhiên, và phần giải thích DropoutNet ở trên phải viết lại.

Khởi tạo đặt sao cho `base ≡ 1` và `delta ≡ 0`, nên mô hình lúc chưa học **trùng
khít** bản không cổng — có test khẳng định điều đó bằng cách sao thẳng trọng số
dùng chung sang bản không cổng rồi so điểm, chứ không gieo cùng seed (bật cổng
tạo thêm tham số nên tiêu thêm số ngẫu nhiên, gieo cùng seed là một phép so sai).

### 9.9 Kết quả: bỏ ngẫu nhiên làm toàn bộ việc, và tái tham số hoá một mình **làm tệ hơn**

Bốn hàng, đọc đúng theo bốn cách đã chốt ở §9.8:

| cấu hình | all | new_user | sparse_user | warm |
|---|---|---|---|---|
| `feed` — không cổng | 0,1169 | **0,3771** | 0,1268 | 0,1039 |
| cổng một phần + khớp hậu kiểm | 0,1192 | 0,2710 | 0,1279 | **0,1099** |
| cổng hai phần, **không** bỏ ngẫu nhiên | 0,1171 | **0,1774** | 0,1332 | 0,1041 |
| cổng hai phần **+ bỏ ngẫu nhiên** | **0,1202** | 0,3533 | **0,1387** | 0,1051 |

**Dự đoán số 4 bị bác bỏ, và bác bỏ theo chiều tôi không lường.** Tôi đã viết:
nếu `nodrop` một mình đã hồi thì công thuộc về tái tham số hoá. Nó không hồi —
nó **tệ hơn cả bản cổng một phần** (0,1774 so với 0,2710) và chỉ bằng 47% của
`feed`. Đưa quan hệ độ-dài-lịch-sử vào cấu trúc, tự nó, làm *xấu* đi.

Lý do, và nó làm lập luận §9.8 **chặt hơn** chứ không đổ: ở bản một phần, giá
trị phát cho người đọc mới là một đường hồi quy khớp *sau khi* huấn luyện xong,
nên nó bị ép phải nằm trong vùng `g_u` thật. Ở `nodrop`, `base` là tham số học
được nhưng học trong điều kiện **luôn có** `delta_u`, nên giá trị độc lập của nó
không bị ràng buộc gì cả — `base` và `delta_u` tự do dịch cho nhau. Đưa cấu trúc
vào không đủ; điều quyết định là **huấn luyện trong đúng điều kiện lúc phục vụ**.
Đó là lập luận DropoutNet ở dạng gọn nhất, và `nodrop` là phản chứng của phiên
bản yếu hơn ("chỉ cần viết lại công thức").

**Quy công sạch nhất của cả đồ án:** 0,1774 → 0,3533, **+99,2%** ở `new_user`,
và hai cấu hình này **cùng khởi tạo đến từng bit** — cùng kiến trúc, cùng số
tham số, cùng thứ tự rút số ngẫu nhiên, khác đúng một chỗ là `delta_u` có bị bỏ
hay không. Không cần ghép cặp nhiều seed để tin một hiệu ứng gấp đôi, khi nhiễu
seed đo được ở §10 là ~7% tương đối.

**Lặp lại trên bộ dữ liệu thứ hai (Goodreads, 2016, sách in).** Cùng pipeline,
chỉ đổi tầng đọc dữ liệu: 0,0779 → 0,1580, **+102,8%** ở `new_user` — gần y
nguyên +99,2% ở đây, trên một bộ mà đầu thời điểm chỉ còn 3 trục thật thay vì 5
và mỗi người đọc chỉ có ~36 quan sát thay vì ~233. Và phần phản chứng cũng lặp
lại: bản không bỏ ngẫu nhiên *tệ hơn cả bản không cổng* ở `new_user` trên cả hai
nhánh (0,1774 so với 0,3771 ở đây; 0,0779 so với 0,2322 bên Goodreads). Đây là
dự đoán **P2** đã ghi trước ở `docs/goodreads-arm.md` mục 3, phán quyết ở mục 8.
Ngược lại, lợi thế *tổng* của cổng so với bản không cổng (bảng ghép cặp ngay
dưới) **không** lặp lại: bên Goodreads cổng mất ở cả bốn lát, và cơ chế của chỗ
mất được đo ở mục 10 của tài liệu đó — cổng khuếch đại đầu thời điểm, mà đầu
thời điểm gần như chỉ học *phổ biến toàn kỳ*, trong khi đỉnh danh mục toàn kỳ của
Goodreads lạc hậu (hit@10 5,76% so với 10,24% của cửa sổ 60 ngày).

**Dự đoán 1 và 2 đúng một nửa mỗi cái.** Khoảng hụt ở `new_user` đóng được
**81%** (0,2710 → 0,3533 trên tổng 0,2710 → 0,3771), nhưng `warm` chỉ giữ lại
**21%** phần tăng của cổng một phần (+1,2% so với +5,8%). Điều đó hợp lý và
đáng ghi: bỏ `delta_u` 30% số bước làm yếu đúng cái số hạng mà người đọc nhiều
hưởng lợi. Nên **bên trong thiết kế cổng vẫn có một đánh đổi thật** — nó chỉ
không nằm ở chỗ tôi tưởng. Bù lại `sparse_user` tăng **+9,4%** so với `feed` và
`all` đạt **0,1202**, số cao nhất đồ án có trên nhánh temporal.

**Một lỗi chẩn đoán suýt thành "phát hiện".** Lần quét cổng đầu tiên trên mô
hình mới ra **phẳng tuyệt đối**: cả chín giá trị đều 0,3533. Đọc vội thì đó là
một kết quả lớn ("mô hình mới bất biến với cổng"). Nó là **lỗi của tôi**: hàm
quét ghi đè bằng `trend_fit`, mà `trend_fit` chỉ có đường cũ đọc; blob hai phần
đọc `trend_base`, nên lệnh ghi đè là no-op. Đã sửa để ghi đè đi vào đúng đường
mà blob đó phục vụ, và thêm một `assert` kiểm lại cổng sau khi ghi đè — một
phép quét không tự kiểm là một phép quét có thể phẳng vì sai.

**Và trục hoành của phép quét không so được giữa hai mô hình.** `feed` không có
cổng, tức cổng hằng **1,0** vĩnh viễn; nhưng ở mô hình có cổng, thang của
`item_bias` học được đã khác (cổng phục vụ là 2,25, không phải 1,32 như bản một
phần). Nên "cùng giá trị cổng" không có nghĩa là "cùng lượng độ phổ biến được
phục vụ". Phép so đúng là trong **không gian kết quả**: cặp (NDCG `new_user`,
phủ danh mục), không đánh chỉ số theo giá trị cổng. Ở §9.7 tôi đã dùng `feed`
làm đường phẳng để chứng minh lệnh ghi đè không chạm vào nó — việc đó đúng — mà
không nói rõ rằng **giá trị** cổng thì không so ngang được.

| tại phủ danh mục = 0,0599 (điểm của `feed`) | NDCG `new_user` |
|---|---|
| `feed` | **0,3771** |
| frontier cổng **hai phần** | 0,3498 |
| frontier cổng **một phần** | 0,2944 |

Đọc được hai điều ngược nhau, và cả hai đều phải nói:

- Frontier mới **nằm ngoài** frontier cũ ở **mọi** mức phủ danh mục (0,3498 so
  0,2944 tại 0,0599; 0,2711 so 0,2000 tại 0,1008). Đóng được **67%** khoảng
  cách tới `feed` trong không gian kết quả. Vậy §9.7 đúng về *hình dạng* (vẫn
  đơn điệu, vẫn không có cực trị trong) nhưng *vị trí* của đường cong là tính
  chất của một mô hình có khuyết điểm, không phải một định luật.
- Nhưng `feed` vẫn **nằm ngoài** frontier mới, hơn **7,8%** tại cùng mức phủ.
  Và `discover` đạt phủ 0,1506 trong khi cổng dù hạ về 0 cũng chỉ tới 0,1261 —
  tức `discover` ở ngoài tầm của cái núm. **Với người đọc mới, cổng bị trội
  hoàn toàn:** `feed` chiếm đầu chính xác, `discover` chiếm đầu đa dạng, và cả
  đường cong của cổng nằm *bên trong* hộp hai điểm đó.

**Kết luận, và nó không phải cái tôi đi tìm.** Giá trị của cổng nằm **toàn bộ**
ở người đọc *có* lịch sử (`all` +2,8%, `sparse_user` +9,4%, `warm` +1,2% so với
`feed`), và **bằng không hoặc âm** ở người chưa có gì. Ranh giới không phải "ít
hay nhiều lịch sử" — `sparse_user` vẫn thắng đậm — mà đúng là **có hay không có
lịch sử**. Đó là một quy tắc định tuyến đo được, không phải một vá: hai mô hình
trội nhau trên hai vùng rời nhau.

Chưa làm gì với kết luận đó, vì ba con số +2,8% / +9,4% / +1,2% đến từ **hai
lần huấn luyện khác nhau**, mà bật cổng thì thêm tham số, tiêu thêm số ngẫu
nhiên, nên hai bản không cùng khởi tạo — đúng cái bẫy ở §10. Đang chạy phép kiểm
ghép cặp 4 seed (`--seeds 0 1 2 3`, mới thêm vào `train_embedding.py`) cho cả
hai biến thể. Quy tắc đọc, chốt trước: chỉ đổi mặc định nếu `all` **và**
`sparse_user` đều dương với `t` vượt ngưỡng `df=3` là 3,18; nếu `warm` không
vượt ngưỡng thì bỏ hẳn tuyên bố "+1,2% ở warm" khỏi tài liệu thay vì để nó ở
dạng "có xu hướng".

### 9.10 Phép kiểm ghép cặp 4 seed: hai kết quả lập được, hai kết quả không

Cùng 4 seed cho cả hai biến thể (`--seeds 0 1 2 3`, mới thêm vào
`train_embedding.py`), nên nhiễu khởi tạo triệt tiêu trong hiệu số:

| lát | `feed` | cổng | chênh | t | ngưỡng df=3 | |
|---|---|---|---|---|---|---|
| `all` | 0,1175 | 0,1196 | **+1,9%** | **+4,27** | 3,18 | **lập được**, 4/4 seed |
| `new_user` | 0,3744 | 0,3549 | **−5,2%** | **−5,79** | 3,18 | **lập được**, 0/4 seed |
| `sparse_user` | 0,1283 | 0,1339 | +4,4% | 2,60 | 3,18 | chưa lập được, 4/4 dương |
| `warm` | 0,1035 | 0,1055 | +1,9% | 2,62 | 3,18 | chưa lập được, 4/4 dương |

**Và nó bắt tôi tự đính chính bốn con số tôi vừa công bố.** Một giờ trước tôi
đã viết +9,4% ở `sparse_user` và +2,8% ở `all` vào README, `service.py`,
`schemas.py` và frontend, lấy từ **một** lần huấn luyện. Sự thật trên 4 seed:

| lát | số 1 seed đã công bố | 4 seed ghép cặp | lệch |
|---|---|---|---|
| `sparse_user` | +9,4% | **+4,4%** | phóng đại **2,1×** |
| `all` | +2,8% | +1,9% | phóng đại 1,5× |
| `new_user` | −6,3% | −5,2% | 1,2× |
| `warm` | +1,2% | +1,9% | thiếu 0,6× |

Một lần chạy không chỉ ồn — nó ồn **lệch về phía con số đẹp**, vì tôi đọc nó
sau khi đã kỳ vọng cổng thắng. Đã sửa cả bốn chỗ.

**Giữ nguyên quy tắc đã chốt, không diễn giải lại.** Quy tắc viết trước khi chạy
là: *chỉ đổi mặc định sang `gated` nếu `all` **và** `sparse_user` đều vượt
t = 3,18*. `sparse_user` được 2,60. Vậy **`gated` không thành mặc định** — dù
`all` vượt rõ và dù cả 4 seed đều dương ở `sparse_user`. Và theo đúng quy tắc
ấy, tuyên bố "+1,2% ở `warm`" bị **bỏ hẳn** khỏi tài liệu thay vì để ở dạng "có
xu hướng dương".

**Và tôi không thêm seed để đuổi theo ngưỡng.** `sparse_user` t = 2,60 ở df=3;
thêm 2 seed thì ngưỡng df=5 tụt về 2,57 và nó sẽ "đạt" gần như chắc chắn. Nhưng
đó là **lần nhìn thứ hai**, và quyết định thêm dữ liệu *sau khi* thấy kết quả
làm tỉ lệ dương tính giả cao hơn con số p in ra. Hiệu chỉnh Bonferroni cho hai
lần nhìn đòi p < 0,025, tức t > 3,16 ở df=5 — xa hơn 2,60 nhiều. Nên câu trả lời
đúng là *chưa lập được*, không phải *chạy thêm cho đến khi lập được*.

(Cùng một thước phải áp lại cho §10: ở đó tôi **đã** mở rộng 3 → 5 seed sau khi
nhìn. Hiệu chỉnh hai lần nhìn đòi p < 0,025, tức t > 3,495 ở df=4; mặt `art` đo
được t = 4,10 nên **vẫn vượt**. Kết luận ở §10 giữ nguyên, nhưng con số p = 0,015
phải đọc là p danh nghĩa, không phải tỉ lệ dương tính giả thật.)

### 9.11 Định tuyến: lời đáp đúng cho hai kết quả ngược chiều

Hai thứ *lập được* ở §9.10 ngược chiều nhau — cổng hơn ở `all`, kém ở
`new_user` — nên không tồn tại một mô hình tốt hơn ở cả hai vùng, và câu hỏi
"chọn mô hình nào" là câu hỏi sai. `RoutedTwoTower`: `feed` cho người **chưa
có** lịch sử, `gated` cho người **đã có**.

Đây là thay đổi ở **vế phục vụ**, nên đo được **ghép cặp hoàn hảo** — cùng trọng
số, chỉ khác đường chấm, không huấn luyện lại, không nhiễu seed trong phép so
(`eval_saved.py --route two_tower_feed.pt two_tower_gated.pt`, 0,2 phút):

| | all | new_user | sparse_user | warm | phủ catalog | manhua |
|---|---|---|---|---|---|---|
| `feed` | 0,1169 | **0,3771** | 0,1268 | 0,1039 | 0,0336 | 0,01 |
| `gated` | **0,1202** | 0,3533 | **0,1387** | **0,1051** | **0,0347** | **0,02** |
| **`auto`** | **0,1202** | **0,3771** | **0,1387** | **0,1051** | **0,0347** | **0,02** |

Lấy **max ở mọi lát**. Phải nói rõ phần nào là phát hiện và phần nào là số học:
*việc* mỗi lát nghiêng về engine nào là phát hiện thực nghiệm (§9.10, với hai
hàng lập được); còn *việc* bản định tuyến bằng max thì gần như là số học, vì mỗi
lát được phục vụ hoàn toàn bởi một engine. Giá trị của bảng này không nằm ở chỗ
nó bất ngờ mà ở chỗ nó **được đo thật** thay vì được lập luận.

Ranh giới là **CÓ hay KHÔNG CÓ** lịch sử, không phải ít hay nhiều — `sparse_user`
(người đọc thưa) nghiêng về cổng, nên "ít lịch sử thì dùng `feed`" sẽ sai. Mã
định tuyến đọc đúng độ dài lịch sử đã lưu, nên một người có tên trong bảng mà
lịch sử rỗng vẫn đi đường người-đọc-mới; có test cho đúng trường hợp đó.

**Giá phải trả, đo chứ không đoán.** Ở 150 người, `auto` trông kém `feed` về đa
dạng nước (1,73 so 1,76 nước; phủ nước 0,671 so 0,685) — tôi đã suýt ghi đó là
cái giá của định tuyến. Đo lại ở 800 người thì khoảng cách **biến mất**: 1,742
so 1,741 và 0,678 so 0,675. Vậy 1,6 điểm phần trăm kia là nhiễu lấy mẫu, và
định tuyến **không** tốn gì về đa dạng. Bậc 4 giữ nguyên: p95 7,5–8,7 ms, trùng
tác giả 0,003, lý do 100%, onboarding 1 bìa.

`auto` thành **mặc định**, và lý do nó không vi phạm quy tắc ở §9.10: quy tắc đó
chặn việc đưa `gated` lên mặc định, và `gated` **không** lên mặc định. `auto` là
một đối tượng khác, xét theo tiêu chí riêng — không hồi quy ở bất kỳ lát nào,
cộng một mức hơn **đã lập được** ở `all`. Nó cũng không tốn thêm bộ nhớ: dùng
lại đúng hai blob service đã nạp.

## 10. Mặt `art`: cùng một dữ liệu, hai phép kiểm, hai kết luận trái ngược

Bậc cuối của thang ablation (`+art`, 205 → 220 chiều) cho +11% ở lát `new_item`,
còn khoảng dao động giữa các seed là 0,0142–0,0191 — rộng hơn chính hiệu ứng.
Đây là đúng tình huống mà một con số đơn lẻ không nói được gì, và cũng là chỗ dễ
kết luận sai theo **cả hai** hướng.

Hai phép kiểm trên **đúng cùng một bộ số** (5 seed × 2 bậc):

| phép kiểm | cách tính | kết luận |
|---|---|---|
| độc lập (**sai** ở đây) | lệch chuẩn từng nhóm rồi so hai trung bình | chênh +0,0021 < nhiễu 0,0011·2 → "không tách được" |
| ghép cặp theo seed (đúng) | lấy hiệu *trong từng seed* rồi kiểm hiệu | `t = 4,10`, `df = 4`, p ≈ 0,015 → **có thật** |

Lý do phép độc lập sai: hai cấu hình được huấn luyện trên *cùng* một seed, nên
chúng không độc lập. Nhiễu seed là một số hạng **cộng chung** cho cả hai nhánh;
lấy hiệu thì nó triệt tiêu, còn lấy lệch chuẩn từng nhóm thì nó bị đếm vào làm
nhiễu lần thứ hai. Trong 5 seed, lệch chuẩn *trong nhóm* là 0,0010–0,0017, còn
lệch chuẩn của *hiệu* chỉ 0,0011 — và **5/5 seed đều dương**, điều gần như không
thể xảy ra nếu hiệu ứng bằng 0 (xác suất 1/32 chỉ tính riêng dấu).

Hai bài học về quy trình, cả hai đều là lỗi tôi đã mắc trong chính mục này:

- **Không dừng ở "có xu hướng dương".** Ở 3 seed đầu, `t = 4,11` nhưng ngưỡng
  `df = 2` là 4,30, nên kết quả là *chưa đủ*. Thiếu ở đây không phải hiệu lượng
  mà là **bậc tự do**: thêm 2 seed thì `t` gần như không đổi (4,10) nhưng ngưỡng
  tụt xuống 2,776 và câu trả lời đổi. Nếu lúc đó công bố "có xu hướng" thì đã
  bịa một kết luận không có trong dữ liệu; nếu bỏ `art` thì đã bỏ oan một mặt
  thật.
- **Độ phủ pha loãng mọi kết luận về hiệu lượng.** Chỉ 57,8% truyện có ảnh bìa,
  nên +13,8% đo được là mức *trên toàn lát*, không phải mức trên những truyện
  `art` thật sự nói được điều gì. Con số đúng để nói về nét vẽ phải là mức trên
  nhóm có bìa; +13,8% là chặn dưới.
