# Tiêu chí thành công — chốt trước khi có kết quả

Văn bản này viết **trước** khi chạy đánh giá trên dữ liệu thật, cố ý. Nếu chốt
ngưỡng sau khi đã xem số, mọi kết quả đều "đạt" — chỉ cần chọn ngưỡng vừa khít
cái mình đo được. Ngày chốt: 2026-11-06, lúc crawl mới xong 35%.

Mỗi bậc là cổng cho bậc sau. Hỏng ở bậc dưới thì số liệu bậc trên vô nghĩa.
Mỗi tiêu chí đi kèm **việc phải làm nếu không đạt** — để lúc không đạt thì đã
có đường đi, chứ không phải ngồi hạ chuẩn.

---

## Bậc 0 — Dữ liệu có dùng được không

| Đo | Ngưỡng | Nếu không đạt |
|---|---|---|
| Người đọc sau lọc k-core | ≥ 5.000 | Hạ `min_item` xuống 3, báo rõ là đã hạ |
| Tương tác dương sau lọc | ≥ 500.000 | Crawl thêm user |
| Tỉ lệ truyện CN trong lõi | ≥ 8% | **Báo thẳng là manhua thưa**, không nói là phủ đều |
| Tỉ lệ truyện KR trong lõi | ≥ 20% | Tăng `--weights` cho KR, crawl bù |
| Không user nào >95% một nước | — | Lấy mẫu bị sập về một nhóm, phải lấy lại |

Bậc này **không** phải tiêu chí chất lượng mô hình. Nó chỉ xác nhận cái tôi
đang đo là hệ gợi ý cho cả ba loại truyện, không phải hệ gợi ý manga có lẫn vài
bộ manhwa.

---

## Bậc 1 — Có hơn gợi ý tầm thường không

Chuẩn thật trong recsys không phải "Recall@10 > 0". Nó là **gợi ý theo độ phổ
biến**. Truyện tranh có phân phối đuôi dài rất nặng: chỉ cần trả về 10 bộ nổi
nhất là đã đúng kha khá, vì ai cũng đọc chúng. Mô hình không vượt được mức đó
thì chưa xây được gì.

| Đo (lát `all`, k=10) | Ngưỡng | Nếu không đạt |
|---|---|---|
| Recall@10 hybrid vs phổ biến | **≥ +50% tương đối** | Lỗi ở ALS hoặc ở nhãn tín hiệu, không phải ở đặc trưng |
| NDCG@10 hybrid vs chỉ-ALS | > 0 | Tầng xếp lại đang làm hại, phải tắt |
| Recall@10 lát `warm` | ≥ 0,15 | CF chưa học được gì, xem lại `confidence` |

---

## Bậc 2 — Đặc trưng nội dung có đáng tồn tại không

Đây là luận điểm của cả đồ án. Nếu bốn mặt nội dung không nâng được gì so với
ALS trơn, thì đồ án này là một bộ lọc cộng tác bình thường có thêm phần trang
trí, và phải nói đúng như thế.

| Đo | Ngưỡng | Nếu không đạt |
|---|---|---|
| Lát `new_user`: đủ mặt vs chỉ-ALS | **≥ +20% tương đối Recall@10** | Tiền đề content-based sai với dữ liệu này — báo cáo đúng vậy |
| Lát `new_item`: chỉ-mặt vs phổ biến | > 0 | ALS về lý thuyết không chấm nổi truyện mới, nên mọi điểm ở đây là của nội dung; bằng 0 nghĩa là các mặt vô dụng |
| Lát `sparse_user`: thang ablation | tăng đơn điệu qua 6 bậc | Mặt nào làm giảm thì bỏ mặt đó, đừng giữ cho đẹp bảng |
| Mỗi mặt (`theme`/`pace`/`creator`/`art`) | đóng góp biên > 0 | Mặt âm thì bỏ và ghi lại là đã bỏ |

Thang ablation: phổ biến → ALS → +theme → +creator → +pace → đủ bốn mặt.

---

## Bậc 3 — Đặc trưng nét vẽ có thật là nét vẽ không

Mặt rủi ro nhất. 15 thống kê thủ công rút từ **ảnh bìa** có thể chỉ đang đo
nhiễu nén JPEG và màu nền, chứ không đo nét vẽ. Phép kiểm không cần nhãn tay:
láng giềng gần nhất trong không gian `art` có cùng hoạ sĩ không? Nhãn hoạ sĩ
có sẵn miễn phí từ AniList và **không** được dùng để huấn luyện đặc trưng này,
nên nó là phép kiểm độc lập.

| Đo | Ngưỡng | Nếu không đạt |
|---|---|---|
| Truy hồi cùng hoạ sĩ: precision@5 | **≥ 2× mức ngẫu nhiên** | Chuyển sang Gram matrix (torch 2.10 CPU đã có) |
| Sau khi chuyển Gram | ≥ 2× | Bỏ hẳn mặt `art`, ghi rõ là ảnh bìa không đủ và cần ảnh trang trong |

Hai điều cần nói thẳng trong báo cáo dù kết quả ra sao:
- Ảnh bìa **không phải** nét vẽ trong truyện. Bìa thường do hoạ sĩ vẽ riêng,
  tô màu, bố cục khác hẳn trang truyện trắng đen. Đây là giới hạn của nguồn dữ
  liệu, không phải của phương pháp.
- Một hoạ sĩ vẽ nhiều bộ cùng kiểu nên phép kiểm này có thể đang thưởng cho
  việc nhận ra *bộ truyện* chứ không phải *kiểu vẽ*. Vì vậy ngưỡng là 2×, không
  phải "cao hơn ngẫu nhiên là được".

---

## Bậc 4 — Có dùng được như một sản phẩm không

Một hệ đúng về số mà vô dụng khi dùng thì vẫn là thất bại.

| Đo | Ngưỡng | Nếu không đạt |
|---|---|---|
| Trùng người sáng tác trong top-10 | ≤ 0,30 | Tăng `creator_penalty` |
| Số nước trong top-10 trung bình | ≥ 2 | Thêm ràng buộc đa dạng theo nước |
| Mọi gợi ý đều có lý do | 100% | Lỗi ở explainer, phải sửa |
| Thời gian gợi ý mỗi người | < 300 ms | Giảm `n_candidates` |
| Người đọc mới chọn bìa → có hồ sơ | ≤ 5 lựa chọn | Xem lại lưới onboarding |

---

## Điều này **không** chứng minh được

Nói trước để không bị hỏi ngược lúc bảo vệ:

- **Không có đánh giá người thật.** Mọi con số là đánh giá offline trên lịch sử
  đã có. Offline tốt lên không đảm bảo người dùng thấy hay hơn — đây là hạn chế
  đã biết của cả ngành, không phải lỗi riêng.
- **Nhịp truyện đang đo bằng đại diện, không phải bằng cốt truyện.** Số chương,
  thời gian ra, tỉ lệ bỏ dở và **bỏ ở chương nào** là dấu vết của nhịp, không
  phải bản thân nhịp. Muốn đo nhịp thật phải đọc nội dung từng chương — ngoài
  phạm vi và ngoài giấy phép.
- **Thiên lệch lấy mẫu còn đó.** User lấy từ hoạt động công khai trên các bộ
  phổ biến, nên đây là người đọc tích cực của truyện nổi, không phải người đọc
  nói chung.
- **Không so được với hệ của AniList.** Không có hệ gợi ý nào của họ để làm
  chuẩn, nên không có ý "tốt hơn AniList".

---

## Trạng thái

Đo trên dữ liệu thật ngày 2026-11-06/07 (9.229 người đọc, 34.608 truyện trong
lõi, 1,52 triệu tương tác train). Ngưỡng không bị đổi sau khi thấy số, trừ một
chỗ đánh dấu rõ bên dưới.

Bảng dưới là lần chạy **hai tầng ALS + xếp lại** (bản đầu). Mô hình **hai tháp**
chấm toàn danh mục đo sau, xem mục "Hai tháp" ở cuối.

| Chỉ tiêu | Ngưỡng | Đo được | |
|---|---|---|---|
| 0 · KR trong lõi | ≥ 20% | 24,5% | đạt |
| 0 · CN trong lõi | ≥ 8% | 7,0% | **không đạt** |
| 0 · không ai >95% một nước | không ai | 24,2% người vượt | **không đạt** — ngưỡng này đặt cứng quá, sửa sau khi thấy số: 88,5% người đọc ≥2 nước mới là chỉ số có nghĩa |
| 1 · Recall@10 đủ mặt vs phổ biến | ≥ +50% | 0,0381 vs 0,0201 = +90% | đạt |
| 1 · NDCG@10 hybrid vs chỉ-ALS | > 0 | 0,1072 vs 0,1072 | **không đạt** (bằng nhau) |
| 1 · Recall@10 lát warm | ≥ 0,15 | 0,0415 | **không đạt** |
| 2 · new_user đủ mặt vs chỉ-ALS | ≥ +20% | 0,0345 vs 0,0346 | **không đạt** |
| 2 · new_item vs phổ biến | > 0 | 0 mọi cấu hình | **không đạt** |
| 2 · sparse_user tăng đơn điệu | đơn điệu | 0,0489 → 0,0505 → 0,0519 → 0,0485 → 0,0522 | **không đạt** (pace làm giảm) |
| 3 · cùng hoạ sĩ precision@10 | ≥ 2× ngẫu nhiên | 11,8× | đạt, xem lưu ý |
| 4 | — | chưa đo | — |

Lưu ý bậc 3: 11,8× đạt ngưỡng nhưng có nhiễu cùng-bộ-truyện. Đã tách (mục H,
`docs/research-notes.md`): 35% số lần trúng là cùng bộ; phần khác-bộ còn
**7,6×** so với ngẫu nhiên — vẫn trên ngưỡng 2×, nên mặt nét vẽ đứng vững.

Cách đọc: mặt nội dung chưa chứng minh được đóng góp. Tổng thể chỉ +2,7%
tương đối Recall (0,0371 → 0,0381), chưa kiểm ý nghĩa thống kê. Hai chỗ hỏng là
cấu trúc, không phải tham số: ứng viên chỉ lấy từ top-400 của ALS nên truyện
mới không bao giờ vào được, và người đọc mới chưa có vector ALS nên CF rơi về
phổ biến (điểm trùng khít 0,3453).

---

## Hai tháp, chấm toàn danh mục (2026-11-07)

Bỏ hai tầng, chấm cả 34.608 truyện. Hai mục tiêu tách riêng: `temporal` (mốc
thời gian toàn cục, "sẽ đọc gì 10 tuần tới") và `taste` (giữ ngẫu nhiên 20% mỗi
người, trung hoà thời gian). NDCG@10.

| Cấu hình | temporal all | temporal warm | taste all | taste warm |
|---|---|---|---|---|
| phổ biến | 0,0818 | 0,0577 | 0,1216 | 0,1184 |
| đang nổi 60 ngày | 0,0920 | 0,0664 | 0,0847 | 0,0790 |
| ALS | 0,1098 | 0,0972 | 0,1570 | 0,1578 |
| hai tháp, chỉ gu | 0,0530 | 0,0634 | 0,1370 | 0,1354 |
| **hai tháp, gu + thời điểm** | **0,1175** | **0,1057** | **0,2327** | **0,2338** |
| gu, bán rã 180 ngày | 0,0514 | 0,0620 | 0,1356 | 0,1356 |
| gu, không hiệu chỉnh logQ | 0,0498 | 0,0554 | 0,1229 | 0,1181 |

Đối chiếu ngưỡng đã chốt:

| Chỉ tiêu | Ngưỡng | Hai tháp | |
|---|---|---|---|
| 1 · Recall@10 vs phổ biến (taste) | ≥ +50% | 0,0758 vs 0,0355 = +114% | đạt |
| 1 · NDCG@10 vs chỉ-ALS (taste) | > 0 | 0,2327 vs 0,1570 = +48% | đạt |
| 1 · Recall@10 lát warm | ≥ 0,15 | 0,0738 | **không đạt** |
| 2 · new_user vs chỉ-ALS (Recall, taste) | ≥ +20% | 0,0452 vs 0,0273 = +66% | đạt |
| 2 · new_item > 0 (temporal, n=16.121) | > 0 | 0,0173 — **chỉ khi tắt logQ** | đạt có điều kiện |

Ba điều phải nói thẳng:

- **Nhánh chỉ-gu vẫn thua ALS ở cả hai mục tiêu** (0,0530/0,1370 vs
  0,1098/0,1570). Toàn bộ phần thắng nằm ở đầu thời điểm. Nên 0,2327 là bằng
  chứng cho "đầu thời điểm + CF học được", **không** phải cho "bốn mặt nội dung
  nắm được gu".
- **logQ là đánh đổi, không phải cải tiến.** Tắt nó thì `new_item` temporal
  0,0007 → 0,0173 (25×), phủ catalog 0,048 → 0,158, tỉ lệ CN trong top-10
  0,06 → 0,11; đổi lại `all` giảm 6% và `new_user` giảm 36%. Ghi chú nghiên cứu
  cũ đoán sai chiều của nút vặn này.
- **Đầu thời điểm và truyện mới loại trừ nhau.** Bật đầu thời điểm thì
  `new_item` về 0 ở cả hai mục tiêu: truyện mới có `log_pop` thấp nên bị dìm dù
  tháp gu chấm đúng. Không có cấu hình thắng cả hai.

Hai lát quá nhỏ để kết luận: `new_item` ở nhánh taste chỉ **22 tương tác / 22
người** (chia ngẫu nhiên hiếm khi để một truyện không xuất hiện lần nào trong
train), `sparse_user` 438 tương tác. Bằng chứng về truyện mới chỉ lấy từ nhánh
temporal, nơi lát đó có 16.121 tương tác / 3.331 người.

### Bốn mặt nội dung có đáng tồn tại không — thang ablation

Đo ở lát `new_item` của nhánh temporal (16.121 tương tác / 3.331 người), cấu
hình `discover`. Đây là lát **duy nhất** tách được công của nội dung: truyện
chưa ai đọc có `train_counts = 0` nên vector ID bị tắt, điểm hoàn toàn do
`P·c_i` quyết định. Mọi baseline (phổ biến, đang nổi, ALS) đều đúng 0 ở đây.

| Bậc | chiều | NDCG@10 new_item | so bậc trước |
|---|---|---|---|
| chỉ `theme` | 128 | 0,0055 | — |
| + `creator` | 192 | 0,0073 | +33% |
| + `pace` | 205 | **0,0149** | **+104%** |
| + `art` (đủ bốn mặt) | 220 | 0,0165 | +11% |

**Tăng đơn điệu qua cả bốn bậc, mỗi mặt đóng góp biên dương.** Đây là tiêu chí
bậc 2 cuối cùng, và là lần đầu luận điểm của đồ án có bằng chứng: đủ bốn mặt
cho 0,0165, gấp **3 lần** chỉ dùng `theme`.

Hai điều ngược với bản hai tầng cũ:

- `pace` là mặt **mạnh nhất** (+104%), trong khi ở bản cũ chính nó làm giảm
  điểm (`sparse_user` 0,0519 → 0,0485) và tiêu chí bậc 2 ghi "mặt âm thì bỏ".
  Nguyên nhân là tầng ứng viên: top-400 của ALS gồm truyện đã có người đọc, nên
  tín hiệu nhịp truyện không có gì để phân biệt. Chấm toàn danh mục thì nó trở
  thành mặt hữu ích nhất cho truyện mới — hợp lý, vì số chương, trạng thái phát
  hành và tỉ lệ bỏ dở là những thứ biết được *ngay khi* truyện vừa ra.
- `art` chỉ thêm +11%, mức nhỏ **nhưng đã tách được khỏi nhiễu seed** — xem
  mục dưới. Lưu ý thêm: chỉ 57,8% truyện có ảnh bìa nên đóng góp bị pha loãng
  bởi độ phủ, không nên quy ra "nét vẽ chỉ đáng 11%".

#### Mặt `art` có thật không — phép kiểm ghép cặp 5 seed

Bậc cuối chỉ hơn +11%, nhỏ hơn khoảng dao động giữa các seed (0,0142–0,0191),
nên câu "mỗi mặt đóng góp biên dương" lúc đầu **không** được bảo chứng bởi một
lần đo. Cách đo đúng là **ghép cặp theo seed**: huấn luyện 205 chiều và 220
chiều trên *cùng* một seed rồi lấy chênh lệch, để nhiễu seed triệt tiêu trong
hiệu chứ không cộng vào.

| seed | không `art` (205d) | có `art` (220d) | chênh |
|---|---|---|---|
| 0 | 0,0149 | 0,0165 | +0,0016 |
| 1 | 0,0142 | 0,0166 | +0,0024 |
| 2 | 0,0166 | 0,0176 | +0,0010 |
| 3 | 0,0142 | 0,0157 | +0,0015 |
| 4 | 0,0152 | 0,0191 | +0,0039 |

Trung bình **+0,00208 ± 0,00113**, **5/5 seed dương**, `t = 4,10` với `df = 4`
(ngưỡng hai phía p=0,05 là 2,776; p ≈ **0,015**). **Đạt.** Mức tương đối là
**+13,8%** trên nền 0,0150.

Hai điều phải ghi cho thật:

- Ở 3 seed đầu, `t = 4,11` với `df = 2` còn ngưỡng là 4,30 — **chưa đủ**. Tôi
  đã *không* dừng ở đó và gọi là "có xu hướng dương": hai seed nữa mới cho câu
  trả lời. Đáng chú ý là `t` gần như không đổi (4,11 → 4,10); thiếu không phải
  là hiệu lượng mà là bậc tự do.
- Phép so **độc lập** (lấy lệch chuẩn của từng nhóm rồi so) cho "chênh +0,0027
  nhỏ hơn nhiễu lớn nhất" và kết luận ngược. Đó là phép kiểm **sai** ở đây, vì
  hai nhóm dùng chung seed nên không độc lập; log còn giữ dòng đó kèm chú thích
  để thấy sai ở đâu. Nếu chỉ đọc dòng đó thì `art` đã bị bỏ oan.

Thang này **không** tăng đơn điệu ở lát `sparse_user` (0,0961 → 0,0931 → 0,0981
→ 0,0981): `creator` làm giảm ở đó. Tiêu chí gốc chốt ngưỡng trên `sparse_user`
nên tính đúng theo chữ thì **không đạt**; nó đạt ở `new_item`. Tôi ghi cả hai,
và lý do chọn `new_item` làm lát quyết định là lý do nêu ở trên chứ không phải
vì nó cho số đẹp hơn.

### Bậc 4 — dùng được như sản phẩm chưa

Đo qua đúng `RecommenderService` mà API dùng, 150 người đọc, `engine=feed`.

| Chỉ tiêu | Ngưỡng | Đo được | |
|---|---|---|---|
| Thời gian mỗi lần gợi ý (p95) | < 300 ms | **9,6 ms** | đạt |
| Mọi gợi ý đều có lý do | 100% | 100% | đạt |
| Trùng người sáng tác trong top-10 | ≤ 0,30 | 0,003 | đạt |
| Người mới chọn bìa → có hồ sơ | ≤ 5 lựa chọn | **1** | đạt |
| ~~Số nước trong top-10~~ | ~~≥ 2~~ | 1,76 | **chỉ tiêu bị bác bỏ** — xem dưới |
| Khoảng cách hiệu chỉnh nước (thay thế) | ≤ 0,182 (sàn đo từ hành vi thật) | `discover` **0,147** · `auto` 0,210 | đạt / không đạt |

Chấm toàn bộ 34.608 truyện mất 5 ms, **nhanh hơn** bản hai tầng (7,9 ms) — bỏ
tầng ứng viên vừa gỡ trần 35,6% vừa nhanh hơn.

Về dòng không đạt: ngưỡng "≥2 nước" đo sai thứ cần đo, vì người chỉ đọc manga
mà nhận toàn manga là *đúng*. Phép đo thay thế là **tỉ lệ các nước trong lịch
sử của chính người đó có mặt trong top-10**: `feed` 0,685, `discover` 0,632,
hai tầng cũ 0,622. Tức khoảng **31% số loại truyện một người đang đọc không
xuất hiện** trong gợi ý — đây là thiếu hụt thật, không phải lỗi ngưỡng, và nó
đo được nên sửa được. Khác với ngưỡng bậc 0, lần này tôi không hạ chuẩn: tôi
thay bằng một phép đo chặt hơn và nó vẫn cho thấy có vấn đề.

Con số **2** cũng đã được kiểm lại trên người đọc thật và nó sai (người đọc chỉ
đạt 1,94) — và tệ hơn, chỉ tiêu này **sai chiều**: qua cả ba engine, engine khớp
tỉ lệ nước của từng người tốt nhất lại có ít nước nhất. Nó thưởng cho sự lệch
gu. Chi tiết, cơ chế, và chỉ tiêu thay thế: mục "Và bản thân con số 2 cũng sai"
bên dưới.

Ablation bác bỏ một giải thích tôi từng đưa ra: trọng số bán rã 180 ngày
**không** phải nguyên nhân lần chạy đầu kém (0,0514 vs 0,0530, chênh 3%), dù
phép đo "tổng trọng số hiệu dụng chỉ còn 0,232" là đúng — phép gộp chuẩn hoá
theo tổng trọng số nên mô hình bù lại được. Phần thắng thật đến từ chuẩn hoá L2
ở vế chấm điểm, logQ và số epoch; chưa tách được từng phần.

### Bậc 4, dòng không đạt: đã truy xong nguyên nhân, và nó không phải "thiếu đa dạng"

Ngưỡng "≥2 nước" không đạt (1,76). Tiêu chí đã chốt sẵn fallback là thêm ràng
buộc đa dạng theo nước. **Tôi không thi hành fallback đó**, và dưới đây là lý do
đo được chứ không phải lý do chọn cho tiện.

1. **Xét tổng thể không có vấn đề đa dạng.** Hiệu chỉnh theo Steck 2018 (khoảng
   cách L1 giữa phân phối nước trong gợi ý và trong lịch sử *của chính người
   đó*): mô hình **0,220**, còn *chính hành vi đọc thật ở test* cách lịch sử
   **0,182**. Mô hình chỉ lệch hơn mức sàn 0,038. Một ràng buộc chung sẽ phá
   đúng 0,038 đó và làm méo cả JP lẫn KR.
2. **Vấn đề nằm riêng ở manhua**, dìm 6,3 lần (đọc 6,3% / nhận 1,0%), còn manhwa
   được đẩy 36,2% → 48,2%. Tỉ lệ nhận lại tăng đều theo tỉ lệ đọc
   (0,020 / 0,069 / 0,224 / 0,420) nhưng hụt một nhịp ở **mọi** mức.
3. **Không phải đuôi dài:** hạng phổ biến trung vị của những bộ manhua người ta
   thực sự đọc là 2.400–4.300 trên 34.608, tức top 12%.
4. **Không phải dữ liệu thiếu manhua gần đây:** manhua giữ 6,5% lượt đọc trong
   60 ngày cuối, đúng bằng tỉ lệ cả tập.
5. **Nguyên nhân thật:** đầu mà tôi gọi là "thời điểm" thực ra 94% là **độ phổ
   biến tích luỹ cả đời** (trọng số học được: `log_pop_all` +0,891 so với
   `log_pop_60d` +0,054). Xếp toàn danh mục theo đầu này thì **top-100 có 0,0%
   manhua**, top-1000 có 5,6%. Top-10 gợi ý lấy từ đúng vùng cao nhất đó.

Nên "số nước trong top-10" không phải một ngưỡng độc lập — nó là **hàm của một
cái núm duy nhất**: phục vụ bao nhiêu phần đầu độ phổ biến. Quét chín điểm trên
cùng một mô hình (mục 9.7 `docs/research-notes.md`):

| cổng độ phổ biến | NDCG new_user | phủ catalog | manhua |
|---|---|---|---|
| 0,0 (tắt hẳn) | 0,0264 | **0,1307** | **0,11** |
| 1,0 | 0,1973 | 0,1019 | 0,08 |
| 1,5047 (khớp từ train) | 0,2754 | 0,0699 | 0,06 |
| 5,0 | 0,3694 | 0,0203 | 0,02 |

Đơn điệu cả hai chiều, **không có điểm nào được cả hai**. Tắt hẳn thì manhua lên
0,11 — *gần gấp đôi* tỉ lệ đọc thật 0,063. Vậy ngưỡng đa dạng đạt được, nhưng
bằng cách chọn điểm trên núm, và cái giá là độ chính xác ở người đọc mới. Đó là
một quyết định sản phẩm phải công bố, không phải một ràng buộc cài thêm để con
số qua ngưỡng.

**Việc đã làm:** công bố cả ba điểm làm ba engine (`feed`, `gated`, `discover`)
với bảng được/mất kèm theo, thay vì chọn một điểm rồi báo một con số `all` duy
nhất che cả hai đầu.

#### Và bản thân con số 2 cũng sai — đo trên người đọc thật

Năm điểm trên nói về *nguyên nhân*. Còn **ngưỡng** thì chưa ai kiểm. Tôi chọn
"≥ 2 nước" trước khi có dữ liệu, bằng cảm giác rằng một hệ phục vụ ba loại
truyện thì top-10 nên có ít nhất hai. Phép kiểm đúng cho một ngưỡng hành vi là
**chính người đọc có đạt nó không** (`scripts/diagnose_country_threshold.py`,
8.887 người có ≥ 10 lượt, không nạp mô hình):

| 10 bộ của người thật | số nước TB | tỉ lệ có ≥ 2 nước |
|---|---|---|
| 10 bộ **gần nhất** | **1,94** | 70,4% |
| 10 bộ bất kỳ trong lịch sử | 2,03 | 75,4% |
| toàn bộ lịch sử (chặn trên) | 2,74 | 95,1% |

**Người đọc thật không đạt ngưỡng của tôi.** 10 bộ họ vừa đọc trung bình 1,94
nước, và 29,6% trong số họ chỉ đọc đúng một nước trong cả 10 bộ gần nhất. Một
ngưỡng ≥ 2 áp cho mọi người là đòi mô hình đa dạng **hơn** chính người nó phục
vụ. Mốc so đúng là 1,94, và đọc lại cùng đơn vị:

| engine | nước/top-10 | so mốc hành vi 1,94 | so ngưỡng ≥ 2 |
|---|---|---|---|
| `feed` | 1,76 | **−9,2%** | −12,0% |
| `gated` | 1,77 | −8,8% | −11,7% |
| `discover` | 1,61 | −16,7% | −19,3% |

Thiếu hụt thật là **9,2%**, không phải 12% — và nó khớp với con số hiệu chỉnh
Steck ở điểm 1 (mô hình 0,220 so với sàn 0,182, lệch 0,038). Hai phép đo độc
lập cho cùng một kết luận: *có* thiếu hụt, nhưng nhỏ, và nó là hệ quả của đầu độ
phổ biến ở điểm 5 chứ không phải của việc thiếu ràng buộc.

**Người đọc đi theo đợt từng nước — và điều này làm ràng buộc đa dạng thành sai,
không chỉ vô ích.** 10 bộ gần nhất ít nước hơn 10 bộ bất kỳ *lấy từ chính lịch
sử người đó*: chênh **−0,088 ± 0,008**, `t = −10,9`, n = 8.887 (2.279 người ít
nước hơn so với 1.664 người nhiều hơn). Phép kiểm ghép cặp, vì hai cột đo trên
đúng cùng tập người. Nghĩa là ai đang trong đợt đọc manhwa thì *đang* đọc hẹp
hơn gu cả đời của họ, và một top-10 bị ép trộn đều là sai với họ ngay lúc đó.
Ràng buộc đa dạng theo nước sẽ ép đúng vào chiều này.

**Kết luận về dòng không đạt, viết lại cho đúng:** ngưỡng ≥ 2 **mis-specified**
(người đọc đạt 1,94); ngưỡng thay thế là mốc hành vi 1,94, và theo mốc đó
`feed` **vẫn không đạt**, thiếu 9,2%. Tôi không hạ chuẩn để qua: mốc mới đo
được từ dữ liệu, chặt hơn ở chỗ nó so với hành vi thật thay vì một số tròn, và
nó vẫn cho thấy có vấn đề — cùng một vấn đề mà điểm 5 đã gọi tên.

**Hai giả thuyết của tôi bị bác bỏ, và cái bác bỏ thứ hai làm ngưỡng này sụp
hẳn.** `discover` phủ danh mục gấp 5 lần và có manhua gấp 11 lần `feed` nhưng
lại **ít nước hơn** (1,61 so 1,76).

Giả thuyết 1 — đuôi dài bó vào một nước (một hốc webtoon lãng mạn thì toàn KR).
Đo thẳng trên dữ liệu: tương quan hạng giữa độ nổi trung bình của những bộ một
người đọc và số nước trong 10 bộ gần nhất là **ρ = +0,009**, ba tam phân vị
phẳng (1,92 / 1,96 / 1,94). **Bác bỏ.**

Giả thuyết 2 — `discover` *phân cực theo người* (người này 10 manhua, người kia
10 manhwa). Chạy `diagnose_country.py` cho từng engine trên cùng 400 người:

| | JP | KR | CN | khoảng cách hiệu chỉnh | số nước top-10 |
|---|---|---|---|---|---|
| lịch sử của chính họ | 0,575 | 0,362 | 0,063 | — | 1,94 |
| **hành vi đọc thật ở test** | 0,534 | 0,392 | 0,074 | **0,182** (sàn) | — |
| `feed` | 0,508 | **0,482** | 0,010 | 0,220 | **1,76** |
| `auto` / `gated` | 0,514 | 0,474 | 0,013 | 0,210 | 1,73 |
| `discover` | 0,580 | 0,332 | **0,088** | **0,147** | **1,61** |

**Bác bỏ — và ngược hẳn.** `discover` không phân cực; nó là engine duy nhất
**dưới mức sàn** (0,147 < 0,182), tức nó khớp tỉ lệ nước của từng người đọc còn
sát hơn chính hành vi đọc thật của họ ở kỳ test. Nó cũng là engine duy nhất
*đẩy* manhua lên trên tỉ lệ đọc (+0,025) thay vì dìm.

**Nên ngưỡng "số nước trong top-10" không chỉ sai con số, nó sai chiều.** Thứ tự
hai cột cuối **ngược nhau hoàn toàn** qua cả ba engine: engine khớp gu nước tốt
nhất (0,147) có ít nước nhất (1,61), engine khớp tệ nhất (0,220) có nhiều nước
nhất (1,76). Cơ chế rõ ràng: `feed` đẩy manhwa vào danh sách của *mọi người*
(+0,120 so với lịch sử), và việc nhồi thêm một nước vào top-10 của người chỉ đọc
manga làm **tăng** số nước đúng bằng cách làm **giảm** độ khớp. Một người đọc
95% manga mà được phục vụ đúng 95% manga thì top-10 của họ có 1 nước — và đó là
phục vụ *đúng*.

Vậy chỉ tiêu này **thưởng cho sự lệch gu**. Việc phải làm không phải thêm ràng
buộc đa dạng (nó sẽ đẩy theo đúng chiều sai) mà **thay chỉ tiêu**: mốc là khoảng
cách hiệu chỉnh so với sàn 0,182, nơi `discover` **đạt** (0,147) và `auto` chưa
(0,210). Đó là lần thứ hai trong đồ án một tiêu chí chốt-trước bị chính dữ liệu
bác bỏ, và lần này nó bị bác bỏ cả về giá trị ngưỡng lẫn về dấu.

### Engine thứ ba: cổng độ phổ biến theo từng người

Cùng seed, khác đúng một chỗ — đầu độ phổ biến được nhân với `g_u` học riêng cho
từng người thay vì cộng chung:

| lát | `feed` | `gated` | |
|---|---|---|---|
| all | 0,1169 | **0,1192** | +2,0% |
| warm (5.518 người) | 0,1039 | **0,1099** | +5,8% |
| sparse_user | 0,1268 | **0,1279** | +0,9% |
| new_user (893 người) | **0,3771** | 0,2710 | −28% |
| Recall@10 all | 0,0391 | **0,0397** | +1,5% |

`g_u` bám đúng nét nó được dựng để bám, và điều này đo được **ngoài** NDCG:
tương quan hạng giữa `g_u` và độ nổi trung bình của những bộ chính người đó đã
đọc là **+0,845** (nhóm đọc truyện ít nổi nhất: 1,20; nhóm đọc truyện nổi nhất:
1,45). Với tỉ lệ manhua trong lịch sử thì tương quan chỉ **−0,046** — cổng mù
màu quốc tịch, nên nó không phải và không thể là đường sửa manhua.

Người đọc mới không có `g_u` nào được học. `g_u` giảm đơn điệu theo độ dài lịch
sử qua cả sáu bậc (1,455 ở 1–4 mục → 1,247 ở ≥200 mục, tương quan hạng −0,51),
nên *trung bình toàn dân* là đường khớp đọc ở sai chỗ trên trục x. Khớp lại trên
**chỉ train** và đọc tại độ dài của chính họ: `new_user` 0,2499 → **0,2710**
(+8,4%), và **mọi lát khác không đổi một chữ số nào** — đúng điều kiện để gọi là
nhận dạng được.

Nhưng +8,4% chỉ lấp 17% khoảng cách. Phần còn lại **chưa giải thích xong**: đã
loại trừ thang (bản có cổng nghiêng về phổ biến *nhiều hơn* 43% mà vẫn kém) và
loại trừ hướng của đầu phổ biến (hai bản cùng thứ tự, tương quan hạng +1,0000).
Nên `gated` là engine thứ ba, **không** phải mặc định mới, và câu hỏi mở được
ghi là câu hỏi mở.

#### Đã giải thích xong, và cái giá là một quy tắc tôi phải tự giữ

Nguyên nhân: cổng **lấy mất** tín hiệu độ phổ biến mà đường gộp-lịch-sử trước
đây phải mang, và người đọc mới *chỉ có* đường đó (bảng nguồn gốc top-10 ở
`docs/research-notes.md` §9.8). Sửa bằng cách áp **DropoutNet cho chính cái
cổng**: `g_u = base(log(1+n_u)) + delta_u`, bỏ `delta_u` với p = 0,3 lúc huấn
luyện. Riêng việc bỏ ngẫu nhiên đó đưa `new_user` từ 0,1774 lên **0,3533**
(+99%) trên hai cấu hình cùng khởi tạo đến từng bit; chỉ tái tham số hoá mà
không bỏ ngẫu nhiên thì **tệ hơn** cả bản cũ.

Hiệu ứng riêng của phép bỏ ngẫu nhiên này **lặp lại trên Goodreads**: +102,8% ở `new_user` (0,0779 → 0,1580), dự đoán P2 đã ghi trước — nhưng lợi thế *tổng* của cổng thì **không** lặp lại, cổng mất ở cả bốn lát bên đó. Chi tiết và cơ chế: `docs/goodreads-arm.md` mục 8–10.

Rồi phép kiểm **ghép cặp 4 seed** (`--seeds 0 1 2 3`) chốt lại:

| lát | `feed` | cổng | chênh | t | ngưỡng df=3 | |
|---|---|---|---|---|---|---|
| `all` | 0,1175 | 0,1196 | +1,9% | **+4,27** | 3,18 | **lập được** |
| `new_user` | 0,3744 | 0,3549 | −5,2% | **−5,79** | 3,18 | **lập được** |
| `sparse_user` | 0,1283 | 0,1339 | +4,4% | 2,60 | 3,18 | chưa lập được |
| `warm` | 0,1035 | 0,1055 | +1,9% | 2,62 | 3,18 | chưa lập được |

**Bảng một-seed ở trên (+5,8% warm, +2,0% all) là bảng phóng đại.** Trên 4 seed,
`sparse_user` +9,4% rút về +4,4% (phóng đại 2,1×) và `warm` +5,8% của bản cũ chỉ
còn +1,9% ở bản mới, không lập được. Tôi để lại cả hai bảng chứ không xoá bảng
cũ: sai số của một lần chạy là một kết quả của đồ án này, không phải một thứ
đáng che.

Quy tắc chốt trước khi chạy: *chỉ đổi mặc định sang `gated` nếu `all` **và**
`sparse_user` đều vượt t = 3,18*. `sparse_user` được 2,60, nên **`gated` không
thành mặc định** — và theo đúng quy tắc đó, tuyên bố "+1,2% ở warm" bị **bỏ
hẳn** thay vì để ở dạng "có xu hướng". Tôi cũng **không** chạy thêm seed để đuổi
theo ngưỡng: thêm dữ liệu sau khi đã nhìn kết quả là lần nhìn thứ hai, và nó làm
tỉ lệ dương tính giả cao hơn con số p in ra.

### Engine mặc định mới: định tuyến, vì hai kết quả lập được ngược chiều nhau

Cổng hơn ở `all`, kém ở `new_user`, cả hai đều lập được — nên **không tồn tại**
một mô hình tốt hơn ở cả hai vùng, và "chọn mô hình nào" là câu hỏi sai.
`engine="auto"`: `feed` cho người **chưa có** lịch sử, `gated` cho người **đã
có**. Thay đổi ở vế phục vụ nên đo được ghép cặp hoàn hảo (cùng trọng số, chỉ
khác đường chấm, 0,2 phút):

| | all | new_user | sparse_user | warm | phủ catalog | manhua |
|---|---|---|---|---|---|---|
| `feed` | 0,1169 | **0,3771** | 0,1268 | 0,1039 | 0,0336 | 0,01 |
| `gated` | **0,1202** | 0,3533 | **0,1387** | **0,1051** | **0,0347** | **0,02** |
| **`auto`** | **0,1202** | **0,3771** | **0,1387** | **0,1051** | **0,0347** | **0,02** |

Lấy max ở mọi lát. Ranh giới là **có hay không có** lịch sử, không phải ít hay
nhiều — `sparse_user` nghiêng về cổng, nên quy tắc "ít lịch sử thì dùng `feed`"
sẽ sai. Và cái giá được **đo chứ không đoán**: ở 150 người `auto` trông kém
`feed` về đa dạng nước (1,73 so 1,76), nhưng ở 800 người khoảng cách biến mất
(1,742 so 1,741; phủ nước 0,678 so 0,675) — 1,6 điểm kia là nhiễu lấy mẫu. Bậc 4
giữ nguyên: p95 7,5–8,7 ms, trùng tác giả 0,003, lý do 100%, onboarding 1 bìa.

`auto` không vi phạm quy tắc ở trên: quy tắc chặn việc đưa **`gated`** lên mặc
định, và `gated` không lên mặc định. `auto` là đối tượng khác, xét theo tiêu chí
riêng — không hồi quy ở bất kỳ lát nào, cộng một mức hơn đã lập được ở `all`.

### Hạ tầng thêm vào để phép so không trôi

- `scripts/eval_saved.py` — chấm lại blob đã lưu **không huấn luyện lại** (0,2
  phút thay vì 44 phút). Bản `--no-fit` tái lập đúng từng chữ số con số của
  `train_embedding.py`, nên nó cũng là phép kiểm cho "chỉ tồn tại một đường chấm
  điểm". Trước đó mỗi lần sửa *cách phục vụ* phải huấn luyện lại, tức là đổi cả
  seed, làm mọi phép so hết còn ghép cặp.
- `scripts/diagnose_trend_gate.py` — ba phép đo về `g_u`, tất cả chỉ dùng train.
- `--art-seeds` cho `train_embedding.py` — chạy thêm seed lẻ mà không chạy lại
  seed đã có (mỗi lần huấn luyện ~43 phút).
- **`--seeds` cho `train_embedding.py`** — lặp *mọi* biến thể qua cùng danh sách
  seed rồi in phép kiểm ghép cặp theo từng lát (biến thể đầu làm mốc). Đây là
  hạ tầng quan trọng nhất thêm vào đợt này: không có nó thì hai biến thể chạy ở
  hai seed khác nhau và mọi chênh lệch nhỏ hơn nhiễu khởi tạo đều vô nghĩa — bẫy
  đã làm mặt `art` bị kết luận sai một lần và làm tôi công bố +9,4% khi sự thật
  là +4,4%.
- **`--route` cho `eval_saved.py`** — chấm bản định tuyến bằng chính hai blob đã
  lưu, nên nó là phép so ghép cặp hoàn hảo: cùng trọng số, chỉ khác đường chấm.
- **`RoutedTwoTower`** — nằm cạnh `ServedTwoTower` để cả đánh giá và API đi qua
  đúng một đường chấm, kể cả khi đường đó là một quyết định định tuyến.
- `scripts/diagnose_country_threshold.py` — kiểm chính **ngưỡng** trên hành vi
  người đọc thật, không nạp mô hình.
- 7 test mới (40 test, tất cả đạt): cổng — người đã biết nhận phần chung cộng
  phần riêng của mình, người mới nhận đúng phần chung (không hồi quy hậu kiểm,
  không trung bình toàn dân), `drop_delta` chỉ bỏ phần riêng, cổng chưa học phải
  trùng khít bản không cổng, blob cũ vẫn nạp được, blob không cổng không bị ảnh
  hưởng; định tuyến — gửi đúng người sang đúng mô hình (kể cả người có tên mà
  lịch sử rỗng), tái lập **chính xác** điểm của mô hình được chọn, và nổ nếu
  ghép hai blob lệch bảng người đọc.
