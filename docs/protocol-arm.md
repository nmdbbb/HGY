# Nhánh "hướng ban đầu": đổi GIAO THỨC, giữ nguyên dữ liệu

> **So hai hướng:** `docs/so-sanh-hai-huong.md` chạy đúng Item-KNN và
> Session-KNN của nhánh này **dưới giao thức của nhánh dữ liệu**, nên khác biệt
> giao thức bị triệt tiêu hoàn toàn. Kết quả: trên Goodreads họ láng giềng hơn
> cả ALS lẫn hai tháp ở mọi lát trừ `new_item`; trên AniList thì ngược lại. Thứ
> tự mô hình là tính chất của bộ dữ liệu, không của hướng nào.

Đồ án có hai nhánh đối chứng, và chúng hỏi hai câu khác nhau. Trộn chúng vào một
bảng là cách chắc chắn nhất để đọc sai cả hai:

| nhánh | đổi gì | giữ gì | câu hỏi |
|---|---|---|---|
| `docs/goodreads-arm.md` | **dữ liệu** (AniList -> Goodreads) | giao thức | phương pháp có lặp lại trên bộ khác không |
| nhánh này | **giao thức** (mốc thời gian chung -> 70/10/20 theo từng người, k-core 10/5, gộp bộ) | dữ liệu (Goodreads) | kết quả đổi bao nhiêu khi đổi cách đo |

Nhánh này cài đúng giao thức mà đề xuất đồ án (bản 22/10/2026) chốt, để (a) so
được với mốc HGN công bố trên cùng bộ Goodreads-Comics, và (b) biết phần nào của
những con số thấp ở nhánh kia là **tính chất của giao thức** chứ không của mô
hình. Một mô hình không đổi, đo bằng hai giao thức, cho hai con số khác bậc —
nên câu "NDCG 0,06" chỉ có nghĩa khi đi kèm giao thức.

Mã: `src/truyen_reco/data/protocol_peruser.py`, `scripts/prepare_protocol.py`,
`scripts/run_protocol_baselines.py`, test ở `tests/test_protocol.py`.

## 1. Bốn chỗ khác giao thức, mỗi chỗ là một lựa chọn của đề xuất

| # | nhánh Goodreads hiện tại | nhánh này (đề xuất) |
|---|---|---|
| dương | nhãn suy từ trạng thái đọc + tiến độ | **rating >= 4** (score >= 80); `read` và `either` có sẵn làm phép thử |
| k-core | 5/5 | **10/5** |
| chia tập | **một mốc thời gian chung** cho cả tập | **70/10/20 theo thời gian của từng người** |
| đơn vị mục | tác phẩm (`work_id`, đã gộp ấn bản) | thêm lựa chọn **gộp tập thành bộ** |

Chỗ nặng nhất là cách chia. Với mốc chung, kỳ test là một *thời kỳ khác* và có
cả người đọc chưa từng xuất hiện trong train (lát `new_user` tồn tại được là vì
thế). Với 70/10/20 theo từng người, mọi người test đều có lịch sử, mục test là
phần sau của chính chuỗi đọc của họ, và không có lát người-mới nào cả. Hai cách
đó đo hai bài toán khác nhau; cách thứ hai là cách HGN và phần lớn văn liệu
dùng, nên nó cần có để đối chiếu.

## 2. Gộp bộ truyện: trường `series` KHÔNG dùng được trực tiếp

Đề xuất ghi ở mục 3.1 bước 4: gộp theo `series`/`work_id`, "Naruto Vol.1…72
thành một item". Hai nửa của câu đó không bằng nhau:

- **`work_id` gộp ấn bản** (cùng một quyển in lại, dịch sang tiếng khác). Tầng
  đọc dữ liệu đã làm: 89.411 `book_id` -> 62.944 `work_id`, rơi 29,6% dòng
  trùng. Đây *không* phải gộp bộ.
- **`series` không gộp được bộ**, vì trong dữ liệu này mỗi *tập* có series_id
  riêng. Đo thật: series 155838 "The Sandman" ứng với đúng **một** work trong
  tập comics (`series_works_count` toàn Goodreads là 33), còn các tập khác nằm ở
  series_id khác, và tên "The Sandman" xuất hiện ở nhiều series_id. Gộp theo
  series_id cho 39.910 nhóm trên 39.910 tác phẩm — tức không gộp được gì.

Nên phải chọn chiến lược khác, và hai chiến lược khả dụng khác nhau **về bản
chất**, không phải về tham số (`scripts/build_series_map.py` và mục gộp của
`protocol_peruser.series_groups`):

| chiến lược | cách nhóm | tác phẩm trong nhóm nhiều tập | **% tương tác** | mục còn lại | nhóm lớn nhất |
|---|---|---|---|---|---|
| `volume` (**mặc định**) | ngoặc `"(Tên bộ #n)"` trong tiêu đề | 9.147 | **36,8%** (dương: 42,0%) | 55.475 (−11,8%) | One Piece 86, Naruto 71, Fairy Tail 60 |
| `franchise` | tên bộ lấy từ series_id | 37.224 | 78,9% (dương: 84,2%) | 31.717 (−49,6%) | **Batman 316**, The Walking Dead 203, "Marvel Masterworks" 168 |

`volume` bắt đúng thứ đề xuất nói: chuỗi tập được đánh số. `franchise` phủ rộng
hơn nhưng gộp quá tay — Batman thành **một** mục gồm 316 tác phẩm trải mấy thập
kỷ và nhiều mạch truyện độc lập, còn "Marvel Masterworks" là một *dòng xuất
bản*, không phải một bộ. Gộp như thế thì xoá luôn tín hiệu thật, nên `franchise`
giữ lại làm phép thử độ bền, không phải để chọn con số dễ chịu hơn.

Con số đáng nhớ: **36,8% tương tác** nằm trên những tác phẩm có tập khác cùng bộ
trong danh mục. Nghĩa là "gợi tập kế tiếp" không phải một lỗi nhỏ ở rìa — nó là
hơn một phần ba dữ liệu.

## 3. Quy mô sau tiền xử lý, so với mốc HGN

`python scripts/prepare_protocol.py`:

| chỉ số | `none` | `volume` | HGN công bố |
|---|---|---|---|
| dòng dương (rating >= 4) | 3.267.868 | 3.267.868 | — |
| số bộ được gộp | — | 3.190 | — |
| dòng bị dồn khi gộp | — | **810.643** | — |
| dòng sau k-core 10/5 | 2.779.017 | 1.949.825 | 2.411.314 |
| người đọc | 58.413 | 48.636 | 34.445 |
| mục | 37.144 | 31.222 | 33.121 |
| mật độ | 0,128% | 0,128% | 0,211% |
| train / val / test | 1.919.974 / 280.040 / 579.003 | 1.344.081 / 196.524 / 409.220 | — |
| lịch sử train trung vị | 16 | 14 | — |

**Không ép cho trùng HGN.** Số người của tôi gấp 1,70 lần họ với số dòng chỉ gấp
1,15 lần, và nguyên nhân đo được: tầng đọc dữ liệu gộp ấn bản theo `work_id`
trước khi lọc, nên mỗi mục có nhiều lượt hơn, vượt ngưỡng `item >= 5` nhiều hơn,
rồi kéo theo nhiều người vượt ngưỡng `user >= 10`. HGN lọc ở mức `book_id`. Khác
nhau ở bước đó thì quy mô không thể trùng, và việc chỉnh ngưỡng lọc cho tới khi
quy mô trùng chính là chọn tiền xử lý theo con số mình muốn. Mốc HGN
(Recall@10 0,1743 / NDCG@10 0,1927) vì vậy là **điểm tham chiếu**, không phải
đích.

## 4. Ba lỗi giao thức, cả ba lộ ra từ dữ liệu chứ không từ đọc lại code

Cả hai đều thuộc loại không báo gì cả, chỉ làm sai số.

**(1) `0,7 + 0,1 = 0,7999999999999999`.** Mốc cắt val tính bằng
`floor(n * (0,7 + 0,1))`, nên người có đúng 10 mục — nhóm đông nhất, vừa qua
ngưỡng k-core — cho `floor(7,999...) = 7`, trùng mốc train, và **tập val của họ
rỗng**. 70/10/20 lặng lẽ thành 70/0/30 ở đúng nhóm đó. Đo bằng cách dựng lại
cả hai lần: val 269.143 -> 280.040 dòng (**+10.897**, tức 3,9% tập val đã
nằm sai chỗ), test 590.059 -> 579.003, train +159. Sửa bằng cộng epsilon trước khi `floor`, cộng một ràng buộc val và
test mỗi tập ít nhất một mục. Test `test_moi_nguoi_con_it_nhat_mot_muc_test` giữ
chỗ này; chính nó bắt lỗi, không phải tôi đọc lại code.

**(2) Đáp án chứa mục không mô hình nào với tới được.** Đề xuất ghi: loại truyện
đã đọc và **cùng series** khỏi *rổ ứng viên*. Làm đúng câu đó mà không làm gì
thêm thì một mục test là tập kế tiếp của bộ đang đọc bị loại khỏi ứng viên nhưng
vẫn nằm trong đáp án — không ai với tới được, nhưng nó vẫn vào mẫu số của
Recall và NDCG. Hai hậu quả: mọi mô hình bị trừ điểm vì một thứ không phải lỗi
của mô hình, và **tỉ lệ bị trừ khác nhau giữa bản gộp và bản không gộp** — tức
đúng phép so mà nhánh này tồn tại để làm thì bị méo. Quy tắc đã sửa:
`đáp án = đáp án ∩ rổ ứng viên` (`protocol_peruser.reachable_test`), và số dòng
bị bỏ được **in ra** thay vì nằm im.

**(3) Biến thể "không gộp" chạy mà KHÔNG có phép loại cùng bộ.** Dấu hiệu nằm
ngay trong log: `bỏ vì cùng bộ/đã đọc **0**`. Nguyên nhân: `run()` chỉ dựng bản
đồ bộ truyện *khi bật gộp*, nên biến thể `none` mất hẳn phép loại — đúng chỗ nó
quan trọng nhất, vì bản đã gộp thì các tập đã là một mục nên chẳng còn gì để
loại. Hậu quả không phải sai số nhỏ: Item-KNN được **0,1537** (80% mốc HGN) nhờ
phần điểm "gợi tập kế tiếp", rồi về **0,0990** khi phép loại chạy thật. Sửa bằng
cách dựng bản đồ bộ không phụ thuộc lựa chọn gộp; test
`test_ban_do_bo_duoc_dung_ca_khi_khong_gop` giữ chỗ. Số trung vị mục bị loại mỗi
người: 18 -> **43**. Lần chạy bị nhiễm được giữ lại làm nhóm đối chứng ở mục 8,
vì nó đo đúng một biến.

Cả ba lỗi cùng một họ: **không có gì báo lỗi**. Chương trình chạy xong, in một
bảng đẹp, và bảng đó sai. Thứ bắt được chúng là một con số trông vô hại trong
log (`0 dòng bị loại`), một test về trường hợp biên (người có đúng 10 mục), và
một phép tự hỏi "mẫu số này gồm những gì".

## 5. Chốt trước khi chạy mô hình

Ở thời điểm viết mục này tôi **đã** thấy ba mốc rẻ nhất trên bản `none` (chạy
với bản chia còn lỗi và chưa cắt đáp án: ngẫu nhiên 0,0004, phổ biến 0,0314,
điểm Bayes 0,0062), nên ba cấu hình đó không được tính là dự đoán. Item-KNN,
User-KNN và ALS thì **chưa chạy lần nào** trên nhánh này.

| # | dự đoán | bác bỏ nếu |
|---|---|---|
| **Q1** | Item-KNN ngang bậc với ALS (chênh < 20% tương đối) ở NDCG@10, vì dữ liệu thưa và đề xuất cũng ghi "Item-KNN tinh chỉnh tốt thường cạnh tranh sát mô hình neural" | một trong hai hơn bên kia > 50% tương đối → "KNN ngang MF trên dữ liệu thưa" là câu nói theo quán tính, không đúng ở đây |
| **Q2** | Cấu hình tốt nhất trong năm mốc L2 này **không đạt** mốc HGN: NDCG@10 <= 0,10, tức dưới một nửa 0,1927. Lý do: HGN là mô hình chuỗi có gating, và giao thức ở đây còn loại thêm ứng viên cùng bộ (HGN không loại) nên bài toán khó hơn | một mốc L2 đạt >= 0,1927 → nghi can đầu tiên là **rò dữ liệu**, phải soi lại phép loại trừ trước khi mừng |
| **Q3** | Gộp bộ (`volume`) làm NDCG@10 **giảm** ở mọi mô hình, và giảm mạnh nhất ở Item-KNN — mô hình bám đồng xuất hiện nhiều nhất, nên mất nhiều nhất khi phần "tập kế tiếp" bị xoá khỏi cả ứng viên lẫn đáp án | gộp bộ làm **tăng** NDCG ở đa số mô hình → phần điểm từ "tập kế tiếp" không phải thứ tôi nghĩ, và mục 2 phải viết lại |
| **Q4** | Mốc `phổ biến` ở giao thức này **yếu hơn tương đối** so với ở giao thức mốc-thời-gian-chung, vì chia theo từng người thì mục test của mỗi người nằm rải theo gu riêng chứ không dồn vào cùng một thời kỳ nổi | `phổ biến` mạnh hơn (so theo tỉ số với mô hình tốt nhất cùng giao thức) → thiên lệch phổ biến không phụ thuộc cách chia, và mục 10 của `goodreads-arm.md` phải nói lại cho hẹp hơn |

### Thêm Q5, ghi trước khi chạy mô hình nội dung

Mục 7.5 của đề xuất kỳ vọng: "gợi ý theo nội dung thường thua lọc cộng tác về
NDCG tổng thể nhưng thắng ở truyện ít tương tác". Đó là **hai** tuyên bố, nên
tách thành hai điều kiện và đo bằng hai lát trên **cùng một hàm gợi ý**: lát
chung, và lát chỉ gồm những dòng test có mục nằm ngoài 20% truyện đầu bảng.

| # | dự đoán | bác bỏ nếu |
|---|---|---|
| **Q5a** | Nội dung **thua** Item-KNN ở NDCG@10 chung, và thua rõ: <= 70% của Item-KNN | nội dung ngang hoặc hơn Item-KNN ở lát chung → lọc cộng tác không có lợi thế như kỳ vọng trên bộ này |
| **Q5b** | Nội dung **hơn** Item-KNN ở lát mục ít tương tác | nội dung cũng thua ở lát đó → câu "nội dung cứu long-tail" không đúng trên bộ này, và mục 7.5 của đề xuất phải viết lại cho hẹp hơn |

### Thêm Q6, ghi trước khi chạy Session-KNN

Đề xuất đặt câu hỏi "thứ tự đọc có giá trị?" (mục 7.3) và trả lời nó bằng
GRU4Rec/SASRec — vài giờ huấn luyện. Có một phép thử rẻ hơn hẳn trả lời *cùng*
câu đó trước khi bỏ tiền: **Session-KNN** chính là Item-KNN với hồ sơ chỉ gồm
`L = 5` mục gần nhất. Hai mô hình dùng cùng một ma trận tương đồng, khác đúng
một chỗ là *dùng bao nhiêu lịch sử và theo thứ tự nào*.

| # | dự đoán | bác bỏ nếu |
|---|---|---|
| **Q6** | Session-KNN (L=5) **thua** Item-KNN (toàn hồ sơ) ở NDCG@10, vì (a) mốc thời gian của bộ này là `date_added` — lúc *thêm vào kệ*, một ước lượng yếu cho thứ tự đọc thật — và (b) lịch sử trung vị chỉ 16 mục nên cắt còn 5 là bỏ hai phần ba tín hiệu | Session-KNN **ngang hoặc hơn** Item-KNN → thứ tự gần đây mang nhiều thông tin hơn cả hồ sơ đầy đủ, và lúc đó mô hình chuỗi (L6) xứng đáng với thời gian huấn luyện |

Thêm một biến thể có giảm trọng số theo bậc (0,7 mỗi bậc lùi về trước) để tách
"dùng ít lịch sử" khỏi "ưu tiên gần đây": nếu bản giảm trọng số hơn bản đều thì
*thứ tự* có giá trị kể cả khi *lượng* lịch sử ít.

**Cách đọc phép quét L, chốt trước khi xem kết quả.** Nếu Q6 bị bác bỏ thì còn
một giả thuyết cạnh tranh phải loại: chia 70/10/20 theo từng người làm kỳ test
**nằm sát ngay sau** kỳ train, nên mục gần nhất gần về thời gian một cách máy
móc — đó là tính chất của *giao thức*, không phải của hành vi đọc. Quét
`L = 1, 2, 3, 5, 10, 20, 50, 150` ở cả hai biến thể (trọng số đều và giảm 0,7),
đọc theo đúng ba trường hợp này:

1. **NDCG đạt đỉnh ở L nhỏ nhất (1–2) rồi giảm đơn điệu** → tín hiệu gần như
   hoàn toàn là "gần nhau về thời gian"; không kết luận được gì về *thứ tự*.
2. **NDCG đạt đỉnh ở L trung gian (5–20)** → có hai lực ngược chiều: quá ít lịch
   sử thì thiếu tín hiệu, quá nhiều thì hồ sơ bị loãng. Lúc đó "thứ tự có giá
   trị" là kết luận có cơ sở, và nó còn cho biết nên cho mô hình chuỗi nhìn lại
   bao xa.
3. **L = 150 trọng số đều ≈ Item-KNN (0,0990)** là phép **kiểm tính đúng đắn của
   bản cài**: lịch sử trung vị chỉ 16 mục, nên với gần hết người đọc thì "150 mục
   gần nhất, trọng số đều" *chính là* toàn bộ hồ sơ. Hai số đó lệch nhiều thì lỗi
   nằm ở code, không phải ở dữ liệu, và mọi dòng Session-KNN phải bỏ.

Lưu ý khi đọc Q5b: lát ít tương tác có **tập người đọc khác** (chỉ những ai có ít
nhất một mục ngoài đầu bảng trong test), nên con số của nó không so trực tiếp
được với lát chung — chỉ so *giữa các mô hình trong cùng lát*.

Quy tắc đọc giữ nguyên như nhánh kia: chênh lệch nhỏ hơn nhiễu khởi tạo thì phải
ghép cặp theo seed; "lặp lại" là cùng dấu và cùng kết luận, không phải cùng con
số; và dự đoán bị bác bỏ thì ghi là bị bác bỏ.

## 6. Kết quả bậc thang L2+L3, hai biến thể

NDCG@10, xếp hạng **toàn danh mục**, mẫu 2.000 người test (seed 0), đáp án đã
cắt về rổ ứng viên. Cùng một bộ ứng viên và cùng phép loại cho mọi cấu hình.

| cấu hình | NDCG@10 | NDCG ít-lượt | Recall@10 | HR@10 | phủ | mới lạ | ARP | long-tail |
|---|---|---|---|---|---|---|---|---|
| **không gộp bộ** (`none`, 36.241 mục) | | | | | | | | |
| ngẫu nhiên | 0,0001 | 0,0001 | 0,0001 | 0,0010 | 0,424 | 12,05 | 0,500 | 0,991 |
| phổ biến | 0,0345 | **0,0000** | 0,0374 | 0,1290 | 0,001 | 3,12 | 1,000 | 0,000 |
| điểm Bayes | 0,0062 | **0,0000** | 0,0076 | 0,0290 | 0,001 | 6,85 | 0,966 | 0,409 |
| **Item-KNN** | **0,0990** | 0,0258 | **0,1117** | 0,2790 | 0,082 | 5,74 | 0,967 | 0,320 |
| User-KNN | 0,0900 | 0,0101 | 0,1073 | 0,2790 | 0,042 | 4,85 | 0,988 | 0,163 |
| ALS | 0,0756 | **0,0010** | 0,0924 | 0,2625 | 0,053 | 5,59 | 0,988 | 0,295 |
| BPR-MF (20 epoch) | 0,0537 | 0,0068 | 0,0650 | 0,1960 | 0,050 | 5,04 | 0,985 | 0,197 |
| BPR-MF (60 epoch) | 0,0597 | 0,0111 | 0,0726 | 0,2105 | 0,066 | 5,39 | 0,980 | 0,271 |
| nội dung (TF-IDF kệ + tác giả) | 0,0557 | **0,0422** | 0,0606 | 0,1485 | **0,156** | **10,17** | 0,727 | 0,925 |
| **gộp bộ** (`volume`, 30.387 mục) | | | | | | | | |
| phổ biến | 0,0372 | 0,0000 | 0,0398 | 0,1600 | 0,001 | 2,94 | 1,000 | 0,000 |
| điểm Bayes | 0,0046 | 0,0000 | 0,0055 | 0,0190 | 0,001 | 6,89 | 0,955 | 0,403 |
| **Item-KNN** | **0,1102** | 0,0177 | **0,1204** | 0,3385 | 0,087 | 5,24 | 0,973 | 0,243 |
| User-KNN | 0,0930 | 0,0044 | 0,1063 | 0,3205 | 0,043 | 4,47 | 0,991 | 0,125 |
| ALS | 0,0888 | **0,0002** | 0,1034 | 0,3175 | 0,052 | 5,26 | 0,991 | 0,234 |
| BPR-MF (20 epoch) | 0,0598 | 0,0052 | 0,0677 | 0,2490 | 0,051 | 4,75 | 0,988 | 0,162 |
| nội dung | 0,0636 | **0,0462** | 0,0645 | 0,1665 | **0,198** | **10,16** | 0,723 | 0,910 |

Cột **NDCG ít-lượt** đo trên cùng hàm gợi ý nhưng chỉ lấy những dòng test có mục
nằm ngoài 20% truyện đầu bảng (165.937/471.620 dòng, 32.122 người ở bản `none`).
Nó **không** so được với cột NDCG chung — tập người đọc và tập đáp án đều khác —
chỉ so *giữa các mô hình trong cùng một cột*.

Thứ tự **Item-KNN > User-KNN > ALS > BPR-MF > phổ biến** giữ nguyên ở cả hai
biến thể. Hai mô hình đơn giản nhất của cả đồ án đang đứng đầu nhánh này — và
sau mục 11, mô hình dẫn đầu là **Session-KNN có giảm trọng số (0,1276)**, tức
Item-KNN với hồ sơ chỉ gồm vài mục gần nhất. Không có mạng nơ-ron nào trong hai
cái đó.

## 7. Phán quyết Q1–Q4

**Q1 — không bị bác bỏ, nhưng khoảng tôi ghi thì sai.** Item-KNN hơn ALS
**+31,0%** (`none`) và **+24,1%** (`volume`). Điều kiện bác bỏ (> 50%) không
xảy ra, nên dự đoán "cùng bậc" đứng; nhưng khoảng xác nhận tôi tự đặt (< 20%)
thì không đạt. Ghi đúng như đo được: Item-KNN hơn ALS khoảng một phần tư, ổn
định qua cả hai biến thể — không phải "ngang nhau".

**Q2 — xác nhận, và phán quyết này treo trên một chi tiết giao thức.** Cấu hình
tốt nhất trên biến thể so được với HGN (`none`) đạt **0,0990**, đúng dưới ngưỡng
0,10 tôi ghi trước, tức 51% mốc HGN (0,1927). Nhưng phải ghi thêm cho đủ:
**trước khi sửa lỗi 3** (mục 4), chính Item-KNN cho **0,1537** — con số đó sẽ
làm Q2 trông như bị bác bỏ. Thứ quyết định không phải mô hình mà là một dòng
code quyết định có loại ứng viên cùng bộ hay không. Đây chính là lý do nhánh này
tồn tại.

**Q3 — BỊ BÁC BỎ.** Tôi dự đoán gộp bộ làm NDCG **giảm** ở mọi mô hình. Thực tế
nó **tăng** ở 5 trong 6 cấu hình:

| | `none` | `volume` | chênh |
|---|---|---|---|
| phổ biến | 0,0345 | 0,0372 | **+7,8%** |
| điểm Bayes | 0,0062 | 0,0046 | −25,8% |
| Item-KNN | 0,0990 | 0,1102 | **+11,3%** |
| User-KNN | 0,0900 | 0,0930 | **+3,3%** |
| ALS | 0,0756 | 0,0888 | **+17,5%** |
| BPR-MF (bản mặc định, không chệch) | 0,0537 | 0,0598 | **+11,4%** |

Điều kiện bác bỏ đã ghi là "gộp bộ làm tăng NDCG ở đa số mô hình". Nó tăng.
**Ghi là bị bác bỏ.** Và lý do đo được chứ không phải đoán: tôi đã lẫn hai phép
khác nhau.

- **Loại ứng viên cùng bộ** (bản `none`) lấy đi phần điểm dễ *và* để lại một bài
  toán còn khó hơn: 17,4% dòng test là tập kế tiếp của bộ người ta đang đọc, nên
  chúng bị cắt khỏi đáp án, phần còn lại là những bộ *mới hẳn* — khó hơn.
- **Gộp bộ** (bản `volume`) làm phần "tập kế tiếp" biến mất theo một đường khác:
  tập 2 và tập 1 thành *cùng một mục*, nên mục test đó dồn vào chính mục đã có
  trong train và không còn là một câu hỏi nữa. Đáp án còn lại 98,5% (so với
  81,5%), và danh mục co từ 36.241 xuống 30.387 mục — xếp hạng trên danh mục nhỏ
  hơn thì dễ hơn.

Tức gộp bộ không phải "bài khó hơn" mà là **bài sạch hơn và nhỏ hơn**. Câu ở
mục 2 ("gộp bộ xoá phần điểm dễ") phải đọc lại: *loại* mới xoá điểm dễ, còn
*gộp* xoá cả điểm dễ lẫn câu hỏi không trả lời được, rồi thu nhỏ danh mục — và
tổng ba thứ đó là dương.

**Q4 — xác nhận.** So mốc `phổ biến` với ALS, mô hình duy nhất chạy ở cả hai
giao thức:

| | giao thức mốc thời gian chung | giao thức đề xuất (`none`) |
|---|---|---|
| phổ biến | 0,0461 | 0,0345 |
| ALS | 0,0734 | 0,0756 |
| **tỉ số phổ biến / ALS** | **0,628** | **0,456** |

Chia theo từng người thì mục test của mỗi người nằm rải theo gu riêng chứ không
dồn vào cùng một thời kỳ nổi, nên một bảng xếp hạng chung yếu đi hẳn — đúng
chiều đã ghi trước.

Và một con số không thuộc dự đoán nào nhưng đáng ghi: **ALS gần như không đổi
giữa hai giao thức** (0,0734 so với 0,0756), trong khi mốc `phổ biến` đổi 25%.
Phải đọc ở mức "gói giao thức" chứ không phải "một biến": hai bên khác nhau cả
định nghĩa dương, ngưỡng k-core, cách chia, lẫn phép loại ứng viên.

## 8. Giá của "gợi tập kế tiếp", đo trực tiếp

Lần chạy bị nhiễm vì lỗi 3 không bỏ đi: nó là **nhóm đối chứng** cho đúng một
biến — có hay không loại ứng viên cùng bộ, trên cùng dữ liệu `none`, cùng mô
hình, cùng seed.

| cấu hình | KHÔNG loại cùng bộ | CÓ loại | chênh |
|---|---|---|---|
| Item-KNN | 0,1537 | 0,0990 | **−35,6%** |
| User-KNN | 0,1413 | 0,0900 | **−36,3%** |
| ALS | 0,1227 | 0,0756 | **−38,4%** |
| phổ biến | 0,0299 | 0,0345 | **+15,4%** |

Hơn một phần ba điểm của mọi mô hình lọc cộng tác trên bộ này đến từ việc gợi
**tập kế tiếp của bộ người ta đang đọc**. Đề xuất gọi đó là "chỉ số cao nhưng
gợi ý vô dụng"; đây là con số của nó: 35–38%.

Chiều ngược của mốc `phổ biến` (+15,4%) xác nhận phép loại không phải một hình
phạt đều tay: khi đáp án còn lại là những bộ mới hẳn, bảng xếp hạng chung lại
*khá hơn* tương đối, vì phần bị cắt đi là phần mà chỉ mô hình cá nhân hoá với
tới được.

Chú ý khi đọc: hai cột không chỉ khác nhau ở rổ ứng viên mà còn ở **mẫu số** —
đáp án rút từ 579.003 xuống 471.620 dòng (81,5%), vì đáp án phải nằm trong rổ
ứng viên (mục 4, lỗi 2). Nên "−36%" là chênh lệch giữa **hai định nghĩa chỉ
số**, không phải giữa hai mô hình. Cả hai định nghĩa đều có thể bảo vệ được;
đề xuất chọn định nghĩa thứ hai, và nhánh này theo đề xuất.

## 9. BPR-MF sập về bảng phổ biến, và thủ phạm là số hạng chệch theo item

Bảng mục 6 có một dòng đáng ngờ chứ không đáng tin: BPR-MF đạt **0,0344**, gần
như đúng bằng mốc `phổ biến` (**0,0345**). Hai mô hình khác hẳn nhau về cấu trúc
mà trùng số đến chữ số thứ tư thì không phải trùng hợp. Ba chỉ số ngoài độ chính
xác nói thẳng chuyện gì đã xảy ra:

| | NDCG@10 | phủ catalog | mới lạ | ARP | long-tail |
|---|---|---|---|---|---|
| `phổ biến` | 0,0345 | 0,0010 | 3,12 | 0,9998 | 0,0000 |
| BPR-MF **có** chệch item | 0,0344 | 0,0053 | 3,26 | **0,9995** | **0,0067** |
| BPR-MF **không** chệch (mặc định mới) | **0,0537** | **0,0501** | 5,04 | 0,9854 | 0,1965 |
| ALS (để so) | 0,0756 | 0,0527 | 5,59 | 0,9879 | 0,2953 |

Bản có chệch gợi gần như **chỉ** đầu danh mục: ARP 0,9995 và 0,67% mục long-tail,
tức nó không cá nhân hoá gì cả. Bỏ số hạng chệch đi: NDCG **+56%**, độ phủ gấp
**9,5 lần**, tỉ lệ long-tail từ 0,7% lên 19,7%.

Lý do là một câu về hàm mất, không phải về siêu tham số. BPR tối ưu theo **cặp**:
`x_ui - x_uj`. Với số hạng chệch, hiệu đó chứa `bias_i - bias_j`, mà hiệu này
**không phụ thuộc người đọc** — nó là đúng một bảng xếp hạng chung cho mọi
người. Đó là hướng hạ hàm mất *rẻ nhất* (mỗi bước cập nhật hai số thực thay vì
hai vector 64 chiều), nên bộ tối ưu đi vào đó trước và phần nhân tử bị bỏ đói.
Trên bộ này "bảng xếp hạng chung tốt nhất" chính là độ phổ biến, nên mô hình hội
tụ về đúng mốc `phổ biến`.

**Đây là lỗi của bản tôi cài, không phải của phương pháp.** BPR-MF bản công bố
(Rendle và cộng sự, UAI 2009) không có số hạng chệch theo item, và giờ tôi hiểu
vì sao chứ không chỉ biết là nó không có. Mặc định đã đổi thành **tắt**; cờ
`--bpr-bias` giữ lại để chạy lại chính phép đo trên, không phải để dò tham số.

Hàm mất vẫn còn giảm ở epoch 20 (0,6556 -> 0,1556 -> 0,0973 -> 0,0753 ->
0,0648), nên "chưa học xong" là một giả thuyết cạnh tranh thật — và nó đã được
kiểm riêng, không để treo:

| bản có chệch | loss cuối | NDCG@10 | phủ | ARP | long-tail |
|---|---|---|---|---|---|
| 20 epoch | 0,0648 | 0,0344 | 0,0053 | 0,9995 | 0,0067 |
| 60 epoch | 0,0467 | 0,0384 | 0,0064 | 0,9993 | 0,0083 |

Gấp ba số epoch, hàm mất giảm thêm 28%, mà NDCG chỉ thêm **+11,6%** và ARP vẫn
đứng ở 0,9993. Tức huấn luyện lâu hơn **không** thoát khỏi chỗ sập; bỏ số hạng
chệch thì thoát (+56%). Giả thuyết "chưa học xong" bị loại với tư cách nguyên
nhân chính, và kết luận ở trên không dựa vào số epoch mà dựa vào **ba chỉ số
ngoài độ chính xác trùng khít mốc phổ biến** — thứ mà thêm epoch không giải thích
được.

## 10. Q5: cả hai nửa đều xác nhận, nhưng một nửa có nguy cơ là lập luận vòng

**Q5a — xác nhận.** Nội dung đạt 0,0557 ở lát chung, bằng **56,3%** của Item-KNN
(0,0990), dưới ngưỡng 70% đã ghi trước. Lọc cộng tác hơn nội dung ở độ chính xác
tổng thể, đúng như mục 7.5 của đề xuất.

**Q5b — xác nhận, và chênh lệch rất lớn.** Trên lát mục ít tương tác, thứ tự đảo
hẳn:

| mô hình (bản `none`) | NDCG chung | NDCG ít-lượt | **tỉ lệ ít-lượt / chung** |
|---|---|---|---|
| nội dung | 0,0557 | **0,0422** | **0,76** |
| Item-KNN | 0,0990 | 0,0258 | 0,26 |
| BPR-MF (60 epoch) | 0,0597 | 0,0111 | 0,19 |
| User-KNN | 0,0900 | 0,0101 | 0,11 |
| ALS | 0,0756 | 0,0010 | **0,013** |
| phổ biến | 0,0345 | 0,0000 | 0,00 |
| điểm Bayes | 0,0062 | 0,0000 | 0,00 |

Con số đáng nhớ nhất của cả nhánh này: **ALS lấy 98,7% điểm của nó từ 20% truyện
đầu bảng.** Ra khỏi vùng đó nó gần như không gợi đúng gì (0,0010), trong khi mô
hình nội dung giữ được ba phần tư phong độ. `phổ biến` và `điểm Bayes` ra **đúng
0,0000** — với `phổ biến` thì đó là định nghĩa, nhưng `điểm Bayes` có 40,9% mục
long-tail trong danh sách mà vẫn không trúng mục nào: nó chọn truyện *điểm cao*,
không phải truyện người ta *đọc*.

### Chỗ phải tự bắt: một phần của Q5b là hiệu ứng cơ cấu

Mô hình nội dung gợi 92,5% mục long-tail (ARP 0,727), còn ALS gợi 29,5%
(ARP 0,988). Một mô hình chỉ gợi mục ít lượt thì **đương nhiên** trúng nhiều hơn
trên một đáp án chỉ gồm mục ít lượt. Nếu dừng ở bảng trên thì tôi đang khen một
mô hình vì chính cái tôi dùng để chọn lát.

Chuẩn hoá thô cho đúng chỗ đó — chia NDCG ít-lượt cho tỉ lệ long-tail trong danh
sách, tức "được bao nhiêu điểm trên mỗi mục long-tail đã dám gợi":

| | NDCG ít-lượt | long-tail trong danh sách | tỉ số thô |
|---|---|---|---|
| **Item-KNN** | 0,0258 | 0,320 | **0,081** |
| User-KNN | 0,0101 | 0,163 | 0,062 |
| nội dung | 0,0422 | 0,925 | 0,046 |
| BPR-MF (60 epoch) | 0,0111 | 0,271 | 0,041 |
| ALS | 0,0010 | 0,295 | 0,003 |

Đọc đúng mức: **Item-KNN chính xác hơn trên mỗi mục long-tail nó gợi**; lợi thế
của mô hình nội dung ở lát ít-lượt phần lớn đến từ việc nó *sẵn sàng* gợi mục ít
lượt, không phải từ việc nó chọn khéo hơn. Hai câu đó khác nhau và chỉ câu thứ
nhất là lời khen về mô hình.

Phép chia này là **thô**, phải ghi rõ: chia một chỉ số xếp hạng cho tỉ lệ thành
phần danh sách không phải một phép chuẩn hoá có cơ sở, nó chỉ đủ để thấy chiều
của hiệu ứng. Muốn kết luận chắc thì phải đo độ chính xác *có điều kiện* trên
từng mục được gợi, và đó là việc chưa làm.

Nhưng kết luận dùng được cho hệ lai ghép thì đã đủ chắc, và nó khớp với thiết kế
mà đề xuất vạch ở mục 5.9: **hai lớp mô hình này không thay thế nhau mà bù nhau**
— lọc cộng tác giữ độ chính xác ở đầu danh mục, nội dung mở được phần đuôi
(độ phủ 0,156 so với 0,053 của ALS, độ mới lạ 10,17 so với 5,59). Đó chính là
lý do để lai ghép, và giờ nó là một con số chứ không phải một câu trong sách.

### BPR-MF: thiết kế 2×2 đầy đủ

| NDCG@10 | 20 epoch | 60 epoch |
|---|---|---|
| **có** chệch item | 0,0344 | 0,0384 |
| **không** chệch | 0,0537 | 0,0597 |

Hai hiệu ứng gần như cộng tính, không tương tác: bỏ chệch được +56% / +55%, thêm
epoch được +12% / +11%. Nên hai nguyên nhân tách bạch, và nguyên nhân lớn là cấu
trúc hàm mất chứ không phải ngân sách huấn luyện. Ngay cả ở cấu hình tốt nhất của
chính nó (0,0597), BPR-MF vẫn **dưới ALS** (0,0756) và dưới Item-KNN (0,0990)
trên bộ này.

## 11. Q6 bị BÁC BỎ: hai mục gần nhất hơn cả hồ sơ đầy đủ

| | NDCG@10 | NDCG ít-lượt | phủ | mới lạ | long-tail |
|---|---|---|---|---|---|
| Item-KNN (toàn hồ sơ) | 0,0990 | 0,0258 | 0,082 | 5,74 | 0,320 |
| Session-KNN (L=5) | 0,1206 | 0,0415 | 0,140 | 6,89 | 0,523 |
| **Session-KNN (L=5, giảm 0,7)** | **0,1271** | **0,0468** | **0,145** | 6,99 | 0,540 |

Điều kiện bác bỏ đã ghi là "Session-KNN ngang hoặc hơn Item-KNN". Nó hơn
**+28,4%**, và hơn ở *mọi* cột: độ chính xác, độ chính xác trên mục ít lượt, độ
phủ, độ mới lạ. **Q6 bị bác bỏ.**

### Phép quét L, đọc theo đúng ba trường hợp đã chốt

| L | NDCG@10 (đều) | NDCG@10 (giảm 0,7) | NDCG ít-lượt (đều) | phủ (đều) | long-tail (đều) |
|---|---|---|---|---|---|
| 1 | 0,1218 | 0,1218 | **0,0545** | **0,194** | 0,654 |
| **2** | **0,1244** | 0,1254 | 0,0505 | 0,169 | 0,603 |
| 3 | 0,1241 | 0,1253 | 0,0457 | 0,154 | 0,569 |
| 5 | 0,1206 | 0,1271 | 0,0415 | 0,140 | 0,523 |
| 10 | 0,1144 | **0,1276** | 0,0352 | 0,117 | 0,438 |
| 20 | 0,1066 | 0,1276 | 0,0314 | 0,099 | 0,377 |
| 50 | 0,1004 | 0,1275 | 0,0277 | 0,090 | 0,341 |
| 150 | 0,0995 | 0,1275 | 0,0260 | 0,084 | 0,323 |
| *Item-KNN* | *0,0990* | — | *0,0258* | *0,082* | *0,320* |

**Phép kiểm tính đúng đắn của bản cài: ĐẠT.** L=150 trọng số đều cho 0,0995 so
với Item-KNN 0,0990 (lệch 0,5%), mục ít-lượt 0,0260 so với 0,0258, phủ 0,0835 so
với 0,0824. Thang hội tụ về đúng Item-KNN khi L lớn — đúng như phải thế, vì lịch
sử trung vị chỉ 16 mục. Nếu hai số đó lệch nhau thì mọi dòng Session-KNN đã phải
bỏ; chúng không lệch.

**Đỉnh ở L=2, rồi giảm đơn điệu.** Đó là **trường hợp 1** trong ba trường hợp đã
chốt ở mục 5, và trường hợp 1 nói rõ: tín hiệu gần như hoàn toàn là *gần nhau về
thời gian*, **không kết luận được gì về thứ tự đọc**. Nên tôi không được phát
biểu "thứ tự đọc có giá trị" — dù nó là câu dễ viết và là câu đề xuất mong đợi.
Một phần hiệu ứng này là **tính chất của giao thức**: chia 70/10/20 theo từng
người thì kỳ test nằm sát ngay sau kỳ train, nên mục cuối của train gần kỳ test
một cách máy móc.

Phát biểu được, và nó vẫn mạnh: **ưu tiên gần đây đáng +26 đến +29% mà không
thêm một tham số mô hình nào.** Tách công của hai thành phần:

- cắt hồ sơ còn 2 mục gần nhất: 0,0990 -> 0,1244, **+25,7%**;
- thay cắt cứng bằng giảm trọng số 0,7: 0,1276, thêm **+2,6%** — và quan trọng
  hơn là **L không còn phải dò**, vì 0,7^k tắt nhanh nên L=10 và L=150 cho cùng
  một số (0,1276 so với 0,1275).

Một chi tiết đáng ghi cho hệ lai ghép: **L=1 cho độ phủ và độ chính xác long-tail
cao nhất** (0,194 và 0,0545) trong khi L=2 cho NDCG chung cao nhất. Tức có một
đánh đổi thật giữa "đúng nhiều" và "đi xa khỏi đầu danh mục", và nó nằm ở đúng
một tham số đọc được.

### Hệ quả cho L6, khắt khe hơn điều kiện tôi tự ghi

Điều kiện ghi trước là "Session-KNN ngang hoặc hơn thì mô hình chuỗi xứng đáng
thời gian huấn luyện". Nó đạt, nên GRU4Rec/SASRec vào danh sách phải chạy. Nhưng
hình dạng phép quét đổi **mốc** mà chúng phải vượt: phần lợi đến từ *trọng số
gần đây*, thứ mà một mô hình chuỗi cũng có sẵn trong kiến trúc. Nên mốc so không
còn là Item-KNN 0,0990 mà là **Session-KNN giảm trọng số 0,1276**. Một mô hình
chuỗi chỉ hơn 0,0990 thì chưa chứng minh được gì về thứ tự; nó phải hơn 0,1276.

Đây là thứ phép quét mua được: 9 phút đánh giá thay cho vài giờ huấn luyện, và
một mốc so đúng thay vì một mốc so dễ dãi.

> **Đính chính, xem mục 12:** cả phép quét này chạy trên **tập test**, nên mọi
> con số trong mục 11 là số *đã được test chọn*. Mốc đúng để so, chọn trên
> validation, là **0,1273** (L=5, giảm 0,7) — lệch 0,0003 so với 0,1276, kết
> luận không đổi, nhưng con số phải lấy từ mục 12.

## 12. Sửa một lỗi phương pháp của chính tôi: chọn tham số trên test

Phép quét `L` ở mục 11 chạy **trên tập test**, rồi tôi báo con số tốt nhất
(0,1276 ở L=10). Đó đúng là loại lỗi mà đồ án này bắt ở chỗ khác — "chọn tham số
sau khi nhìn kết quả" — và mục 7.1 của đề xuất cũng đòi tinh chỉnh trên
validation với ngân sách tương đương cho mọi mô hình. Tập val cho tới lúc đó
chưa được dùng vào việc gì.

`scripts/tune_protocol.py` làm đúng quy trình: mọi cấu hình chấm trên val, **chỉ
cấu hình thắng** được chấm trên test, đọc đúng một lần.

| mô hình | cấu hình val chọn | val NDCG@10 | **test NDCG@10** | số cũ (chọn bằng test) |
|---|---|---|---|---|
| Item-KNN | k=500, hạ trọng số 0 | 0,1634 | **0,1018** | 0,0990 (k=200, hạ 20) |
| Session-KNN | L=5, giảm 0,7 | 0,2471 | **0,1273** | 0,1276 (L=10, giảm 0,7) |

**Lỗi chọn-bằng-test đáng 0,0003** — kết luận định tính không đổi (Session-KNN
hơn Item-KNN ở *mọi* giá trị L đã thử, nên nó không phụ thuộc việc chọn), nhưng
giờ con số có nguồn đúng. Mức hơn, cả hai đều đã tinh chỉnh trên val:
**0,1273 so với 0,1018, +25,0%**.

Một chi tiết phải ghi để không ai đọc nhầm bảng: **val cho NDCG cao gần gấp đôi
test** (0,2471 so với 0,1273). Lý do là cấu trúc chia: val là 10% ngay sau train,
test là 20% cuối, nên val gần kỳ train hơn hẳn. Val dùng để **xếp hạng cấu hình**
thì được; dùng để báo mức thì không.

### Hạ trọng số cặp mỏng làm *xấu đi* trên bộ này

> **Đính chính, xem mục 14:** lời giải thích "hệ số gần đều" ở cuối phần này
> **sai** — 98,8% cặp có  và hệ số trung vị là 0,048. Cơ chế thật
> là một bộ lọc theo độ phổ biến đặt lên tập láng giềng, đo ở mục 14.

| | hạ trọng số 0 | hạ trọng số 20 |
|---|---|---|
| k = 50 | 0,1613 | 0,1579 |
| k = 200 | 0,1605 | 0,1571 |
| k = 500 | **0,1634** | 0,1577 |

Đều đặn ở cả ba mức k, hạ trọng số làm giảm 2–4% tương đối. Mục 5.2 của đề xuất
khuyên "giảm trọng số các cặp có ít truyện chung" — khuyên đó **không đúng ở
đây**, và lý do đo được: sau lọc k-core (mỗi truyện >= 5 người đọc) trên 1,9
triệu tương tác, số người đọc chung của các cặp đã đủ lớn, nên hệ số
`n/(n + 20)` chỉ còn là một độ chệch đều tay. Hạ trọng số là thuốc cho dữ liệu
mỏng; dữ liệu này không mỏng ở chỗ đó.

Số láng giềng `k` thì gần như không quan trọng (0,1613 / 0,1605 / 0,1634 cho
k = 50 / 200 / 500) — một tham số mà đề xuất dành hẳn một dòng để dò.

### Một bug nan, và hiệu ứng thật của nó: gần bằng không

Lưới val in `RuntimeWarning: invalid value encountered in divide`. Nguyên nhân:
hệ số hạ trọng số là `n_chung / (n_chung + shrink)`, nên với `shrink = 0` thì
cặp không đồng xuất hiện cho `0/0 = nan`. Nan rồi lan theo đường tệ nhất:
`np.argpartition(-line, k)` không có thứ tự xác định với nan nên nan có thể
chiếm chỗ trong k phần tử đầu, và mô hình còn **ít láng giềng hơn k** mà không
báo gì — đúng ở cấu hình `shrink = 0` mà val vừa chọn.

Đã sửa (phép chia có điều kiện, 0 người chung cho tương đồng 0), chuyển hàm ra
`src/truyen_reco/models/knn.py` vì nó là *mô hình* chứ không phải một bước chạy,
và thêm 4 test giữ chỗ.

**Nhưng phải ghi hiệu ứng thật, không được để nó trông như một pha cứu bàn:**
sau khi sửa, k=500/hạ 0 đi từ 0,1635 xuống 0,1634 — **lệch 0,0001**. Bộ lọc
`line[picked] > 0` ở bước sau đã chặn nan lại, và vẫn còn đủ láng giềng thật để
kết quả gần như không đổi. Lỗi là thật và hành vi là không xác định, nên sửa là
đúng; nhưng nó **không** làm sai bất kỳ con số nào đã công bố, và nói ngược lại
thì là phóng đại công của mình.

Một test cũng vừa bắt tôi ở chỗ khác: `6/6` trong float32 ra `0,99999994`, không
phải `1,0` chẵn — so bằng dấu `==` trên số thực là một lỗi tôi tự viết vào test.

## 13. Bản GỘP TẬP — cấu hình của chính đề xuất — tinh chỉnh trên val, và một phép đo độ phân giải của lưới

Mục 12 tinh chỉnh bản `none` (không gộp). Nhưng cấu hình mà đề xuất *yêu cầu* là
bản **gộp tập thành bộ** ("Naruto Vol.1…72 thành một item"), nên con số để nộp
phải là của bản đó. `python scripts/tune_protocol.py --dir data/processed_hgn_volume`:

| mô hình | cấu hình val chọn | val NDCG@10 | **test NDCG@10** | bản `none` |
|---|---|---|---|---|
| Item-KNN | k=500, hạ trọng số 0 | 0,1816 | **0,1157** | 0,1018 |
| Session-KNN | L=20, giảm 0,5 | 0,2661 | **0,1369** | 0,1273 |

Gộp tập nâng cả hai: Item-KNN **+13,7%**, Session-KNN **+7,5%**. Nhưng mục 7 đã
đo vì sao con số này **không** được đọc thành "gộp tập làm mô hình tốt hơn": gộp
bỏ đi những điểm dễ (tập kế tiếp của bộ đang đọc), bỏ luôn những đáp án không
với tới được, và co danh mục 36.241 → 30.387. Đó là **hai bài toán khác nhau**,
không phải một bài toán với hai mô hình.

Hạ trọng số lại thua ở **cả ba** mức k trên bản gộp (0,1786/0,1681 ở k=50,
0,1798/0,1687 ở k=200, 0,1816/0,1696 ở k=500) — cộng với bản `none` là **6 trên
6** phép so cùng chiều. Mục 14 đo ra *vì sao*, và lời giải thích tôi viết ở mục
12 sai.

So với mốc tham chiếu HGN (NDCG@10 0,1927): **0,1369 là 71%** của mốc đó, trên
một bộ có mật độ 0,128% so với 0,211%, chấm toàn danh mục, và loại cả mục cùng
bộ khỏi rổ ứng viên. Mục 3 đã ghi vì sao hai quy mô không trùng được; con số 71%
là một *vị trí*, không phải một kết luận về mô hình.

### Lưới tinh chỉnh phân giải được tới đâu? Đo, vì argmax không nói

Trên bản gộp, val chọn L=20/giảm 0,5 với 0,2661 — hơn L=10/giảm 0,5 đúng
**0,0001** và hơn L=5/giảm 0,5 đúng **0,0001**. Lấy argmax ở đó rồi báo "L=20 là
tham số tốt nhất" là một phát biểu không có dữ liệu đỡ. Mục 7.1 của đề xuất nói
"tinh chỉnh trên validation với ngân sách tương đương"; nó **không** nói phải làm
gì khi mặt val bẹt.

`scripts/tuning_noise.py` chạy cùng lưới ở 5 **mẫu người đọc** khác nhau (val
chấm trên 2.000 người, nên đổi mẫu thì số đổi). Nó không bao giờ chạm tập test.

| | độ lệch của MỘT số val | hiệu số GHÉP CẶP theo mẫu |
|---|---|---|
| `volume` | ± 0,0072 – 0,0082 | L≥5 × giảm {0,5; 0,7}: **t = 0,31 – 2,05, không phân biệt được**; L=2: t = 8,1 – 10,6, **khác biệt thật** |
| `none` | ± 0,0059 – 0,0082 | L≥5 × giảm 0,7: **t = 0,22 – 1,23, không phân biệt được**; giảm 0,5: t = 3,85 – 3,93 **thật**; L=2: t = 4,9 – 5,7 **thật** |

Ba điều đọc được, và điều thứ hai là điều quan trọng:

1. **Sai số của một con số val là ±0,007**, tức chữ số thập phân thứ ba trở đi
   của lưới tinh chỉnh là nhiễu mẫu. Lưới in bốn chữ số; chỉ hai chữ số đầu đứng.
2. **Hiệu số ghép cặp chặt hơn 5–50 lần** (nhỏ nhất ±0,0001). Nếu so hai cấu
   hình bằng hai con số rời thì khoảng ±0,007 của chúng trùm lên nhau và mọi thứ
   "không phân biệt được"; ghép cặp theo mẫu thì tách được L=2 khỏi mảng bẹt với
   t = 10,6. Cùng một dữ liệu, hai kết luận trái nhau, khác nhau chỉ ở phép kiểm.
3. **Cấu hình val chọn nằm trong một mảng bẹt**, ở cả hai bản. Nên con số test
   (0,1369 và 0,1273) vẫn là số đọc đúng một lần của cấu hình do val chọn —
   không đổi — nhưng **không được phát biểu** rằng L=20 hơn L=5, hay giảm 0,5
   hơn giảm 0,7 trên bản gộp. Phát biểu được: *L ≥ 5 là một mảng bẹt, L = 2 thật
   sự xấu hơn, và trên bản `none` thì giảm 0,7 thật sự hơn giảm 0,5* (t ≈ 3,9) —
   còn trên bản gộp thì hai hệ số giảm bằng nhau.

Chỗ thứ ba cũng là một **sai lệch so với đề xuất phải ghi**: "ngân sách tinh
chỉnh tương đương" không đủ để chọn tham số nếu không kèm một phép đo độ phân
giải. Báo một điểm ở nơi dữ liệu chỉ đỡ được một mảng là báo chính xác hơn mức
mình có.

## 14. Lời giải thích của tôi ở mục 12 về hạ trọng số là SAI — đo lại ra một cơ chế khác

Mục 12 kết luận: hạ trọng số không ăn thua vì "sau lọc k-core, số người đọc
chung của các cặp đã đủ lớn, nên hệ số `n/(n + 20)` chỉ còn là một độ chệch đều
tay". Câu đó nghe hợp lý và tôi **không** đo nó trước khi viết.

Đo (512 mục mẫu, `scripts/diagnose_shrink.py`):

| | `none` | `volume` |
|---|---|---|
| cặp đồng xuất hiện có `n_chung < 20` | **98,8%** | **98,7%** |
| `n_chung` trung vị | **1** | **1** |
| hệ số `n/(n + 20)` trung vị | **0,048** | **0,048** |

Hệ số không "gần đều"; nó dập gần như **mọi** cặp xuống khoảng 1/20. Lời giải
thích ở mục 12 bị bác bỏ bởi chính dữ liệu nó nói về.

### Giả thuyết thay thế, ghi trước khi chạy, có điều kiện bác bỏ

Hạ trọng số không phải một độ chệch mà là một **bộ lọc theo độ phổ biến đặt lên
tập láng giềng**: `top_k` xếp theo tương đồng *đã hạ*, nên chỉ cặp có `n_chung`
lớn trụ lại, mà `n_chung` lớn gần như đồng nghĩa với "cả hai mục đều nhiều người
đọc". Mục 10 đã đo rằng trên bộ này độ chính xác đến từ việc **đi ra khỏi đầu
danh mục**, nên một bộ lọc đẩy láng giềng về đầu bảng phải làm giảm NDCG.

Điều kiện bác bỏ ghi trong docstring của script: `shrink = 20` cho ARP **thấp
hơn hoặc bằng** và long-tail **cao hơn hoặc bằng** `shrink = 0`.

| `none`, k=500 | NDCG | phủ | ARP | long-tail | mới lạ | **phân vị phổ biến của láng giềng (trung vị)** |
|---|---|---|---|---|---|---|
| hạ trọng số 0 | **0,1634** | 0,1364 | 0,9084 | 0,4691 | 6,83 | **0,6561** |
| hạ trọng số 20 | 0,1577 | 0,0853 | **0,9658** | **0,3226** | 5,77 | **0,8010** |

| `volume`, k=500 | NDCG | phủ | ARP | long-tail | mới lạ | phân vị phổ biến của láng giềng |
|---|---|---|---|---|---|---|
| hạ trọng số 0 | **0,1816** | 0,1519 | 0,9131 | 0,3977 | 6,38 | 0,6711 |
| hạ trọng số 20 | 0,1696 | 0,0915 | **0,9706** | **0,2497** | 5,27 | **0,7950** |

**Không bị bác bỏ, và dấu hiệu sạch hơn tôi mong:** số láng giềng mỗi mục
**y hệt nhau** (trung vị 500, trung bình 446,1 ở `none` và 429,0 ở `volume` cho
cả hai mức hạ trọng số) — nên đây không phải chuyện "hạ trọng số làm mất láng
giềng", mà là chuyện nó chọn **những láng giềng khác**, dịch từ phân vị phổ biến
0,656 lên 0,801. Kéo theo đúng ba hệ quả của một lệch về đầu bảng: ARP tăng,
long-tail giảm 31–37%, độ phủ giảm 37–40%.

Nói gọn: significance weighting là thuốc cho dữ liệu mỏng, nhưng nó trả giá bằng
một **ưu tiên độ phổ biến**, và trên bộ này cái giá đó đắt hơn cái bệnh. Mục 5.2
của đề xuất khuyên dùng nó; khuyên đó không đúng ở đây, và giờ có cơ chế đo được
chứ không phải một câu đoán.

**Một chỗ còn chưa giải thích, ghi là chưa giải thích:** mức mất NDCG của hạ
trọng số gần **gấp đôi** trên bản gộp (−6,6% so với −3,5%) trong khi độ dịch
phân vị phổ biến thì *bằng nhau* (0,671→0,795 so với 0,656→0,801). Nên độ lệch
đầu bảng giải thích được *chiều*, chưa giải thích được *độ lớn khác nhau*. Không
viết thêm một câu hợp lý nào vào chỗ này nữa — mục 12 vừa cho thấy hậu quả của
việc đó.

## 15. Chỗ mục 11 để mở: "gần đây" là tín hiệu thật hay là hệ quả của phép chia

Mục 11 đo được đỉnh ở L=2 rồi giảm đơn điệu, và tự ghi rằng nó **không kết luận
được gì về thứ tự đọc**, vì "một phần hiệu ứng này là tính chất của giao thức:
chia 70/10/20 theo từng người thì kỳ test nằm sát ngay sau kỳ train". Phép quét L
không tách được hai thứ, vì L **luôn** lấy các mục gần nhất — mọi điểm trên đường
cong đổi đồng thời *số mục* và *vị trí mục*.

Phép đo tách chúng ra là giữ nguyên số mục và chỉ đổi *lấy mục nào*.

### Cùng 2 mục, chỉ đổi vị trí

| đầu vào hàm điểm Item-KNN | NDCG@10 | Recall@10 | phủ | long-tail |
|---|---|---|---|---|
| 2 mục **gần nhất** | **0,1240** | 0,1272 | 0,1686 | 0,6047 |
| 2 mục **ngẫu nhiên** | 0,0795 | 0,0906 | 0,1369 | 0,4978 |
| 2 mục **cũ nhất** | 0,0574 | 0,0701 | 0,1037 | 0,3459 |
| 5 mục gần nhất | 0,1209 | 0,1258 | 0,1400 | 0,5271 |
| 5 mục ngẫu nhiên | 0,0934 | 0,1061 | 0,1076 | 0,4131 |
| 5 mục cũ nhất | 0,0707 | 0,0816 | 0,0946 | 0,3226 |
| toàn bộ lịch sử | 0,0989 | 0,1119 | 0,0823 | 0,3217 |

**Hơn kém 2,16 lần với y nguyên số lượng.** Vậy cách giải thích "cộng dồn nhiều
mục khuếch đại độ phổ biến" bị bác bỏ ở vai trò nguyên nhân chính: vị trí mới là
thứ quyết định.

Hai đường cong còn cắt nhau theo chiều có ý nghĩa. Với mục **ngẫu nhiên**, nhiều
hơn thì **tốt lên** (2 → 5: 0,0795 → 0,0934) vì lấy trung bình bớt nhiễu. Với mục
**gần nhất**, nhiều hơn thì **tệ đi** (0,1240 → 0,1209 → 0,0989) vì phải với tay
về mục cũ. Nên sự sụt giảm khi dùng cả lịch sử không phải "nhiều mục thì tệ": nó
là **mục cũ gây hại hơn cả một mẫu ngẫu nhiên** — lịch sử đầy đủ (0,0989) vẫn hơn
5 mục ngẫu nhiên (0,0934), trong khi 5 mục cũ nhất chỉ còn 0,0707.

*Ô tự kiểm:* chia điểm cho số mục cho ra **0,1240**, không lệch chữ số nào — chia
cho hằng số thì không được đổi thứ hạng, và nó không đổi.

### Cơ chế: mục cũ của một người là mục phổ biến

Trên **cùng** tập người đọc (24.147 người có ≥ 20 mục trong train):

| | phân vị độ phổ biến | trung vị số người đọc |
|---|---|---|
| 2 mục **cũ nhất** của họ | 0,9385 | **1.160** |
| 2 mục **mới nhất** của họ | 0,8691 | **203** |

Chênh **5,7 lần**. Người ta mở danh sách bằng truyện ai cũng đọc rồi mới đi vào
truyện hẹp. Đó là tính chất của cách danh sách đọc hình thành, đo trực tiếp trên
dữ liệu, không phải tính chất của mô hình nào. Bốn bước nối liền:

1. mục cũ là mục phổ biến (1.160 người đọc), mục mới là mục hẹp (203);
2. test là 20% mục mới nhất, nên **test chính là phần hẹp nhất** của mỗi người;
3. chấm từ mục cũ thì láng giềng rơi vào đầu danh mục (long-tail 0,346), trượt
   đúng phần test đang hỏi;
4. nên cửa sổ gần đây hơn lịch sử đầy đủ.

### Phán quyết cho chỗ mục 11 để mở

Phần lợi của "gần đây" **không** chỉ là hệ quả của phép chia. Nếu nó chỉ là
khoảng cách thời gian tới kỳ test thì 2 mục cũ nhất và 2 mục ngẫu nhiên phải cho
cùng một con số — chúng lệch 38%. Cái mà "gần đây" đại diện ở đây là **độ hẹp của
mục**, và độ hẹp là thuộc tính của mục, không của phép chia.

Nhưng phán quyết này cũng **thu hẹp** phát biểu được: ưu tiên gần đây có giá trị
vì nó là một *bộ chọn mục hẹp*, không vì nó bắt được *thứ tự đọc*. Hai điều đó
khác nhau ở hệ quả: nếu là độ hẹp thì một mô hình chỉ cần biết độ phổ biến của
mục trong lịch sử là đủ, không cần tới chuỗi. Mục 11 đã từ chối phát biểu về thứ
tự; mục này nói thêm lý do **vì sao** nó phải từ chối.

### Đồng thời: onboarding không còn là chế độ suy giảm

Dòng "onboarding chọn sở thích" là dòng duy nhất còn mang trạng thái *một phần*
trong bảng phương pháp, vì `cold_start.py` dựng được hồ sơ từ vài lựa chọn nhưng
chưa bao giờ bị chấm như một bộ gợi ý. Đường cong trên đóng chỗ đó, và kết quả
ngược chiều trực giác thông thường: **hỏi người đọc mới 2 lựa chọn cho 0,1240,
trong khi biết cả lịch sử của họ cho 0,0989.** Onboarding ở hệ này không phải
cái giá phải trả cho người mới — nó là chế độ tốt hơn.

Một chỗ phải ghi rõ để không đọc quá: thí nghiệm giữ **nguyên** tập loại trừ của
mọi dòng khác (loại cả train và val), nên nó là "bộ gợi ý bị bịt mắt bớt lịch sử"
chứ không phải "người đọc mới hoàn toàn". Mô phỏng người mới cho đúng tinh thần
thì chỉ nên loại 2 mục đã lộ, nhưng khi đó rổ ứng viên to hơn rổ của mọi dòng
khác và con số rơi vì lý do chẳng liên quan tới onboarding. Chọn so được với
bảng, và ghi rõ cái giá của lựa chọn đó.

Script: `scripts/run_onboarding.py --picks 1 2 3 5 10 20 --mechanism`.

## 16. Ngân sách epoch là một yếu tố gây nhiễu, và nó đổi cả THỨ HẠNG chứ không chỉ mức

Sáu mô hình nơ-ron ở mục 5.3–5.8 của đề xuất lần đầu được chạy với ngân sách 8
epoch, vì mỗi epoch trên `processed_hgn_none` tốn từ 22 giây (GRU4Rec) tới 1.660
giây (LightGCN). Con số thu được đã đủ để xếp hạng, và tôi gần như đã báo chúng.

Chỗ chặn lại là một quan sát về **chính quy trình chọn**, không về con số: khi
chấm val mỗi 2 epoch trong ngân sách 8, *cả năm* mô hình đều chọn đúng epoch cuối
cùng. Năm lần như nhau thì không phải tình cờ — nó là dấu hiệu val vẫn đang lên ở
mọi mô hình, nghĩa là thứ đang chặn là ngân sách, không phải kiến trúc. Nên thay
vì báo, tôi chạy lại với ngân sách lớn và một quy tắc dừng sớm.

### Bảng: cùng mô hình, cùng dữ liệu, cùng giao thức — chỉ đổi ngân sách

NDCG@10 trên test, 2.000 người, rổ toàn danh mục. Dấu † = val chọn đúng epoch
cuối, tức là con số đó là **chặn dưới**, không phải giá trị của mô hình.

| Mô hình | ngân sách 8 | ngân sách 20 | ngân sách lớn | epoch được chọn |
|---|---|---|---|---|
| GRU4Rec | 0,0200 | 0,0430† | **0,0744** | 320 / 400 |
| NeuMF | 0,0500† | 0,0590† | **0,0672** | 64 / 120 |
| GRU4Rec (lượt 120) | — | — | 0,0622† | 120 / 120 |
| SASRec | — | 0,0514 | 0,0504 | 20 / 120 |
| LightGCN | 0,0473† | — | **0,0793** | 90 / 100 |
| CL4SRec | — | 0,0388† | 0,0455 | 72 / 120 |
| NRMS-style | — | 0,0203† | 0,0321 | 52 / 120 |

Đọc theo cột thì chỉ thấy "nhiều epoch hơn cho số cao hơn", một chuyện ai cũng
đoán được. Chỗ đáng ghi nằm ở **thứ hạng**:

* Ở ngân sách 8, NeuMF hơn GRU4Rec **2,50 lần** (0,0500 so với 0,0200), và
  GRU4Rec đứng **cuối** bảng nơ-ron.
* Ở ngân sách lớn, GRU4Rec hơn NeuMF 1,11 lần và đứng **đầu** bảng nơ-ron.

Vậy một bảng chạy ở ngân sách nhỏ không chỉ sai *mức*; nó sai **thứ tự**, và sai
theo hướng khó phát hiện nhất: nó ưu ái mô hình học nhanh (NeuMF cập nhật mỗi cặp
người–mục) và trừng phạt mô hình học chậm mà trần cao (GRU4Rec cập nhật mỗi
chuỗi). Kết luận "mô hình chuỗi không phù hợp dữ liệu này" rút ra từ ngân sách 8
sẽ là một kết luận sai, và sai vì cấu hình thí nghiệm, không vì dữ liệu.

### Hai hàng *không* đổi, và vì sao chúng mới là chỗ đọc được

SASRec đi 0,0514 (epoch 18) → 0,0504 (epoch 20 trong ngân sách 120): gần như
không nhích, và quy tắc dừng sớm cắt nó sau 4 lần chấm không khá hơn. Đó là dấu
vân tay của một mô hình **thật sự đã bẹt**, phân biệt được với một mô hình bị
chặn bởi ngân sách. Nên phát biểu về SASRec (NDCG 0,0504 với Recall chỉ 0,0215 —
đỉnh bảng rất tập trung, phần còn lại trượt) là phát biểu về mô hình.

NRMS-style đi 0,0203 → 0,0321, tăng 1,58 lần, nhưng **vẫn đứng cuối**. Nó gặp
LSA nội dung (0,0328) và dưới TF-IDF (0,0557) — ba đường độc lập tới cùng một
trần nội dung. Trần đó không phải hệ quả của ngân sách: tăng ngân sách 6 lần
không chạm tới nó. Đây là chỗ hai loại kết luận tách nhau rõ nhất: ngân sách nhỏ
làm **thứ hạng giữa các mô hình ID** sai, nhưng **không** tạo ra trần nội dung và
cũng không xoá được nó.

### Mất mát train hội tụ xong rất lâu trước khi val hội tụ

Đường GRU4Rec cho một con số cụ thể cho câu "dừng khi loss bẹt":

| epoch | 10 | 50 | 100 | 200 | 320 | 370 |
|---|---|---|---|---|---|---|
| loss train | 0,0997 | 0,0319 | 0,0213 | 0,0149 | 0,0121 | 0,0112 |
| val NDCG | 0,0351 | 0,0904 | 0,1209 | 0,1406 | **0,1732** | 0,1673 |

Từ epoch 50 tới 320, loss train chỉ còn giảm 2,6 lần trong khi val NDCG tăng
1,92 lần. Riêng khoảng 100 → 320: loss giảm 43%, val **tăng** 43%. Dừng theo loss
train ở quanh epoch 100 — chỗ mà bằng mắt loss đã "gần bẹt" — mất 43% NDCG. Loss
BPR không phải đại diện của NDCG ở đây, và chênh lệch không nhỏ.

### Quy tắc dừng sớm cũng là một siêu tham số, với hiệu ứng 2 lần

Đường val dao động ±0,01 giữa hai lần chấm liền nhau (epoch 190 cho 0,1470, epoch
200 cho 0,1406), nên "kiên nhẫn" bao nhiêu lần chấm quyết định luôn kết quả. Chạy
lại quy tắc trên chính đường val đã ghi:

| kiên nhẫn (lần chấm, mỗi 10 epoch) | dừng ở epoch | val tốt nhất giữ được | so với 0,1732 |
|---|---|---|---|
| 1 | 60 | 0,0904 | 52% |
| 2 | 130 | 0,1250 | 72% |
| 3 | 300 | 0,1627 | 94% |
| 4 | 360 | 0,1732 | 100% |
| 5 | 370 | 0,1732 | 100% |

Kiên nhẫn 1 — mặc định của rất nhiều đoạn mã huấn luyện — cho đúng một nửa. Đây
là lý do mọi dòng nơ-ron trong bảng ghi kèm *cả* ngân sách *và* epoch được chọn:
không có hai con số đó thì con số NDCG không đọc được.

### Chỗ để mở ở trên đã đóng: LightGCN tăng 1,68 lần

LightGCN là hàng cuối cùng còn đọc ở biên (epoch 8/8, val vẫn đang lên). Nó đắt
nhất cả bảng vì `propagate()` phải chạy lại mỗi bước cập nhật, nên 8 epoch đã
mất 3,7 giờ. Lượt chạy lại dùng lô 16.384 thay cho 4.096 — ít bước hơn 4 lần,
mỗi bước một lần lan truyền — với lr 3e-3 bù lại, ngân sách 100 epoch: 340 giây
một epoch thay vì 1.660, tức 7,9 giờ cho 100 epoch thay vì 46 giờ.

Kết quả: **0,0473 → 0,0793**, gấp 1,68 lần, và LightGCN thành mô hình nơ-ron
**cao nhất** trong sáu mô hình, trên cả GRU4Rec 0,0744. Đọc theo val thì epoch 8
cho 0,0491 còn epoch 90 cho 0,1102 — con số cũ không sai ở phép đo, nó chỉ là
một mô hình chưa học xong được đem đi chấm.

Mức độ bẹt, nói đúng: val đi 0,0486 (epoch 5) → 0,1034 (epoch 60) rồi chỉ còn
0,1099 (epoch 100) — 40 epoch cuối thêm 6%, với dao động ±0,002 quanh mức đó.
Epoch được chọn là 90 chứ không phải 100, nên nó không còn là "đọc ở biên" theo
nghĩa chặt, nhưng cũng chưa phải một chỗ bẹt sạch như SASRec. Nếu cho thêm ngân
sách thì nhiều khả năng còn nhích, mức nhích nhỏ.

Một chỗ phải đọc kèm con số này: **LightGCN mua độ chính xác bằng cách nghiêng
về phần đầu danh mục.** Tỉ lệ long-tail của nó là 0,334 và ARP 0,970 — thấp nhất
và cao nhất trong sáu mô hình nơ-ron (GRU4Rec 0,620; SASRec 0,675; CL4SRec
0,945). Con số 0,334 ấy gần như trùng với Item-KNN chạy trên **lịch sử đầy đủ**
(long-tail 0,322, NDCG 0,0989), và điều đó khớp với mục 15: gộp qua toàn bộ lịch
sử của một người thì điểm dồn về chỗ phổ biến. LightGCN lan truyền trên cả đồ
thị hai phía, tức là dạng gộp-toàn-lịch-sử triệt để nhất trong bảng, nên nó rơi
đúng vào chế độ đó.

### Hệ quả cho cả bảng: trần nơ-ron ở cấu hình này là 62%

Khi đã sửa cả ba lỗi cấu hình (mất mát ở mọi vị trí cho GRU4Rec, cửa sổ chuỗi
đúng độ dài, ngân sách epoch đủ cho cả sáu mô hình), mô hình nơ-ron tốt nhất
trên `processed_hgn_none` đạt **0,0793**. Xếp cạnh toàn bộ bảng cùng cấu hình:

| | NDCG@10 |
|---|---|
| Đa dạng hoá MMR (lai ghép) | 0,1284 |
| Song song có trọng số | 0,1283 |
| Session-KNN | 0,1273 |
| Onboarding 2 lựa chọn (Item-KNN) | 0,1240 |
| Item-KNN Pearson | 0,1020 |
| User-KNN (cosine) | 0,0900 |
| **LightGCN (tốt nhất trong 6 mô hình nơ-ron)** | **0,0793** |
| iALS / ALS | 0,0756 |
| GRU4Rec | 0,0744 |
| NeuMF | 0,0672 |
| SASRec | 0,0504 |
| CL4SRec | 0,0455 |
| NRMS-style | 0,0321 |

Không mô hình nơ-ron nào lọt vào tám hạng đầu, và mô hình tốt nhất trong sáu
mô hình đó bằng **62%** của số cao nhất. Đây là kết quả đã được báo trong tài liệu
(Dacrema, Cremonesi & Jannach, RecSys'19) và ở đây nó lặp lại trên dữ liệu truyện
tranh.

Nhưng phát biểu phải thu hẹp đúng chỗ, vì **ngân sách tinh chỉnh không tương
đương**: các mô hình láng giềng và lai ghép được chọn siêu tham số trên validation
(k, co ngót, trọng số trộn, λ), còn các mô hình nơ-ron chỉ được chọn **epoch**,
mọi siêu tham số khác để mặc định (lr 1e-3, 64 chiều, 1 mẫu âm, dropout 0,2).
Chính mục này vừa cho thấy riêng việc chọn epoch đã đổi GRU4Rec 3,7 lần, nên
không có cơ sở để giả định lr, số chiều hay số mẫu âm vô hại.

Kết luận đứng được là: *ở cấu hình mặc định và cùng một ngân sách công sức, các
phương pháp láng giềng rẻ hơn nhiều mà cao hơn rõ rệt* — không phải *nơ-ron kém
hơn láng giềng trên dữ liệu này*. Phát biểu thứ hai cần một lưới tinh chỉnh
tương đương cho sáu mô hình nơ-ron, và với 380–1.660 giây mỗi epoch thì nó không
nằm trong tầm của môi trường này; ghi lại ở đây như một chỗ còn thiếu, không lấp
bằng một con số mặc định.

### CL4SRec hội tụ xong rồi vẫn thấp hơn SASRec — và chỗ nó đi mất đo được

CL4SRec là SASRec cộng thêm một số hạng NT-Xent trên hai biến thể của cùng một
chuỗi (cắt / che / đảo). Ở ngân sách 20 nó cho 0,0388 so với SASRec 0,0514, và
lúc đó con số còn ở biên nên chưa kết luận được. Lượt ngân sách 120 dừng sớm ở
epoch 88 với val tốt nhất epoch 72 — **đã bẹt thật** — và cho 0,0455. Hai mô
hình giờ đều hội tụ, nên so được:

| | NDCG@10 | Recall@10 | độ phủ | tỉ lệ long-tail |
|---|---|---|---|---|
| SASRec | **0,0504** | 0,0215 | 0,0398 | 0,675 |
| CL4SRec (= SASRec + NT-Xent) | 0,0455 | **0,0236** | **0,0547** | **0,945** |

Số hạng tương phản **không** làm mô hình tệ đi theo mọi hướng: Recall cao hơn
10%, độ phủ cao hơn 37%, tỉ lệ long-tail nhảy từ 0,675 lên **0,945**. Nó làm NDCG
thấp hơn 10% trong khi tìm được *nhiều* mục đúng hơn — nghĩa là mục đúng bị xếp
thấp hơn trong danh sách 10.

Đó đúng là hành vi mà số hạng NT-Xent được thiết kế để tạo ra: nó đẩy biểu diễn
của các mục ra xa nhau (thành phần *uniformity*), nên gợi ý bị dàn mỏng ra khỏi
phần đầu danh mục. Trên dữ liệu này, phần đầu danh mục là chỗ test thưởng điểm
NDCG, nên dàn mỏng là một phép đánh đổi có giá, không phải một cải tiến. Ghi kèm:
λ = 0,1 là giá trị mặc định của bài gốc, **không** được dò trên val — và vì chính
λ điều khiển độ mạnh của phép đánh đổi này, một λ nhỏ hơn rất có thể cho cả hai
chiều tốt hơn. Đây là chỗ thiếu, ghi lại chứ không lấp.

Hệ quả cho bảng: cả sáu mô hình nơ-ron giờ đều được đọc ở một epoch do val
chọn bên trong ngân sách, không còn hàng nào đọc ở biên: NeuMF 64/120,
GRU4Rec 320/400, SASRec 20/120, NRMS 52/120, CL4SRec 72/120, LightGCN 90/100.
Bảy ô còn mang cờ “ở biên” trong `budget.csv` đều là các lượt ngân sách
nhỏ để lại làm đối chứng, không phải con số được báo cho mô hình nào.

## 17. Sáu cấu hình dữ liệu có cần sáu không — đo, rồi bỏ bớt

Câu hỏi đáng hỏi: sáu cấu hình tiền xử lý là cần thiết hay chỉ là chuyện đã lỡ
làm? Và nếu cần ít hơn thì sao không gộp tất cả thành một bộ chính quy?

Trả lời bằng đo, không bằng khẳng định. Với mỗi cặp cấu hình, lấy các mô hình đã
chạy ở **cả hai** rồi hỏi hai câu: thứ hạng có đổi không (tương quan Spearman
trên hạng), và mức lệch điển hình là bao nhiêu. Bỏ các mốc có NDCG < 0,01 ra
khỏi phép tính độ lệch, vì Random đi từ 0,0001 sang 0,0002 là "+100%" mà không
nói gì về cấu hình.

| Cặp | mô hình chung | ρ hạng | lệch trung vị | đầu bảng mỗi bên |
|---|---|---|---|---|
| GR khớp ↔ GR đầy đủ (A) | 8 | **0,98** | 3,3% | Session-KNN / Session-KNN |
| GR B không gộp ↔ GR B gộp bộ | 9 | **1,00** | 11,4% | Session-KNN / Session-KNN |
| AniList ↔ GR khớp (A) | 8 | **−0,26** | 33,2% | DropoutNet / Session-KNN |
| AniList ↔ GR đầy đủ (A) | 8 | **−0,24** | 35,3% | DropoutNet / Session-KNN |

Ba phán quyết khác nhau:

**Hai cấu hình là đối chứng đã xong việc.** Gấp 2,15 lần dữ liệu (GR khớp → GR
đầy đủ) cho ρ = 0,98 và lệch trung vị 3,3%: quy mô không đổi kết luận nào. Gộp
3.190 bộ truyện (B không gộp → B gộp bộ) cho ρ = **1,00** trên 9 mô hình: thứ
hạng y hệt, mọi mô hình thật lên 3–17%. Hai cấu hình này vẫn phải tồn tại — chúng
là cách *chứng minh* hai câu trên — nhưng chúng không phải hai bộ kết quả song
song cần mang theo mọi bảng. Trong trang trình bày chúng được dán nhãn "đối
chứng" thay vì đứng ngang hàng với cấu hình chính.

Chỗ phải cẩn thận ở cặp thứ hai: mức "+3–17%" **không** đọc được là "gộp bộ thì
tốt hơn". Gộp làm danh mục co từ 36.241 xuống 30.387 mục và tập test nhỏ lại, nên
đó là một bài toán dễ hơn, không phải cùng bài toán làm tốt hơn. Cái đọc được chỉ
là ρ = 1,00: *thứ tự* không đổi.

**Một cặp là lý do không được gộp.** AniList và Goodreads, cùng giao thức, cùng
code, đã khớp quy mô (0,96× tương tác, 0,99× danh mục), vẫn cho ρ = **−0,26** —
thứ hạng gần như nghịch nhau. DropoutNet đứng đầu AniList (0,1202) và gần cuối
Goodreads (0,0564); Session-KNN tầm trung ở AniList (0,0957) và đứng đầu Goodreads
(0,0996). Nếu gộp tất cả thành một bộ chính quy, hoặc chọn một bộ duy nhất, bài
này đã báo một kết luận chắc nịch mà bộ kia nói ngược lại. Đây là chỗ duy nhất
trong toàn bộ bảng mà một kết luận thực sự đổi khi đổi dữ liệu, nên nó là cặp
không được bỏ.

Cơ chế của chỗ nghịch nhau đã đo ở mục 3: thứ duy nhất còn lệch sau khi khớp quy
mô là **quan sát trên mỗi người** (232,6 so với 35,8), và đó là thứ quyết định mô
hình nào thắng. Gộp hai bộ lại sẽ trộn hai phân bố đó thành một con số không mô
tả ai.

**Một cấu hình không phải biến thể mà là bài toán khác.** Tập dự đoán điểm
(101.196 người, 4,13 triệu dòng) không lọc ngưỡng rating, vì chính phép lọc
rating ≥ 4 của giao thức B làm cột điểm chỉ còn hai giá trị và bài dự đoán điểm
biến mất. Nó cho MAE/RMSE, không cho NDCG, nên nó không nằm cùng thang với năm
cấu hình kia.

**Số tối thiểu:** ba cấu hình chính (AniList·A, GR khớp·A, GR B không gộp) cộng
một tập cho bài dự đoán điểm. Hai cấu hình còn lại là đối chứng, mỗi cái trả lời
đúng một câu rồi thôi. Script: `scripts/export_matrix.py` ghi `agree.csv`, bảng
trên sinh thẳng từ đó.

## 18. "Trần nơ-ron" không phải tính chất của mô hình — nó là tính chất của bộ dữ liệu

Mục 16 kết lại bằng một con số: mô hình nơ-ron tốt nhất (LightGCN 0,0793) chỉ
bằng **62%** số cao nhất trên bảng, và không mô hình nơ-ron nào lọt vào tám hạng
đầu. Lúc đó cả sáu mô hình nơ-ron chỉ chạy trên **một** cấu hình — Goodreads · B
không gộp — nên "trần 62%" và "trần nơ-ron của bài này" là cùng một câu, không
tách được.

Giờ tách được. Sáu mô hình đã chạy xong trên **AniList · B**: cùng code, cùng
phép chia 70/10/20 theo thời gian của từng người, cùng cách chọn epoch trên
validation, cùng chấm toàn danh mục.

| | Goodreads · B (lịch sử TV 16) | AniList · B (lịch sử TV 90) |
|---|---|---|
| Cao nhất toàn bảng | Đa dạng hoá MMR 0,1284 | **LightGCN 0,1220** |
| Nơ-ron tốt nhất | LightGCN 0,0793 | LightGCN 0,1220 |
| Trần nơ-ron | **62%** | **100%** |
| Nơ-ron trong 10 hạng đầu | 1 / 10 | **4 / 10** |

Thứ hạng từng mô hình nơ-ron, Goodreads → AniList: LightGCN 9 → **1**,
SASRec 17 → 7, NeuMF 13 → 9, GRU4Rec 11 → 10, CL4SRec 18 → 18,
NRMS-style 22 → 19. Không mô hình nào tụt; năm trên sáu đi lên.

### Không phải chuyện một mô hình may mắn

Gộp theo họ mô hình (NDCG@10 trung bình trong họ):

| Họ | Goodreads · B | AniList · B |
|---|---|---|
| lai ghép (3 mô hình) | 0,1088 — hạng 1 | 0,0922 — hạng 3 |
| láng giềng (5) | 0,1027 — hạng 2 | 0,0982 — hạng 1 |
| tri thức (3) | 0,0742 — hạng 3 | 0,0575 — hạng 5 |
| **nhân tố ẩn (5)** | **0,0616 — hạng 6** | **0,0981 — hạng 2** |
| chuỗi (3) | 0,0568 — hạng 5 | 0,0849 — hạng 4 |
| nội dung (2) | 0,0442 — hạng 7 | 0,0286 — hạng 7 |
| mốc (3) | 0,0136 | 0,0325 |

Họ **nhân tố ẩn** — ALS, BPR-MF, NeuMF, LightGCN, NRMS, tức là mọi phương pháp
ước lượng một vectơ cho từng người — đi từ hạng 6 lên hạng 2. Họ **láng giềng**
gần như không nhích (0,1027 → 0,0982). Đó là hình dạng của một hiệu ứng về
**số quan sát trên mỗi người**, không phải về kiến trúc: láng giềng chỉ cần một
hàng của ma trận, nhân tố ẩn cần đủ dòng để vectơ của một người hội tụ. 16 mục
không đủ; 90 mục thì đủ.

### Loại trừ ngân sách epoch

Mục 16 cho thấy ngân sách nhỏ đổi cả thứ tự, nên phải kiểm chính chỗ đó. Trên
AniList · B **cả sáu lượt đều dừng sớm** vì val hết lên, không lượt nào chạm
đáy ngân sách:

| Mô hình | Epoch val chọn / ngân sách | Epoch cuối cùng chạy | Test NDCG@10 |
|---|---|---|---|
| LightGCN | 55 / 100 | 75 | 0,1220 |
| SASRec | 36 / 300 | 52 | 0,0992 |
| NeuMF | 16 / 120 | 32 | 0,0946 |
| GRU4Rec | 50 / 800 | 100 | 0,0905 |
| CL4SRec | 4 / 300 | 20 | 0,0651 |
| NRMS-style | 12 / 300 | 28 | 0,0622 |

Không có ô "ở biên" nào — khác bảng ở mục 16, nơi 7 trong 15 ô là chặn dưới.
Nên các số AniList là giá trị ở chỗ bẹt.

Điều đáng ghi thêm: **AniList bẹt sớm hơn nhiều**. GRU4Rec cần epoch 320 trên
Goodreads nhưng chỉ 50 trên AniList; NeuMF 64 → 16. Đếm theo *số đích tính mất
mát* thì hai bộ gần bằng nhau (Goodreads 16 mục × 58.413 người ≈ 934k;
AniList 90 × 8.724 ≈ 785k), nên chênh lệch không nằm ở khối lượng mỗi epoch.
Nó nằm ở chỗ tín hiệu mỗi chuỗi rõ hơn: cùng một lượng gradient, bài dễ hơn thì
hội tụ sớm hơn. Suy ra "mô hình X cần ngân sách lớn" là phát biểu về **cặp mô
hình–dữ liệu**, không phải về mô hình.

### Một chỗ phải nói rõ

SASRec và CL4SRec chấm trên **1.410** người đọc, bốn mô hình kia và toàn bộ
phương pháp cổ điển chấm trên **2.000** — hai mô hình chuỗi bỏ người có lịch sử
ngắn hơn ngưỡng chuỗi. Các người đọc còn lại có lịch sử dài hơn trung bình, nên
hạng 7 của SASRec có thể được ưu ái nhẹ. Nó không ảnh hưởng tới câu chính:
LightGCN, mô hình đứng hạng 1, chấm trên đủ 2.000 người.

### Hệ quả cho cách đọc mục 16

Câu "cấu hình mặc định không bắt kịp láng giềng" ở mục 16 vẫn đúng **trên
Goodreads**. Câu "mô hình nơ-ron không bắt kịp trong bài này" thì sai: đổi bộ
dữ liệu sang lịch sử dày, chính LightGCN đứng đầu. Mọi chỗ nói "trần nơ-ron
62%" từ đây phải kèm hai chữ "ở bộ này".

Script: `scripts/run_anilist_b.sh` nhóm `neural`; log ở
`<scratchpad>/anl/n_*.log`; kết quả `data/processed_anilist_b/neural_*.json`.
Tương quan hạng giữa hai bộ trên cả 24 phương pháp: Spearman **+0,56**.

## 19. Crawl AniList mở rộng tới trần thật — mục 18 chạy lại, ba kết luận giữ nguyên, một đổi

Câu hỏi "AniList chỉ có vậy thôi à" (8.724 người) có câu trả lời đo được: **không**.
Hai tham số của crawl tự đặt ra một trần thấp hơn trần thật của AniList:

* `discover` chỉ lấy top 2.000 truyện/nước, nhưng AniList có **5.000** (`total=5000,
  lastPage=100` khi hỏi trang 41 — còn hơn một nửa catalogue chưa từng được quét).
* `stage_users` chỉ quét 3 trang hoạt động mỗi nhóm 25 truyện. Quét thêm trang 4–6
  **trên đúng một nhóm đã quét** vẫn ra 46–106 người mới mỗi nước — 3 trang là số tự
  chọn, không phải điểm hoạt động cạn.

Mở `--per-country 5000` và `--pages-per-chunk 8`, chạy hết toàn bộ 15.000 truyện × 3
nước thì dừng tự nhiên (không chạm target đặt cố ý rất cao) — đó mới là trần thật của
phương pháp này: **37.282 người thô → 33.754 sau lọc lõi 10/5** (từ 8.724, ×3,9).
Lịch sử trung vị giảm nhẹ 90 → 79 (crawl sâu hơn kéo vào cả người đọc kém tích cực
hơn một chút — tỉ lệ dưới 10 mục đo được tăng từ 3,6% lên 7,6%, vẫn cách xa 40% của
Goodreads). Toàn bộ 24 phương pháp và 6 mô hình nơ-ron chạy lại trên quy mô mới.

### Ba kết luận của mục 18 đứng vững

1. **Trần nơ-ron đổi theo bộ dữ liệu, không phải tính chất cố định của mô hình.**
   Trên Goodreads vẫn 62%, 1/10 hạng đầu là nơ-ron. Trên AniList quy mô mới: họ
   nơ-ron có 4/10 hạng đầu, và một mô hình nơ-ron (không còn là LightGCN — xem dưới)
   đứng hạng 1 toàn bảng.
2. **Họ nhân tố ẩn hưởng lợi từ lịch sử dày, hướng không đổi**: NDCG trung bình
   0,0616 (hạng 6/7) ở Goodreads → 0,0969 (hạng 2/7) ở AniList quy mô mới. Nhưng ở
   quy mô lớn hơn này, bốn họ đầu bảng AniList (láng giềng 0,0970, nhân tố ẩn
   0,0969, lai ghép 0,0963, chuỗi 0,0944) **nén lại trong 3% của nhau** — ở quy mô
   8.724 người chúng còn tách nhau rõ (láng giềng 0,0982 so với lai ghép 0,0922, cách
   6,5%). Mẫu đọc rộng và ít lệch hơn thì ưu thế của một họ mô hình mờ đi.
3. **AniList bẹt sớm hơn Goodreads nhiều**, vẫn đúng và còn rõ hơn: ở quy mô mới
   GRU4Rec chốt epoch 50/800 (trước 50/800 — không đổi), SASRec 8/300 (trước 36/300,
   nhanh hơn), LightGCN 30/100 (trước 55/100, nhanh hơn). Không ô nào ở biên.

### Một kết luận đổi: quán quân không còn là LightGCN

| | Goodreads · B (lịch sử TV 16) | AniList · B, 8.724 người | AniList · B, 33.754 người |
|---|---|---|---|
| Cao nhất toàn bảng | MMR 0,1284 | LightGCN 0,1220 | **SASRec 0,1314*** |
| Nơ-ron tốt nhất | LightGCN 0,0793 | LightGCN 0,1220 | SASRec 0,1314* |
| Trần nơ-ron | 62% | 100% | 100%* |
| Nơ-ron trong 10 hạng đầu | 1/10 | 4/10 | 4/10 |

\* **Có điều kiện mẫu phải đọc kèm, và lần này nó đổi câu trả lời chính.** SASRec và
CL4SRec chấm trên **1.281** người đủ lịch sử cho chuỗi 50 mục, không phải 2.000 như
22 phương pháp khác. Ở quy mô 8.724 người cũ, điều này chỉ ảnh hưởng tới hạng 7
(SASRec) — không đổi câu "LightGCN đứng đầu". Ở quy mô 33.754 người, nó ảnh hưởng
tới **hạng 1**: SASRec đứng đầu toàn bảng, nhưng trên một mẫu khác (dễ hơn, lịch sử
dài hơn trung bình) với phần còn lại. Trong nhóm 23 phương pháp chấm trên đúng
2.000 người, hạng đầu là **User-KNN** (0,1195), rồi **LightGCN** (0,1186, nơ-ron),
**ALS** (0,1177) — kết luận "nơ-ron thắng" vẫn đứng (LightGCN hạng 3/24, không cần
điều kiện mẫu), chỉ là tên mô hình đứng đầu tuyệt đối phụ thuộc vào việc có chấp
nhận chênh mẫu 1.281 so với 2.000 hay không.

### Tương quan hạng

Spearman trên cả 24 phương pháp, hai bộ cùng giao thức B: **+0,52** ở quy mô mới
(trước +0,56 ở 8.724 người). Dao động quanh +0,5 khi quy mô đổi; hướng "đồng ý vừa
phải, không nghịch nhau" không đổi.

Script: `scripts/crawl_anilist.py users --pages-per-chunk` (tham số mới, mặc định 3,
dùng 8 cho lần này); `scripts/reingest_anilist.py` (mới — chỉ đọc lại interactions/
media.parquet, không chạy lại ALS/feature/cover của `train.py`); `scripts/run_anilist_b.sh`
không đổi. Log crawl ở `<scratchpad>/anl/users_big.log`, `lists_big.log`; log huấn
luyện ở `<scratchpad>/anl3/`.
