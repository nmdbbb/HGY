# Nhánh đối chứng Goodreads: phương pháp có lặp lại trên bộ dữ liệu khác không?

Nhánh AniList đã chốt bốn phát hiện. Câu hỏi của tài liệu này: chúng là tính
chất của **phương pháp** hay của **bộ dữ liệu AniList**? Chỉ một bộ dữ liệu thứ
hai trả lời được, và đó là lý do duy nhất nhánh này tồn tại — không phải để có
thêm một bảng số, cũng không phải để tìm bộ nào cho điểm cao hơn.

## 0. Giấy phép và trích dẫn — bắt buộc

Bộ Goodreads của McAuley Lab (UC San Diego) **chỉ dùng cho mục đích học thuật**,
**không phát hành lại**, **không dùng thương mại**. `data/` đã nằm trong
`.gitignore` nên không có file dữ liệu nào được commit.

Trích dẫn bắt buộc trong báo cáo:

> Mengting Wan, Julian McAuley. *Item Recommendation on Monotonic Behavior
> Chains.* RecSys 2018.
>
> Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley. *Fine-Grained
> Spoiler Detection from Large-Scale Review Corpora.* ACL 2019.

## 1. Thiết kế: một bộ chuyển đổi, không phải hệ thứ hai

`src/truyen_reco/data/ingest_goodreads.py` trả về **đúng hợp đồng cột** của
`ingest.py` bên AniList. Mọi thứ sau tầng đọc dữ liệu — `preprocess`,
`build_features`, ALS, hai tháp, `eval_saved`, `measure_product` — chạy không
sửa một dòng:

```bash
python scripts/train.py --source goodreads --out-dir data/processed_gr \
       --cover-dir data/raw/covers_gr
python scripts/diagnose_goodreads.py            # so bản chất hai bộ
python scripts/train_embedding.py --out-dir data/processed_gr --epochs 15 --save
```

Lý do phải làm vậy chứ không viết pipeline riêng: nếu mỗi bộ có code riêng thì
chênh lệch kết quả **không quy được về dữ liệu**, vì nó cũng có thể do code.
`tests/test_goodreads.py::test_hai_nguon_cho_cung_bo_cot` canh đúng chỗ đó —
nó đỏ là lúc mọi so sánh trong tài liệu này mất nghĩa.

### Hợp đồng cột gồm cả KIỂU của hai khoá

Lần dựng đầu, test trên chỉ so **tên** cột, và `user` của Goodreads là mã số
`int32` nội bộ (dồn vào `array` cho đỡ tốn bộ nhớ) thay vì chuỗi id gốc như
nhánh AniList. Cả pipeline chạy hết: đọc dữ liệu, 27.999 ảnh bìa, bốn mặt đặc
trưng, ALS 64 nhân tố — rồi chết ở **dòng ghi `index.json`**, vì `json` không
nhận khoá `numpy.int32`. Mất một tiếng rưỡi huấn luyện cho một lỗi kiểu dữ
liệu, và chỗ chết nằm xa nhất có thể so với nguyên nhân.

Nên hợp đồng bây giờ gồm hai điều kiện về kiểu, có test canh:
`user` là **chuỗi**, `media_id` là **số nguyên**, ở cả hai nguồn. Cột `user`
xuất ra là id gốc của Goodreads, ánh xạ lại từ mã số bằng cách lập chỉ mục một
mảng `object` — 7,3 triệu con trỏ (58 MB) trỏ vào 342 nghìn chuỗi đã có sẵn
trong bảng băm, không nhân bản chuỗi nào.

Bài học chung hơn, và nó áp cho cả bảng chỉ tiêu ở mục 2: **thiếu dữ liệu phải
hiện ra là `N/A`/`nan`, không được hiện ra là một con số.** Cùng nhánh này còn
hai chỗ nữa cùng kiểu lỗi, đã sửa trước khi chạy: `measure_product.py` báo
"số nước trong top-10" là KHÔNG ĐẠT (danh mục không có trường xuất xứ thì
đếm được 0 nước), và `metrics.evaluate` báo tỉ lệ manhua `0,00` vì
`{id: None}` vẫn là dict "truthy" — đọc lên hệt như "mô hình không gợi ý
manhua nào", một tuyên bố sai về mô hình.

### Những chỗ không ánh xạ được

| Trường | Goodreads | Hệ quả |
|---|---|---|
| `country` | **không có trường nào** | mọi chỉ số theo nước là **N/A**. `country_code` trong dữ liệu là nước của *ấn bản* (gần như toàn bộ `US`), không phải nơi tác phẩm ra đời |
| `progress` | không có tiến độ trong quyển | `catchup` không tính được, `mean_drop_position` rỗng |
| `status` | `is_read` nhị phân | không có DROPPED, không có PAUSED → `drop_rate` và `pause_rate` bằng 0 trên toàn danh mục |
| `pub_status` | không có | gán `FINISHED` cho tất cả: một quyển đã in thì đã xong |

Không suy xuất xứ từ nhà xuất bản hay từ kệ "manga". Đó là heuristic có lỗi và
**không tách được manhwa khỏi manhua**; để `None` là cách duy nhất không nói
sai. `tests/test_goodreads.py::test_khong_doan_xuat_xu` canh điều này.

### Những chỗ ánh xạ được nhưng đổi đơn vị

| Trường | Cách quy đổi |
|---|---|
| `score` | rating 1–5 × 20 → thang 100 của AniList (mốc trung tính 70 ứng với rating 3,5). `rating = 0` là **chưa chấm**, quy về `None` — đúng cách `ingest.py` xử lý `score = 0` của AniList |
| `chapters` | ← `num_pages`. Hai bên đo "tác phẩm dài bao nhiêu" bằng hai đơn vị; `build_pace` log rồi chuẩn hoá nên chỉ thứ tự và độ phân tán có nghĩa. Dùng `num_pages` là để nhánh đối chứng chạy với tín hiệu nó **thật sự có** |
| `tag_ranks` | `count` của `popular_shelves` quy về % so với kệ nhiều nhất của chính tác phẩm đó, bắt chước rank % của AniList |
| `media_id` | ← `work_id`, **không phải** `book_id` (xem mục 2) |

## 2. Đo trên dữ liệu thật, không đọc tài liệu

Chạy ngày 2026-11-10 trên `goodreads_books_comics_graphic.json.gz`
(89.411 ấn bản). Đây là bảng mục 9 của [data-sources.md](data-sources.md), nay
đã điền.

| Câu hỏi | Ngưỡng báo động | Đo được | |
|---|---|---|---|
| Gộp `work_id` giảm bao nhiêu item | > 30% → bắt buộc gộp | **29,6%** (89.411 → 62.944) | sát ngưỡng, vẫn phải gộp |
| Tỉ lệ tác phẩm tách được vai hoạ sĩ | < 20% → không dựa được | **40,6%** | **đạt** — tài liệu cũ bi quan hơn dữ liệu |
| Ảnh bìa tải được bao nhiêu | < 70% → nhánh nét vẽ lung lay | 150/150 URL | đạt |
| Ảnh bìa rộng bao nhiêu pixel | trung vị < 300px → Gram matrix vô dụng | **98px** như URL ghi trong dataset | **không đạt** |
| …sau khi sửa hậu tố URL `m` → `l` | | **309px** rộng, 450px cao, **69,1%** đạt ≥ 300px rộng — đo trên 2.500 ảnh **đã tải thật**, không phải trên URL | **đạt** |
| Tỉ lệ ảnh placeholder | > 15% → cần bước lọc riêng | **37,4%** (23.527 tác phẩm dùng **cùng một** ảnh xám 111×148) | phải lọc |
| Biết độ dài tác phẩm | — | 73,6%, trung vị 160 trang | dùng được |

Hai dòng ảnh bìa là chỗ **dữ liệu nói ngược tài liệu cũ**. Mục 9 cảnh báo đúng
rằng URL trong dataset chỉ cho ảnh 98px — quá nhỏ để đo độ dày nét hay mật độ
screentone, vì chính thông tin đó đã bị resize xoá mất. Nhưng cách sửa mà chính
mục 9 đề xuất ("thử sửa tham số kích thước trong URL") **có hiệu lực**: đổi
`/books/<ts>m/` thành `/books/<ts>l/` cho ảnh 309×450. Nên mặt nét vẽ trên
Goodreads **không chết**, nó chỉ cần một dòng xử lý URL. Còn 37,4% ảnh
placeholder thì bỏ hẳn, không tải: để lại sẽ tạo một cụm giả 23.527 tác phẩm có
vector nét vẽ **giống hệt nhau**, đúng bẫy 5.

### Danh mục này thật ra chứa gì

Đếm theo kệ sách trên 62.944 tác phẩm:

| Kệ | Số tác phẩm | Tỉ lệ |
|---|---|---|
| `comics` | 51.215 | **81,4%** |
| `graphic-novels` | 42.759 | 67,9% |
| `manga` | 16.821 | 26,7% |
| `superheroes` | 8.313 | 13,2% |
| `manhwa` | 843 | **1,34%** |
| `manhua` | 148 | **0,235%** |

Đây là phiên bản **đo được** của khẳng định ở mục 1 [data-sources.md](data-sources.md).
Tỉ lệ manhua trên manga là **0,0088**, tức khoảng **113 bộ manga cho mỗi bộ
manhua**. Với đề tài yêu cầu cả ba loại, con số đó không phải "hơi ít" mà là
không có. Đổi nguồn sang AniList là quyết định đúng, và giờ nó có một con số
đứng sau thay vì một lập luận.

## 3. Chốt trước khi chạy mô hình

Viết trước khi huấn luyện bất kỳ mô hình nào trên `data/processed_gr`, để không
thể đọc kết quả rồi mới chọn câu chuyện. Mỗi dòng có điều kiện **bác bỏ** rõ
ràng; nếu bị bác bỏ thì ghi là bị bác bỏ, không diễn giải lại.

| # | Dự đoán | Bác bỏ nếu |
|---|---|---|
| **P1** | Mặt `art` **vẫn** đóng góp biên dương nhưng **nhỏ hơn** +13,8% của AniList, vì ảnh 309×450 thay vì 460×643 — tức khoảng **45% số pixel** của AniList — và phủ thấp hơn (62,6% tác phẩm có ảnh thật, trong đó 69,1% đạt ≥300px rộng; đo trên 2.500/27.999 ảnh đã tải, 0 ảnh placeholder lọt qua) | `art` đóng góp **nhiều hơn** trên Goodreads → câu chuyện "phụ thuộc độ phân giải" của tôi sai |
| **P2** | DropoutNet cho cổng độ phổ biến **vẫn** cải thiện `new_user` mạnh (> 30% tương đối), vì cơ chế của nó là *huấn luyện trong đúng điều kiện lúc phục vụ* — một tính chất của quy trình, không phải của dữ liệu | hiệu ứng < 10% tương đối hoặc âm → cơ chế phụ thuộc bộ dữ liệu, không phải cơ chế chung |
| **P3** | Chấm toàn danh mục **vẫn** hơn xếp lại top-400 của ALS, với mức hơn ít nhất bằng trên AniList, vì lập luận trần thu hồi là lập luận cấu trúc | hệ hai tầng ngang hoặc hơn → trần thu hồi không phải nguyên nhân thật |
| **P4** | Mặt `pace` đóng góp **gần như không gì** (< 2% tương đối), vì `drop_rate`, `pause_rate`, `mean_drop_position` bằng 0 trên toàn danh mục | `pace` đóng góp ngang AniList → tín hiệu bỏ-dở không phải thứ làm `pace` hoạt động, và mục giải thích `pace` bên AniList phải viết lại |

Thêm **P5**, viết sau khi đo hình dạng dữ liệu (`probe_goodreads_shape.py`)
nhưng **trước** khi huấn luyện bất kỳ hai tháp nào trên bộ này. Phép đo đó cho
một khác biệt lớn đến mức phải chốt dự đoán riêng:

| | AniList | Goodreads |
|---|---|---|
| người đọc (train) | 6.535 | 88.204 — **13,5×** |
| tương tác (train) | 1.520.121 | 3.150.196 — 2,1× |
| tác phẩm | 34.036 | 47.769 |
| **độ dài lịch sử trung vị** | **117** | **13** — ngắn hơn 9 lần |
| **% người đọc có < 10 mục** | **6,6%** | **40,0%** |

(Bảng trên là hình dạng của **toàn bộ** bộ dữ liệu. Mọi mô hình ở các mục
sau chạy ở **quy mô khớp** 40.855 người đọc / 1.462.766 tương tác — xem mục
4 — giữ nguyên lịch sử trung vị 13 và 39,9% người đọc dưới 10 mục. Mục 13
dùng cả hai quy mô, nên chỗ này phải rõ ngay từ đây.)

Nguyên nhân không phải chất lượng dữ liệu mà là **đơn vị đếm**: AniList ghi
nhận từng *bộ truyện* mà người ta theo dõi qua hàng trăm chương, Goodreads ghi
nhận từng *quyển đã in*. Một người đọc 13 quyển trên Goodreads không ít đọc hơn
một người theo 117 bộ trên AniList.

| # | Dự đoán | Bác bỏ nếu |
|---|---|---|
| **P5** | Cổng độ phổ biến sẽ **có lợi rộng hơn** trên Goodreads, vì lát `sparse_user` ở đây là 40% người đọc chứ không phải 6,6%, và trên AniList chính lát thưa là nơi cổng thắng (+4,4%, dù chưa lập được). Kéo theo: **định tuyến `auto` có thể không còn cần thiết** — nếu cổng thắng cả ở `new_user` thì không còn hai kết quả ngược chiều để phải định tuyến | cổng **vẫn** mất ở `new_user` với mức tương tự AniList (−5%) → chỗ mất đó không phải do lát thưa nhỏ, mà là tính chất của chính cơ chế cổng, và lời giải thích bên AniList phải chặt hơn |

### Một điều kiện biên phải ghi trước, không phải sau

Đầu **thời điểm** của hai tháp có 5 cột (`train_embedding.TIME_COLUMNS`). Trên
Goodreads, mọi quyển đã in đều được gán `pub_status = FINISHED`, nên hai cột
`is_releasing` và `is_finished` là **hằng số**; sau phép chuẩn hoá
`(x − mean) / max(std, 1e-6)` chúng thành 0 toàn bộ. Đầu thời điểm vẫn chạy,
vẫn không báo gì, nhưng nó chỉ còn **3 trục thật** thay vì 5: độ phổ biến toàn
kỳ, độ phổ biến 60 ngày, và tuổi tác phẩm.

Điều này ràng buộc cách đọc **P2** và **P5**, vì cả hai biến thể `feed` và
`feed_gated` đều dựa vào đầu thời điểm: nếu chúng yếu hơn trên nhánh này thì
*một phần* nguyên nhân đã biết trước và không được tính thành "cơ chế không
lặp lại". Biến thể `discover` bỏ hẳn đầu thời điểm nên không chịu ảnh hưởng —
nó là mốc so sạch.
`scripts/diagnose_goodreads.py` in số cột bẹt của cả hai nhánh để con số này
nằm trong kết quả chứ không nằm trong trí nhớ.

Quy tắc đọc kết quả, chốt luôn tại đây:

- Mọi kết luận về chênh lệch nhỏ hơn nhiễu khởi tạo phải đi qua **phép kiểm
  ghép cặp theo seed** (`--seeds`), không phải một lần chạy. Nhánh AniList đã
  một lần công bố +9,4% từ một lần chạy mà sự thật ghép cặp là +4,4% — phóng
  đại 2,1 lần, và phóng đại **về phía con số đẹp**.
- Ngưỡng là `p = 0,05` hai phía trên phân phối t với `df = số seed − 1`. Không
  thêm seed sau khi đã nhìn kết quả; nếu có thêm thì phải ghi là lần nhìn thứ
  hai và áp hiệu chỉnh Bonferroni.
- "Lặp lại" nghĩa là **cùng dấu và cùng kết luận ở ngưỡng đã chốt**, không phải
  "cùng con số". Hai bộ dữ liệu khác nhau thì giá trị tuyệt đối khác nhau là
  điều đương nhiên; cái phải giống là *kết luận*.
- Nhánh này chạy ở **quy mô khớp** `--sample-users 160000`, chốt trước khi có
  bất kỳ kết quả mô hình nào và dựa trên một phép **đo** chứ không phải một phép
  chia (mục 4). Nếu một dự đoán bị **bác bỏ**, chạy lại *đúng dự đoán đó* ở quy
  mô đầy đủ (88.204 người đọc) để biết nó bị bác bỏ vì **dữ liệu** hay vì **quy
  mô**, và ghi cả hai kết quả. Nếu một dự đoán được **xác nhận** thì không chạy
  lại ở quy mô khác: chạy lại chỉ những kết quả mình không thích rồi lấy con số
  dễ chịu hơn chính là "chọn quy mô sau khi nhìn kết quả".

## 4. Chọn quy mô: đo, không ngoại suy

`probe_goodreads_shape.py` ban đầu đề xuất `--sample-users 42562`, bằng cách
chia số tương tác của AniList cho độ dài lịch sử trung bình. Lần chạy thử ở
3.000 người đọc cho thấy phép chia đó sai về bản chất: 58.865 tương tác thô chỉ
còn **18.765** sau lọc lõi, 3.000 người đọc còn **578**, và 62.944 tác phẩm còn
**1.598** — tức 2,5% danh mục. Nguyên nhân là `min_item = 5`: mỗi tác phẩm cần 5
người đọc mới trụ được, nên cắt mỏng số người làm danh mục rụng trước, rồi người
đọc mất mục và rụng theo `min_user = 5`. Sụt giảm là **cấp số nhân**.

`scripts/scale_curve_goodreads.py` chạy thật lọc lõi ở năm mức (seed 0):

| mẫu người | train | người | truyện | lịch sử tv | <10 mục | lát thưa |
|---|---|---|---|---|---|---|
| 20.000 | 150.427 | 4.822 | 9.647 | 12 | 41,1% | 33,0% |
| 40.000 | 345.998 | 9.935 | 16.715 | 12 | 40,5% | 33,1% |
| 80.000 | 707.296 | 20.190 | 24.230 | 13 | 39,8% | 31,7% |
| **160.000** | **1.462.766** | 40.855 | **33.527** | **13** | **39,9%** | **32,6%** |
| tất cả | 3.150.196 | 88.204 | 43.306 | 13 | 40,0% | 32,7% |
| *AniList* | *1.520.121* | *6.535* | *34.036* | *117* | *6,6%* | — |

Con số đúng là **160.000**, gấp **3,8 lần** phép ngoại suy. Ở mức đó nhánh
Goodreads khớp AniList trên **cả hai** chiều quy mô — tương tác `0,96×` và danh
mục `0,99×` — trong khi vẫn giữ nguyên **tính chất riêng** của nó: lịch sử trung
vị 13, 39,9% người đọc dưới 10 mục, lát thưa 32,6%, không lệch so với quy mô đầy
đủ. Đó chính là lý do lấy mẫu theo *người đọc* chứ không theo *dòng*: nó giảm số
người mà không bóp méo hình dạng lịch sử của từng người.

Nên quyết định quy mô ở đây không phải một đánh đổi: khớp quy mô **bỏ được hai
yếu tố gây nhiễu** (số tương tác, cỡ danh mục) mà **không đổi** thứ đang muốn so
sánh, và tốn một nửa thời gian huấn luyện. Quy mô đầy đủ vẫn còn đó, làm phép
kiểm tra thứ hai cho bất kỳ dự đoán bị bác bỏ.

Một chỗ còn lệch và không khớp được: số **người đọc** vẫn 40.855 so với 6.535
(6,3×). Đó không phải thiếu sót của phép lấy mẫu mà là hệ quả số học của việc
khớp số tương tác khi lịch sử ngắn hơn 9 lần — muốn khớp cả hai thì phải đổi độ
dài lịch sử, tức đổi đúng cái đang nghiên cứu.

## 5. Kế hoạch chạy và phân bổ seed — chốt trước

Ngân sách là thứ hữu hạn nên phải chia trước khi nhìn kết quả, nếu không thì
việc "chạy thêm seed cho cái nào" sẽ do con số đã thấy quyết định.

Mỗi seed cần tối đa 7 lần huấn luyện: ba bậc thang mặt (`chỉ theme`,
`+creator`, `+pace`), `discover` (= bậc đủ bốn mặt), và ba biến thể `feed`,
`feed_gated_nodrop`, `feed_gated`. Trên AniList một lần huấn luyện mất ~43 phút
ở 1,52 triệu tương tác; nhánh này khớp quy mô nên chi phí phải tương đương, và
**bước 3 dưới đây đo lại con số đó** trước khi cam kết phần còn lại.

| bước | chạy gì | vì sao |
|---|---|---|
| 1 | `train.py --source goodreads --sample-users 160000` | artifact của nhánh |
| 2 | `diagnose_goodreads.py` | bản chất hai bộ + **trần thu hồi top-400** = tiền đề của P3, rẻ |
| 3 | một lần `train_embedding.py --facet-ablation --variants feed feed_gated_nodrop feed_gated` ở seed 0 | ước lượng điểm cho **cả năm** dự đoán, và **đo thời gian thật** mỗi lần huấn luyện |
| 4 | thêm seed theo phân bổ dưới | chỉ chỗ nào chênh lệch nhỏ hơn nhiễu khởi tạo |

Phân bổ seed, chốt bằng **độ lớn hiệu ứng dự kiến từ nhánh AniList**, không
bằng kết quả của nhánh này:

| dự đoán | hiệu ứng dự kiến | số seed | phép kiểm |
|---|---|---|---|
| **P1** mặt `art` | ~+14% tương đối | **5** (`--art-seeds 0 1 2 3 4`) | ghép cặp, `t` ở `df=4`, ngưỡng 2,776 |
| **P4** mặt `pace` | < 2% | **4** (bậc `+creator` vs `+pace`) | ghép cặp, `df=3` |
| **P5** cổng rộng hay hẹp | ±5% theo lát | **4** (`feed` vs `feed_gated`) | ghép cặp từng lát, `df=3` |
| **P2** DropoutNet | > 30% ở `new_user` | **2** (`nodrop` vs `gated`) | chỉ dấu + khoảng giữa hai seed |
| **P3** hai tầng vs toàn danh mục | cấu trúc | **1** + trần thu hồi | so trần, không cần seed |

P1 dùng **5** seed chứ không phải 4, vì phép kiểm bên AniList là 5 seed
(`t = 4,10`, `df = 4`, p ≈ 0,015 — mục 10 [research-notes.md](research-notes.md)).
Muốn hai nhánh cho hai *kết luận* so được với nhau thì phép kiểm phải cùng bậc
tự do và cùng ngưỡng; chạy 4 seed ở đây rồi so với 5 seed ở kia là so hai ngưỡng
khác nhau. Hiệu ứng bên AniList chỉ **0,0021 NDCG tuyệt đối** với lệch chuẩn của
hiệu là 0,0011 — ở mức đó, một bậc tự do đổi là đổi câu trả lời.

Lý do P2 chỉ 2 seed trong khi P1 năm: ngưỡng ghép cặp là bắt buộc **khi chênh
lệch nhỏ hơn nhiễu khởi tạo**. +30% ở `new_user` lớn hơn nhiễu seed đã đo trên
AniList cả bậc độ lớn, nên thêm seed ở đó chỉ làm hẹp một khoảng đã rõ dấu,
trong khi +14% của `art` thì *không* — chính mặt `art` đã một lần bị kết luận
sai vì đọc một lần chạy. Nếu bước 3 cho P2 một hiệu ứng **dưới 10%** thì nó rơi
vào vùng cần ghép cặp và được nâng lên 4 seed; quy tắc đó ghi ở đây trước, nên
nó không phải là chọn sau khi nhìn.

## 6. Bản chất hai bộ dữ liệu, đo sau khi khớp quy mô

`scripts/diagnose_goodreads.py`, chạy trên hai nhánh đã khớp quy mô. Đây là
phần **dữ liệu**, chưa có mô hình nào trong này — nên nó là tiền đề để đọc
P1–P5, không phải kết quả của chúng.

| | AniList | Goodreads | ghi chú |
|---|---|---|---|
| tương tác train | 1.520.121 | 1.462.766 | khớp 0,96× |
| truyện | 34.036 | 33.527 | khớp 0,99× |
| người đọc | 6.535 | 40.855 | 6,3× — không khớp được, xem mục 4 |
| **độ thưa** | 0,00683 | **0,00107** | thưa hơn **6,4 lần** trên mỗi ô |
| Gini lượt/truyện | 0,710 | **0,734** | đầu độ phổ biến có *nhiều* việc hơn |
| lượt vào 1% truyện đầu | 0,189 | **0,273** | |
| lát `new_item` | 16.121 | **43.873** | 2,7× — lát này không hề mỏng |
| lát `sparse_user` | 22.224 | **59.554** | 2,7× |
| tag/kệ mỗi truyện | 5,9 | **36,5** | kệ Goodreads giàu hơn nhiều |
| tách được vai hoạ sĩ | 97,2% | **41,7%** | ảnh hưởng mặt `creator`, không phải mặt `art` |
| phủ mặt `art` | 57,8% | 62,3% | tương đương |
| **cột nhịp bẹt** | 2/13 | **7/13** | |
| **cột thời điểm bẹt** | 0/5 | 2/5 | |
| **trần thu hồi top-400** | 35,3% | **28,8%** | |

Ba chỗ đáng kể, và cả ba đều **đi trước** kết quả mô hình:

1. **P4 có tiền đề mạnh hơn mức đã chốt.** Tôi ghi trước rằng ba cột
   (`drop_rate`, `pause_rate`, `mean_drop_position`) sẽ bằng 0. Đo ra **7/13**
   cột bẹt: thêm `log_volumes`, `duration_years`, `is_finished`,
   `completion_rate`. Tức mặt `pace` trên nhánh này không phải "thiếu một tín
   hiệu" mà là **mất hơn nửa số trục**. Dự đoán không đổi, nhưng nếu `pace` vẫn
   đóng góp thì điều đó còn khó giải thích hơn tôi tưởng lúc chốt.
2. **P3 có tiền đề mạnh hơn.** Trần thu hồi của rổ 400 ứng viên là **28,8%** so
   với 35,3% của AniList — hệ hai tầng ở đây bị chặn trên **thấp hơn**, nên nếu
   chấm toàn danh mục *không* thắng thì lập luận trần thu hồi sai ở chỗ căn bản.
3. **P5 có tiền đề mạnh hơn.** Gini cao hơn và 1% truyện đầu hút 27,3% lượt
   (so với 18,9%), cộng với 39,9% người đọc dưới 10 mục — cả hai đều là điều
   kiện để cổng độ phổ biến có lợi *rộng hơn*, đúng chiều P5 đã chốt.

Một chỗ đi **ngược** chiều thuận lợi và phải ghi rõ: chỉ **41,7%** tác phẩm tách
được vai hoạ sĩ (AniList 97,2%). Mặt `creator` vẫn phủ 98,9% nhờ vai tác giả,
nhưng phần "cùng hoạ sĩ" trong đó mỏng hơn nhiều — nên chỉ số `trùng tác giả`
giữa hai nhánh **không so trực tiếp được**, và tôi sẽ không so.

## 7. P3 bị BÁC BỎ — và lý do không phải là trần thu hồi

> Kiểm lại ở quy mô đầy đủ (88.204 người đọc): phán quyết đứng nguyên, xem mục 14.

Phép so rẻ nhất lại là phép so lật được một dự đoán. `scripts/evaluate.py` chấm
hệ **hai tầng** (ALS lấy 400 ứng viên, các mặt xếp lại) trên cùng artifact đã
khớp quy mô, không huấn luyện gì thêm; `feed` là hai tháp **chấm toàn danh mục**.

NDCG@10, seed 0:

| | `all` | `warm` | `sparse_user` | `new_user` | `new_item` |
|---|---|---|---|---|---|
| **AniList** hai tầng (đủ mặt) | 0,1072 | 0,0970 | 0,1235 | 0,3442 | 0,0000 |
| **AniList** `feed` toàn danh mục | **0,1169** | **0,1039** | **0,1268** | **0,3771** | 0,0000 |
| | *+9,0%* | *+7,1%* | *+2,7%* | *+9,6%* | |
| **Goodreads** hai tầng (đủ mặt) | **0,0757** | **0,0576** | **0,0872** | 0,1858 | 0,0000 |
| **Goodreads** `feed` toàn danh mục | 0,0625 | 0,0430 | 0,0716 | **0,2322** | 0,0000 |
| | *−17,4%* | *−25,3%* | *−17,9%* | *+25,0%* | |

Điều kiện bác bỏ đã ghi trước là "hệ hai tầng ngang hoặc hơn". Nó **hơn**, và hơn
nhiều: −17 đến −25% trên ba lát. **P3 bị bác bỏ.** Ghi là bị bác bỏ, không diễn
giải lại thành "chỉ đúng một phần".

Chỗ đáng nói hơn: trần thu hồi ở đây **thấp hơn** AniList (28,8% so với 35,3%,
mục 6). Tức nhánh hai tầng bị chặn trên *chặt hơn* mà vẫn thắng. Nên lập luận
"trần thu hồi quyết định" là **không đủ**: nó đặt một chặn trên đúng, rồi tôi
dùng chặn trên đó như thể nó là một dự báo về thứ hạng. Hai chuyện khác nhau.
Mô hình có thể bị chặn ở 28,8% mà vẫn xếp hạng tốt hơn trong phần nó với tới
được, nếu phần nó với tới được là phần người ta thật sự đọc.

Lời giải thích ứng viên, và nó **đo được** chứ không phải kể cho hợp:
`feed` chỉ thua ở `warm` và `sparse_user` — hai lát của người đọc *đã có lịch
sử* — mà thắng đậm ở `new_user`. Khác biệt giữa hai nhánh ở đúng chỗ đó là **số
quan sát trên mỗi người**: AniList có 1,52 triệu tương tác trên 6.535 người
(~233 mỗi người), Goodreads khớp quy mô có 1,46 triệu trên 40.855 người (~36
mỗi người). Hai tháp học một vector ID cho *mỗi người*, nên nó cần quan sát trên
mỗi người; ALS trong hệ hai tầng cũng vậy, nhưng phần xếp lại dựa vào các mặt
nội dung dùng chung cho mọi người, không cần.

Dự đoán tiếp theo, ghi **trước** khi chạy: nếu nguyên nhân là *số quan sát trên
mỗi người* thì chạy ở quy mô đầy đủ (3,15 triệu trên 88.204 người — vẫn ~36 mỗi
người) sẽ **vẫn bác bỏ P3**. Nếu nguyên nhân là *tổng lượng dữ liệu* thì quy mô
đầy đủ (2,1× dữ liệu) phải thu hẹp khoảng cách rõ rệt. Đây là nghĩa vụ đã ghi ở
mục 3: dự đoán bị bác bỏ thì phải chạy lại ở quy mô kia — và lần chạy đó bây giờ
còn kiểm được một **cơ chế**, không chỉ kiểm quy mô.

## 8. P2 được XÁC NHẬN — gần y nguyên mức của AniList

`feed_gated_nodrop` so với `feed_gated` tách đúng một chỗ: có hay không bỏ ngẫu
nhiên `delta_u` lúc huấn luyện. NDCG@10:

| | `all` | `new_user` | `sparse_user` | `warm` |
|---|---|---|---|---|
| **AniList** cổng hai phần, **không** bỏ ngẫu nhiên | 0,1171 | 0,1774 | 0,1332 | 0,1041 |
| **AniList** cổng hai phần **+ bỏ ngẫu nhiên** | 0,1202 | **0,3533** | 0,1387 | 0,1051 |
| | | ***+99,2%*** | | |
| **Goodreads** cổng hai phần, **không** bỏ ngẫu nhiên | 0,0525 | 0,0779 | 0,0639 | 0,0316 |
| **Goodreads** cổng hai phần **+ bỏ ngẫu nhiên** | 0,0564 | **0,1580** | 0,0659 | 0,0367 |
| | | ***+102,8%*** | | |

Ngưỡng đã ghi trước là "> 30% tương đối". Mức đo được là **+102,8%**, trên một
bộ dữ liệu khác, với đầu thời điểm chỉ còn 3 trục thật thay vì 5 (mục 3).
**P2 được xác nhận.**

Phần lặp lại còn chặt hơn chính dự đoán: ở *cả hai* nhánh, bản không bỏ ngẫu
nhiên **tệ hơn cả bản không cổng** ở `new_user` — AniList 0,1774 so với 0,3771
(47%), Goodreads 0,0779 so với 0,2322 (33,5%). Tức kết luận quy công ở mục 9.9
`docs/research-notes.md` — "đưa cấu trúc vào không đủ, điều quyết định là huấn
luyện trong đúng điều kiện lúc phục vụ" — lặp lại trên bộ thứ hai: cùng dấu, cùng
thứ tự ba cấu hình, và gần như cùng độ lớn.

Về phép kiểm ghép cặp: quy tắc ở mục 3 bắt buộc ghép cặp cho *chênh lệch nhỏ hơn
nhiễu khởi tạo*. Nhiễu seed của cấu hình này trên AniList là 0,3483–0,3589 ở
`new_user` (±1,5%), còn hiệu ứng là +103%. Nên hai seed đã phân bổ cho P2 là phép
xác nhận thêm, không phải chỗ quyết định kết luận; nếu chúng cho khác dấu thì
phải viết lại cả mục này.

## 9. P5 bị BÁC BỎ — cổng không "có lợi rộng hơn", nó mất ở mọi lát

> Kiểm lại ở quy mô đầy đủ: phán quyết chính đứng nguyên (cổng mất ở cả bốn lát), nhưng câu hỏi *cơ chế* thì luật đo của tôi không phân giải được — xem mục 14.

`feed` so với `feed_gated` (AniList: trung bình 4 seed; Goodreads: seed 0):

| | `all` | `new_user` | `sparse_user` | `warm` |
|---|---|---|---|---|
| **AniList** `feed` | 0,1175 | 0,3744 | 0,1283 | 0,1035 |
| **AniList** `feed_gated` | **0,1196** | 0,3549 | **0,1339** | **0,1055** |
| | *+1,9%* | *−5,2%* | *+4,4%* | *+1,9%* |
| **Goodreads** `feed` | **0,0625** | **0,2322** | **0,0716** | **0,0430** |
| **Goodreads** `feed_gated` | 0,0564 | 0,1580 | 0,0659 | 0,0367 |
| | *−9,8%* | *−32,0%* | *−8,0%* | *−14,7%* |

P5 dự đoán cổng **có lợi rộng hơn** trên Goodreads vì lát thưa ở đây là 40% người
đọc chứ không phải 6,6%. Thực tế: cổng **mất ở cả bốn lát**, kể cả chính lát thưa
(−8,0%, nơi AniList được +4,4%), và mất ở `new_user` gấp sáu lần mức AniList.
**P5 bị bác bỏ.** Hệ quả kéo theo mà tôi đã ghi — "định tuyến `auto` có thể không
còn cần thiết" — cũng sai, nhưng sai theo chiều khác hẳn: trên nhánh này không có
gì để định tuyến, vì bản không cổng trội hơn bản có cổng ở mọi lát (và mốc `đang
nổi 60 ngày` còn trội hơn cả hai ở `all` và `warm`, xem mục 10).

### Một kết luận sai mà phép đo gần như đã chống lưng cho

Chỗ đầu tiên tôi nhìn là `scripts/diagnose_trend_gate.py`. Nó in `g_u trung bình
0,0108` cho Goodreads, trong khi `data/processed/trend_gate_diagnosis.json` ghi
`gate_mean 1,3233` cho AniList. Đọc hai số đó cạnh nhau cho một câu chuyện rất
gọn: *cổng đã tự tắt đầu thời điểm trên Goodreads*. Nó sai hai lần:

1. Script in `model.trend.weight`, mà với thiết kế cổng **hai phần** hiện tại đó
   chỉ là phần dư từng người `delta_u`; cổng thật là
   `g_u = base(log(1+độ dài lịch sử)) + delta_u`.
2. Blob JSON của AniList được ghi từ thời cổng **một phần**, nơi `trend.weight`
   *đúng là* g_u. Tức hai con số đem so là hai tham số khác nhau của hai thiết kế
   khác nhau.

Đo lại đúng cách — cùng một đường tính, chạy lại cả hai nhánh — thì không có gì
tắt cả: g_u trung bình **2,667** trên Goodreads so với **2,090** trên AniList, và
thang hiệu dụng của đầu thời điểm gần như không đổi khi bật cổng (AniList
0,894 → 0,448 × 2,341 = 0,937; Goodreads 1,102 → 0,437 × 2,764 = 1,164). Script
giờ in rõ thiết kế của blob đang đọc, in cả `base`, `delta_u` và g_u, nên cái bẫy
này không dựng lại được bằng cách đọc nhầm nhãn.

## 10. Cơ chế của chỗ mất: bốn phép đo, trong đó một phép đo làm yếu câu chuyện

**(1) Cấu trúc cổng lặp lại về bản chất.** Cùng `scripts/diagnose_trend_gate.py`,
chạy trên hai nhánh:

| | AniList | Goodreads |
|---|---|---|
| g_u trung bình | 2,090 | 2,667 |
| g_u của người đọc **mới** (n=0) | 2,341 | 2,764 |
| rho(g_u, độ dài lịch sử) | −0,721 | −0,342 |
| độ dốc khớp `g_u ~ log(1+n)` | −0,0925 | **−0,0440** |
| rho(g_u, độ nổi trung bình truyện đã đọc) | +0,775 | +0,826 |
| `delta_u` std | 0,097 | **0,174** |

Ba tính chất mà thiết kế dựa vào đều còn: g_u giảm đơn điệu theo độ dài lịch sử,
người đọc mới nhận giá trị cao nhất, và g_u bám vào độ nổi của những truyện người
đó đã đọc (+0,83, còn mạnh hơn AniList). Cái đổi là **tỉ lệ giữa phần có cấu trúc
và phần tự do**: độ dốc theo độ dài lịch sử chỉ còn một nửa, còn phần dư từng
người `delta_u` rộng gấp 1,8 lần.

**(2) `delta_u` khớp train như nhau nhưng lặp lại ở kỳ sau kém hơn.** Mục 4 của
script: cùng một thước độ nổi, nhưng tính trên những mục người đó đọc trong
**test**.

| | AniList | Goodreads |
|---|---|---|
| rho(delta_u, độ nổi mục TRAIN) | +0,770 | +0,784 |
| rho(delta_u, độ nổi mục TEST) | +0,485 | +0,364 |
| theo số quan sát train: 0–19 | +0,381 | +0,312 |
| 20–49 | +0,558 | +0,432 |
| ≥150 | +0,420 | +0,493 |
| ô khớp 10–29 mục test & 10–49 quan sát | +0,486 (n=297) | +0,440 (n=1.683) |

Đọc đúng mức: khoảng cách gộp (+0,485 so với +0,364) **phần lớn là hiệu ứng cơ
cấu**, không phải hai nhánh khác bản chất. Khi khớp cả số quan sát train lẫn số
mục test, hai nhánh còn +0,486 so với +0,440 — chênh 0,046 trong khi sai số chuẩn
của Spearman ở n=297 cỡ 0,058, tức **không phân biệt được**. Điều đứng vững là
quan hệ *trong từng nhánh*: `delta_u` lặp lại càng kém khi người đọc càng ít quan
sát, và Goodreads có 40% người đọc nằm đúng vùng mỏng đó. Tham số theo từng người
cần quan sát theo từng người — cùng cơ chế đã ghi cho P3 (mục 7), đo ở một chỗ
khác.

**(3) Đầu danh mục toàn kỳ của Goodreads lạc hậu — nhưng chỉ ở đỉnh.** Mục F3 của
`scripts/diagnose_goodreads.py` (NDCG@10 chỉ nhìn 10 vị trí, nên đo ở đó):

| | AniList | Goodreads |
|---|---|---|
| hit@10 mốc phổ biến toàn kỳ | 9,25% | **5,76%** |
| hit@10 mốc đang nổi 60 ngày | 10,60% | 10,24% |
| tỉ số | 1,15× | **1,78×** |
| tuổi lượt đọc, toàn danh mục | 821 ngày | 940 ngày |
| tuổi lượt đọc, top-400 toàn kỳ | 790 ngày | **1.042 ngày** |
| % lượt ở top-400 cũ hơn 2 năm | 52,4% | 63,7% |

Trên AniList đầu danh mục *trẻ hơn* danh mục; trên Goodreads nó **già hơn**, và
càng lên đỉnh càng già (top-10: 1.113 ngày, 68,1% cũ hơn 2 năm). Đó là lý do mốc
`đang nổi 60 ngày` đạt 0,0824 ở `all` trên Goodreads — hơn cả ALS (0,0734) và hơn
cả hai tháp (0,0625) — trong khi trên AniList nó là mốc yếu nhất. Chênh lệch này
là tính chất **dữ liệu**, đo được mà không cần mô hình nào.

**(4) Đầu thời điểm học gần như chỉ phổ biến toàn kỳ — ở cả hai nhánh.** Trọng số
đã nhân với g_u của người đọc mới:

| | `log_pop_all` | `log_pop_60d` | `age_years` |
|---|---|---|---|
| AniList `feed` | +0,891 | +0,054 | −0,005 |
| AniList `feed_gated` | +1,047 | +0,059 | −0,007 |
| Goodreads `feed` | +1,101 | +0,050 | +0,009 |
| Goodreads `feed_gated` | **+1,206** | +0,045 | +0,009 |

Nối bốn phép đo thành một chuỗi, mỗi khâu có số: cổng **khuếch đại** đầu thời điểm
(hệ số 2,76 cho người đọc mới) → đầu thời điểm gần như chỉ là *phổ biến toàn kỳ* →
trên Goodreads trục đó lạc hậu ở đúng 10 vị trí đầu → nên người đọc phụ thuộc
nhiều nhất vào bias thời điểm mất nhiều nhất (`new_user` −32,0% so với `all`
−9,8%).

**Phép đo làm yếu câu chuyện, ghi lại nguyên văn.** Nếu lời giải thích là "nhánh
Goodreads cần trục 60 ngày hơn", thì ở cấp **toàn danh mục** nó phải đúng rõ hơn
trên Goodreads. Đo: tương quan hạng giữa mỗi trục và số lượt của truyện trong kỳ
test.

| | `log_pop_all` → test | `log_pop_60d` → test | khoảng cách |
|---|---|---|---|
| AniList | +0,594 | +0,709 | **+0,115** |
| Goodreads | +0,554 | +0,596 | +0,042 |

Ngược chiều: ở cấp toàn danh mục, trục 60 ngày có *lợi thế lớn hơn trên AniList*.
Nguyên nhân đo được: 56,4% truyện Goodreads không có lượt nào trong 60 ngày cuối
(AniList 34,0%), nên cột đó mỏng hơn hẳn. Vậy câu đúng phải hẹp hơn câu tôi muốn
nói: **không** phải "Goodreads cần recency hơn nói chung", mà "chỗ trả giá cho
việc dùng trục toàn kỳ nằm ở đỉnh danh sách, và NDCG@10 chỉ nhìn đúng chỗ đó". Cả
hai nhánh đều đang bỏ qua một trục có ích; nhánh Goodreads bị phạt nặng hơn vì
đỉnh của nó lạc hậu hơn.

### Thí nghiệm thăm dò, đăng ký là THĂM DÒ

Chuỗi trên gợi ra một thay đổi: cho đầu thời điểm trục 60 ngày với trọng số riêng
(hoặc trực giao hoá hai cột phổ biến). Dự đoán ghi trước khi chạy: **cả hai**
nhánh tăng; Goodreads tăng nhiều hơn ở top-10 dù khoảng cách toàn danh mục của nó
nhỏ hơn. Việc này **không** thuộc P1–P5 và được chọn *sau khi* xem kết quả, nên
nó vào mục thăm dò, không được tính là một dự đoán đã lặp lại; nếu chạy thì ghi
riêng, không trộn vào các phán quyết trên.

### Nghĩa vụ chạy lại ở quy mô đầy đủ, giờ gộp hai dự đoán

Mục 3 buộc: dự đoán bị bác bỏ thì chạy lại chính nó ở quy mô đầy đủ (88.204 người
đọc) để tách **dữ liệu** khỏi **quy mô**. P3 và P5 đều bị bác bỏ và đều được giải
thích bằng *số quan sát trên mỗi người*, nên một lần chạy đầy đủ với `feed` và
`feed_gated` kiểm cả hai cùng lúc — và vì quy mô đầy đủ giữ nguyên ~36 quan sát
mỗi người trong khi nhân đôi tổng dữ liệu, nó tách được đúng hai giả thuyết: nếu
cơ chế là quan sát-trên-mỗi-người thì **cả hai vẫn bị bác bỏ**; nếu là tổng lượng
dữ liệu thì khoảng cách phải thu hẹp rõ.

## 11. Cơ chế "quan sát trên mỗi người" tự kiểm được bằng lát có sẵn — và chỉ đúng một phần

Cơ chế tôi ghi cho P3 (mục 7) và P5 (mục 10) là *số quan sát trên mỗi người*.
Cơ chế đó có một hệ quả kiểm được ngay, không cần chạy thêm gì: nó phải lệch
nhiều nhất ở lát mà hai nhánh lệch nhau nhiều nhất về chính con số đó, và lệch ít
nhất ở lát hai nhánh khớp nhau. Mục F4 của `scripts/diagnose_goodreads.py` đo:

| lịch sử train trung vị | AniList | Goodreads | tỉ số |
|---|---|---|---|
| lát `all` | 126 mục | 14 mục | 9,0× |
| lát `warm` | 142 mục | 32 mục | **4,4×** |
| lát `sparse_user` | 5 mục | 4 mục | **1,25× — khớp** |

Đem so với mức lật của P3 (`feed` toàn danh mục so với hệ hai tầng, mục 7):

| lát | AniList (`feed` hơn) | Goodreads (`feed` kém) | lịch sử hai nhánh |
|---|---|---|---|
| `warm` | +7,1% | **−25,3%** | 142 so với 32 — lệch nhất |
| `all` | +9,0% | −17,4% | 126 so với 14 |
| `sparse_user` | +2,7% | −17,9% | 5 so với 4 — **khớp** |

Thứ tự khớp ở chỗ dự đoán được: lát `warm`, nơi hai nhánh lệch 4,4 lần về số
quan sát, là lát lật mạnh nhất (−25,3%). Nhưng lát `sparse_user` **vẫn lật
−17,9%** dù số quan sát gần như bằng nhau (5 so với 4 mục). Nên:

**Số quan sát trên mỗi người là một cơ chế có thật nhưng KHÔNG đủ.** Phần lật ở
lát thưa phải có nguyên nhân dùng chung cho mọi người đọc, không phụ thuộc lịch
sử — và ứng viên đã đo được nằm ở mục 10: đầu thời điểm học gần như chỉ *phổ biến
toàn kỳ*, mà đỉnh danh mục toàn kỳ của Goodreads lạc hậu (hit@10 5,76% so với
10,24%). Người đọc thưa dựa vào bias chung nhiều hơn vào vector ID của mình, nên
họ chịu đúng phần đó.

Ghi rõ để lần chạy đầy đủ không bị đọc quá mức: quy mô đầy đủ giữ ~36 quan
sát/người nên nó tách được *quan sát-trên-mỗi-người* khỏi *tổng lượng dữ liệu*,
nhưng nó **không** tách được phần nguyên nhân dùng chung này — phần đó chỉ tách
được bằng thí nghiệm thăm dò về trục 60 ngày ở mục 10.

## 12. P4 được XÁC NHẬN, P1 chỉ có ước lượng điểm — và tôi không được gọi đó là phán quyết

Thang ablation từng mặt, cấu hình `discover` (tắt logQ, không đầu thời điểm), đo
ở lát `new_item` — lát duy nhất tách được công của nội dung, vì truyện chưa ai
đọc có vector ID bằng 0 nên điểm hoàn toàn do `P·c_i` quyết định. Seed 0, quy mô
khớp, 183 phút cho bốn bậc.

| bậc | chiều | AniList `new_item` | so bậc trước | Goodreads `new_item` | so bậc trước |
|---|---|---|---|---|---|
| chỉ `theme` | 128 | 0,0055 | — | 0,0224 | — |
| + `creator` | 192 | 0,0073 | +33% | 0,0321 | **+43,3%** |
| + `pace` | 205 | 0,0149 | **+104%** | 0,0311 | **−3,1%** |
| + `art` (đủ bốn) | 220 | 0,0165 | +11% | 0,0330 | +6,1% |

### P4 — xác nhận, và nó xác nhận luôn *lời giải thích* bên AniList

P4 ghi trước: mặt `pace` đóng góp **gần như không gì** (< 2% tương đối) trên
Goodreads, vì `drop_rate`, `pause_rate`, `mean_drop_position` bằng 0 trên toàn
danh mục — Goodreads không ghi "đọc tới chương bao nhiêu rồi bỏ". Điều kiện bác
bỏ: "`pace` đóng góp ngang AniList".

Đo được: **−3,1%**. Không ngang AniList (+104%); không dương; không đáng kể.
**P4 được xác nhận.**

Giá trị của nó không nằm ở con số mà ở chỗ nó **kiểm được một lời giải thích**.
Bên AniList, `pace` là mặt *mạnh nhất* (+104%), và tôi đã giải thích rằng công
thuộc về tín hiệu **bỏ dở / tạm dừng** chứ không phải số chương hay trạng thái
phát hành. Lời giải thích đó có thể kiểm được: lấy một bộ dữ liệu không có tín
hiệu bỏ dở nhưng *vẫn có* số trang và năm phát hành. Nếu công thuộc về số
chương, `pace` phải vẫn chạy. Nó không chạy. Nên lời giải thích đứng, và đứng
bằng một phép thử có thể làm nó sụp.

Đây là kiểu kết quả mà một nhánh đối chứng tồn tại để tạo ra: không phải "bộ nào
cho điểm cao hơn", mà "cơ chế tôi *nói* là nguyên nhân có đúng là nguyên nhân
không".

### P1 — ước lượng điểm phù hợp dự đoán, nhưng CHƯA phải phán quyết

P1 ghi trước: mặt `art` **vẫn** dương nhưng **nhỏ hơn** +13,8% của AniList, vì
ảnh bìa ở đây là 309×450 thay vì 460×643 (~45% số pixel) và phủ thấp hơn.

Đo được ở seed 0: **+6,1%** — dương, và nhỏ hơn +13,8%. Khớp cả hai nửa của dự
đoán.

**Nhưng tôi không được ghi là "xác nhận", vì quy tắc ở mục 3 cấm chính việc đó.**
Trên AniList, chênh lệch của bậc `art` là +11% ở một lần đo, *nhỏ hơn* khoảng dao
động giữa các seed (0,0142–0,0191, tức ±15% quanh trung bình), và phải có phép
**ghép cặp 5 seed** mới kết luận được (+13,8%, t = 4,10, df = 4, p ≈ 0,015). Mức
+6,1% ở đây còn nhỏ hơn thế. Một lần chạy không phân biệt được nó với 0.

Nên trạng thái đúng của P1 là: **ước lượng điểm +6,1%, cùng dấu và nhỏ hơn như
dự đoán, chưa có phép kiểm.** Phép kiểm cần `--art-seeds 0 1 2 3 4` cho hai bậc
cuối, tức 10 lần huấn luyện ở ~62 phút mỗi lần (~10 giờ) — đã phân bổ trong kế
hoạch mục 5 và chưa chạy. Ghi thiếu là ghi thiếu; không nâng ước lượng điểm lên
thành phán quyết chỉ vì nó ra đúng chiều mình mong.

### Một khác biệt ngoài dự đoán, ghi lại vì nó đổi thứ tự các mặt

Trên Goodreads, `creator` là mặt **mạnh nhất** (+43,3%), còn trên AniList nó chỉ
+33% và đứng sau `pace`. Thang Goodreads cũng **không đơn điệu** (lên, xuống,
lên), trong khi thang AniList đơn điệu qua cả bốn bậc.

Điều này khớp với bản chất hai bộ đã đo ở mục 6: Goodreads có trường tác giả dày
(dù vai *vẽ* chỉ tách được 41,7% so với 97,2%) nhưng không có gì về nhịp đọc.
Tức thứ tự đóng góp của các mặt **đi theo trường dữ liệu nào có thật**, không
theo một thứ bậc nội tại của "nét vẽ quan trọng hơn thể loại". Đó là một câu dè
dặt hơn câu tôi viết ở nhánh AniList, và nó là câu đúng hơn.

## 13. Quy mô đầy đủ cho P3 và P5: luật đọc chốt trước, số điền sau

Mục này viết **trước khi** lần chạy quy mô đầy đủ kết thúc. Lý do là mục 3 bắt
buộc: dự đoán nào bị bác bỏ thì phải chạy lại chính nó ở quy mô đầy đủ để biết
nó bị bác bỏ vì **dữ liệu** hay vì **quy mô**. Nhưng nếu tôi đợi có số rồi mới
viết cách đọc số thì phép chạy lại đó vô nghĩa — nó chỉ còn là một cơ hội thứ
hai để kể câu chuyện dễ chịu hơn.

### Vì sao một lần chạy tách được đúng hai giả thuyết

`scale_curve.json` cho một tính chất mà tôi không thiết kế ra, chỉ đo được, và
nó đúng là thứ cần cho phép tách này:

| | khớp quy mô (160k) | đầy đủ (342k) | tỉ lệ |
|---|---|---|---|
| người đọc train | 40.855 | 88.204 | **2,16×** |
| tương tác train | 1.462.766 | 3.150.196 | **2,15×** |
| tác phẩm train | 33.527 | 43.306 | 1,29× |
| **quan sát / người** | **35,80** | **35,71** | **1,00×** |
| lịch sử trung vị | 13 | 13 | — |
| % người đọc < 10 mục | 39,9% | 40,0% | — |

Tổng dữ liệu **gấp đôi** trong khi hình dạng từng người đọc **đứng yên**. Hai
giả thuyết về chỗ P3 và P5 bị bác bỏ tách nhau đúng ở đây:

- **H-người**: cái thiếu là *số quan sát trên mỗi người đọc* (35,7 so với 232,6
  của AniList). Gấp đôi số người đọc không thêm một quan sát nào cho người đã
  có, nên **cả hai dự đoán vẫn bị bác bỏ, và mức gần như không đổi**.
- **H-lượng**: cái thiếu là *tổng lượng dữ liệu* để học biểu diễn. Gấp đôi thì
  khoảng cách phải **thu hẹp rõ**.

### Luật đọc, chốt bằng số, không bằng tính từ

| | mốc ở quy mô khớp | **đảo ngược** nếu | **thu hẹp rõ** nếu | **giữ nguyên** nếu |
|---|---|---|---|---|
| **P3**: hai tháp chấm toàn danh mục so với hệ hai tầng, lát `all` | 0,0625 so với 0,0757 (tỉ lệ **0,826**) | feed ≥ 0,0733 (= hai tầng ở quy mô đầy đủ) | tỉ lệ ≥ **0,913** (lấp một nửa khoảng cách), tức feed ≥ 0,0669 | tỉ lệ còn trong **0,78–0,87** |
| **P5**: cổng so với không cổng, lát `new_user` | 0,1580 so với 0,2322 (tỉ lệ **0,680**) | feed_gated ≥ feed | tỉ lệ ≥ **0,840** | tỉ lệ còn trong **0,63–0,73** |

Ba cột, không hai: nếu kết quả rơi vào khoảng giữa "thu hẹp rõ" và "giữ nguyên"
thì phải ghi là **không phân giải được**, không được kéo về phía nào. Dải
0,78–0,87 cho P3 và 0,63–0,73 cho P5 là ±5% tương đối quanh tỉ lệ cũ — rộng hơn
nhiễu seed đã đo trên nhánh AniList (±15% ở mức NDCG nhỏ thì hẹp hơn ở đây, nên
dải này là dải *thận trọng*, cố tình dễ cho H-lượng).

### Một nửa câu trả lời đã có sẵn, và nó không cần mô hình nào

Ba mốc không-học-biểu-diễn được chấm ở **cả hai** quy mô. Chúng không có tham
số để học nên không thể "được lợi vì nhiều dữ liệu hơn" theo nghĩa biểu diễn —
nhưng ALS thì **có**:

| mốc | khớp quy mô | đầy đủ | đổi |
|---|---|---|---|
| phổ biến | 0,0461 | 0,0422 | −8,5% |
| đang nổi (60 ngày) | 0,0824 | 0,0837 | +1,6% |
| **ALS** (có học, 2,15× dữ liệu) | **0,0734** | **0,0710** | **−3,3%** |

ALS gấp đôi dữ liệu và **tệ đi 3,3%**. Đây là một phép thử của H-lượng mà tôi
không phải đợi: nếu tổng lượng dữ liệu là thứ đang thiếu, mô hình có học rõ
ràng nhất trong ba mốc phải tiến lên. Nó không. (Danh mục cũng nở từ 33.527 lên
43.306 tác phẩm, nên một phần mức giảm là bài toán khó lên chứ không phải mô
hình kém đi — ghi lại để không đọc −3,3% thành một kết luận mạnh hơn nó.)

Nên trước khi có số, dự đoán của tôi là **H-người**, và nó đã có một bằng chứng
không do tôi chọn. Nếu hai tháp ở quy mô đầy đủ lại thu hẹp khoảng cách rõ
trong khi ALS không, thì chính dự đoán này sai và phải ghi là sai.

### Ghi lại một lần mất bài, vì nó đổi cách chạy

Lần chạy đầu của phép đo này chết ở epoch 12/15 của biến thể đầu sau hơn hai
giờ, không traceback, không mã lỗi của Python — tiến trình bị giết cùng phiên
làm việc. Mất sạch, kể cả bảng ba mốc đã chấm xong từ hai giờ trước, vì
`train_embedding.py` chỉ ghi kết quả **một lần ở cuối vòng lặp**.

Máy cũng đang ở trạng thái đáng ghi: 15,7 GB RAM vật lý nhưng 24,5 GB đã cam
kết, tức khoảng 9 GB nằm trên đĩa; tiến trình huấn luyện cam kết 3,75 GB mà chỉ
2,15 GB thường trú. Phần lớn thời gian mỗi epoch là lỗi trang, không phải phép
nhân ma trận. Đó là lý do không chạy song song bất kỳ việc nặng nào khác trong
lúc này, kể cả phép tinh chỉnh nhánh giao thức đang chờ.

Sửa: ghi điểm kiểm tra **sau mỗi biến thể** (`checkpoint()`), và thêm
`--out-name` để một lần chạy tập biến thể khác không ghi đè lặng lẽ lên
`embedding_temporal.json` của lần trước. Không đổi một dòng nào của phép đo.

## 14. Kết quả quy mô đầy đủ: P3 giữ nguyên phán quyết, P5 **không phân giải được** — và lỗi nằm ở luật tôi tự viết

Mục 13 không bị sửa một chữ nào sau khi có số; nó đứng nguyên để đọc được rằng
luật đã chốt trước. Mục này chỉ điền số và áp luật.

7,5 giờ cho hai biến thể ở 88.204 người đọc, seed 0, 15 epoch, cùng code và cùng
giao thức với quy mô khớp. Kết quả ghi ở
`data/processed_gr_full/embedding_temporal_p3p5.json`.

| cấu hình | quy mô | all | new_user | sparse_user | warm | phủ (`all`) |
|---|---|---|---|---|---|---|
| hệ hai tầng (đủ mặt) | khớp | 0,0757 | 0,1858 | 0,0872 | 0,0576 | — |
| hệ hai tầng (đủ mặt) | **đầy đủ** | **0,0733** | 0,1860 | 0,0891 | 0,0614 | — |
| ALS | khớp | 0,0734 | 0,1043 | 0,0838 | 0,0545 | 0,048 |
| ALS | **đầy đủ** | **0,0710** | 0,1021 | 0,0843 | 0,0571 | 0,038 |
| `feed` (toàn danh mục) | khớp | 0,0625 | 0,2322 | 0,0716 | 0,0430 | 0,121 |
| `feed` (toàn danh mục) | **đầy đủ** | **0,0586** | **0,1978** | **0,0715** | **0,0433** | 0,109 |
| `feed_gated` | khớp | 0,0564 | 0,1580 | 0,0659 | 0,0367 | 0,169 |
| `feed_gated` | **đầy đủ** | **0,0546** | **0,1452** | **0,0616** | **0,0375** | 0,132 |

### P3 — luật nói "giữ nguyên", và số rơi đúng vào dải đó

Tỉ lệ `feed` / hệ hai tầng ở lát `all`: **0,826 → 0,7995**. Dải "giữ nguyên" đã
chốt là 0,78–0,87. Số rơi vào trong, và rơi về phía **xa hơn** chứ không gần hơn.

**P3 bị bác bỏ vì dữ liệu, không vì quy mô.** Gấp 2,15× tương tác và 2,16× người
đọc không đưa phép chấm toàn danh mục lại gần hệ hai tầng; nó đẩy ra thêm 3%
tương đối. Nghĩa vụ ở mục 3 đã hoàn thành cho P3, và phán quyết ở mục 7 đứng
nguyên.

### P5 — luật nói "không phân giải được", và tôi phải ghi đúng thế

Tỉ lệ `feed_gated` / `feed` ở lát `new_user`: **0,680 → 0,7341**.

Dải "giữ nguyên" là 0,63–0,73. Ngưỡng "thu hẹp rõ" là 0,840. **0,7341 không nằm
trong dải nào cả** — nó cách mép trên của "giữ nguyên" đúng **0,004** và cách
ngưỡng "thu hẹp rõ" **0,106**.

Theo đúng luật ba cột đã chốt: **không phân giải được**. Tôi không được làm tròn
0,7341 xuống 0,73 để lấy kết luận mình đã dự đoán, dù khoảng cách là 0,004 so với
0,106 — chính vì khoảng cách nhỏ thế nên việc làm tròn mới hấp dẫn, và chính vì
hấp dẫn nên luật mới được viết trước.

### Nhưng chỗ sai thật không phải con số, mà là luật

Đọc lại luật của mục 13 sau khi thấy nó không phân giải được: **nó gán dải cho
một tỉ lệ đo từ MỘT seed.** Mục 3 cấm đúng việc đó bằng chữ của chính nó — "mọi
kết luận về chênh lệch nhỏ hơn nhiễu khởi tạo phải đi qua phép kiểm ghép cặp
theo seed, không phải một lần chạy" — và tôi đã vi phạm nó khi *viết luật*, chứ
không phải khi đọc kết quả.

Tệ hơn: dải 0,63–0,73 được tôi lấy là "±5% tương đối quanh tỉ lệ cũ", một con số
tròn trịa tự chọn, chứ **không** dẫn từ một phép đo nhiễu seed nào trên nhánh
này. Mà 0,7341 là **tỉ lệ của hai tỉ lệ**, mỗi tỉ lệ một seed, nên phương sai
của nó lớn hơn phương sai của từng số. Dải đúng phải rộng hơn 0,63–0,73, và rộng
hơn bao nhiêu thì tôi chưa đo.

Nên trạng thái thật của câu hỏi cơ chế P5 không phải "dữ liệu chưa đủ để trả
lời" mà là **"thiết kế đo tôi viết ra không trả lời được"**. Hai thứ đó khác
nhau: thứ nhất gọi thêm dữ liệu, thứ hai gọi thêm seed. Việc phải làm là
`--seeds 0 1 2 3 4` cho `feed` và `feed_gated` ở quy mô đầy đủ — 10 lần huấn
luyện, khoảng 37 giờ ở tốc độ máy hiện tại — hoặc, nếu không đủ thời gian máy,
đo nhiễu seed ở quy mô khớp rồi dùng nó để đặt dải cho quy mô đầy đủ.

### Phán quyết chính của P5 thì **có** lặp lại, và nó không cần luật kia

Dự đoán P5 là "cổng có lợi **rộng hơn** trên Goodreads". Điều kiện bác bỏ là
"cổng vẫn mất ở `new_user`". Ở quy mô đầy đủ, cổng mất ở **cả bốn lát đo được**:

| lát | khớp quy mô | đầy đủ |
|---|---|---|
| `all` | −9,8% | −6,9% |
| `new_user` | −32,0% | −26,6% |
| `sparse_user` | −8,0% | **−13,9%** |
| `warm` | −14,5% | −13,3% |

**P5 bị bác bỏ ở cả hai quy mô.** Đây là phán quyết không phụ thuộc dải nào: dấu
giống nhau ở tám ô trên tám, và mức thì xa hẳn −5% của AniList.

Một chi tiết **không** ghi trước, nên ghi là quan sát thêm chứ không phải bằng
chứng đã chốt: mức mất **không** thu hẹp đồng đều. Ở `new_user` nó đỡ đi
(−32,0% → −26,6%), nhưng ở `sparse_user` nó **xấu thêm** (−8,0% → −13,9%) —
mà `sparse_user` lại chính là lát mà lập luận của P5 dựa vào. Nếu nhiều dữ liệu
hơn đang giúp cổng một cách hệ thống thì nó phải giúp ở lát thưa. Nó không.
Quan sát này đi ngược chiều với bước 0,680 → 0,7341, và vì cả hai đều là một
seed nên nó **không** giải quyết được gì — nó chỉ cho thấy phép kiểm ghép cặp ở
trên là việc bắt buộc, không phải việc cho đẹp.

### Bằng chứng cơ chế không dựa vào luật đã vỡ

Chẻ mức đổi của `feed` khi đi từ quy mô khớp sang đầy đủ theo từng lát:

| lát | khớp | đầy đủ | đổi |
|---|---|---|---|
| `warm` (có lịch sử) | 0,0430 | 0,0433 | **+0,7%** |
| `sparse_user` | 0,0716 | 0,0715 | **−0,1%** |
| `new_user` (chưa có lịch sử) | 0,2322 | 0,1978 | −14,8% |
| `all` | 0,0625 | 0,0586 | −6,3% |

Toàn bộ mức giảm nằm ở lát **chưa có lịch sử**. Hai lát có lịch sử **đứng yên**
khi tổng dữ liệu gấp 2,15×. Đó là H-người phát biểu trực tiếp nhất mà dữ liệu
cho phép: thêm người đọc không thêm một quan sát nào cho người đã có, và chất
lượng gợi ý cho người đã có không nhúc nhích.

Hai chỗ phải ghi để không đọc bảng này mạnh hơn nó:

1. **Danh mục nở 1,29×** (33.527 → 43.306 tác phẩm), nên bài toán khó lên ở mọi
   lát. "Đứng yên trước một bài toán khó hơn" là một mức tăng nhỏ bị che, không
   phải "không có hiệu ứng" — nhưng độ lớn của mức tăng đó thì không tách được
   mà không giữ danh mục cố định, và giữ danh mục cố định thì không còn là quy mô
   đầy đủ nữa. Ghi là không tách được.
2. Mức giảm −14,8% ở `new_user` cũng phần lớn là hệ quả của danh mục nở: lát đó
   chấm trên toàn danh mục mà không có vector ID nào để dựa vào, nên nó chịu
   29% ứng viên thêm nặng nhất. Không được đọc thành "nhiều dữ liệu làm hỏng
   cold-start".

> **Đính chính, xem `docs/so-sanh-hai-huong.md` mục 6:** câu "ba mô hình có học,
> ba lần không tiến" ngay dưới đây **quá rộng**. Item-KNN là mô hình thứ tư, nó
> cũng học từ dữ liệu, và nó *tiến* **+9,0%** khi tổng dữ liệu gấp đôi. Câu đúng
> hẹp hơn: thêm người đọc làm tốt lên thống kê **phía mục** và không làm gì cho
> biểu diễn **phía người** — ba mô hình không tiến đều là mô hình có một vectơ
> cho mỗi người. Kết luận H-người đứng; lập luận thì phải hẹp lại.

Cùng chiều với nó, và độc lập với hai tháp: **ALS được gấp đôi dữ liệu và tệ đi
3,3%** (0,0734 → 0,0710), hệ hai tầng tệ đi **3,2%** (0,0757 → 0,0733). Ba mô
hình có học, ba lần không tiến. Nếu tổng lượng dữ liệu là thứ đang thiếu thì
ít nhất một trong ba phải tiến.

### Một quan sát thêm, chưa ghi trước: lợi thế độ phủ của cổng teo lại

Cổng vẫn phủ danh mục rộng hơn `feed` ở cả hai quy mô, nhưng khoảng cách hẹp đi:
0,169 so với 0,121 ở quy mô khớp (+40%), còn 0,132 so với 0,109 ở quy mô đầy đủ
(+21%). Ghi lại vì mục 10 có dùng độ phủ của cổng làm một phần lý do giữ nó
trong hệ dù mất NDCG; nếu lợi thế đó teo theo quy mô thì lý do đó yếu đi. Một
seed, chưa kết luận.
