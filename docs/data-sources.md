# Nguồn dữ liệu: kiểm chứng, bẫy và pipeline xử lý

Tài liệu này kiểm lại bộ dữ liệu trong slide đề xuất, chỉ ra những chỗ phải sửa cách hiểu, và đặc tả các bước xử lý. Chưa chốt — mục 9 là các phép kiểm phải chạy trước khi chốt.

Nguồn đối chiếu: [trang dataset chính thức của McAuley Lab](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html), [repo code chính thức](https://github.com/MengtingWan/goodreads), [bản mirror schema trên HuggingFace](https://huggingface.co/datasets/liyucheng/goodreads).

## 0. Quyết định đã chốt

**AniList là nguồn chính, Goodreads Comics & Graphic làm đối chứng.** Môn học không bắt buộc dùng benchmark đã công bố.

Việc tiếp theo, theo thứ tự:

```
python scripts/crawl_anilist.py discover
python scripts/crawl_anilist.py users --target 1000 --weights "KR=0.35,CN=0.35,JP=0.30"
python scripts/crawl_anilist.py lists      # chạy thử ~35 phút, kiểm pipeline
python scripts/crawl_anilist.py media
```

Nếu mẻ 1.000 user trông ổn thì nâng `--target 10000` và chạy lại — crawler resume được, không làm lại phần đã xong. Lần chạy đầy đủ mất khoảng 6 tiếng, nên chạy qua đêm.

Goodreads chỉ đụng tới **sau khi** AniList xong, để so sánh trong báo cáo. Mục 2–9 dưới đây là đặc tả cho phần đối chứng đó, và nhiều bẫy trong đó (chuỗi hành vi, gộp bản in, chia tập theo thời gian) **cũng áp dụng cho AniList**.

## 1. Vì sao đổi nguồn

Mọi con số trong slide đều chính xác (mục 2). Nhưng khi yêu cầu là **manga + manhwa + manhua**, Goodreads Comics & Graphic không đáp ứng được, và không phải vì xử lý sai mà vì **catalog của nó không chứa phần lớn thứ cần gợi ý**.

Lý do gốc: Goodreads lập chỉ mục **sách in có ISBN**, và bộ dữ liệu đóng băng ở **cuối 2017**. Manhwa hiện đại (Solo Leveling, Tower of God, Omniscient Reader) chỉ có bản in tiếng Anh **từ khoảng 2021 trở đi**, tức là *sau* mốc thu thập. Thứ còn lại chỉ là manhwa in giấy thời Tokyopop những năm 2000. Manhua in tiếng Anh năm 2017 gần như không tồn tại. Nghiêm trọng hơn: **Goodreads không có trường nào cho biết truyện đến từ nước nào**, nên ngay cả việc tách ba loại cũng không làm được một cách đáng tin.

Đã kiểm trực tiếp AniList API ngày 2026-11-06 (truy vấn thật, không phải đọc tài liệu):

| Tiêu chí | Goodreads UCSD (2017) | AniList (đã kiểm) |
|---|---|---|
| Tách manga / manhwa / manhua | **Không có trường nào** | `countryOfOrigin` = JP / KR / CN — Solo Leveling trả về `KR` |
| Manhwa hiện đại | Gần như không có (in giấy sau mốc 2017) | Đầy đủ |
| Manhua | Không đáng kể | Có |
| Dữ liệu tương tác | **7.347.630, có sẵn ngay** | **Phải tự thu**, 30 request/phút |
| Chất lượng điểm số | `rating = 0` (chưa chấm) chiếm phần lớn | Mẫu thử: 41/52 mục có điểm |
| Trạng thái đọc | Nhị phân `is_read` | CURRENT / COMPLETED / **DROPPED** / PAUSED / PLANNING |
| **Tiến độ theo chương** | **Không có** | `progress` — đo được "bỏ ở chương 92" |
| Ảnh bìa | Thumbnail, phải đo mới biết dùng được không | `extraLarge` = **460×643**, đủ lớn |
| Tách tác giả / họa sĩ | `role` = "Illustrator", không đều | `Art (chs 1-179)`, `Story (chs 93-201)` — tách rõ, có cả khoảng chương |
| Tag | `popular_shelves` lẫn shelf hành chính, cần stoplist | `tags` có rank %, sạch |
| Số chương / tập / thời gian | Không có số chương | `chapters`, `volumes`, `startDate`, `endDate`, `status` |
| Tính trích dẫn học thuật | Dataset đã xuất bản, có 2 bài để cite | Tự thu, **không citable** |

AniList thắng ở **mọi tiêu chí trừ một**: dữ liệu tương tác không có sẵn mà phải tự thu. Chi phí của việc đó đã đo được và **nhỏ hơn nhiều so với tưởng tượng** — xem mục 10.

Đáng chú ý: AniList làm sống lại ba thứ mà Goodreads giết chết — **drop-off theo chương** (ý tưởng nhịp truyện), **ảnh bìa đủ độ phân giải** (nhánh nét vẽ), và **tách vai Story/Art** (ý tưởng tác giả ↔ nhịp, họa sĩ ↔ nét vẽ). Cả ba đều là trụ của thiết kế trong [content-features.md](content-features.md).

Phần còn lại của tài liệu (mục 2–9) vẫn giữ nguyên giá trị: nếu chọn Goodreads thì đó là những gì phải xử lý, và nhiều bẫy trong đó (chuỗi hành vi, gộp bản in, chia theo thời gian) **cũng áp dụng cho AniList**.

## 2. Xác minh slide

Đã đối chiếu với trang chính thức. Toàn bộ khớp:

| Nội dung slide | Kiểm chứng |
|---|---|
| McAuley Lab (UC San Diego) | Đúng |
| Thu thập cuối 2017, từ kệ sách công khai, ID ẩn danh | Đúng, nguyên văn "collected in late 2017 from only users' public shelves", ID người dùng và review đã ẩn danh |
| 89.411 đầu truyện | Đúng |
| 7.347.630 tương tác | Đúng |
| 542.338 review | Đúng |
| Trường tương tác: user, truyện, rating, đã đọc hay chưa, thời điểm thêm và đọc xong | Đúng về tên trường, **sai về ý nghĩa `rating`** — xem bẫy 1 |

Ba điều slide **chưa nói mà ảnh hưởng trực tiếp tới thiết kế**:

1. **Nhãn thể loại là "fuzzy".** Trang gốc ghi rõ: nhãn genre được rút từ popular shelves của người dùng bằng **so khớp từ khóa đơn giản**. Tức là tập "Comics & Graphic" không phải do người biên tập chọn, mà là kết quả keyword matching — có nhiễu cả hai chiều. Phải lọc, và phải nói điều này trong báo cáo.
2. **Đồ thị con theo genre không tự đủ** ("may not be self-contained"). Thông tin tác giả, work và series **không nằm trong** ba file comics, mà nằm ở các file metadata của **toàn bộ** dataset. Muốn có tác giả/series thì phải tải thêm file lớn rồi join.
3. **Chỉ được dùng cho mục đích học thuật**, không được phát tán lại, không dùng thương mại. Hệ quả: **không commit dữ liệu vào git** (`.gitignore` đã chặn `data/`), và báo cáo **bắt buộc trích dẫn hai bài**: Wan & McAuley, *Item Recommendation on Monotonic Behavior Chains*, RecSys'18; và Wan, Misra, Nakashole & McAuley, *Fine-Grained Spoiler Detection from Large-Scale Review Corpora*, ACL'19.

## 3. Ba chỗ phải sửa cách hiểu

### 3.1 "Comics & Graphic" trộn lẫn, và thiếu đúng thứ cần

Thể loại này trên Goodreads trộn: graphic novel phương Tây, truyện siêu anh hùng Marvel/DC, manga bản dịch tiếng Anh, truyện tranh Pháp-Bỉ, tuyển tập webcomic, truyện tranh báo thiếu nhi.

Vấn đề không nằm ở chỗ trộn — mà ở chỗ **không có cách tách ra**. Goodreads không có trường quốc gia xuất xứ. Muốn lọc lấy truyện châu Á phải suy đoán gián tiếp qua nhà xuất bản (Viz, Kodansha, Yen Press, Seven Seas, Tokyopop), qua `popular_shelves` chứa "manga", hoặc `language_code` — tất cả đều là heuristic có lỗi, và **không cái nào tách được manhwa khỏi manhua khỏi manga**.

Cộng với việc manhwa/manhua gần như vắng mặt vì mốc thời gian 2017 (mục 1), kết luận là: nếu đề tài cần cả ba loại thì **Goodreads không phải nguồn phù hợp**, và không có cách xử lý nào cứu được điều đó.

Nếu vẫn chọn Goodreads thì phải chấp nhận đề tài trên thực tế là "hệ gợi ý truyện tranh nói chung, chủ yếu manga và comics phương Tây" và nói đúng như vậy trong báo cáo.

### 3.2 Item ở đây là "bản in", không phải series, càng không phải chương

Một `book_id` trên Goodreads là **một ấn bản cụ thể**: bìa cứng, bìa mềm, bản omnibus, bản dịch từng thứ tiếng đều là các `book_id` khác nhau của cùng một tác phẩm. Trường `work_id` là thứ gom chúng lại.

Hai hệ quả, cả hai đều nghiêm trọng:

- **Bắt buộc gộp theo `work_id`** trước khi làm bất cứ việc gì. Nếu không, cùng một tập truyện bị tách thành 5 item, tương tác bị chia nhỏ, ma trận thưa đi một cách giả tạo, và CF kém đi mà không ai biết tại sao. 89.411 `book_id` chắc chắn ứng với **ít hơn nhiều** số tác phẩm thật.
- **Không có dữ liệu cấp chương.** Goodreads ghi nhận ở mức tập/quyển. Ý tưởng "người đọc rơi ở chương 3 hay chương 80" trong content-features.md **không thực hiện được**. Thứ thay thế được: **rơi ở tập mấy trong series** — xem mục 7.

### 3.3 Chỉ có ảnh bìa, không có trang truyện

Metadata có `image_url` (ảnh bìa). Không có trang truyện bên trong. Nghĩa là toàn bộ nhóm chỉ số nhịp theo panel (số panel/trang, splash page, mật độ thoại) **không có đầu vào**. Đây là mất mát thật và phải chấp nhận, không nên hứa trong báo cáo.

## 4. Các file cần tải và khóa join

| File | Phạm vi | Dùng để |
|---|---|---|
| `goodreads_books_comics_graphic.json.gz` | Comics | Metadata item: mô tả, tác giả, series, shelves, ảnh bìa |
| `goodreads_interactions_comics_graphic.json.gz` | Comics | Ma trận tương tác, có timestamp |
| `goodreads_reviews_comics_graphic.json.gz` | Comics | Text review, cho nhánh nhịp từ review |
| `goodreads_book_authors.json.gz` | **Toàn bộ** | Tên tác giả theo `author_id` |
| `goodreads_book_works.json.gz` | **Toàn bộ** | Gom ấn bản theo `work_id` |
| `goodreads_book_series.json.gz` | **Toàn bộ** | Thông tin series |
| `goodreads_interactions.csv` + `book_id_map.csv` + `user_id_map.csv` | Toàn bộ, ~4.1GB | Tuỳ chọn: ma trận CF dạng số gọn nhẹ |

Khóa join: `book_id` → `work_id` → `series_id`; `authors[].author_id`; `user_id`.

**Mẹo quan trọng về hiệu năng:** bản `goodreads_interactions.csv` dùng ID số nguyên đã mã hóa lại, nhẹ và nhanh hơn nhiều so với parse JSON. Nếu chỉ cần ma trận CF, dùng file CSV này rồi lọc về comics qua `book_id_map.csv`, thay vì đọc file JSON lớn. Nhưng CSV này **không có timestamp**, nên nhánh nhịp-theo-thời-gian vẫn phải đọc file JSON.

### Schema đã xác minh (file tương tác)

Xác nhận qua bản mirror: `user_id` (chuỗi hex 32), `book_id` (chuỗi số), `is_read` (bool), `rating` (int 0–5), `date_added` và `date_updated` (chuỗi `YYYY-MM-DD HH:MM:SS`), `read_at` (nullable), `started_at` (nullable). File review có thêm `review_text`, `n_votes`, `n_comments`, và `review_id`.

### Schema file metadata — **chưa xác minh, phải tự kiểm**

Trang chính thức không in schema. Các trường *dự kiến* có: `book_id`, `work_id`, `title`, `title_without_series`, `description`, `authors` (danh sách, mỗi phần tử có `author_id` và `role`), `series`, `popular_shelves` (danh sách `{count, name}`), `similar_books`, `image_url`, `isbn`, `isbn13`, `asin`, `num_pages`, `publication_year`, `publisher`, `language_code`, `country_code`, `format`, `is_ebook`, `average_rating`, `ratings_count`, `text_reviews_count`.

Đây là **dự kiến, không phải đã kiểm**. Bước đầu tiên của mục 9 là in ra key thật của bản ghi đầu tiên.

## 5. Mười một bẫy phải xử lý

**Bẫy 1 — `rating = 0` nghĩa là CHƯA CHẤM ĐIỂM, không phải điểm 0.** Slide viết "rating 0–5" như thể là thang 6 mức. Không phải. Dữ liệu mẫu cho thấy rất nhiều bản ghi `is_read = true` mà `rating = 0`: người đọc đã đọc nhưng không chấm sao. Nếu đưa 0 vào model dự đoán rating như một điểm thấp thì **toàn bộ kết quả sai mà vẫn chạy trơn tru** — loại lỗi nguy hiểm nhất. Xử lý: tách `rating = 0` ra khỏi dữ liệu explicit, hoặc coi toàn bài toán là implicit feedback.

**Bẫy 2 — chuỗi hành vi đơn điệu: thêm kệ → đọc → chấm điểm.** Đây chính là nội dung bài RecSys'18 phải trích dẫn. Dữ liệu có ba mức tín hiệu khác hẳn nhau:

| `is_read` | `rating` | Nghĩa thật | Dùng thế nào |
|---|---|---|---|
| false | 0 | Mới cho vào kệ, chưa đọc | Tín hiệu *quan tâm*, yếu, **không phải đã đọc** |
| true | 0 | Đã đọc, không chấm | Implicit dương, mạnh |
| true | 1–5 | Đã đọc và chấm điểm | Explicit |

Rất nhiều bài làm cẩu thả gộp cả ba thành "tương tác". Tách ba mức này ra là việc rẻ và là một điểm cộng rõ khi bảo vệ.

**Bẫy 3 — không gộp `work_id` thì hỏng ngầm.** Đã nói ở 3.2. Phải gộp trước khi chia train/test, nếu không cùng một tác phẩm nằm cả hai bên và gây **rò rỉ dữ liệu**.

**Bẫy 4 — series làm gợi ý trở nên tầm thường.** Truyện tranh đi theo bộ. Gợi ý "tập 4" cho người vừa đọc tập 1–3 thì đúng nhưng vô dụng, và nó sẽ **chiếm hết các vị trí đầu của mọi danh sách**, làm mọi metric đẹp lên một cách giả tạo. Phải quyết định sớm:

- **Gợi ý ở cấp series** (khuyến nghị): gộp mọi tập về một node series, bài toán thành "gợi ý bộ truyện mới", đúng với nhu cầu thật của người đọc.
- Gợi ý cấp tập nhưng loại bỏ các tập cùng series với thứ người dùng đã đọc.

Đây là quyết định mô hình **quan trọng nhất mà bộ dữ liệu này ép phải đưa ra**, ngang với bẫy 3.

**Bẫy 5 — ảnh bìa có ba vấn đề, và nó quyết định nhánh nét vẽ sống hay chết.**

- *Ảnh placeholder*: Goodreads trả về ảnh "nophoto" xám cho sách không có bìa. Hàng trăm item sẽ có **cùng một vector nét vẽ giống hệt nhau**, tạo ra một cụm giả to đùng. Phải phát hiện theo URL và theo hash ảnh rồi loại.
- *Link chết*: URL từ 2017, CDN đã đổi nhiều. Phải đo tỉ lệ tải được **trước** khi cam kết.
- *Độ phân giải* — vấn đề nặng nhất: URL ảnh Goodreads thường là **thumbnail nhỏ**. Gram matrix cần ảnh đủ lớn để thống kê nét/texture có nghĩa; một ảnh rộng 98px thì **không đo được độ dày nét hay screentone**, vì chính thông tin đó đã bị resize xóa mất. Phải đo kích thước thật của ảnh tải về. Nếu phần lớn dưới ~300px chiều rộng thì nhánh nét vẽ phải đổi cách làm (thử sửa tham số kích thước trong URL để lấy bản lớn hơn) hoặc bị cắt khỏi phạm vi.

**Không được giả định nhánh nét vẽ khả thi trước khi có ba con số này.**

**Bẫy 6 — ngày tháng do người dùng tự nhập, rác rất nhiều.** `started_at` null ở hầu hết bản ghi mẫu; `read_at` thưa. Bản mirror còn cho thấy trường thời lượng đọc có giá trị **âm tới -367.531 ngày**, tức là có bản ghi `read_at` sớm hơn `started_at`. Người dùng điền ngày tuỳ tiện khi nhập lại sách cũ. Mọi phân tích theo thời gian phải lọc mạnh: bỏ thời lượng âm, bỏ ngày trước khi Goodreads tồn tại (2007), bỏ ngày sau thời điểm thu thập (cuối 2017). **Dùng `date_added` làm trục thời gian chính** vì nó do hệ thống ghi, đáng tin hơn `read_at` do người dùng khai.

**Bẫy 7 — `popular_shelves` cần stoplist, nhưng chứa vàng.** Các shelf đầu bảng luôn là shelf hành chính: `to-read`, `currently-reading`, `owned`, `default`, `books-i-own`, `favorites`. Phải loại bằng stoplist. Nhưng sau khi lọc, còn lại là tag nội dung do người đọc tự đặt, và đặc biệt **`dnf`, `abandoned`, `couldnt-finish` là tín hiệu nhịp trực tiếp**: tỉ lệ người bỏ dở trên tổng số người xếp kệ chính là thước đo "truyện có giữ chân được không". Đây là thứ thay thế rẻ và tốt cho phần drop-off cấp chương đã mất.

**Bẫy 8 — `similar_books` là cám dỗ, không phải ground truth.** Trường này là gợi ý của chính Goodreads. Dùng nó làm nhãn đúng để đánh giá là **lập luận vòng tròn** (đi đo xem ta có giống hệ gợi ý sẵn có không). Cách dùng hợp lệ: làm **baseline để vượt qua**, hoặc làm nhãn yếu để huấn luyện. Phải nói rõ trong báo cáo là đã không dùng nó làm nhãn đánh giá.

**Bẫy 9 — nhãn thể loại fuzzy.** Tập comics có nhiễu vì được lọc bằng keyword. Nên kiểm thủ công một mẫu 100 item xem bao nhiêu phần trăm thật sự là truyện tranh, rồi báo con số đó trong báo cáo. Rẻ, và biến một điểm yếu thành một phép đo.

**Bẫy 10 — bộ nhớ và định dạng.** File JSON-lines nén gzip. `pandas.read_json` nạp thẳng sẽ nổ RAM. Phải đọc **từng dòng theo chunk**, lọc lấy các trường cần, rồi **ghi ra Parquet ngay từ bước đầu**. Mọi bước sau làm trên Parquet. Bước này làm đúng một lần tiết kiệm cả dự án.

**Bẫy 11 — chia train/test phải theo thời gian, không được random.** Có `date_added` nên chia được theo mốc thời gian (ví dụ tương tác trước 2017-06 là train, sau là test). Chia ngẫu nhiên sẽ cho model nhìn thấy tương lai và **thổi phồng mọi chỉ số**, đồng thời làm lát "người dùng mới"/"truyện mới" trở nên vô nghĩa — mà đó chính là lát quan trọng nhất của đề tài này. Chia theo thời gian còn cho phép đánh giá cold-start một cách trung thực: người dùng xuất hiện lần đầu sau mốc cắt chính là người dùng mới thật.

## 6. Pipeline xử lý

| Bước | Việc | Đầu ra |
|---|---|---|
| B0 | Tải 6 file, kiểm checksum, in key bản ghi đầu mỗi file | Xác nhận schema thật |
| B1 | Parse JSON-lines theo chunk → Parquet, chỉ giữ trường cần | `data/interim/*.parquet` |
| B2 | Gộp ấn bản theo `work_id`; gắn `series_id`; quyết định đơn vị item (series hay work) | Bảng item chuẩn |
| B3 | Tách tương tác thành 3 mức (bẫy 2); lọc ngày rác (bẫy 6) | Bảng tương tác sạch |
| B4 | Lọc nhiễu thể loại; lọc user/item quá thưa (ví dụ bỏ user dưới 5 tương tác) | Tập lõi |
| B5 | Trích đặc trưng nội dung: `popular_shelves` sau stoplist, embedding `description`, tác giả | `v_theme`, `v_creator` |
| B6 | Ảnh bìa: tải, loại placeholder, đo kích thước, rút `v_art` | `v_art` hoặc kết luận cắt nhánh |
| B7 | Chia train/test **theo thời gian**, tách sẵn các lát đánh giá (user mới, item mới, user thưa) | Tập đánh giá |

## 7. Ảnh hưởng ngược lên thiết kế trong content-features.md

Bộ dữ liệu này buộc phải sửa vài chỗ trong tài liệu trước. Ghi lại cho trung thực:

| Ý tưởng cũ | Còn dùng được? | Ghi chú / thay thế |
|---|---|---|
| `v_theme` từ genre, tag, synopsis | **Có** | `popular_shelves` sau stoplist là nguồn tag rất giàu, cộng `description` |
| `v_creator` tác giả/họa sĩ | **Một phần** | Goodreads có `authors[].role`, đôi khi ghi "Illustrator". Phải **đo tỉ lệ có vai trò riêng** trước khi tin vào ý "tác giả ↔ nhịp, họa sĩ ↔ nét vẽ" |
| `v_art` từ ảnh bìa | **Chưa biết** | Phụ thuộc hoàn toàn vào bẫy 5. Phải đo trước |
| Nhịp từ cấu trúc serialize | **Có, nhưng khác cách** | Không có số chương. Thay bằng: số tập trong series, khoảng thời gian xuất bản, `num_pages`, series đã kết thúc chưa |
| Nhịp từ hành vi, drop-off cấp chương | **Không** | Không có dữ liệu chương. **Thay bằng: drop-off cấp tập trong series** — người dùng đọc tập 1,2,3 rồi dừng ở tập 4. Tính được và là tín hiệu nhịp thật |
| Nhịp từ panel trên trang truyện | **Không** | Không có ảnh trang trong. Cắt khỏi phạm vi, hoặc tách thành nghiên cứu riêng trên bộ khác |
| Nhịp từ review | **Có, mạnh** | 542.338 review là nguồn rất tốt. Cộng thêm tỉ lệ shelf `dnf` |
| Onboarding bằng lưới ảnh bìa | **Có, nếu bẫy 5 qua được** | |
| Trọng số theo mặt cho từng người dùng | **Có** | Không phụ thuộc dữ liệu ảnh |

Tức là **ba trong bốn nguồn nhịp vẫn sống**, và nguồn mới (drop-off cấp tập, tỉ lệ DNF) thậm chí trực tiếp hơn. Nhánh rủi ro duy nhất là nét vẽ, và rủi ro đó đo được trong một buổi.

**Nếu đổi sang AniList thì bảng trên không còn mất mát nào**: `progress` cho drop-off *cấp chương* (không phải cấp tập), `extraLarge` 460×643 cho nét vẽ, `chapters`/`volumes`/`startDate`/`endDate` cho cấu trúc serialize, vai `Story`/`Art` tách rõ cho `v_creator`. Chỉ riêng nhịp từ panel là vẫn không có, vì không nguồn nào cung cấp ảnh trang truyện hợp pháp ở quy mô lớn.

## 8. Có nên hợp nhất thêm nguồn khác không

Khuyến nghị: **không, ở giai đoạn đầu.** "Hợp nhất" trong dự án này nên hiểu là join 6 file *bên trong* chính bộ Goodreads (mục 4), và riêng việc đó đã là khối lượng thật.

Về các nguồn ngoài đã bàn trước đây:

- **AniList**: nay không còn là "nguồn bổ sung" mà là **ứng viên thay thế hoàn toàn** (mục 1 và 10). Nếu chọn AniList thì **không join với Goodreads** — ghép hai nguồn phải so khớp mờ theo tên (AniList không dùng ISBN), vừa đắt vừa sinh lỗi, trong khi AniList đã có đủ mọi trường cần. Chọn một, không chọn cả hai.
- **MyAnimeList qua Kaggle**: đã kiểm, **không dùng được.** Các bộ dữ liệu MAL lớn trên Kaggle gần như toàn bộ là **anime**, không phải manga. Các bộ manga chỉ có **điểm trung bình cộng đồng, không có đánh giá theo từng người dùng**, nên không dựng được ma trận tương tác.
- **Manga109**: **không join được với Goodreads.** Manga109 gồm các bộ manga Nhật cũ, phần lớn không có bản tiếng Anh trên Goodreads, nên giao nhau gần như bằng không. Đây là sửa lại khuyến nghị trước đó của tôi: Manga109 chỉ dùng được như một **nghiên cứu riêng biệt** về nhịp panel, không phải nguồn bổ sung cho hệ gợi ý chính.

Mỗi nguồn ghép thêm là một tầng lỗi khớp mới. Với thời lượng một đồ án môn học, một nguồn sạch tốt hơn ba nguồn chắp vá.

## 9. Việc phải làm trước khi chốt

Chạy `scripts/check_data.py` (xem file) và điền bảng này. Mỗi dòng có một ngưỡng báo động:

| Câu hỏi | Cách đo | Ngưỡng báo động |
|---|---|---|
| Schema thật là gì | In key của bản ghi đầu mỗi file | Khác dự kiến ở mục 4 |
| `rating = 0` chiếm bao nhiêu | Phân bố rating | Nếu > 40% thì bắt buộc chuyển sang implicit |
| Ba mức hành vi phân bố ra sao | Bảng chéo `is_read` × (`rating` = 0) | `is_read = false` chiếm đa số → phần lớn là "muốn đọc", không phải đã đọc |
| Gộp `work_id` giảm bao nhiêu item | Số `book_id` so với số `work_id` | Giảm > 30% thì bước gộp là bắt buộc tuyệt đối |
| Bao nhiêu % tương tác có `read_at` hợp lệ | Tỉ lệ null + tỉ lệ ngày vô lý | `read_at` dùng được < 20% → bỏ, chỉ dùng `date_added` |
| **Ảnh bìa tải được bao nhiêu %** | Thử 300 URL ngẫu nhiên, ghi HTTP status | < 70% → nhánh nét vẽ lung lay |
| **Ảnh bìa rộng bao nhiêu pixel** | Đo kích thước thật ảnh tải về | Trung vị < 300px → **Gram matrix không dùng được**, phải đổi cách |
| Bao nhiêu % là ảnh placeholder | Hash ảnh, đếm nhóm trùng lớn nhất | > 15% cần bước lọc riêng |
| Tỉ lệ item có `role` = Illustrator | Đếm trong `authors` | < 20% → ý "tác giả ↔ nhịp, họa sĩ ↔ nét vẽ" không dựa được vào Goodreads |
| Tỉ lệ item thuộc series | Đếm `series` khác rỗng | Cao → quyết định ở bẫy 4 càng cấp thiết |
| Độ thưa ma trận | #tương tác / (#user × #item) | Cho biết CF có khả thi không |
| 50 shelf phổ biến nhất | Đếm `popular_shelves` | Để dựng stoplist ở bẫy 7 |

Ba con số in đậm quyết định **nhánh nét vẽ có tồn tại không**, mà đó là ý tưởng khác biệt nhất của đề tài. Nên chạy chúng trước, trong tuần này, trước khi viết thêm bất kỳ dòng model nào.

## 10. Phương án AniList: đã kiểm bằng truy vấn thật

Mọi con số dưới đây lấy từ truy vấn trực tiếp `https://graphql.anilist.co` ngày 2026-11-06, không phải từ tài liệu.

### 10.1 Những gì đã xác nhận

- **Không cần API key** cho truy vấn đọc. Giới hạn: header trả về `X-RateLimit-Limit: 30` (30 request mỗi phút).
- **Tách ba loại**: `countryOfOrigin` trả về `JP` / `KR` / `CN`. Solo Leveling → `KR`, đúng là manhwa.
- **Một request lấy được toàn bộ danh sách của một người dùng.** Truy vấn `MediaListCollection(userName, type: MANGA)` trả về 52 mục cho một user thử, mỗi mục có `status`, `score`, `progress`, `startedAt`, `completedAt`, cộng metadata truyện.
- **Trạng thái đọc 5 mức**: CURRENT / COMPLETED / DROPPED / PAUSED / PLANNING. Giàu hơn hẳn `is_read` nhị phân của Goodreads.
- **Tiến độ theo chương**: trường `progress`. Mẫu thử có mục DROPPED ở chương 92 — tức là **đo được người đọc bỏ ở chương nào**, đúng thứ Goodreads không có.
- **Tỉ lệ chấm điểm cao**: 41/52 mục có điểm, so với Goodreads nơi `rating = 0` chiếm phần lớn.
- **Ảnh bìa `extraLarge` = 460×643 px** (91 KB). Đủ lớn cho đặc trưng nét vẽ — giải quyết rủi ro lớn nhất ở bẫy 5.
- **Vai trò staff tách rõ**: Solo Leveling trả về `Art (chs 1-179)`, `Story (chs 1-92)`, `Story (chs 93-201)`, `Original Story`. Không chỉ tách Story/Art mà còn **ghi cả khoảng chương khi người sáng tác đổi giữa chừng**.
- **Tag có trọng số**: `tags` kèm `rank` phần trăm, ví dụ `Dungeon 95`, `Full Color 89`. Trong đó `Full Color` là tag *phong cách*, dùng được cho `v_art`.
- **Lấy được username**: `Page.activities(type: MANGA_LIST)` trả về username kèm hoạt động đọc theo chương.

### 10.2 Chi phí thu thập — đo thật, không ước lượng

Đã chạy thử trọn vẹn 4 giai đoạn bằng `scripts/crawl_anilist.py`. Kết quả đo trên 9 user lấy mẫu có chủ đích:

- **178,2 mục/user** (cao hơn nhiều so với ước lượng ban đầu 52).
- 9 user → 1.604 tương tác → 1.132 truyện riêng biệt.
- Giai đoạn `users` rất rẻ: **9 request thu được 315 username**.

Quy đổi cho lần chạy thật:

| Mục tiêu | Thời gian | Tương tác ước tính |
|---|---|---|
| 1.000 user (chạy thử, kiểm pipeline) | ~35 phút | ~180 nghìn |
| **10.000 user (khuyến nghị)** | **~6 giờ, chạy qua đêm** | **~1,8 triệu** |
| 30.000 user | ~17,5 giờ | ~5,3 triệu |

Giai đoạn `discover` mất ~2 phút, `users` ~5 phút, `media` ~20–40 phút. Phần lâu là `lists`.

**10.000 user là quá đủ.** Không cần đuổi theo con số 7,3 triệu của Goodreads.

### 10.3 Mật độ ba loại truyện — đã đo

Lấy mẫu ngẫu nhiên sẽ cho manhwa chỉ vài phần trăm. Cách giải là **lấy mẫu có chủ đích**:

1. `Page.media(countryOfOrigin: KR, sort: POPULARITY_DESC)` → ID manhwa phổ biến (AniList báo tổng 5.000 đầu manhwa). Lặp với `CN` cho manhua.
2. `Page.activities(mediaId_in: [...], type: MANGA_LIST)` → username của **người thật sự đọc manhwa/manhua**. Đã kiểm: lọc theo ID manhwa trả về **100% activity trên truyện KR**.
3. Lấy toàn bộ danh sách của những user đó.

Kết quả đo được, với quota user chia đều ba nước:

| Xuất xứ truyện được đọc | Tỉ lệ |
|---|---|
| JP (manga) | 66,4% |
| KR (manhwa) | **29,6%** |
| CN (manhua) | **4,0%** |

Manhwa đạt gần 30%, thừa sức dựng mô hình. **Manhua chỉ 4% là mỏng** — catalog manhua trên AniList nhỏ hơn và ít người đọc hơn thật. Cách bù: tăng quota user nguồn CN khi chạy giai đoạn `users`:

```
python scripts/crawl_anilist.py users --target 10000 --weights "KR=0.35,CN=0.35,JP=0.30"
```

Lưu ý: người đọc manhua vẫn đọc chủ yếu manga, nên nâng quota CN chỉ đẩy manhua lên được một mức nhất định. Nếu sau khi chạy thật manhua vẫn dưới ~8%, nên nói thẳng trong báo cáo rằng mô hình yếu ở nhóm này thay vì giả vờ phủ đều cả ba.

### 10.4 Chất lượng dữ liệu — đã đo trên mẫu thử

**Độ đầy đủ metadata** (trên 1.132 truyện):

| Trường | Độ phủ | Dùng cho |
|---|---|---|
| `coverImage.extraLarge` | **100%** | `v_art` — nét vẽ |
| `description` | 99% | `v_theme` |
| `tags` (có rank %) | 99% | `v_theme` |
| staff vai **Story** | 100% | `v_creator` — trục nhịp kể |
| staff vai **Art** | 97% | `v_creator` — trục nét vẽ |
| **tách riêng cả Story và Art** | **97%** | Ý tưởng "tác giả ↔ nhịp, họa sĩ ↔ nét vẽ" |
| `chapters` | 61% | `v_pace` (thiếu ở truyện đang ra, hợp lý) |

Con số 97% tách riêng Story/Art là điểm quyết định: trên Goodreads đây là chuyện may rủi, trên AniList nó gần như luôn có.

**Phân bố trạng thái đọc:**

| Trạng thái | Tỉ lệ | Ý nghĩa |
|---|---|---|
| CURRENT | 37,2% | Đang đọc — implicit dương mạnh |
| PLANNING | 29,9% | **Mới định đọc — KHÔNG phải đã đọc** |
| COMPLETED | 27,8% | Đã đọc xong |
| PAUSED | 3,7% | Tạm dừng — tín hiệu nhịp yếu |
| DROPPED | 1,4% | **Bỏ dở — tín hiệu nhịp mạnh nhất** |

Hai điều rút ra:

- **PLANNING chính là bẫy 2 ở dạng khác.** 29,9% số mục là "muốn đọc", không phải đã đọc. Gộp nó vào tương tác dương là lặp đúng sai lầm của `is_read = false` bên Goodreads.
- **DROPPED chỉ 1,4% nhưng sạch**: 21/22 mục DROPPED có `progress`, tức là biết chính xác bỏ ở chương nào (ví dụ đo được: bỏ ở chương 61/109 của *The Second Coming of Gluttony*). Với 1,8 triệu tương tác thì 1,4% là **khoảng 25.000 sự kiện bỏ dở có vị trí chương** — thừa để mô hình hoá nhịp.

**Tỉ lệ chấm điểm: 33,7%.** Thấp hơn con số 79% ở mẫu một user trước đó, vì PLANNING không bao giờ được chấm. Với 1,8 triệu tương tác thì vẫn còn **khoảng 600.000 điểm số tường minh**. Dùng `score(format: POINT_100)` để chuẩn hoá, vì mỗi người dùng AniList có thể đặt thang điểm riêng (10 sao, 100 điểm, 5 sao, mặt cười) và **điểm thô giữa các user không so sánh được với nhau**.

### 10.5 Rủi ro và điều kiện phải chấp nhận

- **Không citable.** Goodreads UCSD là dataset đã xuất bản, có hai bài báo để trích dẫn; dữ liệu tự thu thì không. Nhóm đã xác nhận môn học **không bắt buộc** dùng benchmark công bố, nên điểm này không còn chặn — nhưng vẫn nên mô tả kỹ quy trình thu thập trong báo cáo để người đọc tái lập được.
- **Lệch mẫu**: lấy username từ activity feed thiên về người dùng đang hoạt động. Là hạn chế thật, phải ghi trong báo cáo, nhưng không làm hỏng nghiên cứu.
- **Đạo đức và điều khoản**: API công khai, đọc không cần key, dành cho ứng dụng bên thứ ba. Phải tôn trọng 30 request/phút, khai báo User-Agent, **ẩn danh username** trước khi lưu, và không phát tán lại dữ liệu người dùng thô.
- **Thời gian**: cần viết crawler và chạy qua đêm. Nếu crawl hỏng thì mất vài ngày. Giảm rủi ro: chạy thử 1.000 user trong một giờ để kiểm toàn bộ pipeline trước khi chạy lớn.
- **Mốc dữ liệu mới**, không đóng băng 2017 — vừa là ưu điểm (có manhwa hiện đại) vừa nghĩa là kết quả không so sánh trực tiếp được với các bài dùng Goodreads.
