# So kết quả hai hướng

Hai nhánh của đồ án này được dựng để **không** so thẳng với nhau:

- `docs/goodreads-arm.md` đổi **dữ liệu**, giữ nguyên giao thức → "phương pháp
  có lặp lại không?"
- `docs/protocol-arm.md` giữ nguyên **dữ liệu**, đổi giao thức theo đề xuất đồ
  án → "đổi cách đo thì con số đổi bao nhiêu?"

Trộn hai bảng vào một là cách chắc chắn nhất để đọc sai cả hai. Tài liệu này so
chúng theo cách so được, và nói rõ chỗ nào **không** so được.

## 1. Vì sao con số thô không so thẳng được

Session-KNN ở nhánh giao thức được 0,1273; hai tháp ở nhánh dữ liệu được 0,0625.
Hai số đó **không** nói rằng mô hình thứ nhất tốt gấp đôi, vì năm thứ khác nhau
cùng lúc:

| | nhánh dữ liệu (A) | nhánh giao thức (B) |
|---|---|---|
| dương | mọi tương tác | `rating >= 4` |
| k-core | 5 / 5 | 10 / 5 |
| chia | cắt theo **thời điểm toàn cục** | **70/10/20 theo từng người** |
| rổ ứng viên | loại mục đã đọc | loại mục đã đọc **và cùng bộ** |
| lát đánh giá | all / new_user / new_item / sparse / warm | một lát, thêm lát mục ít lượt |

Giá của riêng phép loại cùng bộ đã đo ở mục 8 nhánh giao thức: Item-KNN
**−35,6%**, ALS −38,4%, còn `phổ biến` thì *tăng* +15,4%. Một mình nó đã đủ làm
mọi so sánh thô vô nghĩa.

## 2. Cầu nối thứ nhất: chuẩn hoá theo ALS (một phép tính, không phải phép đo)

ALS chạy ở **cả hai** nhánh, nên tỉ lệ `mô hình / ALS` trên *cùng* một chân bỏ
được phần lớn độ khó của giao thức.

| chân | quan sát/người | quan sát/mục | `phổ biến` | `đang nổi` | hai tầng | hai tháp | cổng | **Item-KNN** | **Session-KNN** |
|---|---|---|---|---|---|---|---|---|---|
| AniList (A) | **232,6** | 44,7 | 0,74 | 0,84 | 0,98 | **1,07** | **1,09** | 0,85 | 0,87 |
| Goodreads khớp (A) | **35,8** | 43,6 | 0,63 | 1,12 | 1,03 | 0,85 | 0,77 | **1,08** | **1,36** |
| Goodreads đầy đủ (A) | 35,7 | 72,7 | 0,59 | 1,18 | 1,03 | 0,83 | 0,77 | **1,21** | **1,39** |
| Goodreads `none` (B) | 32,9 | 53,0 | 0,46 | — | — | — | — | **1,35** | **1,68** |
| Goodreads `volume` (B) | 27,6 | 44,2 | 0,42 | — | — | — | — | **1,30** | **1,54** |

Hai dòng đầu là **cùng giao thức, cùng code, chỉ khác bộ dữ liệu** — và thứ tự
hai họ mô hình **đảo hẳn**: trên AniList hai tháp hơn ALS còn láng giềng kém
ALS; trên Goodreads đúng ngược lại.

## 3. Cầu nối thứ hai, và đây là một phép ĐO

Tỉ lệ vẫn là phép chuẩn hoá. `scripts/crossarm_knn.py` bỏ hẳn nó: lấy đúng
Item-KNN và Session-KNN của nhánh B, cho chạy dưới **giao thức của nhánh A** —
cùng train/test, cùng lát, cùng phép loại "đã đọc", cùng `metrics.evaluate`,
cùng `catalog_size`. Ba bộ gợi ý (ALS, hai tháp, KNN) đều che lịch sử y như
nhau, đã kiểm trong code.

Chốt trước khi chạy, ghi trong docstring: **điều kiện bác bỏ giả thuyết "họ mô
hình" là Item-KNN không hơn ALS dưới giao thức A.**

### Goodreads, quy mô khớp — giao thức A

| mô hình | all | new_user | new_item | sparse_user | warm |
|---|---|---|---|---|---|
| `phổ biến` | 0,0461 | 0,1043 | 0,0000 | 0,0385 | 0,0299 |
| `đang nổi` (60 ngày) | 0,0824 | 0,0853 | 0,0000 | 0,0730 | 0,0792 |
| ALS | 0,0734 | 0,1043 | 0,0000 | 0,0838 | 0,0545 |
| hệ hai tầng (đủ mặt) | 0,0757 | 0,1858 | 0,0000 | 0,0872 | 0,0576 |
| hai tháp `feed` | 0,0625 | 0,2322 | 0,0000 | 0,0716 | 0,0430 |
| **Item-KNN** (k=500) | **0,0791** | **0,3500** | 0,0000 | **0,0906** | **0,0737** |
| **Session-KNN** (L=5, giảm 0,7) | **0,0996** | **0,3548** | 0,0000 | **0,0948** | **0,0995** |

### Goodreads, quy mô đầy đủ — giao thức A

| mô hình | all | new_user | sparse_user | warm |
|---|---|---|---|---|
| ALS | 0,0710 | 0,1021 | 0,0843 | 0,0571 |
| hệ hai tầng | 0,0733 | 0,1860 | 0,0891 | 0,0614 |
| hai tháp `feed` | 0,0586 | 0,1978 | 0,0715 | 0,0433 |
| **Item-KNN** | **0,0862** | **0,3540** | **0,0955** | **0,0711** |
| **Session-KNN** (L=10) | **0,0988** | **0,3633** | **0,0970** | **0,1075** |

**Không bị bác bỏ.** Item-KNN 0,0791 hơn ALS 0,0734 ở quy mô khớp, và 0,0862 hơn
0,0710 ở quy mô đầy đủ. Session-KNN hơn **mọi** cấu hình của nhánh dữ liệu ở
**mọi** lát trừ `new_item`: hơn hai tháp +59,4%, hơn hệ hai tầng +31,6%, hơn ALS
+35,7%, và ở lát `new_user` hơn hai tháp **+52,8%** với một hồ sơ chỉ 3 mục mồi.

Khoảng cách này lớn hơn nhiễu seed một bậc: bốn seed của hai tháp trên AniList
nằm trong 0,1169–0,1186 (±0,7%), còn chênh lệch ở đây là +59%.

## 4. Nhưng phép đo đó cũng lật lại chính nó trên AniList

Cùng script, cùng giao thức, bộ AniList:

| mô hình | all | new_user | sparse_user | warm |
|---|---|---|---|---|
| ALS | 0,1098 | 0,3167 | 0,1324 | 0,0972 |
| hệ hai tầng | 0,1072 | 0,3442 | 0,1235 | 0,0970 |
| **hai tháp `feed`** (trung bình 4 seed) | **0,1175** | 0,3744 | 0,1283 | **0,1035** |
| Item-KNN | 0,0931 | **0,3816** | 0,1261 | 0,0915 |
| Session-KNN (L=10) | 0,0957 | 0,3718 | 0,1208 | 0,0983 |

Trên AniList, họ láng giềng **kém** ALS (0,0957 so với 0,1098) và kém hai tháp
(0,1175). Trên Goodreads, cùng code và cùng giao thức, nó **hơn** cả hai.

Nên câu trả lời cho "hướng nào đúng" không phải một hướng nào cả: **thứ tự mô
hình là tính chất của bộ dữ liệu**, và nó đảo chiều giữa hai bộ.

## 5. Cơ chế, và vì sao phép khớp quy mô mới làm nó đọc được

Phép khớp quy mô (mục 4 nhánh dữ liệu) cố ý giữ **tổng tương tác** (0,96×) và
**cỡ danh mục** (0,99×) bằng nhau, nên `quan sát/mục` gần như trùng (44,7 so với
43,6) và thứ duy nhất còn lệch là `quan sát/người`: **232,6 so với 35,8 — 6,5
lần.**

Hai họ mô hình ước lượng hai thứ khác nhau:

- **ALS và hai tháp** ước lượng một **vectơ cho mỗi người**. Số tham số trên mỗi
  người là cố định (64 chiều), nên chất lượng của nó đi theo `quan sát/người`.
- **Item-KNN và Session-KNN** ước lượng **đồng xuất hiện giữa các mục**. Chất
  lượng của nó đi theo `quan sát/mục`, và một người đọc 13 quyển vẫn đóng góp đủ
  các cặp.

Ba lần kiểm, cả ba cùng chiều:

1. **Đổi bộ, giữ giao thức và giữ `quan sát/mục`:** 232,6 → 35,8 quan sát/người
   thì thứ tự đảo (mục 3 và 4).
2. **Giữ bộ, tăng `quan sát/mục` 1,67× (43,6 → 72,7) trong khi `quan sát/người`
   đứng yên (35,8 → 35,7):** Item-KNN **+9,0%** (0,0791 → 0,0862), còn ALS
   −3,3%, hệ hai tầng −3,2%, hai tháp −6,3% — và danh mục lúc đó còn nở 1,31×
   nên +9,0% là *dưới* mức thật.
3. **Giữ bộ, đổi giao thức (B):** tỉ lệ so ALS của họ láng giềng vẫn trên 1
   (1,30–1,68). Giao thức khuếch đại chứ không tạo ra hiệu ứng.

Phép kiểm số 2 là phép tách sạch nhất trong cả đồ án, và nó có được **chỉ vì**
quy mô đầy đủ giữ nguyên hình dạng từng người trong khi nhân đôi tổng dữ liệu.

## 6. Đính chính cho mục 14 của `docs/goodreads-arm.md`

Ở đó tôi viết: "Ba mô hình có học, ba lần không tiến. Nếu tổng lượng dữ liệu là
thứ đang thiếu thì ít nhất một trong ba phải tiến."

Lập luận đó **quá rộng, và chính tôi vừa bác nó**: Item-KNN là mô hình thứ tư,
nó cũng học từ dữ liệu, và nó **tiến +9,0%** khi tổng dữ liệu gấp đôi. Câu đúng
phải hẹp hơn: *thêm người đọc làm tốt lên phần thống kê **phía mục**, và không
làm gì cho phần biểu diễn **phía người*** — ba mô hình không tiến đều là mô hình
có một vectơ cho mỗi người. Kết luận H-người đứng, nhưng nó đứng bằng một câu
hẹp hơn câu tôi đã viết, và câu rộng kia phải bỏ.

## 7. Hệ quả thẳng cho "hướng ban đầu"

Hướng ban đầu (đề xuất đồ án) đóng góp đúng thứ mà hướng của tôi bỏ sót: **họ mô
hình láng giềng**. Tôi chưa bao giờ chạy Item-KNN hay Session-KNN dưới giao thức
của mình, nên mốc so duy nhất của hai tháp là ALS và bảng phổ biến — và trên
Goodreads thì cả hai mốc đó đều yếu. Toàn bộ bộ máy của hướng tôi (bốn mặt nội
dung, đầu thời điểm, cổng độ phổ biến, DropoutNet, xếp lại hai tầng) thua một mô
hình **không có tham số học nào** ở mọi lát trừ một.

Lát trừ ra là `new_item`, và nó là lát duy nhất nội dung có chỗ đứng: truyện
chưa ai đọc thì không có đồng xuất hiện, nên Item-KNN và Session-KNN được
**0,0000**, còn cấu hình `discover` của hai tháp được 0,0330. Đó là chỗ duy nhất
hướng của tôi thắng, và nó thắng tuyệt đối chứ không phải thắng điểm.

Nên kết luận không phải "bỏ hai tháp" mà là: **mốc so đúng của nó là
Session-KNN, không phải ALS**, và dưới mốc đúng thì nó chỉ còn biện minh được ở
cold-start phía *mục*. Mọi bảng cũ so hai tháp với ALS đều phải đọc kèm câu này.

## 8. Cái mỗi hướng dạy được mà hướng kia không

| | nhánh dữ liệu (A) | nhánh giao thức (B) |
|---|---|---|
| trả lời | cơ chế nào là của *phương pháp*, cơ chế nào là của *AniList* | giao thức mua/bán bao nhiêu con số, và mô hình nào thật sự hợp dữ liệu |
| kết quả riêng | P2 lặp lại (+102,8%); P4 xác nhận, và xác nhận luôn *lời giải thích* về `pace`; P3 và P5 bị bác bỏ vì dữ liệu chứ không vì quy mô | Q3 và Q6 bị bác bỏ; BPR sập về bảng phổ biến vì số hạng chệch theo item; significance weighting là bộ lọc độ phổ biến trá hình |
| sai lầm tự bắt được | luật đọc P5 gán dải cho một tỉ lệ đo từ một seed, trái chính quy tắc mục 3 | chọn tham số trên test; lưới val bẹt mà vẫn báo một điểm |

Hai hướng không thay thế nhau: hướng B tìm ra mô hình đúng cho dữ liệu này, còn
hướng A là thứ duy nhất nói được *vì sao* — vì nó có hai bộ dữ liệu đã được khớp
quy mô để so.

## 9. Chạy lại

```bash
python scripts/crossarm_knn.py --out-dir data/processed            # AniList
python scripts/crossarm_knn.py --out-dir data/processed_gr         # Goodreads khớp quy mô
python scripts/crossarm_knn.py --out-dir data/processed_gr_full --lengths 2 5 10
```

`L = 5` không phải tham số chọn trên test của nhánh này: nó là giá trị do **tập
validation của nhánh giao thức** chọn (mục 12 `docs/protocol-arm.md`), đem sang
nguyên vẹn. Nhánh dữ liệu không có tập validation, nên mọi phép dò `L` ở đây sẽ
là dò trên test — phép quét `L = 2 / 5 / 10` in ra chỉ để thấy kết luận không
phụ thuộc `L` (cả ba đều hơn ALS), không phải để chọn số đẹp nhất.
