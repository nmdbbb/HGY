# Đặc trưng nội dung cho truyện tranh: nét vẽ, nhịp truyện, tác giả

Tài liệu thiết kế cho phần content-based. Dưới đây là bản thiết kế gốc; phần đã triển khai thật ghi ở ngay mục 0.

## 0. Trạng thái triển khai

Nguồn dữ liệu đã chốt là AniList ([docs/data-sources.md](data-sources.md)), và bốn mặt đã được cài đặt:

| Mặt | File | Cách làm thực tế |
|---|---|---|
| `art` | [`features/art.py`](../src/truyen_reco/features/art.py) | 15 thống kê mực/tông từ ảnh bìa: tỉ lệ mực, mật độ biên, độ dày nét qua distance transform, năng lượng screentone qua FFT dải trung, bão hoà, colorfulness |
| `pace` | [`features/build_features.py`](../src/truyen_reco/features/build_features.py) | Cấu trúc xuất bản + dấu vết hành vi: tỉ lệ bỏ dở, tỉ lệ tạm dừng, tỉ lệ hoàn thành, **vị trí chương trung bình lúc bỏ** |
| `theme` | cùng file | Tag AniList có rank phần trăm, dùng thẳng làm trọng số, qua TF-IDF rồi SVD |
| `creator` | cùng file | Hai không gian SVD tách biệt cho vai Story và vai Art |

Khác biệt so với thiết kế gốc, và lý do:

- **Chưa dùng Gram matrix.** Bản cài đặt đầu dùng thống kê thủ công vì chúng chạy trên CPU, không cần nhãn, và mỗi chiều dịch được thành một câu giải thích. Gram matrix chỉ thêm vào **nếu** phép kiểm truy hồi cùng hoạ sĩ (mục 2.4, đã cài ở `scripts/evaluate.py`) cho thấy thống kê thủ công không đủ. Đo trước, thêm sau.
- **Nhịp từ panel bị cắt** — không nguồn nào cho ảnh trang truyện ở quy mô lớn một cách hợp pháp.
- **Drop-off trở lại ở cấp chương**, không phải cấp tập như lo ngại hồi còn tính dùng Goodreads: AniList có trường `progress`.

## 1. Ý tưởng gốc và chỗ cần làm rõ

Ý tưởng: content-based cho truyện tranh không chỉ là thể loại/tag, mà phải nắm hai yếu tố **nét vẽ** và **nhịp câu chuyện**, cộng thêm **tác giả/họa sĩ**.

Đây là nhận định đúng và là điểm làm dự án khác với một bài recommender thông thường. Hầu hết hệ gợi ý truyện chỉ dùng genre + tag, tức là mô tả *truyện kể về cái gì*, trong khi người đọc thật bỏ một bộ truyện vì *nét vẽ không hợp mắt* hoặc *nhịp quá chậm/quá dài* — hai thứ genre không nói gì về chúng.

**Một giả định cần sửa.** Bạn viết "nhịp câu truyện cần phải biết toàn bộ cốt truyện". Nếu hiểu theo nghĩa phải đọc và hiểu hết nội dung hàng chục nghìn bộ thì dự án sẽ bế tắc ngay — đó là bài toán nghiên cứu, không phải đồ án môn học. Nhưng nhịp truyện **để lại dấu vết ở nhiều chỗ rẻ hơn nhiều so với cốt truyện**:

| Dấu vết của nhịp | Lấy từ đâu | Chi phí |
|---|---|---|
| Độ dài, thời gian serialize, tốc độ ra chương | metadata | Gần như miễn phí |
| Mật độ panel, splash page, lượng thoại mỗi trang | ảnh trang truyện | Trung bình, cần ảnh nội dung |
| Chỗ người đọc bỏ dở, tốc độ cày | log tương tác có timestamp | Miễn phí nếu có log |
| "Nhịp chậm", "kéo dài", "kết gấp" | review của người đọc | Thấp |

Người đọc phàn nàn về nhịp trong review nhiều hơn bất kỳ thuộc tính nào khác, nên review gần như là một oracle sẵn có cho đúng thuộc tính này. Kết luận: **không cần hiểu cốt truyện để đo nhịp truyện.** Đọc hết cốt truyện là con đường đắt nhất tới một tín hiệu có thể lấy bằng ba cách rẻ hơn.

## 2. Nét vẽ

### 2.1 Nguồn tín hiệu

Ảnh. Theo thứ tự dễ lấy: **ảnh bìa** (API công khai cho hầu hết mọi bộ) → **banner/ảnh quảng bá** → **trang truyện bên trong** (khó, xem mục 8).

Bìa chỉ phản ánh một phần nét vẽ: nó thường được vẽ công phu hơn trang thường và hay được tô màu dù bên trong là đen trắng. Đây là hạn chế thật, phải ghi rõ trong báo cáo chứ không nên che. Có một thí nghiệm nhỏ để biến hạn chế này thành đóng góp: với tập con có cả bìa và trang trong (Manga109), kiểm xem vector nét vẽ từ bìa có tương quan với vector từ trang trong không. Trả lời được câu đó là đã trả lời được "dùng bìa có hợp lệ không".

### 2.2 Ba cách rút đặc trưng nét vẽ

**(a) Thống kê mực và tông — thủ công, rẻ, giải thích được**

- Tỉ lệ pixel mực: nét đậm hay thưa.
- Mật độ biên (Canny) và **độ dày nét**: binarize vùng mực rồi lấy distance transform, trung bình + phương sai cho ra nét mảnh tinh tế so với nét to bản.
- Năng lượng screentone/hatching: bộ lọc Gabor hoặc FFT ở dải tần trung — screentone dạng điểm và nét gạch chéo tạo năng lượng tuần hoàn rất đặc trưng.
- Thống kê màu: độ bão hòa trung bình, số màu trội (k-means trên pixel), cân bằng nóng/lạnh. Tách ngay manga đen trắng khỏi manhwa/webtoon màu.

Giá trị lớn nhất của nhóm này không phải độ chính xác mà là **mỗi con số đều dịch được thành một câu cho người dùng đọc**: "truyện này nét dày, nhiều screentone, giống bộ X bạn đã đọc". Nhóm (b) và (c) không làm được việc đó.

**(b) Gram matrix trên VGG — chuẩn mực cho "style tách khỏi nội dung"**

Đưa ảnh qua VGG19 pretrained, lấy feature map ở các lớp conv đầu/giữa (relu1_1 … relu4_1), tính Gram matrix (tương quan giữa các channel) của từng lớp, nối lại rồi giảm chiều bằng PCA.

Gram matrix lấy trung bình trên toàn bộ không gian ảnh, nên nó **bỏ thông tin "cái gì nằm ở đâu" và giữ lại thống kê nét/texture** — đúng định nghĩa nét vẽ. Đây là cơ sở của neural style transfer (Gatys et al.), tức là có nguồn trích dẫn tử tế cho báo cáo. Không cần nhãn, không cần train.

Lưu ý: rất nhạy với chất lượng scan, độ phân giải, nén JPEG. Phải chuẩn hóa trước (xám hóa, resize về cùng cạnh, có thể lọc nhiễu), nếu không model sẽ học ra "nhóm scan nét" thay vì "nhóm vẽ nét".

**(c) Embedding CLIP — mạnh nhưng lẫn nội dung vào style**

CLIP image encoder được train trên lượng ảnh khổng lồ gồm nhiều tranh anime/manga, nên embedding của nó bắt được phong cách khá tốt và chỉ cần một lần forward pass.

Nhược điểm phải biết trước: CLIP **trộn nội dung với phong cách**. Hai bộ đều vẽ robot khổng lồ sẽ nằm gần nhau dù một bộ là nét mực thô ráp kiểu seinen và bộ kia là nét tròn dễ thương. Với gợi ý, phần nào thường đã được genre/tag lo rồi, nên để CLIP làm luôn việc đó là trùng lặp tín hiệu.

Khuyến nghị: dùng **(a) + (b)** làm vector nét vẽ, giữ (c) làm baseline để so sánh trong phần ablation. Nếu (b) không hơn (c) thì đó cũng là một kết quả đáng báo cáo.

### 2.3 Classify nét vẽ như thế nào

Câu hỏi "làm sao classify nó" có hai hướng trả lời, và hướng thứ hai thường tốt hơn cho dự án này.

**Hướng 1 — phân loại có nhãn.** Cần taxonomy. Ba cách lấy nhãn mà không phải ngồi gán tay hàng nghìn bộ:

- *Nhãn yếu từ metadata, miễn phí*: demographic (shounen/shoujo/seinen/josei), thập niên phát hành, xuất xứ (manga/manhwa/manhua), màu hay đen trắng. Bốn trường này tương quan rất mạnh với nét vẽ. Train classifier ảnh → nhãn yếu, rồi lấy **lớp áp cuối** làm embedding nét vẽ. Đây là mẹo "học style qua proxy task", và nhãn không tốn công nào.
- *Lấy chính họa sĩ làm nhãn*: train ảnh → ID họa sĩ (vài trăm lớp). Lớp áp cuối thành không gian "dấu vân tay nét vẽ", nơi các trang cùng một họa sĩ tụ lại. Nhãn cũng miễn phí từ metadata. Rủi ro phải canh: model có thể học nhận **nhân vật quen** thay vì nét vẽ, vì cùng họa sĩ thì vẽ cùng dàn nhân vật. Giảm rủi ro bằng cách train trên crop tránh mặt, hoặc đưa Gram matrix vào thay ảnh thô.
- *Gán tay, nhưng chỉ cho tập đánh giá*: 200–500 bộ với taxonomy khoảng 8 lớp, cả nhóm làm 2–3 tiếng. Dù đặc trưng nét vẽ là không giám sát thì vẫn **cần một tập ground truth** để chứng minh không gian nét vẽ có nghĩa.

**Hướng 2 — không áp taxonomy, để dữ liệu tự chia (ưu tiên).** Chạy k-means/HDBSCAN trên embedding nét vẽ, xem từng cụm rồi đặt tên. Như vậy taxonomy được *phát hiện* thay vì bị áp đặt, và việc đặt tên cụm là phần kể chuyện hay nhất trong báo cáo. Kèm theo đó là một hình vẽ rất mạnh: UMAP 2 chiều của không gian nét vẽ, mỗi điểm thay bằng ảnh bìa thu nhỏ. Hình đó một mình đã thuyết phục người đọc rằng nét vẽ đo được.

Quan trọng hơn: **hệ gợi ý không cần nhãn lớp, nó cần khoảng cách.** Lớp "shoujo-style" chỉ hữu ích khi hiển thị cho người dùng; còn để gợi ý thì `cos(v_art(i), v_art(j))` là đủ và mất ít thông tin hơn.

### 2.4 Cách kiểm chứng nét vẽ mà không cần tới hệ gợi ý

Giữ lại toàn bộ tác phẩm của một số họa sĩ, rồi kiểm xem trong không gian `v_art`, láng giềng gần nhất của một tác phẩm có phải tác phẩm cùng họa sĩ không (precision@k với họa sĩ làm ground truth). Nhãn miễn phí, metric rõ ràng, và **làm được ngay trước khi hệ gợi ý tồn tại**. Nên làm thí nghiệm này sớm nhất có thể: nó quyết định toàn bộ nhánh nét vẽ có đáng theo hay không.

## 3. Nhịp câu chuyện

Bốn nguồn, xếp theo tỉ lệ giá trị trên chi phí.

**(a) Cấu trúc serialize — từ metadata, gần như miễn phí**

Số chương, số tập, ngày bắt đầu/kết thúc, trạng thái (đang ra/hoàn thành/tạm ngưng), số chương trên tháng, số chương mỗi tập. Một bộ shounen 400 chương chạy 15 năm có nhịp khác về bản chất so với một bộ josei 30 chương đã hoàn. Trục "dài và còn tiếp" đối lại "ngắn và đã hoàn" là một phần rất lớn của cái người đọc gọi là nhịp, và nó nằm sẵn trong metadata.

**(b) Nhịp từ hành vi đọc — nếu log có timestamp**

Tốc độ cày (số chương mỗi lượt đọc), **phân bố chương bị bỏ dở** (người đọc rơi ở chương 3 hay chương 80), tỉ lệ đọc lại, khoảng cách giữa hai lần đọc. Một bộ "slow burn" có đường cong rơi-rồi-trụ rất đặc trưng.

Đây là tín hiệu nhịp trung thực nhất vì nó đo trực tiếp trải nghiệm. Nhưng nó là dữ liệu hành vi, nên **truyện mới không có** — đúng lý do vì sao vẫn cần nhịp suy ra từ nội dung ở (a), (c), (d).

**(c) Nhịp từ panel và trang — cần ảnh nội dung, và là phần hay nhất về mặt học thuật**

Trong lý thuyết truyện tranh (McCloud, *Understanding Comics*), nhịp được mã hóa ngay trong cách chia khung. Đo được:

- Số panel trên trang: nhiều thì nhanh và dày thoại, ít thì chậm và nặng không khí.
- Phương sai diện tích panel; tần suất splash page và trang đôi, là dấu hiệu cao trào.
- Mật độ chữ trên panel (diện tích bong bóng thoại chia diện tích panel): nhiều thoại hay nhiều hành động.
- Tỉ lệ panel không có chữ — nhịp lặng, chậm, tả cảnh.

Xếp các chỉ số này theo thứ tự trang trong một chương sẽ được một **chuỗi thời gian, và đó chính là đường cong nhịp** của chương đó. Từ đây tính được vị trí các đỉnh cao trào.

Giá trị: đây là phần biến dự án từ "một recommender nữa" thành một dự án có điểm riêng, và nó có chỗ dựa lý thuyết để trích dẫn. Chi phí: cần ảnh trang trong, nên chỉ khả thi trên tập con. Manga109 có sẵn annotation panel và text, tức là làm được ngay trên 109 bộ mà không phải viết panel detector. Nên đóng gói phần này thành **một nghiên cứu nhỏ trên tập con**, không hứa chạy toàn kho.

**(d) Nhịp từ synopsis và review — LLM gán nhãn mềm**

Dùng phần chữ có sẵn: synopsis, tag, review. Cho LLM chấm mỗi bộ trên vài trục liên quan tới nhịp: chậm rãi hay dồn dập, episodic hay một mạch dài, mật độ twist, biến động cảm xúc, kiểu kết (trọn vẹn / gấp / mở). Kết quả là một vector ngắn, mỗi chiều đều có tên gọi người đọc hiểu được, nên dùng cho phần giải thích rất tốt: "cả hai đều slow burn, nặng tâm lý, ít twist".

Cảnh báo phải xử lý trong báo cáo: **nhãn do LLM gán thì phải được kiểm chứng.** Cho nhóm gán tay 100 bộ rồi báo cáo độ đồng thuận giữa nhãn tay và nhãn LLM. Không có số đó, phần này chỉ là nói suông và sẽ bị hỏi ngay khi bảo vệ.

Riêng review còn là nguồn trực tiếp: người đọc phàn nàn về nhịp bằng chữ rõ ràng ("kéo dài quá", "20 chương đầu chậm", "kết gấp"). Một lexicon từ khóa về nhịp cộng sentiment là đã có tín hiệu tốt, không nhất thiết phải LLM.

## 4. Tác giả và họa sĩ

### 4.1 Một điểm khớp đẹp với chính ý tưởng của bạn

Truyện tranh thường ghi công **tác giả phần truyện và họa sĩ phần tranh riêng biệt** (metadata của AniList phân vai "Story" và "Art"). Nghĩa là:

- Trục **họa sĩ** ánh xạ gần như một-một với yếu tố **nét vẽ**.
- Trục **tác giả** ánh xạ với yếu tố **nhịp và cách kể**.

Hai yếu tố bạn nêu ra có sẵn hai cái neo trong metadata. Đây cũng là nguồn nhãn miễn phí đã nói ở 2.3 và 2.4.

### 4.2 Ba cách dùng, từ thô tới tốt

- **Đặc trưng hạng mục / prior mạnh**: cùng họa sĩ thì gần như chắc chắn trùng nét vẽ. Đơn giản nhưng mạnh, và câu giải thích "vì cùng họa sĩ với bộ bạn thích" vừa đúng vừa dễ hiểu.
- **Embedding người sáng tác (nên dùng)**: đừng one-hot. Hai cách học:
  1. *Từ phía item*: embedding của một họa sĩ bằng trung bình vector nét vẽ các tác phẩm của họ. Khi đó tính được độ giống giữa hai họa sĩ **không chia sẻ người đọc nào**. Quan trọng hơn, nó mở một đường cho **cold-start truyện mới**: một bộ mới của họa sĩ đã biết thì khởi tạo vector nét vẽ bằng trung bình của họa sĩ đó, có ngay từ ngày đầu.
  2. *Từ phía tương tác*: coi người sáng tác như item trong ma trận CF (người dùng đọc từ 2 tác phẩm của X trở lên thì tạo một cạnh), rồi factorize. Bắt được quan hệ "fan của tác giả A cũng thích tác giả B" kể cả khi hai người viết thể loại khác nhau.
- **Góc nhìn đồ thị**: dựng đồ thị không đồng nhất người dùng — truyện — người sáng tác — tag, rồi node2vec/metapath2vec hoặc một GNN nhẹ (LightGCN). Node người sáng tác đóng vai trò **cầu nối lan truyền thị hiếu giữa các item thưa dữ liệu**. Đây là câu trả lời hiện đại và đúng nhất, nhưng cũng tham vọng nhất — để ở giai đoạn sau, không phải giai đoạn đầu.

### 4.3 Cái bẫy phải chặn

Đặc trưng người sáng tác gây **sụp về filter bubble**: hệ sẽ gợi ý toàn bộ khác của cùng họa sĩ, đúng một cách tầm thường mà vô dụng. Phải chặn bằng ràng buộc đa dạng, ví dụ tối đa 1–2 slot cùng người sáng tác trong một danh sách. Và phải **đo** nó: đưa "mức tập trung người sáng tác trong danh sách" vào bộ metric. Có đo cái này là một điểm chững chạc cho báo cáo.

## 5. Ghép tất cả vào hệ gợi ý

### 5.1 Biểu diễn item theo nhiều mặt (multi-facet), không nối thành một vector

Giữ các vector **riêng theo từng mặt**:

| Mặt | Nội dung | Chiều (dự kiến) |
|---|---|---|
| `v_art` | nét vẽ: thống kê mực + Gram, sau PCA | 64–128 |
| `v_pace` | nhịp: cấu trúc serialize + trục từ review/synopsis | 10–30, đa phần đọc hiểu được |
| `v_theme` | genre, tag, embedding synopsis — phần content-based thông thường | 64–256 |
| `v_creator` | embedding tác giả/họa sĩ | 32–64 |

Độ giống giữa hai truyện là **tổng có trọng số của độ giống từng mặt**:

```latex
\mathrm{sim}(i,j) = w_{art}\cos(v_{art}^i, v_{art}^j) + w_{pace}\cos(v_{pace}^i, v_{pace}^j) + w_{theme}\cos(v_{theme}^i, v_{theme}^j) + w_{cre}\cos(v_{cre}^i, v_{cre}^j)
```

Ba lý do vì sao tách mặt tốt hơn nối một vector dài:

1. **Giải thích có được miễn phí.** Biết mặt nào đóng góp nhiều nhất thì sinh được câu lý do: "giống 82% về nét vẽ, 40% về nhịp". Một vector nối không làm được việc này một cách sạch sẽ — mà giải thích được là yêu cầu lõi của đề bài.
2. **Trọng số theo từng người dùng.** Học `w` riêng cho mỗi người. Có người chọn truyện theo nét vẽ (sẵn sàng theo một họa sĩ đi bất cứ đâu), có người chọn theo cốt. Khớp `w` từ lịch sử của họ bằng một hồi quy logistic nhỏ: mặt nào giải thích tốt nhất các bộ họ đã thích. Đây là **điểm có thể gọi là đóng góp riêng của dự án**, và nó cho thêm một câu giải thích đắt giá: "bạn có xu hướng chọn truyện theo nét vẽ".
3. **Suy giảm mềm khi thiếu dữ liệu.** Bộ nào không có ảnh trang trong thì bỏ phần panel của `v_art` rồi chuẩn hóa lại trọng số. Các mặt thiếu được độc lập với nhau.

### 5.2 Chỗ cắm vào kiến trúc hybrid

Hai tầng, theo đúng cách làm trong công nghiệp:

- **Tầng 1, sinh ứng viên**: CF/ALS cho người dùng đã có lịch sử; popularity cộng độ giống theo mặt cho người dùng mới. Lấy ra vài trăm ứng viên.
- **Tầng 2, xếp hạng lại**: một model (LightGBM hoặc hồi quy logistic) với đặc trưng gồm điểm CF, độ giống từng mặt so với tập truyện người dùng đã thích, độ phổ biến, độ mới, trùng người sáng tác, hình phạt đa dạng.

Như vậy **không phải chọn trọng số bằng tay mà học được chúng**, và đó cũng là câu trả lời thật cho "làm sao tận dụng tất cả tín hiệu cùng lúc".

### 5.3 Cộng hưởng với cold-start

Cách tiếp cận theo mặt giải quyết cold-start ở **cả hai phía**, nên nó không phải một nhánh riêng mà là cùng một thiết kế:

- **Người dùng mới**: onboarding bằng cách bày một lưới khoảng 12 ảnh bìa trải đều các cụm nét vẽ, cho họ chọn 3. Chọn bìa là **lấy được sở thích về `v_art` với lịch sử đọc bằng không**, và đó là hành vi tự nhiên — người ta đánh giá truyện qua nét vẽ trong một giây. Thêm một câu hỏi về nhịp (thích bộ ngắn gọn hay bộ dài). Sau khoảng 20 giây đã có vector sở thích theo mặt. Cách này tốt hơn hẳn onboarding kiểu "chọn 3 thể loại", và nó là **hệ quả trực tiếp của ý tưởng nét vẽ** — tức là ý tưởng của bạn tự nó trả tiền cho phần cold-start.
- **Truyện mới**: chưa có tương tác nào, nhưng có bìa, có người sáng tác, có metadata, nên `v_art`, `v_creator`, `v_pace` tính được ngay ngày đầu. Đặc trưng nội dung chính là thứ lấp đúng cái lỗ item cold-start.

## 6. Đánh giá

Phải chứng minh các mặt **đáng công sức**, không chỉ là chúng tồn tại.

- **Ablation**: CF thuần → thêm `theme` → thêm `art` → thêm `pace` → thêm `creator`, đo NDCG@10 và recall@20.
- **Đo theo từng lát dữ liệu, không chỉ đo tổng.** Đây là chỗ dễ sai nhất: nét vẽ có thể giúp rất ít trên toàn bộ tập test nhưng giúp rất nhiều ở lát người dùng mới và truyện mới, và một metric tổng sẽ **che mất** đóng góp đó. Chia lát: người dùng mới, người dùng thưa, truyện mới, truyện ít phổ biến.
- **Metric ngoài độ chính xác**: đa dạng trong danh sách, mức tập trung người sáng tác, độ phủ catalog. Các mặt nội dung phải làm đa dạng tốt lên.
- **Kiểm chứng nét vẽ độc lập**: bài toán truy hồi cùng họa sĩ ở mục 2.4, làm trước khi có recommender.
- **Kiểm chứng nhãn nhịp**: độ đồng thuận giữa nhãn LLM và nhãn gán tay trên 100 bộ (mục 3d).
- **Đánh giá chất lượng giải thích**: cho 10–20 bạn cùng lớp chấm xem lý do gợi ý có thuyết phục không. N nhỏ, nên trình bày như kết quả định tính, đừng tô thành thống kê.

## 7. Lộ trình và phạm vi

Ý tưởng có một bản nghiên cứu và một bản khả thi. Ranh giới nên vạch rõ ngay để không hứa quá khi bảo vệ.

| Mức | Việc | Ghi chú |
|---|---|---|
| **Lõi an toàn, làm trước** | metadata + tag + embedding synopsis, CF, hybrid, onboarding cold-start, giải thích | Chưa cần ảnh. Phải chạy được trước khi làm gì khác |
| **Điểm riêng, khả thi** | `v_art` từ ảnh bìa (thống kê mực + Gram, cụm không giám sát, hình UMAP); `v_pace` từ cấu trúc serialize + nhãn từ review; `v_creator` làm một mặt; trọng số mặt theo người dùng | Đây là phần làm dự án khác biệt. Toàn bộ đều làm được |
| **Mức nghiên cứu, khoanh vùng** | nhịp từ panel trên ảnh trang trong; mô hình đường cong cốt truyện theo chương; GNN trên đồ thị người sáng tác | Làm trên tập con (Manga109) và gọi đúng tên là nghiên cứu nhỏ, đừng hứa chạy toàn kho |
| **Không làm** | đọc/parse toàn bộ cốt truyện hàng nghìn bộ; crawl site truyện lậu để lấy ảnh | Bế tắc về quy mô, và rủi ro bản quyền khi trình bày |

Thứ tự khuyến nghị: lõi an toàn chạy end-to-end trước (kể cả khi chất lượng tầm thường), rồi làm **thí nghiệm truy hồi cùng họa sĩ** ở mục 2.4 để biết nhánh nét vẽ có sống được không, rồi mới xây `v_art` đầy đủ. Không nên làm ngược: dồn ba tuần vào nét vẽ rồi mới phát hiện không có dữ liệu tương tác để đánh giá.

## 8. Chặn lớn nhất: dữ liệu

Toàn bộ thiết kế trên phụ thuộc vào việc có dữ liệu gì, mà nhóm **chưa chốt nguồn dữ liệu** (xem mục 6 của [SPEC.md](../SPEC.md)). Cụ thể, mỗi nhánh cần:

| Nhánh | Dữ liệu bắt buộc |
|---|---|
| CF, và mọi đánh giá offline | ma trận người dùng – truyện (rating hoặc implicit) |
| Nhịp từ hành vi | log tương tác **có timestamp** |
| Nét vẽ | ảnh bìa cho toàn bộ item |
| Nhịp từ panel | ảnh trang trong, ít nhất cho tập con |
| Tác giả/họa sĩ | metadata staff phân vai Story/Art |

Hai lựa chọn thực tế:

1. **Metadata + ảnh bìa từ API công khai** (AniList có GraphQL API công khai với genre, tag kèm phần trăm bình chọn, staff phân vai, số chương/tập, ngày bắt đầu/kết thúc, trạng thái, ảnh bìa; MyAnimeList qua Jikan tương tự). Cho `v_theme`, `v_pace` phần cấu trúc, `v_creator`, và `v_art` từ bìa. **Không** cho dữ liệu tương tác.
2. **Dữ liệu tương tác**: danh sách đọc công khai của người dùng MyAnimeList (có các bộ dữ liệu dẫn xuất sẵn trên Kaggle) cho ma trận người dùng – truyện thật. Cần kiểm xem bản nhóm lấy được có timestamp không, vì điều đó quyết định nhánh nhịp-từ-hành-vi sống hay chết.
3. **Manga109** cho ảnh trang trong có annotation panel và text, dùng cho nghiên cứu nhỏ về nhịp panel. Cấp quyền theo yêu cầu cho mục đích học thuật, nên **xin sớm** vì có thời gian chờ.

Cần kiểm lại trước khi tin: vocabulary tag thực tế của AniList (có bao nhiêu tag mang tính cấu trúc/nhịp), và liệu bộ dữ liệu tương tác chọn được có timestamp hay không.
