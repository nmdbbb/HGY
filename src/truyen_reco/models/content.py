"""Gợi ý theo nội dung (L4 của đề xuất): TF-IDF trên kệ cộng đồng + tác giả.

Đây là lớp mô hình duy nhất trong đồ án chấm được **truyện chưa ai đọc**, nên nó
là nửa còn lại của hai phép so mà đề xuất đặt ra: "nội dung có cứu cold-start
không" và "nội dung thua lọc cộng tác bao nhiêu ở NDCG tổng thể".

## Một cái bẫy phải đo trước khi dựng TF-IDF

`popular_shelves` của Goodreads **không phải** một bộ nhãn thể loại. Đo trên
62.876 tác phẩm:

    từ vựng kệ:                 108.012
    kệ chỉ xuất hiện ở 1 truyện: 51.558  (47,7%)
    phân vị 50 / 90 / 99:        2 / 15 / 163 truyện

Gần một nửa "kệ" là tên riêng hoặc ghi chú cá nhân của một người đọc:
`linseyschussan`, `ff-re-2011-till-2015`, `start-and-finish-the-same-day`,
`returned-library`. TF-IDF thuần sẽ cho đúng những token đó **trọng số IDF cao
nhất**, vì chúng hiếm nhất. Chúng không tạo ra tương đồng nào (chỉ nằm trên một
truyện) nhưng chiếm chỗ trong chuẩn vector, tức làm loãng phần tín hiệu thật.

Nên cắt theo **tần số tài liệu**: giữ kệ xuất hiện trên >= `min_df` truyện.
`min_df = 20` cho 8.370 kệ, đặt ngay trên phân vị 90 (15 truyện) — chọn từ phép
đo, không phải từ cảm giác. Đầu trên *không* cắt: `comics`, `graphic-novels`,
`fiction` xuất hiện gần như khắp nơi nên chính IDF đã hạ chúng xuống, cắt tay
thêm là can thiệp hai lần vào cùng một chỗ.

Trọng số trong tài liệu dùng `tag_ranks` (count của kệ quy về phần trăm so với
kệ nhiều nhất của truyện đó) — đúng câu "trọng số theo số lượt gắn" của đề xuất,
và nó thật sự có thông tin: 857 giá trị khác nhau, không phải một thang hai bậc.

Tác giả vào cùng không gian vector dưới dạng token `story:<id>` / `art:<id>`,
trọng số 1,0. Với truyện tranh thì tác giả là tín hiệu nội dung mạnh (đề xuất
cũng dùng nó để giải thích: "Cùng tác giả Naoki Urasawa"), và để chung một ma
trận thì chỉ cần một phép nhân.
"""

import numpy as np
from scipy import sparse


def build_item_features(media, item_index, merged_of=None, min_df=20,
                        use_authors=True):
    """Ma trận TF-IDF (mục x từ vựng), hàng đã chuẩn hoá L2.

    `merged_of`: `work_id -> id mục đã gộp` (id âm). Mục đã gộp lấy **trung bình
    vector các tập thành viên** rồi chuẩn hoá lại — không cộng dồn, vì cộng dồn
    thì một bộ 86 tập sẽ có chuẩn lớn gấp mấy chục lần một truyện lẻ và mọi phép
    cosine sẽ nghiêng về bộ dài.
    """
    merged_of = merged_of or {}
    document_frequency = {}
    for tags in media["tag_names"]:
        if tags is None:
            continue
        for name in set(tags.tolist() if hasattr(tags, "tolist") else tags):
            document_frequency[name] = document_frequency.get(name, 0) + 1
    vocabulary = {name: i for i, name in
                  enumerate(sorted(n for n, c in document_frequency.items()
                                   if c >= min_df))}
    if use_authors:
        offset = len(vocabulary)
        authors = set()
        for column, prefix in (("staff_story_ids", "story"),
                               ("staff_art_ids", "art")):
            if column not in media.columns:
                continue
            for ids in media[column]:
                if ids is None:
                    continue
                for author_id in (ids.tolist() if hasattr(ids, "tolist") else ids):
                    authors.add(f"{prefix}:{author_id}")
        for i, token in enumerate(sorted(authors)):
            vocabulary[token] = offset + i

    # Cộng dồn theo *mục sau khi gộp*: nhiều work có thể trỏ vào cùng một mục.
    rows, cols, values, members = [], [], [], {}
    for record in media.itertuples(index=False):
        media_id = int(record.media_id)
        target = merged_of.get(media_id, media_id)
        position = item_index.get(target)
        if position is None:
            continue
        members[position] = members.get(position, 0) + 1
        names = record.tag_names
        ranks = record.tag_ranks
        if names is not None and ranks is not None:
            names = names.tolist() if hasattr(names, "tolist") else names
            ranks = ranks.tolist() if hasattr(ranks, "tolist") else ranks
            for name, rank in zip(names, ranks):
                column = vocabulary.get(name)
                if column is not None:
                    rows.append(position)
                    cols.append(column)
                    values.append(float(rank) / 100.0)
        if use_authors:
            for attribute, prefix in (("staff_story_ids", "story"),
                                      ("staff_art_ids", "art")):
                ids = getattr(record, attribute, None)
                if ids is None:
                    continue
                for author_id in (ids.tolist() if hasattr(ids, "tolist") else ids):
                    column = vocabulary.get(f"{prefix}:{author_id}")
                    if column is not None:
                        rows.append(position)
                        cols.append(column)
                        values.append(1.0)

    matrix = sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(len(item_index), len(vocabulary)), dtype=np.float32).tocsr()
    matrix.sum_duplicates()
    # Mục gộp từ nhiều tập: chia cho số tập để thành trung bình.
    counts = np.ones(len(item_index), dtype=np.float32)
    for position, total in members.items():
        counts[position] = total
    matrix = sparse.diags(1.0 / counts) @ matrix

    # IDF tính trên chính ma trận mục (sau gộp), không trên danh mục thô: thang
    # phải khớp với tập mục mà mô hình thật sự xếp hạng.
    present = np.asarray((matrix > 0).sum(axis=0)).ravel()
    idf = np.log((1.0 + matrix.shape[0]) / (1.0 + present)) + 1.0
    matrix = matrix @ sparse.diags(idf.astype(np.float32))
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    matrix = sparse.diags((1.0 / norms).astype(np.float32)) @ matrix
    return matrix.tocsr(), vocabulary


def user_profiles(train, item_matrix, user_index, item_index,
                  time_column="created_at", half_life_rows=20):
    """Hồ sơ người đọc = trung bình có trọng số vector các mục đã đọc.

    Trọng số giảm theo thứ tự *gần đây* chứ không theo ngày: đơn vị thời gian
    của hai nhánh dữ liệu khác nhau (AniList ghi theo chương, Goodreads theo
    quyển), nên dùng hạng ngược trong lịch sử của chính người đó thì cùng một
    tham số có nghĩa ở cả hai. `half_life_rows = 20`: mục thứ 20 tính từ cuối có
    trọng số một nửa mục cuối.
    """
    frame = train[["user", "media_id", time_column]].copy()
    frame["row"] = frame["user"].map(user_index)
    frame["column"] = frame["media_id"].map(item_index)
    frame = frame.dropna(subset=["row", "column"])
    frame = frame.sort_values(["user", time_column], kind="mergesort")
    position_from_end = (frame.groupby("user", sort=False).cumcount(ascending=False))
    weight = np.power(0.5, position_from_end.to_numpy() / float(half_life_rows))
    picker = sparse.csr_matrix(
        (weight.astype(np.float32),
         (frame["row"].astype(np.int32), frame["column"].astype(np.int32))),
        shape=(len(user_index), len(item_index)))
    totals = np.asarray(picker.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    profiles = sparse.diags((1.0 / totals).astype(np.float32)) @ picker @ item_matrix
    return profiles.tocsr()


class ContentModel:
    """Chấm điểm bằng cosine giữa hồ sơ người đọc và vector nội dung của mục."""

    def __init__(self, item_matrix, profiles):
        self.item_matrix = item_matrix
        self.profiles = profiles

    def scores_for(self, row):
        profile = self.profiles[row]
        if profile.nnz == 0:
            return None
        return np.asarray((self.item_matrix @ profile.T).todense()).ravel()
