"""Cổng bám-theo-đang-nổi học được gì, và vì sao người đọc mới bị mất điểm.

    python scripts/diagnose_trend_gate.py

Phép đo trước (cùng seed, chỉ khác một chỗ là cổng):

    lát            không cổng   có cổng
    all                0,1169    0,1192
    warm               0,1039    0,1099
    sparse_user        0,1268    0,1277
    new_user           0,3771    0,2499   <- mất 34%
    CN trong top-10      0,01      0,02

Cổng thắng ở người đọc đã có lịch sử và mất ở người đọc mới. Ba câu hỏi, mỗi
câu một con số, tất cả chỉ dùng dữ liệu train nên không mượn đáp án của test:

  1. Cổng có thật sự được dùng? Nếu `g_u` co về một hằng số thì khác biệt đo
     được không thể do cổng, và tôi phải tìm chỗ khác.
  2. `g_u` có phụ thuộc độ dài lịch sử? Người đọc mới không có `g_u` nào được
     học, nên lúc phục vụ họ nhận trung bình toàn dân. Nếu người ít lịch sử
     trong train có `g_u` khác hẳn trung bình thì chính cái trung bình đó là
     một ước lượng chệch, và chỗ mất 34% có nguyên nhân cấu trúc chứ không
     phải cổng sai.
  3. `g_u` có bám vào nét "thích truyện đang nổi" của người đọc không? Đo bằng
     tương quan giữa `g_u` và độ nổi gần đây trung bình của những bộ chính
     người đó đã đọc. Đây là bằng chứng độc lập với NDCG: nếu tương quan dương
     thì cổng đang đọc đúng cái nét tôi nói nó sẽ đọc.

Và một câu cho manhua: nếu người đọc manhua có `g_u` thấp (vì bảng đang nổi là
manhwa) mà tỉ lệ CN trong top-10 chỉ nhúc nhích 0,01 -> 0,02, thì cổng không
phải chỗ chặn manhua — phải tìm tiếp.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from truyen_reco.models.two_tower import ServedTwoTower  # noqa: E402

COUNTRIES = ("JP", "KR", "CN")


def spearman(a, b):
    """Tương quan hạng, tự viết để không thêm phụ thuộc scipy."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/processed", type=Path)
    parser.add_argument("--gated", default="two_tower_gated.pt")
    parser.add_argument("--plain", default="two_tower_feed.pt")
    args = parser.parse_args()

    gated_path = args.out_dir / args.gated
    if not gated_path.exists():
        sys.exit(f"chưa có {gated_path} — chạy train_embedding.py "
                 "--variants feed_gated --save trước")

    served = ServedTwoTower.load(gated_path)
    model = served.model
    if model.trend is None:
        sys.exit("mô hình này không có cổng")

    # `model.trend.weight` KHÔNG phải g_u. Với thiết kế cổng hai phần hiện tại
    # (`TwoTower.trend_weight`) nó chỉ là phần dư từng người `delta_u`, còn
    # g_u = base(log(1+độ dài lịch sử)) + delta_u. Bản cổng MỘT phần cũ thì
    # `trend.weight` đúng là g_u. Hai thứ lệch nhau cả một bậc độ lớn —
    # delta_u trung bình +0,01 trong khi g_u trung bình +2,67 — nên đọc
    # `trend.weight` của bản hai phần rồi so với blob JSON ghi từ bản một
    # phần là so hai tham số khác nhau, và nó dẫn thẳng tới kết luận sai
    # "cổng đã tắt đầu thời điểm". Vì vậy script tính g_u ĐẦY ĐỦ và in rõ
    # thiết kế của blob đang đọc.
    delta = model.trend.weight.detach().numpy().ravel()
    n_users = len(delta)
    hist_index = served.hist_index.numpy()
    length = (hist_index >= 0).sum(axis=1).astype(float)
    two_part = getattr(model, "trend_base", None) is not None
    if two_part:
        with torch.no_grad():
            lengths = torch.log1p(torch.from_numpy(length.astype(np.float32)))
            base = model.trend_base(lengths.reshape(-1, 1)).numpy().ravel()
            cold = float(model.trend_weight(user_ids=None, n_history=0)
                         .detach().numpy().ravel()[0])
        gate = base + delta
    else:
        base = np.zeros_like(delta)
        gate = delta
        cold = float(gate.mean())

    # ---- 1. cổng có được dùng không -------------------------------------
    print("=" * 78)
    print("1. Phân phối cổng g_u (khởi tạo 1,0 — lệch khỏi 1,0 là do học)")
    print("=" * 78)
    quantiles = np.percentile(gate, [1, 10, 25, 50, 75, 90, 99])
    print(f"  n = {n_users:,} người đọc train | thiết kế: "
          + ("cổng HAI phần, g_u = base(log(1+n)) + delta_u"
             if two_part else "cổng MỘT phần, trend.weight chính là g_u"))
    print(f"  g_u      trung bình {gate.mean():+.4f}   lệch chuẩn {gate.std():.4f}")
    if two_part:
        print(f"  base     trung bình {base.mean():+.4f}   lệch chuẩn {base.std():.4f}")
        print(f"  delta_u  trung bình {delta.mean():+.4f}   lệch chuẩn {delta.std():.4f}")
    print("  phân vị g_u  " + "  ".join(
        f"p{p}={v:.3f}" for p, v in zip([1, 10, 25, 50, 75, 90, 99], quantiles)))
    print(f"  giá trị người đọc MỚI nhận (n_history=0, chưa có delta): {cold:+.4f}")

    # Thang của đầu thời điểm: chỉ tích g_u * bias_i là xác định được, nên so
    # chuẩn của trọng số thời điểm giữa hai bản để biết cổng có bị bù thang.
    plain_path = args.out_dir / args.plain
    if plain_path.exists():
        plain = ServedTwoTower.load(plain_path)
        if plain.model.time is not None and model.time is not None:
            a = float(plain.model.time.weight.norm())
            b = float(model.time.weight.norm())
            print(f"\n  |w_thời điểm| không cổng {a:.4f} | có cổng {b:.4f}"
                  f" | tích với cổng trung bình {b * gate.mean():.4f}")
            print("  (chỉ tích g_u·bias_i là xác định; so để biết thang có "
                  "dịch đi không)")

    print("\n" + "=" * 78)
    print("2. Cổng theo độ dài lịch sử — quyết định ước lượng cho người mới")
    print("=" * 78)
    print(f"  tương quan hạng g_u ~ độ dài lịch sử:     {spearman(gate, length):+.4f}")
    print(f"  tương quan hạng delta_u ~ độ dài lịch sử: {spearman(delta, length):+.4f}"
          + ("   (phần dư từng người, không phải cổng)" if two_part else ""))
    edges = [1, 5, 15, 40, 100, 200, 10 ** 9]
    print(f"\n  {'độ dài lịch sử':<20}{'số người':>10}{'g_u trung bình':>18}"
          f"{'g_u trung vị':>16}{'delta_u tb':>16}")
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (length >= low) & (length < high)
        if mask.sum() < 10:
            continue
        label = f"{low} – {high - 1}" if high < 10 ** 9 else f"≥ {low}"
        print(f"  {label:<20}{int(mask.sum()):>10}{gate[mask].mean():>18.4f}"
              f"{np.median(gate[mask]):>16.4f}{delta[mask].mean():>16.4f}")

    # Hồi quy g_u ~ a + b·log1p(độ dài), chỉ trên train, để có giá trị dự đoán
    # ở độ dài của một người đọc mới (vài lượt chọn lúc onboarding).
    x = np.log1p(length)
    design = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(design, gate, rcond=None)
    print(f"\n  khớp g_u ≈ {coef[0]:.4f} + {coef[1]:+.4f}·log(1+độ dài)")
    print(f"  {'số lượt chọn của người mới':<30}{'g_u dự đoán':>14}")
    for picks in (1, 3, 5, 10, 20):
        print(f"  {picks:<30}{coef[0] + coef[1] * np.log1p(picks):>14.4f}")
    print(f"  {'trung bình toàn dân (đang dùng)':<30}{gate.mean():>14.4f}")

    # ---- 3. cổng có bám nét "thích đang nổi" ----------------------------
    media = pd.read_parquet(args.out_dir / "media_core.parquet",
                            columns=["media_id", "country", "popularity"])
    index = json.loads((args.out_dir / "index.json").read_text(encoding="utf-8"))
    item_index = {int(k): v for k, v in index["item_index"].items()}
    reverse = {v: int(k) for k, v in item_index.items()}

    n_items = served.item_raw.shape[0]
    country_of = np.full(n_items, -1, dtype=np.int8)
    pop_of = np.zeros(n_items, dtype=np.float64)
    codes = {c: i for i, c in enumerate(COUNTRIES)}
    for media_id, country, popularity in media.itertuples(index=False):
        position = item_index.get(int(media_id))
        if position is None or position >= n_items:
            continue
        country_of[position] = codes.get(country, -1)
        pop_of[position] = float(popularity or 0.0)

    # Hạng phổ biến chuyển thành phân vị, để "độ nổi trung bình của những bộ
    # người này đọc" là một con số trong [0, 1] đọc được.
    order = np.argsort(-pop_of)
    percentile = np.empty(n_items)
    percentile[order] = 1.0 - np.arange(n_items) / max(n_items - 1, 1)

    trendiness = np.full(n_users, np.nan)
    cn_share = np.full(n_users, np.nan)
    for row in range(n_users):
        positions = hist_index[row]
        positions = positions[positions >= 0]
        if positions.size == 0:
            continue
        trendiness[row] = float(percentile[positions].mean())
        known = country_of[positions]
        known = known[known >= 0]
        if known.size:
            cn_share[row] = float((known == codes["CN"]).mean())

    valid = ~np.isnan(trendiness)
    print("\n" + "=" * 78)
    print("3. Cổng có bám nét người đọc không (bằng chứng ngoài NDCG)")
    print("=" * 78)
    print(f"  tương quan hạng g_u ~ độ nổi trung bình của truyện đã đọc: "
          f"{spearman(gate[valid], trendiness[valid]):+.4f}")
    ok = valid & ~np.isnan(cn_share)
    print(f"  tương quan hạng g_u ~ tỉ lệ manhua trong lịch sử:          "
          f"{spearman(gate[ok], cn_share[ok]):+.4f}")

    thirds = np.percentile(trendiness[valid], [33, 67])
    print(f"\n  {'nhóm theo độ nổi truyện đã đọc':<34}{'số người':>10}"
          f"{'g_u trung bình':>18}")
    groups = [("đọc truyện ít nổi nhất", trendiness < thirds[0]),
              ("ở giữa", (trendiness >= thirds[0]) & (trendiness < thirds[1])),
              ("đọc truyện nổi nhất", trendiness >= thirds[1])]
    for label, mask in groups:
        mask = mask & valid
        print(f"  {label:<34}{int(mask.sum()):>10}{gate[mask].mean():>18.4f}")

    cn_readers = ok & (cn_share > 0.2)
    if cn_readers.sum() >= 10:
        print(f"\n  người đọc >20% manhua: {int(cn_readers.sum())} người, "
              f"g_u trung bình {gate[cn_readers].mean():.4f} "
              f"(còn lại {gate[ok & ~cn_readers].mean():.4f})")

    # ---- 4. delta_u có lặp lại ở kỳ sau, hay chỉ là ghi nhớ train? -------
    # Mục 3 đo trên chính dữ liệu đã huấn luyện, nên tương quan cao ở đó chưa
    # phân biệt được "bắt đúng nét người đọc" với "khớp nhiễu của 36 lượt".
    # Phép tách: cùng một thước độ nổi, nhưng tính trên những mục người đó đọc
    # trong TEST. delta_u được học mà không thấy test, nên nếu nét là thật thì
    # tương quan phải còn lại phần lớn. Đây là chỗ hai nhánh tách nhau:
    # +0,56 (AniList, ~233 quan sát/người) so với +0,33 (Goodreads, ~36).
    gen = {}
    test_path = args.out_dir / "test.parquet"
    if test_path.exists():
        test = pd.read_parquet(test_path, columns=["user", "media_id"])
        user_index = served.user_index
        test = test.assign(position=test["media_id"].map(item_index))
        test = test.dropna(subset=["position"])
        test["percentile"] = percentile[test["position"].astype(int).to_numpy()]
        per_user = test.groupby("user")["percentile"].agg(["mean", "size"])
        rows = []
        for user, stats in per_user.iterrows():
            row = user_index.get(user, user_index.get(str(user)))
            if row is None or stats["size"] < 5 or np.isnan(trendiness[row]):
                continue
            rows.append((delta[row], gate[row], trendiness[row],
                         stats["mean"], length[row], stats["size"]))
        arr = np.array(rows, dtype=float)
        print()
        print("=" * 78)
        print("4. delta_u có lặp lại ở kỳ sau không (train so với test)")
        print("=" * 78)
        if len(arr) < 50:
            print("  quá ít người đọc có >= 5 mục test — không đo được")
        else:
            gen["n_users"] = int(len(arr))
            gen["rho_delta_trendiness_train"] = spearman(arr[:, 0], arr[:, 2])
            gen["rho_delta_trendiness_test"] = spearman(arr[:, 0], arr[:, 3])
            gen["rho_gate_trendiness_test"] = spearman(arr[:, 1], arr[:, 3])
            gen["obs_median"] = float(np.median(arr[:, 4]))
            print(f"  n = {len(arr):,} người đọc có >= 5 mục test"
                  f" | quan sát train/người: trung vị {np.median(arr[:, 4]):.0f}")
            print(f"  rho(delta_u, độ nổi các mục TRAIN): "
                  f"{gen['rho_delta_trendiness_train']:+.3f}")
            print(f"  rho(delta_u, độ nổi các mục TEST ): "
                  f"{gen['rho_delta_trendiness_test']:+.3f}")
            print(f"  {'quan sát train':<16}{'số người':>10}{'rho với TEST':>16}")
            for low, high in ((0, 20), (20, 50), (50, 150), (150, 10 ** 9)):
                mask = (arr[:, 4] >= low) & (arr[:, 4] < high)
                if mask.sum() < 30:
                    continue
                label = f"{low}-{high - 1}" if high < 10 ** 9 else f">={low}"
                value = spearman(arr[mask, 0], arr[mask, 3])
                gen[f"rho_test_obs_{label}"] = value
                print(f"  {label:<16}{int(mask.sum()):>10}{value:>+16.3f}")
            # Ô khớp cả hai chiều, để so giữa hai nhánh không bị lệch vì số mục
            # test khác nhau (test ít mục thì thước đo bên test cũng ồn hơn).
            cell = ((arr[:, 5] >= 10) & (arr[:, 5] < 30)
                    & (arr[:, 4] >= 10) & (arr[:, 4] < 50))
            if cell.sum() >= 30:
                value = spearman(arr[cell, 0], arr[cell, 3])
                gen["rho_test_matched_cell"] = value
                gen["n_matched_cell"] = int(cell.sum())
                print(f"  ô khớp 10-29 mục test & 10-49 quan sát train: "
                      f"n={int(cell.sum()):,}  rho {value:+.3f}")
            print("  -> tương quan train gần như nhau giữa hai nhánh mà tương quan")
            print("     test lệch hẳn thì chênh lệch là KHỚP QUÁ, không phải nét khác.")

    path = args.out_dir / "trend_gate_diagnosis.json"
    path.write_text(json.dumps({
        "n_users": int(n_users),
        "gate_mean": float(gate.mean()), "gate_std": float(gate.std()),
        "gate_quantiles": {str(p): float(v) for p, v
                           in zip([1, 10, 25, 50, 75, 90, 99], quantiles)},
        "fit_intercept": float(coef[0]), "fit_slope": float(coef[1]),
        "two_part_gate": bool(two_part),
        "delta_mean": float(delta.mean()), "delta_std": float(delta.std()),
        "base_mean": float(base.mean()), "cold_gate": float(cold),
        "rho_gate_length": spearman(gate, length),
        "rho_delta_length": spearman(delta, length),
        "rho_delta_trendiness": spearman(delta[valid], trendiness[valid]),
        "rho_gate_trendiness": spearman(gate[valid], trendiness[valid]),
        "rho_gate_cn": spearman(gate[ok], cn_share[ok]),
        "delta_generalization": gen,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
