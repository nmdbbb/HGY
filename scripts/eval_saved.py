"""Đánh giá các blob đã lưu, không huấn luyện lại.

    python scripts/eval_saved.py two_tower_gated.pt
    python scripts/eval_saved.py two_tower_gated.pt --refit-trend

Có script này vì một thay đổi ở *vế phục vụ* không cần huấn luyện lại: trọng số
y nguyên, chỉ đường chấm điểm đổi. Trước đây muốn xem con số mới phải chạy lại
44 phút, nên mỗi lần sửa cách phục vụ là một lần tốn vô ích — và tệ hơn, là một
lần đổi cả seed, làm phép so hết còn ghép cặp.

`--refit-trend` tính lại đường khớp `g_u ~ log(1+độ dài lịch sử)` rồi ghi vào
blob. Dùng cho blob lưu trước khi `fit_trend_prior` tồn tại; đường khớp tính từ
chính trọng số và lịch sử trong blob, đều là train, nên không có đường nào cho
một con số của test lọt vào.

Hai lát quan trọng nhất ở đây là `new_user` (người đọc mới, nơi cổng đang mất
điểm) và `all`. Các lát khác không thể đổi vì chúng dùng `g_u` riêng của từng
người, không đi qua đường khớp — in ra để chứng minh đúng điều đó.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(min(4, torch.get_num_threads()))

from truyen_reco.evaluation import metrics  # noqa: E402
from truyen_reco.models.two_tower import (  # noqa: E402
    RoutedTwoTower, ServedTwoTower, fit_trend_prior)
from train_embedding import (  # noqa: E402
    build_histories, load_state, make_recommender, preprocess_slices,
    time_features)


def score_all(served, label, report, args, slices, combined, reverse,
              creator_map, country_map, item_index):
    """Chấm một mô hình trên mọi lát rồi in một hàng. Dùng chung cho blob đơn
    và cho bản định tuyến, để hai thứ đó đi qua đúng một đường chấm."""
    recommend = make_recommender(served, reverse, args.k)
    report[label] = {}
    line = f"{label:<34}"
    for slice_name, frame in slices.items():
        scored = metrics.evaluate(
            recommend, frame, combined, k=args.k, max_users=args.max_users,
            creator_map=creator_map, country_map=country_map,
            catalog_size=len(item_index))
        report[label][slice_name] = scored
        line += f"{scored['ndcg']:>12.4f}"
    print(line, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("blobs", nargs="+")
    parser.add_argument("--out-dir", default="data/processed", type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--refit-trend", action="store_true",
                        help="tính lại đường khớp cho người đọc mới và ghi "
                             "vào blob")
    parser.add_argument("--no-fit", action="store_true",
                        help="bỏ qua đường khớp, phát trung bình toàn dân như "
                             "bản đầu — để so cặp")
    parser.add_argument("--gate-sweep", action="store_true",
                        help="CHẨN ĐOÁN: quét cổng người-đọc-mới qua một dải "
                             "giá trị, chỉ lát new_user. Mục đích là trả lời "
                             "một câu có/không — 'có giá trị cổng nào lấy lại "
                             "được 0,3771 không' — chứ KHÔNG phải để chọn giá "
                             "trị: bản phục vụ vẫn dùng đường khớp trên train")
    parser.add_argument("--route", nargs=2, metavar=("COLD", "WARM"),
                        help="thêm một hàng ĐỊNH TUYẾN: blob COLD cho người "
                             "chưa có lịch sử, blob WARM cho người đã có. Đây "
                             "là thay đổi ở vế phục vụ nên so được ghép cặp "
                             "hoàn hảo với chính hai blob đó — cùng trọng số, "
                             "chỉ khác đường chấm.")
    args = parser.parse_args()

    started = time.monotonic()
    state = load_state(args.out_dir)
    item_index = state["item_index"]
    reverse = {v: k for k, v in item_index.items()}
    media, cutoff = state["media"], state["cutoff"]
    train, test = state["train"], state["test"]

    times, train_counts = time_features(media, train, item_index, cutoff)
    history, seed, holdout = build_histories(train, state["cold_users"],
                                             item_index)
    combined = dict(history)
    combined.update(seed)

    slices = dict(preprocess_slices(train, test))
    cold_users = state["cold_users"]
    slices["new_user"] = cold_users[
        [m in holdout.get(u, set())
         for u, m in zip(cold_users["user"], cold_users["media_id"])]]

    creator_map, country_map = {}, {}
    for row in media.itertuples(index=False):
        key = tuple(sorted(set(row.staff_art_ids) | set(row.staff_story_ids)))
        creator_map[row.media_id] = key or None
        country_map[row.media_id] = row.country

    def score_slice(served, slice_name):
        return metrics.evaluate(
            make_recommender(served, reverse, args.k), slices[slice_name],
            combined, k=args.k, max_users=args.max_users,
            creator_map=creator_map, country_map=country_map,
            catalog_size=len(item_index))

    if args.gate_sweep:
        # Câu hỏi: chỗ mất ở `new_user` có nằm trong cổng không. Nếu đỉnh của
        # đường quét vẫn thấp hơn hẳn bản không cổng thì KHÔNG — chỗ mất nằm ở
        # tháp gu, và việc chọn cổng nào cũng không lấy lại được.
        print("CHẨN ĐOÁN quét cổng người đọc mới — chỉ lát new_user")
        print("(không dùng để chọn giá trị; bản phục vụ giữ đường khớp train)")
        for name in args.blobs:
            path = (args.out_dir / name if not Path(name).is_absolute()
                    else Path(name))
            blob = torch.load(path, map_location="cpu", weights_only=False)
            served = ServedTwoTower(blob)
            if served.model.trend is None:
                # Blob không cổng = cổng hằng 1,0 vĩnh viễn. Vẫn phải in MỘT
                # dòng, vì nó là điểm neo để đặt đường quét vào không gian kết
                # quả; bỏ qua nó thì mất mốc so.
                scored = score_slice(served, "new_user")
                print(f"\n{Path(name).stem}  (không cổng = hằng 1,0)")
                print(f"  {'cổng':>8}{'NDCG new_user':>16}{'phủ catalog':>14}"
                      f"{'CN':>7}")
                print(f"  {1.0:>8.4f}{scored['ndcg']:>16.4f}"
                      f"{scored['coverage']:>14.4f}"
                      f"{scored.get('mix_CN', float('nan')):>7.2f}")
                del served, blob
                continue
            # Ghi đè phải đi vào ĐÚNG đường mà blob này phục vụ. Blob cũ đọc
            # `trend_fit`; blob cổng hai phần đọc `trend_base`, nên đặt
            # `trend_fit` lên nó là no-op và bảng quét ra phẳng — lỗi này đã
            # xảy ra thật và suýt bị đọc thành "mô hình mới bất biến với cổng".
            fitted = served.gate(None, 3)
            print(f"\n{Path(name).stem}  (đường phục vụ phát {fitted:.4f} "
                  f"ở 3 lượt chọn; {'khớp hậu kiểm' if served.legacy_gate else 'cổng hai phần'})")
            print(f"  {'cổng':>8}{'NDCG new_user':>16}{'phủ catalog':>14}"
                  f"{'CN':>7}")
            for value in (0.0, 0.5, 0.8937, 1.0, 1.3233, 1.5466, 2.0, 3.0, 5.0):
                if served.legacy_gate:
                    served.trend_fit = {"intercept": value, "slope": 0.0,
                                        "lo": value, "hi": value, "mean": value}
                else:
                    with torch.no_grad():
                        served.model.trend_base.weight.zero_()
                        served.model.trend_base.bias.fill_(value)
                assert abs(served.gate(None, 3) - value) < 1e-5, "ghi đè cổng thất bại"
                scored = score_slice(served, "new_user")
                mark = "  <- đường phục vụ" if abs(value - fitted) < 0.02 else ""
                print(f"  {value:>8.4f}{scored['ndcg']:>16.4f}"
                      f"{scored['coverage']:>14.4f}"
                      f"{scored.get('mix_CN', float('nan')):>7.2f}{mark}")
            del served, blob
        print(f"\n({(time.monotonic() - started) / 60:.1f} phút)")
        return

    header = f"{'blob':<34}" + "".join(f"{s:>12}" for s in slices)
    print("=" * len(header))
    print(f"NDCG@{args.k} — blob đã lưu, chấm toàn danh mục")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    report = {}
    for name in args.blobs:
        path = args.out_dir / name if not Path(name).is_absolute() else Path(name)
        blob = torch.load(path, map_location="cpu", weights_only=False)

        if args.refit_trend and blob["config"].get("trend_gate"):
            fit = fit_trend_prior(
                blob["state_dict"]["trend.weight"], blob["hist_index"])
            blob["trend_fit"] = fit
            torch.save(blob, path)
            print(f"  khớp lại: g_u ≈ {fit['intercept']:.4f} "
                  f"{fit['slope']:+.4f}·log(1+n), cắt vào "
                  f"[{fit['lo']:.3f}, {fit['hi']:.3f}], "
                  f"trung bình cũ {fit['mean']:.4f}")
        if args.no_fit:
            blob = dict(blob)
            blob["trend_fit"] = None

        served = ServedTwoTower(blob)
        # Nhãn phải nói đúng đường nào đang phục vụ, vì ba đường cho ba số
        # khác nhau ở người đọc mới: không cổng / trung bình toàn dân (blob cũ
        # chưa khớp) / hàm độ dài lịch sử (cổng hai phần, học trong lúc train).
        if served.model.trend is None:
            label = Path(name).stem
        elif served.legacy_gate and not served.trend_fit:
            label = Path(name).stem + " (trung bình)"
        else:
            label = Path(name).stem + (" (khớp hậu kiểm)" if served.legacy_gate
                                       else " (cổng hai phần)")
            sample = [(n, served.gate(None, n)) for n in (1, 3, 5)]
            print(f"  {label}: cổng phát cho người mới " + ", ".join(
                f"{n} lượt chọn → {g:.4f}" for n, g in sample))

        score_all(served, label, report, args, slices, combined, reverse,
                  creator_map, country_map, item_index)
        del served, blob

    if args.route:
        cold_path, warm_path = (
            args.out_dir / n if not Path(n).is_absolute() else Path(n)
            for n in args.route)
        routed = RoutedTwoTower(ServedTwoTower.load(cold_path),
                                ServedTwoTower.load(warm_path))
        label = (f"ĐỊNH TUYẾN: {cold_path.stem[len('two_tower_'):]}"
                 f" → {warm_path.stem[len('two_tower_'):]}")
        score_all(routed, label, report, args, slices, combined, reverse,
                  creator_map, country_map, item_index)
        del routed

    print("\n" + "=" * len(header))
    print("NGOÀI ĐỘ CHÍNH XÁC (lát 'all')")
    print("=" * len(header))
    print(f"{'blob':<34}{'phủ catalog':>14}{'trùng tác giả':>15}"
          f"{'KR':>7}{'CN':>7}")
    for label, scored in report.items():
        row = scored["all"]
        print(f"{label:<34}{row['coverage']:>14.4f}"
              f"{row['creator_conc']:>15.3f}"
              f"{row.get('mix_KR', float('nan')):>7.2f}{row.get('mix_CN', float('nan')):>7.2f}")

    path = args.out_dir / "eval_saved.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1,
                               default=float), encoding="utf-8")
    print(f"\n-> {path}   ({(time.monotonic() - started) / 60:.1f} phút)")


if __name__ == "__main__":
    main()
