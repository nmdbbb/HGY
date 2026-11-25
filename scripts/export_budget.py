"""Rút đường huấn luyện và bậc ngân sách epoch ra CSV, đọc TỪ LOG của các lượt.

    python scripts/export_budget.py --logs <log...> --out <thư mục>

Vì sao đọc log mà không đọc JSON: `run_neural_ladder.py` chỉ ghi **một** con số
cuối cho mỗi mô hình, và mỗi lượt ngân sách lại ghi đè lên cùng tệp. Con số của
ngân sách 8 và 20 vì thế không còn trong JSON nữa — nhưng chúng còn nguyên trong
log, dòng từng epoch một. Phát hiện cần tới chúng là: *ngân sách nhỏ không chỉ hạ
mức, nó đổi THỨ HẠNG*, và muốn nói câu đó thì phải có cả ba bậc cạnh nhau.

Script này không tính lại gì, không làm tròn lại gì: nó chỉ đọc đúng những dòng
mà quá trình huấn luyện đã in ra, nên mọi số trong hình truy được về một dòng log
cụ thể. Dạng dòng được khớp:

    GRU4Rec epoch 320/400 loss 0.0121 | val NDCG 0.1732 *
    -> GRU4Rec: val chọn epoch 320, test NDCG 0.0744 | Recall 0.0929

Hai tệp ra:

* `budget.csv` — một dòng cho mỗi (lượt, mô hình): ngân sách, epoch được chọn,
  NDCG/Recall trên test, và cờ `o_bien` khi epoch được chọn **là** epoch cuối
  của ngân sách (lúc đó con số là chặn dưới, không phải giá trị của mô hình).
* `curve.csv` — đường loss train và val NDCG theo epoch, để thấy chỗ loss train
  bẹt xong rất lâu trước khi val bẹt.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

EPOCH = re.compile(
    r"^\s*(?P<model>[A-Za-z0-9-]+(?:-style)?) epoch (?P<epoch>\d+)/"
    r"(?P<budget>\d+) loss (?P<loss>[\d.]+)"
    r"(?: \| val NDCG (?P<val>[\d.]+))?")
CHOSEN = re.compile(
    r"^\s*-> (?P<model>[A-Za-z0-9-]+(?:-style)?): val chọn epoch "
    r"(?P<epoch>\d+), test NDCG (?P<ndcg>[\d.]+) \| Recall (?P<recall>[\d.]+)")
STOPPED = re.compile(r"^\s*dừng sớm: (?P<stale>\d+) lần chấm")


def parse(path):
    """Một log có thể chứa nhiều mô hình; khoá theo (mô hình, ngân sách)."""
    curve, final, stopped = {}, {}, {}
    text = path.read_text(encoding="utf-8", errors="replace")
    budget_of, last_model = {}, None
    for line in text.splitlines():
        hit = EPOCH.match(line)
        if hit:
            model = hit.group("model")
            budget = int(hit.group("budget"))
            budget_of[model] = budget
            last_model = model
            curve.setdefault((model, budget), []).append({
                "epoch": int(hit.group("epoch")),
                "loss": float(hit.group("loss")),
                "val": (float(hit.group("val"))
                        if hit.group("val") else None)})
            continue
        if STOPPED.match(line) and last_model is not None:
            stopped[(last_model, budget_of[last_model])] = True
            continue
        hit = CHOSEN.match(line)
        if hit:
            model = hit.group("model")
            budget = budget_of.get(model)
            if budget is None:
                continue
            final[(model, budget)] = {
                "chon": int(hit.group("epoch")),
                "ndcg": float(hit.group("ndcg")),
                "recall": float(hit.group("recall"))}
    return curve, final, stopped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    budget_rows, curve_rows = [], []
    for path in args.logs:
        if not path.exists():
            print(f"  bỏ qua (không có): {path}")
            continue
        curve, final, stopped = parse(path)
        for key, points in sorted(curve.items(), key=lambda kv: kv[0][1]):
            model, budget = key
            got = final.get(key)
            checked = [p for p in points if p["val"] is not None]
            if got:
                # Ở BIÊN nghĩa là val vẫn đang lên khi hết ngân sách. Chỉ đúng
                # khi lượt đó KHÔNG dừng sớm: dừng sớm thì epoch cuối chạy
                # được không phải epoch cuối của ngân sách.
                at_edge = (not stopped.get(key)
                           and got["chon"] == max(p["epoch"] for p in points))
                budget_rows.append({
                    "mo_hinh": model, "ngan_sach": budget,
                    "epoch_chon": got["chon"], "ndcg": got["ndcg"],
                    "recall": got["recall"],
                    "o_bien": "1" if at_edge else "0",
                    "lan_cham": len(checked),
                    "dung_som": "1" if stopped.get(key) else "0",
                    "log": path.name})
            for point in points:
                curve_rows.append({
                    "mo_hinh": model, "ngan_sach": budget,
                    "epoch": point["epoch"], "loss": point["loss"],
                    "val_ndcg": "" if point["val"] is None else point["val"],
                    "log": path.name})

    budget_rows.sort(key=lambda r: (r["mo_hinh"], r["ngan_sach"]))
    for name, rows, cols in (
            ("budget.csv", budget_rows,
             ["mo_hinh", "ngan_sach", "epoch_chon", "ndcg", "recall",
              "o_bien", "lan_cham", "dung_som", "log"]),
            ("curve.csv", curve_rows,
             ["mo_hinh", "ngan_sach", "epoch", "loss", "val_ndcg", "log"])):
        with (args.out / name).open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name}  {len(rows):,} dòng, {len(cols)} cột")

    edge = [r for r in budget_rows if r["o_bien"] == "1"]
    print(f"  {len(budget_rows)} (mô hình × ngân sách) | {len(edge)} ở biên: "
          + ", ".join(f"{r['mo_hinh']}@{r['ngan_sach']}" for r in edge))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
