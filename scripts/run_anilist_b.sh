#!/usr/bin/env bash
# Chạy thang phương pháp trên AniList theo giao thức B — cùng các script, cùng
# tham số mặc định như `data/processed_hgn_none` của Goodreads.
#
#   bash scripts/run_anilist_b.sh classic   # nhóm không dùng mạng nơ-ron
#   bash scripts/run_anilist_b.sh neural    # sáu mô hình nơ-ron
#
# Phần A của run_ladder_gaps (dự đoán điểm) bị bỏ: chỉ 21,6% dòng AniList có
# điểm, và nó thuộc một bài toán khác, không phải một biến thể của xếp hạng.
set -u
cd "$(dirname "$0")/.."
D=data/processed_anilist_b
M=data/processed/media.parquet
LOG=${LOG_DIR:-logs_anilist_b}
mkdir -p "$LOG"
PY="python -I -W ignore"

step() {  # step <tên> <lệnh...>
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] $name"
  "$@" > "$LOG/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] $name exit=$?"
}

case "${1:-classic}" in
  classic)
    step baselines $PY scripts/run_protocol_baselines.py --dir $D --media $M
    step tune      $PY scripts/tune_protocol.py --dir $D
    step ladder_bc $PY scripts/run_ladder_gaps.py --dir $D --media $M --parts b c
    step more      $PY scripts/run_more_methods.py --dir $D --media $M
    step onboard   $PY scripts/run_onboarding.py --dir $D --media $M \
                      --picks 1 2 3 5 10 20 --mechanism
    ;;
  neural)
    # Ngân sách tính theo SỐ BƯỚC CẬP NHẬT, không theo epoch: AniList-B có 8.724
    # chuỗi, Goodreads-B có 58.413, nên một epoch ở đây là ~6,7 lần ít bước hơn.
    # Lô chuỗi 128 (thay 256) và trần epoch cao hơn để ngân sách không phải là
    # thứ chặn; dừng sớm trên val vẫn quyết định chỗ dừng thật.
    N="$PY scripts/run_neural_ladder.py --dir $D --media $M --threads 8"
    step n_neumf  $N --models neumf    --epochs 120 --check-every 4  --patience 4 --out-name neural_neumf.json
    step n_gru    $N --models gru4rec  --epochs 800 --check-every 10 --patience 5 --seq-batch 128 --out-name neural_gru.json
    step n_sasrec $N --models sasrec   --epochs 300 --check-every 4  --patience 4 --seq-batch 128 --out-name neural_sasrec.json
    step n_nrms   $N --models nrms     --epochs 300 --check-every 4  --patience 4 --seq-batch 128 --out-name neural_nrms.json
    step n_cl4s   $N --models cl4srec  --epochs 300 --check-every 4  --patience 4 --seq-batch 128 --out-name neural_cl4srec.json
    step n_lgcn   $N --models lightgcn --epochs 100 --check-every 5  --patience 4 --batch 16384 --lr 3e-3 --out-name neural_lightgcn.json
    ;;
esac
echo "xong $1"
