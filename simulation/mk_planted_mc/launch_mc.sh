#!/usr/bin/env bash
set -euo pipefail
# Planted Monte Carlo of the Mari-Kurchan model (mk_mc.py): every production run of the Letter and SM.
#   bash launch_mc.sh A   # set A: 29 runs, phi = 0.80-2.00 (SM Sec. S4 A-E, Tables S7-S12, Letter Fig. 2(c,d))
#   bash launch_mc.sh B   # set B: 24 runs, phi = 1.90-3.60 (with set A at 1.80 and 2.00: SM Fig. S5, Table S13)
# Each run is one single-threaded background process; most take 1-2 h of wall time (all 53 runs in parallel: ~2 h).
# Output: set A in data/, set B in data_B/ (run_phi<phi>_N<N>_s<seed>.npz,
# 10-25 MB each, 0.9 GB in total); logs in logs/. Checkpoints every 10 min (--ckpt 600).
cd "$(dirname "$0")"
source ../_jobs.sh
PY=${PYTHON:-python3}
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
OUT=data
run () {  # N phi sweeps norig seed
  local N=$1 phi=$2 T=$3 norig=$4 seed=$5
  local tag=$(printf "phi%.2f_N%d_s%d" $phi $N $seed)
  nohup nice -n 5 "$PY" mk_mc.py --N $N --phi $phi --sweeps $T --norig $norig --seed $seed --delta 0.10 \
      --ckpt 600 --out $OUT/run_${tag}.npz > logs/run_${tag}.log 2>&1 &
  register_job "$!" "run_${tag}"
  echo "launched $tag T=$T norig=$norig pid=$!"
}
case "${1:-}" in
  A)
    for s in 1 2 3; do run 1000 0.80 20000   16 $s; done
    for s in 1 2 3; do run 1000 1.20 50000   16 $s; done
    for s in 1 2 3; do run 1000 1.50 300000  12 $s; done
    for s in 1 2;   do run 2000 1.50 100000  8  $s; done
    for s in 1 2 3; do run 1000 1.58 500000  8  $s; done
    for s in 1 2 3; do run 1000 1.65 1000000 8  $s; done
    for s in 1 2 3; do run 1000 1.72 1000000 8  $s; done
    for s in 1 2;   do run 1000 1.80 1000000 6  $s; done
    run 1000 1.80 3000000 4 3
    for s in 1 2 3; do run 1000 2.00 1000000 6  $s; done
    for s in 1 2;   do run 2000 1.80 300000  4  $s; done
    run 2000 2.00 200000 4 1
    ;;
  B)
    OUT=data_B
    for phi in 1.90 2.10 2.20 2.40 2.60 2.80 3.20 3.60; do
      for seed in 11 12 13; do run 1000 $phi 1000000 4 $seed; done
    done
    ;;
  *) echo "usage: bash launch_mc.sh A|B"; exit 1 ;;
esac

# Block until all children finish; a failed job makes this launcher fail.
wait_jobs
