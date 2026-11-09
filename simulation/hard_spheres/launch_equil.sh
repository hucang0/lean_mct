#!/usr/bin/env bash
set -euo pipefail
# Swap-MC equilibration (hs_equil.py): N=1000, 6 samples per density; N=2000, 3 samples at phi=0.585.
cd "$(dirname "$0")"                      # run from this folder; data/ and logs/ are created here
source ../_jobs.sh
PY=${PYTHON:-python3}                     # override: PYTHON=/path/to/python3 ./launch_equil.sh
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p data/eq logs
run () { # N phi seed neq
  local tag=$(printf "phi%.3f_N%d_s%d" $2 $1 $3)
  nohup nice -n 5 "$PY" -u hs_equil.py --N $1 --phi $2 --seed $3 --neq $4 --delta 0.10 --pswap 0.2 \
     --out data/eq/eq_${tag}.npz > logs/eq_${tag}.log 2>&1 &
  register_job "$!" "eq_${tag}"
}
for s in 1 2 3 4 5 6; do
  run 1000 0.500 $s 100000
  run 1000 0.550 $s 100000
  run 1000 0.570 $s 200000
  run 1000 0.585 $s 300000
  run 1000 0.600 $s 600000
done
for s in 1 2 3; do run 2000 0.585 $s 300000; done

# Block until all children finish; a failed job makes this launcher fail.
wait_jobs
