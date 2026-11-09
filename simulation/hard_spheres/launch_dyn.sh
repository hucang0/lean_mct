#!/usr/bin/env bash
set -euo pipefail
# Non-swap MC dynamics (hs_dyn.py) from swap-equilibrated starts. Production: delta = 0.05 (cube side 0.1 sigma_bar,
# Berthier et al. 2016 convention; acceptance 0.3-0.5), norig = 8 time origins, 12 saves per decade.
# Controls at phi = 0.585: delta = 0.10 (MK value) and 0.025; N = 2000.
cd "$(dirname "$0")"                      # run from this folder; data/ and logs/ are created here
source ../_jobs.sh
PY=${PYTHON:-python3}                     # override: PYTHON=/path/to/python3 ./launch_dyn.sh
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p data/dyn logs
run () { # N phi seed sweeps delta
  local eq=$(printf "data/eq/eq_phi%.3f_N%d_s%d.npz" $2 $1 $3)
  local tag=$(printf "phi%.3f_N%d_d%.3f_s%d" $2 $1 $5 $3)
  ( if [ ! -s "$eq" ]; then echo "Missing equilibrated start: $eq; run launch_equil.sh first" >&2; exit 1; fi
    nice -n 5 "$PY" -u hs_dyn.py --init $eq --sweeps $4 --delta $5 --norig 8 --seed $3 --ckpt 600 \
       --out data/dyn/run_${tag}.npz > logs/dyn_${tag}.log 2>&1 ) &
  register_job "$!" "dyn_${tag}"
}
for s in 1 2 3 4 5 6; do
  run 1000 0.500 $s 200000 0.05
  run 1000 0.550 $s 200000 0.05
  run 1000 0.570 $s 1000000 0.05
  run 1000 0.585 $s 4000000 0.05
  run 1000 0.600 $s 16000000 0.05
done
for s in 1 2 3; do
  run 2000 0.585 $s 4000000 0.05
  run 1000 0.585 $s 4000000 0.10
  run 1000 0.585 $s 10000000 0.025
done

# Block until all children finish; a failed job makes this launcher fail.
wait_jobs
