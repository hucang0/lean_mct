#!/usr/bin/env bash
set -euo pipefail
# Event-driven MD control (hs_edmd.py) from the same swap-equilibrated starts; tick = 0.01 MD time units.
cd "$(dirname "$0")"                      # run from this folder; data/ and logs/ are created here
source ../_jobs.sh
PY=${PYTHON:-python3}                     # override: PYTHON=/path/to/python3 ./launch_md.sh
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p data/md logs
md () { # phi seed tmax
  local eq=$(printf "data/eq/eq_phi%.3f_N1000_s%d.npz" $1 $2)
  local tag=$(printf "phi%.3f_N1000_md_s%d" $1 $2)
  ( if [ ! -s "$eq" ]; then echo "Missing equilibrated start: $eq; run launch_equil.sh first" >&2; exit 1; fi
    nice -n 5 "$PY" -u hs_edmd.py --init $eq --tmax $3 --norig 8 --seed $2 --out data/md/run_${tag}.npz > logs/md_${tag}.log 2>&1 ) &
  register_job "$!" "md_${tag}"
}
for s in 1 2 3 4 5 6; do md 0.585 $s 4000; done
for s in 1 2 3; do md 0.550 $s 600; md 0.600 $s 40000; done

# Block until all children finish; a failed job makes this launcher fail.
wait_jobs
