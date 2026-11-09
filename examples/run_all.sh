#!/usr/bin/env bash
# --release checks published data/figures and launch control without physics runs.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
case "${1:-}" in
  "") tests=(a_restored_gauss b_py_pilot c_figures d_bond_factorization e_hierarchy_gpu f_release) ;;
  --release) tests=(c_figures f_release) ;;
  *) echo "usage: bash run_all.sh [--release]" >&2; exit 2 ;;
esac
mkdir -p _work/logs
status=0
for t in "${tests[@]}"; do
  if [ "$t" = e_hierarchy_gpu ] && ! "$PY" -c 'import torch' >/dev/null 2>&1; then
    echo "e_hierarchy_gpu          SKIP  PyTorch unavailable; install torch to run test (e)."
    continue
  fi
  if [ "$t" = e_hierarchy_gpu ]; then dev=(); else dev=(CUDA_VISIBLE_DEVICES=); fi
  start=$(date +%s)
  if env "${dev[@]}" "$PY" "smoke_$t.py" > "_work/logs/$t.log" 2>&1; then
    label=PASS
  else
    label=FAIL
    status=1
  fi
  result=$(grep -m1 -o 'RESULT:.*' "_work/logs/$t.log" || true)
  printf '%-24s %-5s %4ss   %s\n' "$t" "$label" "$(( $(date +%s) - start ))" "$result"
done
exit "$status"
