# `examples/`: smoke tests

Six short scripts that exercise every component and check its key numbers against the paper. Each prints `CHECK`
lines and ends with `RESULT: PASS` (exit code 0) or `RESULT: FAIL`. Reference outputs from the author's machine are in
`expected/` (timings and the trajectory-based values in (d) vary slightly between machines).

    bash run_all.sh --release             # data/figures and stub launch checks; no physics runs
    bash run_all.sh                       # full suite; numerical tests also run small simulations/solves
    python3 smoke_a_restored_gauss.py     # or one at a time, from any directory

| test | what it runs | key numbers | time |
|---|---|---|---|
| `smoke_a_restored_gauss.py` | `restored_pair/brc_gauss.py`: Gaussian restored-pair closure on MK | φ_c = 1.776241 (1/Z), 3.236939 (Z ≡ 1); MSD(φ = 2) = 0.119 | < 1 s |
| `smoke_b_py_pilot.py` | `py_pilot/mk_sanity.py`; PY Mangeat–Zamponi fold on a coarse grid (N = 2¹⁴, Δr = 2 × 10⁻³) and with `run_folds.fold` on the production grid | MK 1.776241 / 3.236939; PY MZ 0.51845 (both grids); VW MZ 0.53448 | 5 s |
| `smoke_c_figures.py` | two Letter figures, seven SM figures and an extra overview, from `data/` only | ten PDF + PNG pairs written | 7 s |
| `smoke_d_bond_factorization.py` | tiny fresh simulations: MK `mk_mc.py` → `analysis.py` → `conditional.py` and `mk_pearson.py` (N = 300, φ = 1.5); hard spheres `hs_equil.py` → `hs_dyn.py` → `hs_analysis.py` → `hs_summary.py` (N = 300, φ = 0.55) | MK R∥(τ₇) = 2.49 (paper 2.54), Re Γ∥ = 1.80 (1.81), two analysis codes agree to 10⁻⁹; HS τ₇ = 2.6 × 10³ (2.7 × 10³), R∥(τ₇) = 4.20 (4.07), Re Γ∥ = 2.64 (2.61) | 10 s |
| `smoke_e_hierarchy_gpu.py` | `py_compile` + `--help` of all solvers, import check; with a GPU (≥ 16 GB free; the process is capped at 8 GB): `run_phic.py` MK W̃_s N = 2, 3 on (100, 40) and `gmct_dense_gpu.py` PY N = 2; without a GPU: dense N = 2 on the CPU at φ = 2.79 / 2.80 | MK brackets (2.796484, 2.796680), (3.257126, 3.257228) = Table S1; PY φ_c(2) = 0.515912; peak GPU memory 0.025 GB | 12 s |
| `smoke_f_release.py` | regenerate R2 tables from archived data in an unrelated cwd; test all production launchers with success/failure stub jobs | archived tables match; every child waited for; failures propagated; no physics runs | seconds |

Tests (a)–(d) need only numpy, scipy, matplotlib, numba and Pillow; `run_all.sh` hides the GPU from them. Test (e)
needs PyTorch; the wrapper explicitly skips it when PyTorch is absent. Its CPU fallback still requires PyTorch. Logs and work files go to `_work/`; figure outputs, `py_pilot/results/` and numba caches are written next to their scripts (all git-ignored).
No trajectories are shipped, so test (d) generates its own (a few MB) in `_work/`.
