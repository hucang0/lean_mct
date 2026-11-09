# `hierarchy/`: stationary multi-point (GMCT) hierarchy solvers

GPU solvers for the long-time (plateau) limit of the generalized mode-coupling hierarchy, closed at order N, on
Percus–Yevick (PY) hard spheres and on the Mari–Kurchan (MK) liquid. They produce the critical packing fractions
φ_c(N), the frozen Perron constants and the linear-chain bounds of SM Sec. S3 (Tables S1–S5) and the PY reference
series N = 2–7 quoted in the Letter (Fig. 2(a,b); "0.567 at N = 7").

Requires PyTorch with CUDA (`run_phic.py`, `campaign.py`, `gmct_sym2_gpu.py`, `linear_chain_bound.py`,
`certified_liquid.py` and `run_linear.py` hard-code `cuda`; `gmct_dense_gpu.py` falls back to the CPU). All files
import each other from this folder; run them from anywhere with `python3 hierarchy/<script>.py ...`.

| file | role |
|---|---|
| `gmct_dense_gpu.py` | dense solver (fp64): PY statics and vertex, monotone Jacobi iteration from f ≡ 1, correlator (`corr`) and memory (`mem`) closures, bisection CLI |
| `gmct_sym2_gpu.py` | permutation-symmetric solver storing C(M+n−1, n) sorted tuples (N ≥ 5); `bisect` and `extrap` ((1−E)² extrapolation) modes |
| `linear_chain_bound.py` | Collatz–Wielandt lower bound φ_lin(N) (Theorem 3, evaluated in fp64) for PY |
| `certified_liquid.py` | certified-liquid criterion (Theorem 3, evaluated in fp64/fp32) for PY at given φ |
| `campaign.py` | PY: sequence of closure orders N on one grid, each seeded by the previous bracket |
| `mk_vertex.py` | MK statics (S = 1, c = Mayer function) and the vertices W̃_s (`Bself`) and W̃_coll (`Bcoll`) of SM Sec. S2 D, (q,p)-symmetrized; dense iteration with contraction rate E; Perron operator; ρ(T_N) |
| `run_phic.py` | MK bisection campaign φ_c(N) (`dense`: `corr`/`mem`; `sym2`: `corr`) → JSONL |
| `run_linear.py` | MK Perron constants on six grids, PY anchor, ρ(T_N) for N = 2–5 with k-cutoffs → `results/linear_diag.json` |

Outputs are appended to the JSONL file given by `--out` (one `solve` record per density, one `result` record with
the bracket `(lo, hi)` per order). The values used in the paper are collected in `../data/numerics/`
(`results/headline.json`: all φ_c, h → 0 extrapolations, Perron grids and deviations; `figures/F4_grid_convergence.json`;
`results/critical_eigvec.json`) and, for the PY series, as the raw JSONL records in `../data/numerics/py_reference/`.

## MK liquid, benchmark grid (M, q_max) = (100, 40): SM Table S1 (upper block), Table S5, Letter Fig. 2(a,b)

`--variant Bself` is the derived single-leg vertex W̃_s, `Bcoll` the collective variant W̃_coll. With `--closure mem`
the code order N evolves f_1..f_N and carries the published label N+1 (`--mem_general_N2` uses the general branch at
code order 2, i.e. label 3). Each order is seeded with the previous order's liquid bound; `hi = lo * hi_factor`.

    mkdir -p results
    # W_s, correlator closure: N = 2, 3, 4 (dense fp64), 5 (sym2 fp64), 6 (sym2 fp32)
    python3 run_phic.py --variant Bself --closure corr --Ns 2,3,4 --lo 2.7 --hi 2.9 --out results/phic_Bself_corr_M100.jsonl
    python3 run_phic.py --variant Bself --closure corr --Ns 5 --solver sym2 --lo 3.69 --hi_factor 1.25 --out results/phic_Bself_corr_M100_N5.jsonl
    python3 run_phic.py --variant Bself --closure corr --Ns 6 --solver sym2 --dtype float32 --conv_tol 5e-7 --warm \
        --lo 4.036 --hi_factor 1.25 --tol 0.03 --out results/phic_Bself_corr_M100_N6.jsonl
    # W_s, memory closure: labels 3, 4 and 5
    python3 run_phic.py --variant Bself --closure mem --mem_general_N2 --Ns 2,3 --lo 2.79 --hi 3.6 --out results/phic_Bself_mem_M100.jsonl
    python3 run_phic.py --variant Bself --closure mem --Ns 4 --lo 3.73 --hi_factor 1.25 --out results/phic_Bself_mem_M100_N4.jsonl
    # W_coll
    python3 run_phic.py --variant Bcoll --closure corr --Ns 2,3,4 --lo 2.3 --hi 2.5 --out results/phic_Bcoll_corr_M100.jsonl
    python3 run_phic.py --variant Bcoll --closure corr --Ns 5 --solver sym2 --lo 3.018 --hi_factor 1.25 --out results/phic_Bcoll_corr_M100_N5.jsonl
    python3 run_phic.py --variant Bcoll --closure corr --Ns 6 --solver sym2 --dtype float32 --conv_tol 5e-7 --warm \
        --lo 3.3245 --hi_factor 1.25 --tol 0.03 --out results/phic_Bcoll_corr_M100_N6.jsonl
    python3 run_phic.py --variant Bcoll --closure mem --mem_general_N2 --Ns 2,3 --lo 2.41 --hi 3.2 --out results/phic_Bcoll_mem_M100.jsonl

Expected brackets (W̃_s, corr): N = 2 (2.796484, 2.796680), 3 (3.257126, 3.257228), 4 (3.689037, 3.689156),
5 (4.098436, 4.098549), 6 (4.477437, 4.493203). The first two are checked by `../examples/smoke_e_hierarchy_gpu.py`.

## MK grid study: SM Table S1 (lower rows), Table S2, Fig. S2

    for G in "200 40" "400 40" "100 80" "200 80"; do set -- $G
      python3 run_phic.py --variant Bself --closure corr --Ns 2,3 --M $1 --qmax $2 --lo 2.5 --hi 3.1 --out results/phic_Bself_corr_M$1_q$2.jsonl
      python3 run_phic.py --variant Bcoll --closure corr --Ns 2,3 --M $1 --qmax $2 --lo 2.1 --hi 2.7 --out results/phic_Bcoll_corr_M$1_q$2.jsonl
    done
    # N = 4 on the refined grids (bisection tolerance 1e-3)
    python3 run_phic.py --variant Bself --closure corr --Ns 4 --M 200 --qmax 40 --lo 3.64  --hi 3.70  --tol 1e-3 --out results/phic_Bself_corr_M200_q40_N4.jsonl
    python3 run_phic.py --variant Bcoll --closure corr --Ns 4 --M 200 --qmax 40 --lo 2.98  --hi 3.03  --tol 1e-3 --out results/phic_Bcoll_corr_M200_q40_N4.jsonl
    python3 run_phic.py --variant Bself --closure corr --Ns 4 --M 400 --qmax 40 --lo 3.655 --hi 3.680 --tol 1e-3 --out results/phic_Bself_corr_M400_q40_N4.jsonl
    python3 run_phic.py --variant Bcoll --closure corr --Ns 4 --M 400 --qmax 40 --lo 2.990 --hi 3.010 --tol 1e-3 --out results/phic_Bcoll_corr_M400_q40_N4.jsonl

The Aitken h → 0 extrapolation of Table S2 is computed from these brackets (stored in `headline.json`,
`h_to_0_extrapolation`).

## MK Perron constants and transfer matrices: SM Tables S3, S4 (N ≤ 5)

    python3 run_linear.py              # -> results/linear_diag.json (about 7 min)

This gives λ_P, C_1 = w_max/(φ ω_min), w_max and k_peak on the grids (100,40), (200,40), (400,40), (800,40), (100,80),
(200,80) for both vertices, the PY values at φ_c(2) = 0.515914 (Table S3, bottom rows), and ρ(T_N), r_N and
φ_lin(N) for N = 2–5 (Table S4). The entries N = 6–8 of Table S4 and the multi-grid shape factors of Table S6 came from
a fused-operator variant of `mk_vertex.rho_TN` that is not part of this repository.

## PY hard spheres, N = 2–7: SM Tables S4 (PY row) and S5, Letter text

    python3 gmct_dense_gpu.py --M 100 --N 2 --closure corr --lo 0.5155 --hi 0.5165 --tol 4e-6 --out results/py_corr_M100.jsonl
    python3 gmct_dense_gpu.py --M 100 --N 3 --closure corr --lo 0.5315 --hi 0.5325 --tol 4e-6 --out results/py_corr_M100.jsonl
    python3 campaign.py --M 100 --closure corr --Ns 4,5 --lo 0.5318 --hi 0.6 --out results/py_corr_M100.jsonl
    # N = 6, 7: glass side by (1-E)^2 extrapolation from glass points, liquid side certified (Theorem 3)
    python3 gmct_sym2_gpu.py --M 100 --N 6 --lo 0.556 --hi 0.600 --mode extrap --npts 4 --warm --out results/py_sym2_M100.jsonl
    python3 gmct_sym2_gpu.py --M 100 --N 7 --lo 0.563 --hi 0.600 --mode extrap --npts 4 --warm --out results/py_sym2_M100.jsonl
    python3 certified_liquid.py --M 100 --N 6 --phis 0.561 --out results/py_certified.jsonl
    python3 certified_liquid.py --M 100 --N 7 --phis 0.564,0.5665 --dtype float32 --out results/py_certified.jsonl
    # rigorous linear-chain lower bounds phi_lin(N)
    for N in 2 3 4 5 6; do python3 linear_chain_bound.py --M 100 --N $N --out results/py_linear_chain_M100.jsonl; done

Glass points closer to the transition (N = 6: 0.5645, 0.56325, 0.562125, 0.5615625; N = 7: 0.568125) were added with
further single-point runs, `gmct_sym2_gpu.py --N <N> --lo <phi> --hi <phi> --mode extrap --npts 1 --warm`.
Results (`../data/numerics/py_reference/`): φ_c = 0.515912 (N = 2; MCT value 0.51591213), 0.531885 (3),
0.544170 (4), 0.553815 (5), (0.5610, 0.5616) for N = 6 and (0.5665, 0.5681) for N = 7, quoted as 0.5613(3) and
0.5671(10) in Table S5. φ_lin(N) = 0.2854, 0.3074, 0.3487, 0.3797, 0.4033 for N = 2–6.

## Runtimes and GPU memory

Measured on one NVIDIA RTX PRO 6000 Blackwell (96 GB), PyTorch 2.9.1 + CUDA 12.8.

| case | solver / precision | peak GPU memory | wall time |
|---|---|---|---|
| MK, N = 2–4, (100,40) | dense fp64 | < 0.1 GB | 10 s (N = 2) to 15 min (N = 4) |
| MK, N = 5, (100,40) | sym2 fp64 | 1.3 GB | 1.2–1.8 h |
| MK, N = 6, (100,40) | sym2 fp32 | 3.7 GB | 0.6–1.5 h |
| MK, memory closure label 5 | dense fp64 | 4.9 GB | 1.9 h |
| MK, N = 2, 3 on (200,40)–(200,80) | dense fp64 | ≤ 1 GB | 5–17 min per grid |
| MK, N = 4 on (400,40) | dense fp64 | 3.6 GB | ≈ 1 h |
| `run_linear.py` | dense + sym2 fp64 | a few GB | 7 min |
| PY, N = 2–5 | dense fp64 | < 1 GB | seconds to minutes |
| PY, N = 6 | sym2 fp64 | ≈ 8 GB | 8–85 min per density |
| PY, N = 7 | sym2 fp64 | ≈ 60 GB | 1–2 h per density |
| PY certified liquid, N = 7 | sym2 fp32 | ≈ 30 GB | 1–1.7 h per density |

**Shared GPUs.** N = 7 needs most of a 96 GB card; check `nvidia-smi` before launching. To cap a run, call
`torch.cuda.set_per_process_memory_fraction(...)` before the solver starts, as `../examples/smoke_e_hierarchy_gpu.py`
does (8 GB cap, N = 2, 3 only).
