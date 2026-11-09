# More particles or the pair? An exact test of how to improve mode-coupling theory

Code and data for the Letter by Hu Cang (University of California, Irvine) and its Supplemental Material (SM).
Figure numbering follows the Letter (two figures) and its Supplemental Material (seven figures).

The wavevector-diagonal hierarchy dilutes caging memory as particle order increases. On the Mari–Kurchan (MK)
liquid with exactly known equilibrium statics, orders N = 2–6 move the predicted arrest from φ_c = 2.80 to 4.49,
away from the Gaussian mean-field benchmark φ_G = 1.78. Restoring pair impenetrability at N = 2 reverses that drift.
Its Gaussian MSD reduction recovers the replica/cavity equation (φ_G = 1.776) and predicts the mean plateau MSD at
eight arrested densities within 5%, without fitted parameters. The wavevector-resolved and static closures are
reported separately; their assumptions and plateau predictions should not be conflated with the Gaussian reduction.

## Paper item → script → data file

Paths in the last column are relative to `data/`. Script paths are relative to the repository root.

| paper item | scripts | archived data |
|---|---|---|
| Letter Fig. 1 (schematic) | `figures/letter/fig0_cartoon_v2.py` | annotations from Table S1 and the exclusion-order series |
| Letter Fig. 2(a,b); Tables S1, S5 | `hierarchy/run_phic.py`; `figures/letter/fig1_runoff.py` | `numerics/results/headline.json`, `restore/kres_series_n*.json`, `restore/rescatter_series_gauss.json` |
| Letter PY series N = 2–7; Table S5 | `hierarchy/gmct_dense_gpu.py`, `campaign.py`, `gmct_sym2_gpu.py`, `certified_liquid.py`; `hierarchy/py_extrapolation/` | `numerics/py_reference/*.jsonl`; extrapolation data at `hierarchy/py_extrapolation/results/n058.json` outside `data/` |
| Letter Fig. 2(c,d); Tables S11, S12, S14 | `simulation/mk_planted_mc/`, `simulation/hard_spheres/`; `figures/letter/fig1_runoff.py` | `simulation/results/curves_averaged.json`, `simulation/reanalysis/conditional_summary.json`, `simulation/hs_bond/*` |
| Fig. S1 | `figures/sm/F1_statics.py` | `numerics/results/critical_eigvec.json` |
| Fig. S2; Table S2 | `hierarchy/run_phic.py`; `figures/sm/F4_grid.py` | `numerics/figures/F4_grid_convergence.json`, `numerics/results/headline.json` |
| Table S3 | `hierarchy/run_linear.py` | `numerics/results/headline.json` (`perron_grids`) |
| Table S4 | `hierarchy/run_linear.py` (MK N ≤ 5), `linear_chain_bound.py` (PY) | `numerics/py_reference/linear_chain_M100.jsonl`; MK radii not archived |
| Tables S7–S9 | `simulation/mk_planted_mc/analysis.py` | run/relaxation-time/fit tables not archived; regenerate from trajectories |
| Fig. S3; Table S10 | `figures/sm/F5_simulation_Fs.py` | `simulation/results/curves_averaged.json`, `simulation/results/plateau_table.json` |
| Fig. S4 | `figures/sm/F6_factorization.py` | `simulation/results/curves_averaged.json` |
| Fig. S5; Table S13; Sec. S4 D–E | `restored_pair/`; `simulation/mk_planted_mc/collect_sim.py`, `plateau_analysis.py`; `figures/sm/F9_restore_msd.py` | `restore/gauss_curves_dense.json`, `restore/vng_scan.json`, `restore/diag_kres.json`, `restore/sim_summary.json`, `restore/plateau/*.json` |
| Fig. S6; Table S14; Sec. S4 F | `simulation/hard_spheres/`; `figures/sm/F7_hs_vs_mk_bond.py` | `simulation/hs_bond/*` |
| Triplet, co-mobility, packing and CNA analysis (S4 F) | `simulation/hard_spheres/r2_*.py` | `simulation/r2_bond/*` |
| Tables S15–S16; Sec. S5 B–D; Letter Fig. 2(b) | `restored_pair/brc_gauss.py`, `brc_kres.py`, `kres_series.py`, `rescatter_series.py`, `vng*.py` | `restore/*.json`, `restore/kres_bisect/*.json` (ℓ → ∞, 1.784(3)) |
| Fig. S7; Sec. S5 E | `figures/sm/F8_fig18_overlay.py` | `brc/verify/fig18_digitized.json`, `brc/verify/fig18_curve_mine.dat` |
| Table S17; Sec. S5 F | `py_pilot/run_folds.py`, `make_summary.py` | `py_pilot/folds.json`, `py_pilot/summary_table.json` |
| Formal theorems; Sec. S1 B–D | `lean/verify.sh` | pinned Lean/Mathlib sources in `lean/` |

`data/README.md` describes the archived results and the missing raw outputs. Trajectories are not included.
`figures/letter/fig3_restore.py` is retained as an extra diagnostic overview, not a numbered Letter figure.

## Layout

    lean/          Lean 4 proofs, pinned toolchain/Mathlib, verify.sh
    hierarchy/     PY/MK hierarchy solvers, bisection drivers, Perron and chain bounds
    restored_pair/ Gaussian, k-resolved and static pair closures; exclusion-order series
    simulation/    MK planted Monte Carlo, ordinary hard spheres, bond reanalysis
    py_pilot/      PY / Verlet–Weis Gaussian pair closures
    data/          archived aggregate results (CC BY 4.0); no trajectories
    figures/       manuscript plots and one extra overview, reading data/ only
    examples/      six smoke tests, including release checks without physics runs

Run commands from the indicated component folder. Production scripts resolve their default data paths relative
to their own location. `hierarchy/py_extrapolation/check_mode.py` expects its component folder as the working directory.
The vendored `simulation/hard_spheres/lib/` helper routines keep the R2 analysis self-contained.

## Quick start

    python3 -m pip install -r requirements.txt
    bash examples/run_all.sh --release          # figures + published R2 tables + stub launcher tests; no physics runs
    python3 figures/letter/fig1_runoff.py        # Letter Fig. 2 (four panels)
    python3 figures/sm/F9_restore_msd.py         # SM Fig. S5

For all numerical smoke tests (including small new simulations and hierarchy solves):

    bash examples/run_all.sh

Test (e) is explicitly skipped if PyTorch is unavailable. Install PyTorch separately to exercise the hierarchy,
including its CPU fallback. `environment.yml` provides the tested conda setup with PyTorch; the tested CUDA wheel
configuration is documented in `requirements.txt`. GPU production runs require a CUDA-enabled PyTorch installation.

For formal verification, install [elan](https://github.com/leanprover/elan), then:

    cd lean && lake exe cache get && ./verify.sh

## Requirements and runtime

- Python ≥ 3.10; numpy, scipy, matplotlib, numba and Pillow. Paper environment: Python 3.10.19, numpy 2.2.6,
  scipy 1.15.2, matplotlib 3.10.8, numba 0.62.1, Pillow 12.3.0.
- Optional hierarchy dependency: PyTorch (paper environment: 2.9.1 + CUDA 12.8). Most production drivers require CUDA;
  `gmct_dense_gpu.py` and smoke test (e) support a CPU fallback.
- Lean toolchain: `leanprover/lean4:v4.30.0`, with the pinned Mathlib commit fetched by Lake.

| component | approximate runtime |
|---|---|
| release checks | seconds; no simulation or solver runs |
| complete smoke suite | about a minute, depending on compilation and hardware |
| figure scripts | about 1 s per figure |
| Lean verification | about 20 s with Mathlib cache; initial cache download about 5 GB |
| Gaussian closures | seconds |
| static cage / k-resolved scans | tens of minutes / hours |
| MK production | 53 single-core runs, typically 1–2 h each; parallel within each A/B batch |
| hard-sphere production | 33 equilibration, 39 MC-dynamics and 12 MD jobs; parallel within each stage |
| hierarchy production | seconds to hours; PY N = 7 needs about 60 GB GPU memory |

Production launchers wait for every child and return nonzero if any child fails. Use `&&` between stages, as shown
in the component READMEs. Logs are written under each component's `logs/` folder.

The paper's hierarchy results used an NVIDIA RTX PRO 6000 Blackwell (96 GB). Smoke test (e) caps GPU memory at 8 GB
and uses CUDA only with at least 16 GB free. The default release checks use the CPU.

## How to cite

A public manuscript identifier is not recorded in this release. The citation below uses the current manuscript title;
add the final publication details when available:

> H. Cang, “More particles or the pair? An exact test of how to improve mode-coupling theory” (2026), manuscript.

    @unpublished{Cang2026,
      author = {Cang, Hu},
      title  = {More particles or the pair? An exact test of how to improve mode-coupling theory},
      year   = {2026},
      note   = {Manuscript}
    }

Code and data: <https://github.com/hucang0/lean_mct>. Metadata: `CITATION.cff`.

## License

Code: MIT (`LICENSE`). Archived data in `data/`: CC BY 4.0 (`DATA_LICENSE.md`).
