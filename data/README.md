# `data/`: archived figure data and selected table results

JSON, JSONL, CSV and text files only (no trajectories). Licensed CC BY 4.0 (`../DATA_LICENSE.md`).
Paths below are relative to this directory. Producer scripts and any missing collection drivers are identified below.

| file | content | produced by | paper |
|---|---|---|---|
| `numerics/results/headline.json` | MK critical points φ_c(N) for both vertices and closures, all grids; h → 0 extrapolations; frozen Perron constants on all grids; Perron-law deviations; PY reference ratios | `hierarchy/run_phic.py`, `run_linear.py` (collected) | Letter Fig. 2(a,b); Tables S1–S3, S5 |
| `numerics/results/critical_eigvec.json` | critical N = 2 eigenvectors, Perron vectors and plateaus f_1*(k) | `hierarchy/mk_vertex.py` (`perron`, `solve_dense`; driver not included) | Fig. S1 |
| `numerics/figures/F4_grid_convergence.json` | φ_c(2), φ_c(3) vs grid spacing h and q_max | `hierarchy/run_phic.py` (grid study) | Fig. S2, Table S2 |
| `numerics/py_reference/*.jsonl` | raw PY records: `validation` (N = 2, 3), `runs_M100_corr` (N = 4, 5), `runs_M100_mem`, `runs_sym2_M100` (N = 6, 7), `certified_liquid` (N = 2–7), `linear_chain_M100` (φ_lin, N = 2–6) | `hierarchy/gmct_dense_gpu.py`, `campaign.py`, `gmct_sym2_gpu.py`, `certified_liquid.py`, `linear_chain_bound.py` | Tables S4 (PY row), S5; Letter text |
| `simulation/results/curves_averaged.json` | MK planted MC, sample-averaged lag curves: MSD, α₂, F_s(k), C(π), pair/triple chain correlators (contact, random, ∥, ⊥), contact survival | `simulation/mk_planted_mc/analysis.py` | Letter Fig. 2(c,d); Figs. S3, S4; Table S11 |
| `simulation/results/plateau_table.json` | plateau heights f_s(k) at the flattest log-slope | `analysis.py` | Fig. S3(b), Table S10 |
| `simulation/reanalysis/conditional_summary.json` | bond-conditioned decomposition R_glob, R_marg, Γ, Im φ₂, contact persistence (values at τ₇, 10², 10⁴ and full curves) | `simulation/mk_planted_mc/conditional.py` | Letter Fig. 2(c,d); Table S12 |
| `simulation/hs_bond/summary_table.json`, `fig_curves.json`, `hs_phi*_N*_d*.json` | ordinary hard spheres: all bond observables at τ₇, t*, F_s = 0.45 and full curves, per (φ, N_p, δ) group (δ = −1: MD) | `simulation/hard_spheres/hs_summary.py` | Letter Fig. 2(d); Fig. S6, Table S14 |
| `simulation/hs_bond/mk_reanalysis_pearson.json` | MK trajectories re-analysed with the hard-sphere code; Pearson coefficients | `simulation/hard_spheres/mk_pearson.py` | Fig. S6(c), Table S14 |
| `simulation/hs_bond/eq_checks.json` | equilibration and crystallization checks | `simulation/hard_spheres/eq_checks.py` | Sec. S4 F |
| `simulation/hs_bond/tables.md`, `fig_panel_*.csv` | Markdown tables (Table S14 and controls); CSV panels of R vs F_s | `make_tables.py`, `make_fig.py` | Table S14 |
| `restore/sim_summary.json` | MK planted MC at ten densities φ = 1.8–3.6: plateau MSD, f_s(k), Gaussian and bias-corrected cage-mixture predictions, per run | `simulation/mk_planted_mc/collect_sim.py` | SM Fig. S5; Table S13 |
| `restore/gauss_curves.json`, `gauss_curves_dense.json` | Gaussian closure with (`brc`) and without (`diag`, Z ≡ 1) the bond hard core: φ_c, σ²(φ), MSD, f_G(k) | `restored_pair/gauss_curves.py` | SM Fig. S5(a,b); Tables S13, S15 |
| `restore/rescatter_series_gauss.json` | Gaussian folds vs exclusion order ℓ; diagonal hierarchy φ_c(N) | `restored_pair/rescatter_series.py` | Letter Figs. 1(c), 2(b); Table S16 |
| `restore/kres_series_n{1,2,3,5,10}.json` | k-resolved folds vs ℓ (bisection logs) | `restored_pair/kres_series.py` | Letter Fig. 2(b); Table S16 |
| `restore/vng_scan.json`, `vng_series.json`, `vng_single_fold.json` | static non-Gaussian (Bethe) cage: density scans, folds, ℓ truncations | `restored_pair/vng_scan.py`, `vng_series.py`, `vng_single_fold.py` | SM Fig. S5(a); Tables S13, S15, S16 |
| `restore/kres_bisect/kres_bisect_kcNone_M100_q40.json`, `kres_bisect_<φ>_kcNone_M100_q40.json` | k-resolved restored closure at ℓ → ∞: bisection log (bracket 1.78125 liquid / 1.7875 glass; 1.784375 unresolved after 1500 iterations) and the f(k) state at each trial density | `restored_pair/brc_kres.py bisect 1.70 1.80 --nmax 5` | Letter Fig. 2(b); Tables S15, S16 |
| `restore/plateau/plateau_step1_phi{1.72,1.80,2.00}.json`, `plateau_final.json` | per-run plateau observables and cage decomposition (f_meas, f_gauss, f_isomix, f_anisomix at k = 3, 5, 7, 10; lag curves) for φ = 1.72–2.0 and φ = 1.9–3.6 | `simulation/mk_planted_mc/plateau_analysis.py` | SM Sec. S4 D–E, Table S13 |
| `restore/diag_kres.json` | k-resolved diagonal N = 2 plateaus f(k) | `restored_pair/diag_kres.py` | SM Fig. S5(a,b); Table S13 |
| `brc/verify/fig18_digitized.json` | points read off Fig. 18 of Mari and Kurchan, J. Chem. Phys. 135, 124504 (2011) (numbers only) | digitized by hand | Fig. S7, Sec. S5 E |
| `brc/verify/fig18_curve_mine.dat` | the exact Gaussian replica curve I(m = 0.96, K)/[d(m − 1)], d = 3 | evaluation script not included | Fig. S7, Sec. S5 E |
| `py_pilot/*.json` | PY / Verlet–Weis Gaussian closures: folds and branches (`folds`, `summary_table`), MK checks (`mk_sanity`, `mk_paths_check`), structure checks, convergence (`conv_mz`), diagnostics (`diag_b`, `analysis`) | `py_pilot/*.py` | Table S17, Sec. S5 F |

"Collected" means that the values were gathered from the JSONL outputs of the solvers by a collection script that is
not part of this repository; every number in those files is printed by the solvers themselves.

Not shipped: the MK run, relaxation-time and fit tables (`run_table.json`, `tau_table.json`, behind Tables S7–S9)
and the factorization table (`factorization_table.json`); `simulation/mk_planted_mc/analysis.py` writes them from the
trajectories. Also not shipped: the MK transfer-matrix radii ρ(T_N) behind Tables S4 and S6 (`hierarchy/run_linear.py`
recomputes N = 2–5).

## R2 bond reanalysis (SM S4 F)

`simulation/r2_bond/` contains all nine archived R2 results: `comob.json`, `xi_fits.json`, `times.json`,
`triplets.json`, `packing.json`, `pack_thresholds.json`, `cna.json`, `tables_r2.md`, and `key_numbers.json`.
Their producers are the `r2_*.py` scripts in `../simulation/hard_spheres/`; see that folder's README for commands.
These immutable data are tracked by Git. Regenerated results go to the component's ignored `results/` directory.
Some JSON measurements use Python's `NaN` for undefined estimates; use `json.load` or handle these markers explicitly.
