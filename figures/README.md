# Manuscript figure scripts

Each script reads archived results from `../../data/`, resolved relative to its own location, and writes PDF/PNG
outputs next to itself. No simulation or physics solver is run. Python needs numpy, matplotlib and Pillow.
Numbering follows the Letter and its Supplemental Material.

| script | manuscript figure | inputs under `data/` |
|---|---|---|
| `letter/fig0_cartoon_v2.py` | Letter Fig. 1: schematic | annotations from Table S1 and exclusion-order series |
| `letter/fig1_runoff.py` | Letter Fig. 2(a–d): Perron test, N versus ℓ, MK contact ratios, HS versus MK | `numerics/results/headline.json`, `restore/*series*.json`, `simulation/results/curves_averaged.json`, `simulation/reanalysis/conditional_summary.json`, `simulation/hs_bond/fig_curves.json` |
| `sm/F1_statics.py` | Fig. S1: MK statics and eigenvectors | `numerics/results/critical_eigvec.json` |
| `sm/F4_grid.py` | Fig. S2: grid convergence | `numerics/figures/F4_grid_convergence.json`, `numerics/results/headline.json` |
| `sm/F5_simulation_Fs.py` | Fig. S3: self correlators and plateaus | `simulation/results/curves_averaged.json`, `simulation/results/plateau_table.json` |
| `sm/F6_factorization.py` | Fig. S4: factorization ratios and conditioned recoil | `simulation/results/curves_averaged.json`, `simulation/reanalysis/conditional_summary.json` |
| `sm/F9_restore_msd.py` | Fig. S5: MSD and cage-mixture comparison | `restore/gauss_curves_dense.json`, `restore/vng_scan.json`, `restore/diag_kres.json`, `restore/sim_summary.json` |
| `sm/F7_hs_vs_mk_bond.py` | Fig. S6: ordinary hard spheres versus MK | `simulation/results/curves_averaged.json`, `simulation/reanalysis/conditional_summary.json`, `simulation/hs_bond/fig_curves.json`, `simulation/hs_bond/mk_reanalysis_pearson.json` |
| `sm/F8_fig18_overlay.py` | Fig. S7: Gaussian replica curve comparison | `brc/verify/fig18_digitized.json`, `brc/verify/fig18_curve_mine.dat` |
| `letter/fig3_restore.py` | extra diagnostic overview, not a numbered manuscript figure | `restore/*.json` |

Run `bash examples/run_all.sh --release` from the repository root to regenerate all ten outputs and verify the
published R2 tables and launcher completion handling. Figure-file creation is checked by `smoke_c_figures.py`;
this does not by itself establish pixel identity across matplotlib/font versions.
