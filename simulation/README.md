# `simulation/`

Two independent simulation codes; each folder is self-contained apart from the shared launcher helper `_jobs.sh` (run its scripts from anywhere).

| folder | system | paper |
|---|---|---|
| `mk_planted_mc/` | Mari–Kurchan liquid, planted Monte Carlo, N_p = 1000 (2000), φ = 0.8–3.6 | Letter Fig. 2(c,d); SM Sec. S4 A–E, Tables S7–S13, Figs. S3–S5 |
| `hard_spheres/` | ordinary polydisperse 3D hard spheres, swap-MC equilibration, MC and MD dynamics, φ = 0.50–0.60 | Letter Fig. 2(d); SM Sec. S4 F, Table S14, Fig. S6 |

`hard_spheres/analysis.py` is an identical copy of `mk_planted_mc/analysis.py`, so that both systems are analysed with
the same contact definition, k shells, binning and time origins. Trajectories are not part of the repository; the
launch scripts regenerate them (`mk_planted_mc/launch_mc.sh`, `hard_spheres/launch_*.sh`).

## Scripts in `hard_spheres/`

| script | role | SM |
|---|---|---|
| `hs_poly_sim.py` | swap-MC equilibration + Metropolis MC production across φ = 0.50–0.60 | Sec. S4 F, Table S14, Fig. S6 |
| `hs_poly_585_deep.py` | event-driven Newtonian MD, N_p = 2000 finite-size check, and MC step-size controls (δ = 0.025, 0.10 σ̄) at φ = 0.585, 0.55, 0.60 | Sec. S4 F, Table S14 |
| `hs_channels.py` | pair, triplet/loop, and shared-mobility fluctuation channel decomposition | Sec. S4 F, Table S14, Fig. S6 |
| `hs_referee_controls.py` | polydispersity, radical-Voronoi packing, and 1551 common-neighbour-analysis (CNA) icosahedral controls | Sec. S4 F |
