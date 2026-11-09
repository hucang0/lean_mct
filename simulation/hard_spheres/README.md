# `simulation/hard_spheres/`: bond-resolved pair correlations in ordinary 3D hard spheres

Swap-equilibrated polydisperse hard spheres (23 % polydispersity, P(σ) ∝ σ⁻³, mean diameter 1; the model of
Berthier, Coslovich, Ninarello and Ozawa, PRL 116, 238002 (2016)), standard Metropolis MC dynamics and an event-driven
MD control, analysed with exactly the bond observables used for the Mari–Kurchan liquid. Supports Letter Fig. 2(d),
SM Sec. S4 F, Table S14 and Fig. S6. CPU only (numpy, scipy, numba; matplotlib for `make_fig.py`).

| file | role |
|---|---|
| `hs_core.py` | numba engine: polydisperse sizes, cell-built Verlet list (exact rebuild at skin/2), translation and swap moves, virtual-inflation pressure Z, BMCSL equation of state |
| `hs_struct.py` | Steinhardt Q₆, ten Wolde crystal fraction, scaled g(r) |
| `hs_equil.py`, `launch_equil.sh` | soft-quench start, then swap-MC equilibration → `data/eq/` |
| `hs_dyn.py`, `launch_dyn.sh` | standard (non-swap) MC dynamics with the MK schedule → `data/dyn/` |
| `hs_edmd.py`, `launch_md.sh` | event-driven Newtonian MD control → `data/md/` |
| `hs_analysis.py` | per-run bond observables (contact ∥/⊥/orientation-averaged, random pairs, non-contact shells, common-neighbour classes, k ≈ 5, 10, Pearson coefficients) → `curves_*.npz` |
| `hs_summary.py` | per-sample ratios, mean ± s.e.m., values at τ₇, t* and F_s(7) = 0.45 → `hs_*.json`, `summary_table.json`, `fig_curves.json` |
| `mk_pearson.py` | the same bond routines on MK trajectories (validation against SM Tables S11–S12, plus Pearson coefficients) → `mk_reanalysis_pearson.json` |
| `eq_checks.py` | equilibration and crystallization checks → `eq_checks.json` |
| `make_tables.py` | Markdown tables (Table S14 and controls) → `tables.md` |
| `make_fig.py` | comparison figure `figs/hs_vs_mk_bond.{pdf,png}` and the CSV panels `results/fig_panel_*.csv` |
| `analysis.py` | identical copy of `../mk_planted_mc/analysis.py` (k shells, log-lag binning, crossing times) |

All outputs used in the paper are in `../../data/simulation/hs_bond/` (`summary_table.json`, `fig_curves.json`,
`hs_phi*_N*_d*.json`, `mk_reanalysis_pearson.json`, `eq_checks.json`, `tables.md`, `fig_panel_*.csv`).
`make_fig.py` and `make_tables.py` read that folder by default (`--hs-results` to override).

## Reproduce

    ./launch_equil.sh && ./launch_dyn.sh && ./launch_md.sh
    # Each stage blocks until its jobs finish and returns nonzero on failure (33 + 39 + 12 jobs).
    NPROC=12 python3 hs_analysis.py data/curves data/dyn/run_*.npz data/md/run_*.npz
    python3 hs_summary.py data/curves results
    python3 mk_pearson.py results/mk_reanalysis_pearson.json \
        --mk-data ../mk_planted_mc/data --tau-table ../mk_planted_mc/results/tau_table.json
    python3 eq_checks.py results/eq_checks.json
    python3 make_tables.py --hs-results results > results/tables.md
    python3 make_fig.py 0.585 --hs-results results

`PYTHON=/path/to/python3 ./launch_equil.sh` selects the interpreter. Production settings: N_p = 1000 (six samples
per density) at φ = 0.50, 0.55, 0.57, 0.585, 0.60, plus N_p = 2000 (three samples) at 0.585; swap-MC equilibration of
1–6 × 10⁵ sweeps; MC dynamics with δ = 0.05 (controls δ = 0.10 and 0.025 at 0.585) for 2 × 10⁵ to 1.6 × 10⁷ sweeps
(the longest takes about 25 min); MD at φ = 0.55, 0.585, 0.60. Trajectories (0.3 GB) are not shipped.
`mk_pearson.py` needs the MK planted-MC trajectories of `../mk_planted_mc/` (set A) and their `tau_table.json`.

A seconds-long version of the whole chain (N = 300, φ = 0.55) is `../../examples/smoke_d_bond_factorization.py`;
it recovers τ₇ ≈ 2.6 × 10³ sweeps and R∥(τ₇) ≈ 4.2 (Table S14, N = 1000: 2.7 × 10³ and 4.07).

## Definitions (identical to SM Tables S11–S12)

- k shells: lattice vectors 2πn/L with |n|² closest to k ≈ 3, 5, 7, 10, at most 24 directions (mod sign).
- Centre-of-mass drift removed per frame. Contact pairs: r_ij(0) < 1.05 σ_ij, σ_ij = (σ_i + σ_j)/2, r̂ = (x_i − x_j)/r.
- φ₂ = ⟨e^{ik·(Δx_i − Δx_j)}⟩; R = Re φ₂/F_s(k₇)²; R_marg = Re(F^c_i F^c_j*)/F_s²; Γ = φ₂/(F^c_i F^c_j*);
  k = k₇ r̂ (∥), k ⊥ r̂ (⊥) or the lattice k₇ set (orientation average); 4000 random pairs per origin.
- Ratios per sample, then mean ± s.e.m. over samples; points with F_s ≤ 0.15 dropped.
- Times: τ₇ [F_s(k₇, τ₇) = 1/e]; t* (minimum of d ln MSD/d ln t in the β regime); t(F_s = 0.45), the MK φ = 2.0
  plateau value (matched-decorrelation comparison).
- corr∥, corr⊥: Pearson coefficient of the bond-projected displacements e·Δx_i, e·Δx_j over contact pairs.

## R2: triplets, co-mobility and packing (SM S4 F)

Published aggregates are in `../../data/simulation/r2_bond/`. To reproduce the tables using these archived results
(no trajectories or new simulations needed), run from this folder:

    python3 r2_tables.py                         # writes results/tables_r2.md and results/key_numbers.json
    python3 r2_figs.py                           # diagnostic plots under figs/

Readers prefer a regenerated file under `results/` when present and otherwise use its archived copy under `data/`.
Move old local results aside to explicitly recheck the published data. The archived files are never overwritten.

To regenerate the R2 aggregates, first complete the HS and MK production stages documented above, then:

    NPROC=8 python3 r2_comob.py all
    python3 r2_sum_comob.py
    NPROC=8 python3 r2_triplet.py all
    python3 r2_sum_trip.py
    NPROC=8 python3 r2_pack.py struct
    NPROC=8 python3 r2_pack.py dyn
    python3 r2_sum_pack.py
    NPROC=8 python3 r2_cna.py run
    python3 r2_sum_cna.py
    python3 r2_tables.py

Trajectory inputs default to this folder's `data/dyn/`, `data/md/` and `../mk_planted_mc/data/`.
Missing inputs fail with an error instead of silently generating an empty result; incomplete trajectory checkpoints
are rejected. Raw caches are written to `cache/` and excluded from Git.

Optional path overrides (environment variables):

| variable | default relative to this folder |
|---|---|
| `R2_HS_DATA` | `data/dyn` |
| `R2_HS_MD_DATA` | `data/md` |
| `R2_MK_DATA` | `../mk_planted_mc/data` |
| `R2_MK_TAU` | `../mk_planted_mc/results/tau_table.json` |
| `R2_HS_RESULTS` | `../../data/simulation/hs_bond` |
| `R2_CACHE` | `cache` |
| `R2_RESULTS` | `results` |
| `R2_FIGURES` | `figs` |

Paths supplied as relative overrides are resolved against the caller's working directory. For example:

    R2_MK_DATA=/path/to/mk/trajectories R2_HS_DATA=/path/to/hs/trajectories python3 r2_comob.py all

`lib/` contains vendored analysis helpers with repository-relative paths. The Gaussian–Markov triplet null,
normalized covariance residual ΔJ and Pearson residual ΔPb are separate quantities; see the definitions in
`r2_sum_trip.py` and the archived `tables_r2.md`.
