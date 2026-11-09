# `py_pilot/`: Gaussian pair closures for ordinary 3D hard spheres (PY and Verlet–Weis structure)

Tests whether the restored-pair weight 1/Z, which fixes the under-trapping of the diagonal kernel on the
Mari–Kurchan (MK) liquid, carries over to ordinary d = 3 hard spheres, where multi-body packing enters through the
cavity function y(r). It compares Gaussian (MSD-level) closures built on Percus–Yevick (PY) and Verlet–Weis (VW)
structure with the m → 1 Gaussian replica equation of Mangeat and Zamponi (MZ), PRE 93, 012609 (2016).
Supports SM Sec. S5 F and Table S17. CPU only (numpy, scipy); every script runs in under a minute.

## Conventions (as in `../restored_pair/brc_gauss.py`, SM Sec. S5 B)

- Diameter 1, ρ = 6φ/π. Cage: isotropic Gaussian with per-component variance A (= σ² of the SM); f_s(k) = e^{−k²A};
  MSD plateau = 6A; relative variance s = 2A; γ_s is the 3D Gaussian of variance s.
- Z(c) = γ_s ⋆ Θ(· − 1) and J = dZ/dc (closed forms); q(c) = γ_s ⋆ g (MZ Eq. 25 at m = 1).
- Every closure reads Λ(A; φ) = ρ A K(A; φ) = 1 with K = (1/3)∫d³c [integrand]; φ_c is the fold, the glass branch
  the smaller-A root:

| variant | integrand of K | meaning |
|---|---|---|
| `a` | y J²/Z | cavity-weighted centre measure, bare contact force, weight 1/Z |
| `a0` | y J² | the same without 1/Z |
| `amix` | y(1)² J²/q | exact-marginal pair measure |
| `MZ` | q′²/q | potential of mean force = m → 1 replica (MZ Eq. 30) |
| `MZ0` | q′² | the same without 1/q |
| `b`, `b0` | J_c²/Z, J_c² with J_c = d(γ_s ⋆ c_PY)/dc | MCT vertex as effective Mayer function (`b` is ill-posed) |
| `GMCTV` | (1/6π²)∫k⁴c²S e^{−k²s} | Gaussian/Vineyard reduction of tagged MCT |

- PY structure: analytic Wertheim–Thiele c(r), one-shot OZ inversion by DST-I; VW: Verlet–Weis g(r) with
  Carnahan–Starling contact value. Radial grid N = 2¹⁷, Δr = 5 × 10⁻⁴ (φ_c converged to 10⁻⁵).

## Files

| file | role | output (`results/`, created next to the script) |
|---|---|---|
| `py_structure.py` | PY c, ĉ, S, ĥ, y, g; Verlet–Weis g with interior-y choices | |
| `gauss_py.py` | Z, J closed forms; DST convolutions; all closures; fold and branch solvers | |
| `mk_sanity.py` | MK sanity check with `brc_gauss_mk.py` (1.776241 restored, 3.236939 with Z ≡ 1) | `mk_sanity.json` |
| `test_mk_paths.py` | every closure route of `gauss_py.py` reduces to the MK numbers when y ≡ 1 | `mk_paths_check.json` |
| `check_structure.py` | g(1⁺), S(0), h = −1 inside the core, ĉ numeric vs analytic, VW contact | `structure_checks.json` |
| `scan_py.py` | coarse bracket scan (shows the VW `continue` interior pathology) | stdout |
| `run_folds.py` | φ_c and glass-branch MSD for all variants, PY and VW | `folds.json` |
| `make_summary.py` | condenses `folds.json` into the table behind SM Table S17 | `summary_table.json` |
| `conv_mz.py` | grid and cutoff convergence of the MZ and (a) folds; real-space kernel check | `conv_mz.json` |
| `diag_b.py` | divergence of variant (b); sign of Z_c | `diag_b.json` |
| `analyze.py` | MZ Eq. (30) literal vs integrated-by-parts form; region decomposition of 1/q vs 1/Z | `analysis.json` |
| `brc_gauss_mk.py` | identical copy of `../restored_pair/brc_gauss.py` (MK reference) | |
| `szamel_d3_ref.py` | independent implementation of Szamel, PRL 119, 155502 (2017), Eqs. (9)–(11) for MK, d = 3 | stdout |

The shipped outputs are in `../data/py_pilot/` under the same names.

## Reproduce SM Table S17

    python3 mk_sanity.py && python3 test_mk_paths.py && python3 check_structure.py
    python3 run_folds.py && python3 make_summary.py
    python3 conv_mz.py && python3 diag_b.py && python3 analyze.py

Expected: PY MZ φ_c = 0.5184 (published 0.5216), VW 0.5345 (0.5367); variant (a) 0.376 (PY); GMCTV 0.4088 (PY).
`make_summary.py` reproduces `../data/py_pilot/summary_table.json` exactly from `../data/py_pilot/folds.json`
(`python3 make_summary.py ../data/py_pilot/folds.json summary_table_check.json`).
`../examples/smoke_b_py_pilot.py` runs the MK sanity check and the MZ folds in a few seconds.
