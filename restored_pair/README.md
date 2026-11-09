# `restored_pair/`: stationary closures with the bond hard core restored

Stationary (plateau) closures of the N = 2 memory kernel on the Mari–Kurchan (MK) liquid in which the intra-pair hard
core of a bonded neighbour is kept: the memory is weighted by the cage-exclusion factor 1/Z(c) ≥ 1 (SM Sec. S5). The
matched ablation Z ≡ 1 removes the hard core and gives back the wavevector-diagonal theory. Supports SM Fig. S5,
Fig. 1(c), and SM Tables S13, S15, S16 and Sec. S5 A–E. CPU only (numpy, scipy).

| file | role | SM |
|---|---|---|
| `brc_gauss.py` | Gaussian cage at the MSD level: closed forms of Z(c) and J(c), φ(σ²), fold φ_c, glass branch σ²(φ); with Z ≡ 1 (`diag`) | S5 B, Table S15 |
| `gauss_curves.py` | glass branch of both Gaussian closures vs φ: MSD = 6σ², f_G(k) at k = 3, 5, 7, 10 | Fig. S5(a,b), Table S13 |
| `rescatter_series.py` | Gaussian closure with 1/Z truncated at exclusion-series order ℓ, Y_ℓ = Σ_{j<ℓ}(1−Z)^j | Letter Fig. 2(b), Table S16 |
| `brc_kres.py` | wavevector-resolved restored closure on the Franosch grid (M = 100, q_max = 40): Legendre/Hankel kernel, admissible-cage projection, fixed-point iteration, bisection | S5 D, Table S15 |
| `kres_series.py` | the k-resolved closure truncated at exclusion-series order ℓ (ℓ = 1 is the diagonal N = 2 kernel) | Letter Fig. 2(b), Table S16 |
| `diag_kres.py` | k-resolved diagonal N = 2 hierarchy (hard core off), plateau f(k) | Fig. S5(a,b), Table S13 |
| `vng.py` | static non-Gaussian Bethe-entropy cage (variational, m → 1, on finite radial ball u ≤ r_max = 4.0; infinite-space Gaussian restriction recovers Eq. S27): `full` (−ln Z), `single` (1 − Z), `n<ℓ>` truncations | S5 D |
| `vng_scan.py`, `vng_single_fold.py`, `vng_series.py` | density scans and folds of the static closure | Fig. S5(a), Tables S13, S15, S16 |

Every script writes into `results/` next to itself (created on first use). The files used in the paper are in
`../data/restore/` under the same names.

## Commands and runtimes (one CPU, numpy with a multithreaded BLAS)

    python3 -c "import brc_gauss as b; print(b.critical('brc'), b.critical('diag'))"   # 1.7762, 3.2369; < 1 s
    python3 brc_gauss.py              # convergence checks, curves, single-k variant -> results/brc_gauss.json (minutes)
    python3 gauss_curves.py           # -> results/gauss_curves.json, gauss_curves_dense.json            (seconds)
    python3 rescatter_series.py       # -> results/rescatter_series_gauss.json                           (seconds)
    python3 diag_kres.py              # -> results/diag_kres.json                                        (seconds)
    python3 vng_scan.py               # -> results/vng_scan.json                                         (about 30 min)
    python3 vng_single_fold.py        # -> results/vng_single_fold.json                                  (about 10 min)
    python3 vng_series.py             # -> results/vng_series.json                                       (about 1 h)
    # k-resolved series, l = 1, 2, 3, 5, 10 (bisection n lo hi steps; each step 5-60 min, 2-6 h per l)
    python3 kres_series.py 1  2.70 2.90 5
    python3 kres_series.py 2  1.90 2.60 6
    python3 kres_series.py 3  1.80 2.30 6
    python3 kres_series.py 5  1.78 2.10 6
    python3 kres_series.py 10 1.78 1.95 5
    # k-resolved restored closure (l -> infinity), phi_c = 1.784
    python3 brc_kres.py check          # Parseval, Legendre vs direct quadrature, Z(c) vs closed form
    python3 brc_kres.py bisect 1.70 1.80 --nmax 5   # 5 densities, 10-60 min each; Table S15 quotes 1.784(3)

Expected results: φ_c = 1.776241 (restored, σ²_c = 0.04531) and 3.236939 (Z ≡ 1); restored MSD plateau 0.119 at
φ = 2.0; ℓ-series folds 2.769–2.775 (ℓ = 1), 2.184–2.195 (2), 2.003–2.019 (3), 1.885–1.890 (5), 1.812–1.817 (10);
static folds 1.7887 (`full`) and 3.3339 (`single`).

`brc_kres.py` builds Hankel matrices (0.13 GB at the defaults) and caches them in `cache/` for reuse. The fold of the
ℓ → ∞ closure (1.784) is entered as a number in Letter Fig. 2(b); its bisection log is archived in `../data/restore/kres_bisect/` (liquid at 1.78125, glass at 1.7875; 1.784375 unresolved after 1500 iterations, hence 1.784(3)).
Expected output of the Gaussian check is in `../examples/expected/a_restored_gauss.txt`.
