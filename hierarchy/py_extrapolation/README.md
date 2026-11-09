# PY extrapolation beyond N = 7 (SM Sec. S1 E, Table S5 'PY law' rows)

Scripts for SM Sec. S1 E and Table S5 (rows 'PY law', N=8-13). CPU only.

- `n058_estimate.py results/n058.json`: lamP(phi) = r(A), A(k,q) = (2/Om1^2(k)) sum_p W~(k;q,p) on the
  Luo-Janssen grid (M=100, qmax=40), same PY vertex as `../gmct_dense_gpu.py`; schematic
  Lambda_c(N) for f_n/(1-f_n) = (Lam/n) f_{n+1}, f_N = f_1 f_{N-1}; inverts lamP(phi_c(N))/lamP(phi_c(2)) = Lambda_c(N)/4.
- `check_mode.py`: correlator-closure deviations from the law (fit, extrapolation) and Perron-vector drift.

Validation: reproduces SM Table S5 memory-closure ratios to 4 decimals (1.2173, 1.4074, 1.5883, 1.7783) and
Lambda_c(N)/4 for N=2-7. Law vs Luo-Janssen memory closure N=3-6: |dphi| <= 0.0008.

Result (law ~ memory closure / correlator closure with its deficit extrapolated linearly in N):
0.563 at N=6/7, 0.58 at N=9/10, 0.59 at N=11/13, 0.60 at N=13/>=15. Lambda_c(N) ~ N at large N
(20.07, 30.04, 40.03 at N=20, 30, 40) and ln lamP is near-linear in phi over the accessible range phi <= 0.60,
giving an approximately logarithmic rise without a preferred stopping order.
Caveats: extrapolations beyond the solved orders N<=7; PY lamP(phi) becomes irregular above ~0.61 (51.9, 54.3, 55.4, 60.7 at 0.61-0.64, 122.6 at 0.66).
