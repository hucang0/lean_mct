#!/usr/bin/env python3
"""Version (b): MCT vertex c_PY(r) as effective Mayer function, J_c = d/dc (gamma_s * c_PY).
Diagnostics: (1) the restored integral int d^3c J_c^2/Z with the hard-core Z diverges because grad c_PY =
y(1) delta(r-1) rhat - Theta(1-r) grad y is nonzero INSIDE the core where Z ~ exp(-(1-c)^2/2s): show K_b vs a
floor on Z and vs an inner cutoff c_min. (2) The self-consistent 'non-overlap' Z_c = 1 + gamma_s*c_PY (what
1/Z would be if c were a Mayer function) is negative: c_PY = -y < -1 inside the core."""
import math, json
import os
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
import numpy as np
from gauss_py import Grid, Closures, Z_gauss, J_gauss
import py_structure as PS
G = Grid(); PG = PS.Grid(G.N, G.dr)
out = {}
for phi in (0.40, 0.45, 0.50):
    st = PS.py_structure(phi, PG)
    C = Closures(G, st['rho'], st['y'], st['hk'], st['ck'], st['Sk'], y1=st['g1'])
    for A in (0.005, 0.01, 0.02):
        s = 2 * A
        Zc, _ = G.r_from_k_with_deriv(st['ck'] * np.exp(-G.k ** 2 * s / 2))
        Zc = 1 + Zc
        rows = {}
        for zf in (1e-2, 1e-4, 1e-8, 1e-16, 1e-32):
            rows[f"zfloor={zf:g}"] = C.K('b', A, zfloor=zf)
        # inner cutoff version (exact Z, integrate only c > c_min)
        Z = Z_gauss(C.c, s)
        _, Jc = G.r_from_k_with_deriv(st['ck'] * np.exp(-G.k ** 2 * s / 2)); Jc = Jc[C.ic]
        for cmin in (0.9, 0.8, 0.7, 0.6, 0.5):
            m = C.c > cmin
            rows[f"cmin={cmin}"] = float(np.sum(C.w[m] * Jc[m] ** 2 / Z[m]))
        rows['K_b0'] = C.K('b0', A); rows['K_a'] = C.K('a', A)
        rows['min_Zc'] = float(np.min(Zc[C.ic])); rows['Zc_at_0.9'] = float(np.interp(0.9, G.r, Zc))
        out[f"phi={phi},A={A}"] = rows
        print(f"phi={phi} A={A}: rho*A*K_b with Z floor 1e-2,1e-4,1e-8,1e-16,1e-32 = " +
              ", ".join(f"{st['rho']*A*rows[f'zfloor={z:g}']:.3g}" for z in (1e-2, 1e-4, 1e-8, 1e-16, 1e-32)) +
              " | c_min 0.9..0.5 = " + ", ".join(f"{st['rho']*A*rows[f'cmin={c}']:.3g}" for c in (0.9, 0.8, 0.7, 0.6, 0.5)) +
              f" | rho*A*K_b0={st['rho']*A*rows['K_b0']:.3g} rho*A*K_a={st['rho']*A*rows['K_a']:.3g} | min Z_c={rows['min_Zc']:.2f}")
json.dump(out, open(os.path.join(HERE, 'results', 'diag_b.json'), 'w'), indent=1)
