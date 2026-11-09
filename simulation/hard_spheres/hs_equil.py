#!/usr/bin/env python3
"""Prepare equilibrium configurations of polydisperse HS (Delta = 23%) with swap Monte Carlo.

Protocol (Berthier et al., PRL 116, 238002 (2016)): with probability 1 - alpha = 0.2 a swap move (exchange the
diameters of two random particles, accepted if no overlap), else a translation in a cube of half-width delta.
1. random positions at the target phi, overlaps removed by minimising a harmonic overlap energy with diameters
   inflated by 0.3% (unjammed: U -> 0), then deflated -> strictly non-overlapping (an out-of-equilibrium quench);
2. (no compression step needed);
3. swap-MC equilibration at fixed phi for n_eq sweeps; record reduced pressure Z (virtual inflation),
   Steinhardt Q6 / crystal-like fraction, and the swap-dynamics self-correlation Fs(k=7) and MSD from the
   start of step 3 (decorrelation check: the final configuration must be independent of the quench).
Output .npz: x (unwrapped), sig, L, phi, history arrays.
"""
import argparse, os, sys, time
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hs_core import HS, make_sizes, box_for_phi, nb_seed, pressure_Z, Z_bmcsl
from hs_struct import q6_stats


def soft_quench(sig, L, rng, infl=1.003, tol=1e-14):
    """Random positions at the target density, then L-BFGS minimisation of the harmonic overlap energy
    U = sum_{i<j} (1 - r_ij / (infl sigma_ij))^2 for overlapping pairs (unjammed below phi_J => U -> 0).
    The inflation factor guarantees strictly non-overlapping hard spheres after deflation."""
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree
    N = len(sig)
    s_inf = sig * infl

    def fg(flat):
        x = flat.reshape(N, 3)
        xw = x - L * np.floor(x / L)
        xw = np.clip(xw, 0, np.nextafter(L, 0))
        P = cKDTree(xw, boxsize=L).query_pairs(s_inf.max(), output_type="ndarray")
        d = x[P[:, 0]] - x[P[:, 1]]
        d -= L * np.rint(d / L)
        r = np.linalg.norm(d, axis=1)
        sij = 0.5 * (s_inf[P[:, 0]] + s_inf[P[:, 1]])
        m = r < sij
        P, d, r, sij = P[m], d[m], r[m], sij[m]
        h = 1 - r / sij
        U = np.sum(h ** 2)
        gp = (-2 * h / sij / np.maximum(r, 1e-12))[:, None] * d     # dU/dx_i
        g = np.zeros((N, 3))
        np.add.at(g, P[:, 0], gp)
        np.add.at(g, P[:, 1], -gp)
        return U, g.ravel()

    x = rng.uniform(0, L, (N, 3))
    for it in range(20):
        r = minimize(fg, x.ravel(), jac=True, method="L-BFGS-B", options=dict(maxiter=20000, ftol=1e-20, gtol=1e-14))
        x = r.x.reshape(N, 3)
        if r.fun < tol:
            break
    return x, float(r.fun)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--phi", type=float, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--neq", type=int, default=100000, help="swap-MC equilibration sweeps at fixed phi")
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--pswap", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(1000 + a.seed)
    nb_seed(int(rng.integers(0, 2 ** 31 - 1)))
    t0 = time.time()
    sig = make_sizes(a.N, rng)
    Lt = box_for_phi(sig, a.phi)
    x, U = soft_quench(sig, Lt, rng)
    s = HS(x, sig, Lt, skin=0.3, maxnb=160, swap=True)
    Zc, nov, _ = pressure_Z(s.x, s.sig, s.L)
    print(f"[init] N={a.N} poly={sig.std()/sig.mean():.4f} smin={sig.min():.3f} smax={sig.max():.3f} "
          f"phi={s.phi():.5f} L={Lt:.4f} soft-quench U={U:.2e} overlaps={nov} Z={Zc:.3f} ({time.time()-t0:.0f}s)", flush=True)
    assert nov == 0
    # equilibration at fixed phi
    nrec = 40
    every = max(a.neq // nrec, 1)
    x_start = s.x.copy()
    kv = 7.0
    hist = []
    st0 = s.stats.copy()
    for r in range(nrec + 1):
        if r > 0:
            s.run(every, a.delta, a.pswap)
        Z, nov, _ = pressure_Z(s.x, s.sig, s.L)
        d = s.x - x_start
        d -= d.mean(0)
        r2 = (d ** 2).sum(1)
        rr = np.sqrt(r2)
        fs = float(np.mean(np.sinc(kv * rr / np.pi)))
        Q6, xfrac = q6_stats(s.x, s.sig, s.L)
        st = s.stats - st0
        hist.append([r * every, Z, fs, r2.mean(), Q6, xfrac, nov])
        if r % 4 == 0:
            print(f"  eq t={r*every} Z={Z:.3f} (BMCSL {Z_bmcsl(s.sig, a.phi):.3f}) Fs7(swap)={fs:.3f} MSD={r2.mean():.3f} "
                  f"Q6={Q6:.3f} xtal={xfrac:.3f} acc_tr={st[0]/max(st[1],1):.3f} acc_sw={st[2]/max(st[3],1):.4f} "
                  f"rebuilds={st[4]} ({time.time()-t0:.0f}s)", flush=True)
        assert nov == 0
    hist = np.array(hist)
    st = s.stats - st0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, x=s.x, sig=s.sig, L=s.L, phi=s.phi(), N=a.N, seed=a.seed, neq=a.neq, delta=a.delta, pswap=a.pswap,
             hist=hist, hist_cols=np.array(["t", "Z", "Fs7_swap", "MSD_swap", "Q6", "xtal_frac", "overlaps"]),
             acc_tr=st[0] / max(st[1], 1), acc_sw=st[2] / max(st[3], 1), Z_bmcsl=Z_bmcsl(s.sig, a.phi),
             wall=time.time() - t0)
    print(f"[saved] {a.out} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
