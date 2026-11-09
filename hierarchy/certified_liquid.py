#!/usr/bin/env python3
"""Certified-liquid criterion (computer-assisted sharpening of the chain bound).

Let Phi be the monotone hierarchy map (all levels, correlator closure).  The maximal fixed point f*
satisfies f* <= Phi^m(1) =: g^(m) for every m (monotone iteration from f ≡ 1).  Using the upper
bound g^(m)_{N-1} in the closure instead of 1 gives f_1 <= T_N[g] f_1 with
    T_N[g] v = L_1 ... L_{N-1} P_g v,   (P_g v)(K) = (1/N) sum_i v(k_i) g_{N-1}(K \\ k_i).
If rho(T_N[g^(m)]) < 1 for some m, the only fixed point is f ≡ 0: phi is certified liquid.
In the liquid phase g^(m) -> 0 so the certificate always fires eventually; the iteration count
needed grows as phi -> phi_c^-.  This turns the (weak) chain bound into a lower bound on phi_c(N)
that is as sharp as the iteration budget allows (modulo floating-point rigor).
"""
import argparse, json, time
import numpy as np, torch
from gmct_dense_gpu import build_vertex
from gmct_sym2_gpu import SymHierarchy2


def apply_Tg(H, v, g):
    L = H.L
    f = [None] * (L + 1); f[1] = v; f[L] = g
    if L == 1:                                   # N=2: T[g] v (k) = sum_{qp} W~ (v_q g_p + v_p g_q)/2 / Om  = sum W~ v_q g_p / Om
        return torch.einsum('kqp,q,p->k', H.Wt, v, g) / H.Om
    out = H.top_memory(f)
    for n in range(L - 1, 0, -1):
        f[n + 1] = out; out = H.level_memory(n, f)
    return out


def rho_Tg(H, g, v0=None, iters=60):
    v = torch.ones(H.M, device=H.dev, dtype=H.dt) if v0 is None else v0
    v = v / v.max(); lo = hi = None
    for _ in range(iters):
        w = apply_Tg(H, v, g)
        if float(w.max()) <= 0: return 0.0, 0.0, 0.0, v
        r = w / v; lo, hi = float(r.min()), float(r.max()); v = w / w.max()
        if hi - lo < 1e-4 * hi: break
    return float(w.max()), lo, hi, v


def certify(H, phi, max_iters=20000, check_every=25, log_every=0):
    """Iterate Phi from 1; every check_every iterations test rho(T[g]) < 1 (certified via CW upper bound)."""
    Wt, Om, k, S, c = build_vertex(H.M, 40.0, phi, H.dev, H.dt); H.set_phi(Wt, Om)
    f = [None] + [torch.ones(H.counts[n], device=H.dev, dtype=H.dt) for n in range(1, H.L + 1)]
    v = None
    for it in range(1, max_iters + 1):
        fnew = [None] * (H.L + 1)
        for n in range(1, H.L):
            Mn = H.level_memory(n, f); fnew[n] = Mn / (1 + Mn)
        if H.L == 1:                                    # N = 2: plain MCT map
            Mn = torch.einsum('kqp,q,p->k', H.Wt, f[1], f[1]) / H.Om
        else:
            Mn = H.top_memory(f)
        fnew[H.L] = Mn / (1 + Mn)
        f = fnew
        if it % check_every == 0:
            rho, lo, hi, v = rho_Tg(H, f[H.L], v)
            if log_every and it % log_every == 0:
                print(f"    it={it:6d} f1max={float(f[1].max()):.4e} rho(T[g])~{rho:.4f} CW[{lo:.4f},{hi:.4f}]", flush=True)
            if hi < 1.0:
                return dict(certified=True, iterations=it, rho=rho, cw_hi=hi, f1max=float(f[1].max()))
    return dict(certified=False, iterations=max_iters, rho=rho, cw_hi=hi, f1max=float(f[1].max()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=100); ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--phis', type=str, required=True); ap.add_argument('--max_iters', type=int, default=20000)
    ap.add_argument('--out', default='certified_liquid.jsonl')
    ap.add_argument('--dtype', default='float64', help='float32 halves memory (needed for N=7 at M=100)')
    a = ap.parse_args()
    dev = torch.device('cuda'); dt = getattr(torch, a.dtype)
    H = SymHierarchy2(a.M, a.N, dev, dt, chunk_elems=300_000_000)
    with open(a.out, 'a') as log:
        for phi in [float(x) for x in a.phis.split(',')]:
            t0 = time.time(); r = certify(H, phi, max_iters=a.max_iters); r.update(M=a.M, N=a.N, phi=phi, seconds=time.time() - t0, dtype=a.dtype)
            print(f"N={a.N} phi={phi:.6f}: certified_liquid={r['certified']} after {r['iterations']} iterations "
                  f"(rho={r['rho']:.4f}, f1max={r['f1max']:.3e}, {r['seconds']:.1f}s)", flush=True)
            log.write(json.dumps(r) + "\n"); log.flush()
