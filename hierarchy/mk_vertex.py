#!/usr/bin/env python3
"""MK (Mari-Kurchan, lambda = infinity) statics plugged into the diagonal GMCT plateau hierarchy.

Reuses gmct_dense_gpu.py, gmct_sym2_gpu.py and linear_chain_bound.py from this folder READ-ONLY (imported;
build_vertex is monkeypatched at runtime, never edited).

Variants (MK statics of SM Sec. S2):
  Bself : S = 1, c = f~(k) (Mayer function), tagged one-leg vertex, Omega_1^2 = k^2
          W_s(k;q,p) = rho h^5/(16 pi^2) (q p/k^3) (k^2+q^2-p^2)^2 f~(q)^2
  Bcoll : S = 1, c = f~(k), symmetric collective vertex, Omega_1^2 = k^2
          W(k;q,p)   = rho h^5/(32 pi^2) (q p/k^3) [(q^2-p^2+k^2) f~(q) + (p^2-q^2+k^2) f~(p)]^2
  PY    : Percus-Yevick hard spheres (original build_vertex), reference.

IMPORTANT: the Bself vertex is not symmetric in q <-> p.  All contractions in the solvers are against
functions symmetric in (q,p) (every f_n is symmetric), but gmct_dense_gpu.top_level_corr and
gmct_sym2_gpu.top_memory use "2 * T1" which silently assumes W~(k;q,p) = W~(k;p,q).  We therefore
hand the solvers the (q,p)-symmetrised vertex  W~_sym = [W~(k;q,p) + W~(k;p,q)]/2, which is exactly
equivalent for every term of the hierarchy and for the frozen Perron operator.

Grid: k = h (j + 1/2), j = kcut .. M-1, h = qmax/M.  kcut > 0 removes the kcut lowest wavenumbers
(k-cutoff diagnostic); the triangle mask is unchanged for the remaining triples.
"""
from __future__ import annotations
import os, sys, math, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))   # sibling solver modules live in this folder
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import gmct_dense_gpu as G                     # read-only import
from gmct_sym2_gpu import SymHierarchy2        # read-only import
from linear_chain_bound import apply_T          # read-only import

LAMBDA_C = {2: 4.0, 3: 4.8284, 4: 5.6046, 5: 6.3652, 6: 7.1310, 7: 7.9154, 8: 8.7261, 9: 9.5669, 10: 10.4381}
PHI_D_MK_MC, PHI_D_MK_REPLICA = 1.82, 1.65
PHI_C_PY_MCT = 0.515914


def mayer_ft(k):
    return -4.0 * np.pi / k ** 3 * (np.sin(k) - k * np.cos(k))


def grid(M, qmax, kcut=0):
    h = qmax / M
    khat = np.arange(kcut, M) + 0.5
    return h, khat, h * khat


def build_vertex_np(M, qmax, phi, variant, kcut=0, symmetrize=True):
    """numpy (Wt, Om, k).  Wt has shape (M-kcut,)*3 and is (q,p)-symmetrised unless symmetrize=False."""
    h, khat, k = grid(M, qmax, kcut)
    rho = 6 * phi / np.pi
    if variant == 'PY':
        c = G.py_ck(k, phi); S = 1.0 / (1.0 - rho * c)
    elif variant in ('Bself', 'Bcoll'):
        c = mayer_ft(k); S = np.ones_like(k)
    else:
        raise ValueError(variant)
    Om = k ** 2 / S
    K = khat[:, None, None]; Q = khat[None, :, None]; P = khat[None, None, :]
    mask = (P >= np.abs(K - Q) + 0.5 - 1e-9) & (P <= K + Q - 0.5 + 1e-9)
    pref = rho * h ** 5 / (32 * np.pi ** 2)
    if variant == 'Bself':
        brack = math.sqrt(2.0) * (K ** 2 + Q ** 2 - P ** 2) * c[None, :, None]
        Wt = pref * (Q * P / K ** 3) * brack ** 2 * S[None, :, None]
    else:
        brack = (Q ** 2 - P ** 2 + K ** 2) * c[None, :, None] + (P ** 2 - Q ** 2 + K ** 2) * c[None, None, :]
        Wt = pref * (Q * P / K ** 3) * brack ** 2 * S[None, :, None] * S[None, None, :]
    Wt = np.where(mask, Wt, 0.0)
    if symmetrize:
        Wt = 0.5 * (Wt + np.swapaxes(Wt, 1, 2))
    return Wt, Om, k


def build_vertex_torch(M, qmax, phi, variant, device, dtype, kcut=0):
    Wt, Om, k = build_vertex_np(M, qmax, phi, variant, kcut)
    return torch.tensor(Wt, device=device, dtype=dtype), torch.tensor(Om, device=device, dtype=dtype), k


def patched_build_vertex(variant, kcut=0):
    """Drop-in replacement for gmct_dense_gpu.build_vertex (same signature / return tuple)."""
    def f(M_, qmax_, phi_, device_, dtype_):
        Wt, Om, k = build_vertex_np(M_, qmax_, phi_, variant, kcut)
        return (torch.tensor(Wt, device=device_, dtype=dtype_), torch.tensor(Om, device=device_, dtype=dtype_),
                k, np.ones_like(k), mayer_ft(k) if variant != 'PY' else G.py_ck(k, phi_))
    return f


# ----------------------------------------------------------------------------- frozen Perron operator
def perron(Wt, Om, k):
    """A(k,q) = 2 sum_p W~_sym(k;q,p)/Omega_1^2(k)  (= frozen-memory derivative at f = 1).  Dense eig."""
    A = 2.0 * Wt.sum(axis=2) / Om[:, None]
    ev, vec = np.linalg.eig(A)
    i = int(np.argmax(ev.real))
    v = np.abs(vec[:, i].real); v = v / v.max()
    w = Wt.sum(axis=(1, 2))
    kp = int(np.argmax(v))
    return dict(lamP=float(ev[i].real), k_peak=float(k[kp]), k_peak_idx=kp, w_max=float(w.max()),
                k_wmax=float(k[int(np.argmax(w))]), omega_min=float(Om.min()), C=float(w.max() / Om.min()),
                Mfrozen_max=float((w / Om).max()), vP=v.tolist(), w=w.tolist())


# ----------------------------------------------------------------------------- dense iteration with E
def iterate_dense(Wt, Om, N, closure, f=None, maxit=100000, tol=1e-11, liquid_floor=1e-7, log_every=0):
    """Identical fixed-point map to gmct_dense_gpu.iterate (same building blocks), plus the asymptotic
    contraction rate E = geometric mean of the last 20 ratios d_it/d_{it-1} (as in gmct_sym2_gpu)."""
    M = Om.shape[0]
    general_N2 = (closure == 'mem_general')          # memory closure, general branch also at N = 2 (label 3)
    if general_N2: closure = 'mem'
    ndyn = N - 1 if closure == 'corr' else N
    Omn = [None] + [G.omega_n(Om, n) for n in range(1, ndyn + 1)]
    if f is None:
        f = [None] + [torch.ones((M,) * n, device=Om.device, dtype=Om.dtype) for n in range(1, ndyn + 1)]
    ratios = []; dprev = None; E = float('nan')
    for it in range(1, maxit + 1):
        fnew = [None] * (ndyn + 1); Mlev = [None] * (ndyn + 1)
        for n in range(1, ndyn):
            H = G.contract_qp(Wt, f[n + 1])
            Mlev[n] = G.assemble_level(Om, Omn[n], H, n)
            fnew[n] = Mlev[n] / (1.0 + Mlev[n])
        if closure == 'corr':
            n = ndyn
            H = G.top_level_corr(Wt, f[1], f[N - 1], N)
            Mlev[n] = G.assemble_level(Om, Omn[n], H, n)
            fnew[n] = Mlev[n] / (1.0 + Mlev[n])
        else:
            if N == 2 and not general_N2:
                H = G.top_level_corr(Wt, f[1], f[1], 2)
                Mlev[1] = G.assemble_level(Om, Omn[1], H, 1)
                fnew[1] = Mlev[1] / (1.0 + Mlev[1])
                fnew[2] = fnew[1][:, None] * fnew[1][None, :]
            else:
                X = torch.zeros_like(Omn[N])
                for i in range(N):
                    ratio = torch.movedim(Omn[N - 1].unsqueeze(0).expand((M,) + (M,) * (N - 1)), 0, i) / Omn[N]
                    shape = [1] * N; shape[i] = M
                    X += ratio ** 2 * torch.movedim(Mlev[N - 1].unsqueeze(0).expand((M,) + (M,) * (N - 1)), 0, i) \
                         * f[1].reshape(shape)
                X /= (N - 1)
                fnew[N] = X / (1.0 + X)
        d = float(torch.max(torch.abs(fnew[1] - f[1])))
        if dprev is not None and dprev > 0: ratios.append(d / dprev)
        dprev = d; f = fnew
        fmax = float(f[1].max())
        if len(ratios) >= 5: E = float(np.exp(np.mean(np.log(ratios[-20:]))))
        if log_every and it % log_every == 0:
            print(f"    it={it:6d}  f1max={fmax:.6e}  dmax={d:.3e}  E~{E:.5f}", flush=True)
        if fmax < liquid_floor:
            return dict(f=f, status='liquid', iterations=it, dmax=d, f1max=fmax, E=E)
        if d < tol:
            return dict(f=f, status='glass' if fmax > 1e-3 else 'liquid', iterations=it, dmax=d, f1max=fmax, E=E)
    return dict(f=f, status='unresolved', iterations=maxit, dmax=d, f1max=fmax, E=E)


def solve_dense(M, qmax, N, closure, phi, variant, device, dtype, kcut=0, **kw):
    Wt, Om, k = build_vertex_torch(M, qmax, phi, variant, device, dtype, kcut)
    t0 = time.time(); r = iterate_dense(Wt, Om, N, closure, **kw); r['seconds'] = time.time() - t0
    r['f1'] = r['f'][1].detach().cpu().numpy().tolist(); r['k'] = k.tolist(); del r['f']
    r.update(phi=phi, M=M, qmax=qmax, N=N, closure=closure, variant=variant, solver='dense', kcut=kcut)
    r['gpu_peak_GB'] = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0.0
    return r


def solve_sym(H, M, qmax, N, phi, variant, kcut=0, warm=None, **kw):
    Wt, Om, k = build_vertex_torch(M, qmax, phi, variant, H.dev, H.dt, kcut)
    H.set_phi(Wt, Om)
    t0 = time.time(); r = H.iterate(f=warm, **kw); r['seconds'] = time.time() - t0
    r['f1'] = r['f'][1].detach().cpu().numpy().tolist(); r['k'] = k.tolist()
    fstate = r.pop('f')
    r.update(phi=phi, M=M, qmax=qmax, N=N, closure='corr', variant=variant, solver='sym2', kcut=kcut)
    r['gpu_peak_GB'] = torch.cuda.max_memory_allocated() / 1e9
    return r, fstate


# ----------------------------------------------------------------------------- linear chain rho(T_N)
def spectral_radius_T(H, iters=300, tol=1e-9, v0=None):
    """Power iteration for rho(T_N) with Collatz-Wielandt sandwich (same as linear_chain_bound, more iterations)."""
    v = torch.ones(H.M, device=H.dev, dtype=H.dt) if v0 is None else v0.clone()
    v = v / v.max(); rho = None; lo = hi = float('nan')
    for it in range(iters):
        w = apply_T(H, v)
        ratio = w / v
        lo, hi = float(ratio.min()), float(ratio.max())
        rho_new = float(w.max() / v.max())
        v = w / w.max()
        if rho is not None and abs(rho_new - rho) < tol * rho_new and (hi - lo) < 1e-6 * hi:
            rho = rho_new; break
        rho = rho_new
    return rho, lo, hi, v, it + 1


def rho_TN(M, qmax, phi, variant, N, device, kcut=0, dtype=torch.float64, chunk_elems=200_000_000):
    Wt, Om, k = build_vertex_torch(M, qmax, phi, variant, device, dtype, kcut)
    Meff = Wt.shape[0]
    H = SymHierarchy2(Meff, N, device, dtype, chunk_elems=chunk_elems); H.set_phi(Wt, Om)
    t0 = time.time(); rho, lo, hi, v, nit = spectral_radius_T(H); dt_ = time.time() - t0
    out = dict(M=M, qmax=qmax, kcut=kcut, kmin=float(k[0]), Meff=Meff, phi=phi, variant=variant, N=N, rho=rho,
               cw_lo=lo, cw_hi=hi, power_iters=nit, seconds=dt_, v_peak_k=float(k[int(torch.argmax(v))]))
    del H; torch.cuda.empty_cache()
    return out
