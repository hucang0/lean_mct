#!/usr/bin/env python3
"""Bond-resolved pair-factorization analysis of polydisperse-HS MC trajectories (hs_dyn.py output).

Definitions are those of the MK analysis (../mk_planted_mc/analysis.py and conditional.py; SM Tables S11-S12):
 * k shells: lattice vectors 2 pi n / L with |n|^2 closest to k_target (k in units of the mean diameter), at most
   24 directions (mod sign), targets 3, 5, 7, 10; factorization at k7.
 * displacements dx_i(t) = x_i(t0 + t) - x_i(t0) with the centre-of-mass drift removed per frame (unweighted mean).
 * contact pairs at the origin: r_ij(0) < 1.05 sigma_ij, sigma_ij = (sigma_i + sigma_j)/2; rhat = (x_i - x_j)/r, i < j.
 * parallel: k = k7 rhat; transverse: k = k7 * unit(rhat x e), e = x-axis (or y-axis if |rhat_x| >= 0.9);
   orientation-averaged: the lattice k7 set.
 * phi2(-) = < e^{i k.(dx_i - dx_j)} >, phi2(+) = < e^{i k.(dx_i + dx_j)} >, bond-conditioned F^c_i = <e^{i k.dx_i}>,
   F^c_j = <e^{i k.dx_j}> over the same pairs (complex).
   R = Re phi2 / Fs(k7)^2 (global isotropic Fs);  R_marg = Re(F^c_i F^c_j*) / Fs^2;  C = phi2 / (F^c_i F^c_j*).
 * random pairs: 4000 random (i, j != i) per origin; contact return probability P(|r_ij(t)| < 1.05 sigma_ij | contact at 0).
 * averaging: per (origin, frame) over pairs, binned in log lag (12 per decade) over origins (Accum); ratios per
   sample, then mean +- s.e.m. over samples (conditional.per_sample_ratio, den_min = 0.15).
Extra (controls, HS-specific):
 * non-contact shells 1.05 < r/sigma_ij < 1.3 ('n' = rest of first shell), 1.3 < r/sigma_ij < 1.5 ('g' = gap at the
   first minimum of g) and 1.6 < r/sigma_ij < 2.1 ('s' = second shell);
 * contact pairs binned by their number of common neighbours n_c (neighbour: r < 1.35 sigma_ij), parallel only;
 * parallel contact correlator at k5 and k10 (normalised by Fs(k5)^2, Fs(k10)^2);
 * random pairs with k along their own separation vector (sanity: must give 1);
 * model-free Pearson coefficient of the bond-projected displacements u_i = dx_i.e, u_j = dx_j.e (e = k/|k|),
   rho_e = cov(u_i, u_j)/sqrt(var u_i var u_j) over pairs of a group (moments stored per origin/frame).
Usage: hs_analysis.py out_dir file1.npz [file2.npz ...]   (writes out_dir/curves_<tag>.npz, cached)
"""
import os, sys, glob
import numpy as np
from multiprocessing import Pool
from scipy.spatial import cKDTree
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis as A          # identical copy of ../mk_planted_mc/analysis.py: lattice_kvecs, Accum, PER_DECADE, crossing_time, plateau_estimate

K_TARGETS = [3.0, 5.0, 7.0, 10.0]
CONTACT = 1.05
NB_CUT = 1.35
SHELLS = {"n": (1.05, 1.3), "g": (1.3, 1.5), "s": (1.6, 2.1)}
NC_MAX = 10
N_RAND = 4000


def pairs_upto(x, sig, L, smax):
    xw = x - L * np.floor(x / L)
    xw = np.clip(xw, 0, np.nextafter(L, 0))
    t = cKDTree(xw, boxsize=L)
    P = t.query_pairs(smax * sig.max(), output_type="ndarray")
    P = np.sort(P, axis=1)
    d = x[P[:, 0]] - x[P[:, 1]]
    d -= L * np.rint(d / L)
    r = np.linalg.norm(d, axis=1)
    s = r / (0.5 * (sig[P[:, 0]] + sig[P[:, 1]]))
    keep = s < smax
    return P[keep], d[keep], r[keep], s[keep]


def perp_of(rhat):
    tmp = np.cross(rhat, np.where(np.abs(rhat[:, [0]]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
    return tmp / np.linalg.norm(tmp, axis=1)[:, None]


def common_neighbours(N, P, s, ci, cj):
    nb = P[s < NB_CUT]
    adj = [set() for _ in range(N)]
    for a, b in nb:
        adj[a].add(b); adj[b].add(a)
    return np.array([len(adj[i] & adj[j]) for i, j in zip(ci, cj)], dtype=np.int64)


def bond_obs(v, lab, dx, ci, cj, kk):
    ai, aj = (dx[ci] * kk).sum(1), (dx[cj] * kk).sum(1)
    pm, pp = np.exp(1j * (ai - aj)).mean(), np.exp(1j * (ai + aj)).mean()
    Fi, Fj = np.exp(1j * ai).mean(), np.exp(1j * aj).mean()
    v.update({f"{lab}_pm_re": pm.real, f"{lab}_pm_im": pm.imag, f"{lab}_pp_re": pp.real, f"{lab}_pp_im": pp.imag,
              f"{lab}_Fi_re": Fi.real, f"{lab}_Fi_im": Fi.imag, f"{lab}_Fj_re": Fj.real, f"{lab}_Fj_im": Fj.imag})
    # model-free displacement statistics projected on the unit vector of k: moments for the Pearson coefficient
    kn = np.linalg.norm(kk, axis=1)
    ui, uj = ai / kn, aj / kn
    v.update({f"{lab}_ui": ui.mean(), f"{lab}_uj": uj.mean(), f"{lab}_uii": (ui * ui).mean(),
              f"{lab}_ujj": (uj * uj).mean(), f"{lab}_uij": (ui * uj).mean()})


def iso_obs(v, lab, dx, ci, cj, K):
    Pi, Pj = dx[ci] @ K.T, dx[cj] @ K.T
    Fi_k, Fj_k = np.exp(1j * Pi).mean(0), np.exp(1j * Pj).mean(0)
    v[f"{lab}_pm_re"] = np.cos(Pi - Pj).mean()
    v[f"{lab}_pp_re"] = np.cos(Pi + Pj).mean()
    v[f"{lab}_prod_pm_re"] = (Fi_k * np.conj(Fj_k)).real.mean()
    v[f"{lab}_prod_pp_re"] = (Fi_k * Fj_k).real.mean()


def process(args):
    path, outdir = args
    tag = os.path.basename(path).replace("run_", "").replace(".npz", "")
    cache = os.path.join(outdir, f"curves_{tag}.npz")
    d = np.load(path)
    if os.path.exists(cache):
        c = np.load(cache)
        if int(c["sweeps"]) == int(d["sweeps"]):
            return cache
    traj = d["traj"].astype(np.float64)
    t_save = d["t_save"].astype(np.int64)
    sig = d["sig"].astype(np.float64)
    L, N, phi = float(d["L"]), int(d["N"]), float(d["phi"])
    T = int(d["sweeps"])
    origins = d["origins"].astype(np.int64)
    origins = origins[origins < t_save[-1]]
    rng = np.random.default_rng(12345)
    kv, kmag = {}, {}
    for kt in K_TARGETS:
        kv[kt], kmag[kt] = A.lattice_kvecs(kt, L)
    K7, k7 = kv[7.0], kmag[7.0]
    nb = int(np.ceil(A.PER_DECADE * np.log10(T))) + 2
    acc = A.Accum(nb)
    # per-origin Fs7 curves (aging check)
    per_orig = {}
    ncont, ncs, nshell = [], [], {k: [] for k in SHELLS}
    for t0 in origins:
        i0 = int(np.searchsorted(t_save, t0))
        assert t_save[i0] == t0
        x0 = traj[i0]
        P, dv, r, s = pairs_upto(x0, sig, L, 2.1)
        cm = s < CONTACT
        ci, cj = P[cm, 0], P[cm, 1]
        rvec = dv[cm]
        rhat = rvec / r[cm][:, None]
        sij_c = 0.5 * (sig[ci] + sig[cj])
        ncont.append(len(ci))
        kpar, kperp = k7 * rhat, k7 * perp_of(rhat)
        k5par, k10par = kmag[5.0] * rhat, kmag[10.0] * rhat
        nc = np.minimum(common_neighbours(N, P, s, ci, cj), NC_MAX)
        ncs.append(np.bincount(nc, minlength=NC_MAX + 1))
        sh = {}
        for lab, (lo, hi) in SHELLS.items():
            m = (s > lo) & (s < hi)
            sh[lab] = (P[m, 0], P[m, 1], dv[m] / r[m][:, None])
            nshell[lab].append(int(m.sum()))
        ri = rng.integers(0, N, N_RAND)
        rj = (ri + rng.integers(1, N, N_RAND)) % N
        dr = x0[ri] - x0[rj]; dr -= L * np.rint(dr / L)
        rrhat = dr / np.linalg.norm(dr, axis=1)[:, None]
        po = per_orig.setdefault(int(t0), [])
        for f in range(i0 + 1, len(t_save)):
            lag = int(t_save[f] - t0)
            b = int(np.rint(A.PER_DECADE * np.log10(lag)))
            dx = traj[f] - x0
            dx = dx - dx.mean(0)
            r2 = (dx ** 2).sum(1)
            v = {"msd": r2.mean(), "r4": (r2 ** 2).mean()}
            for kt in K_TARGETS:
                v[f"Fs{int(kt)}"] = np.cos(dx @ kv[kt].T).mean()
            po.append((lag, v["Fs7"], v["msd"]))
            # random pairs (lattice k7 set) and random pairs with k along their own separation
            pi_, pj_ = dx[ri] @ K7.T, dx[rj] @ K7.T
            v["r_pm"] = np.cos(pi_ - pj_).mean(); v["r_pp"] = np.cos(pi_ + pj_).mean()
            a1, a2 = (dx[ri] * rrhat).sum(1) * k7, (dx[rj] * rrhat).sum(1) * k7
            v["rL_pm"] = np.cos(a1 - a2).mean()
            bond_obs(v, "rL", dx, ri, rj, k7 * rrhat)
            # contact pairs
            bond_obs(v, "L", dx, ci, cj, kpar)
            bond_obs(v, "T", dx, ci, cj, kperp)
            bond_obs(v, "L5", dx, ci, cj, k5par)
            bond_obs(v, "L10", dx, ci, cj, k10par)
            iso_obs(v, "I", dx, ci, cj, K7)
            rt = np.linalg.norm(rvec + dx[ci] - dx[cj], axis=1) / sij_c
            v["end105"] = (rt < 1.05).mean(); v["end12"] = (rt < 1.2).mean()
            # common-neighbour resolved, parallel, per n_c value
            ai, aj = (dx[ci] * kpar).sum(1), (dx[cj] * kpar).sum(1)
            epm, ei, ej = np.exp(1j * (ai - aj)), np.exp(1j * ai), np.exp(1j * aj)
            cnt = np.bincount(nc, minlength=NC_MAX + 1).astype(float)
            for q, arr in (("pm", epm), ("Fi", ei), ("Fj", ej)):
                sre = np.bincount(nc, weights=arr.real, minlength=NC_MAX + 1)
                sim = np.bincount(nc, weights=arr.imag, minlength=NC_MAX + 1)
                for n in range(NC_MAX + 1):
                    v[f"nc{n}_{q}_sre"] = sre[n]; v[f"nc{n}_{q}_sim"] = sim[n]
            for n in range(NC_MAX + 1):
                v[f"nc{n}_cnt"] = cnt[n]
            ui, uj = ai / k7, aj / k7
            for q, arr in (("ui", ui), ("uj", uj), ("uii", ui * ui), ("ujj", uj * uj), ("uij", ui * uj)):
                sm = np.bincount(nc, weights=arr, minlength=NC_MAX + 1)
                for n in range(NC_MAX + 1):
                    v[f"nc{n}_{q}_s"] = sm[n]
            # shells
            for lab, (si_, sj_, sh_hat) in sh.items():
                bond_obs(v, f"{lab}L", dx, si_, sj_, k7 * sh_hat)
                bond_obs(v, f"{lab}T", dx, si_, sj_, k7 * perp_of(sh_hat))
                iso_obs(v, f"{lab}I", dx, si_, sj_, K7)
            acc.add(b, lag, **v)
    res = acc.result()
    # aging: Fs7 curves of the first and the last origin
    keys = sorted(per_orig)
    first = np.array(per_orig[keys[0]]); last = np.array(per_orig[keys[-1]])
    np.savez(cache, tag=tag, phi=phi, N=N, L=L, sweeps=T, acc=float(d["acc"]), delta=float(d["delta"]),
             seed=int(d["seed"]), norig=len(origins), n_contacts=np.mean(ncont), nc_hist=np.sum(ncs, 0),
             n_shell_n=np.mean(nshell["n"]), n_shell_g=np.mean(nshell["g"]), n_shell_s=np.mean(nshell["s"]), kmag=np.array([kmag[k] for k in K_TARGETS]),
             nvec=np.array([len(kv[k]) for k in K_TARGETS]), Z=d["Z"], aging_first=first, aging_last=last,
             partial=bool(d["partial"]), **res)
    print(f"  processed {tag}: phi={phi:.4f} N={N} T={T} contacts/origin={np.mean(ncont):.0f} "
          f"<n_c>={np.average(np.arange(NC_MAX+1), weights=np.sum(ncs,0)):.2f}", flush=True)
    return cache


def main():
    outdir = sys.argv[1]
    files = [f for f in sys.argv[2:] if not f.endswith(".tmp.npz")]
    os.makedirs(outdir, exist_ok=True)
    nproc = int(os.environ.get("NPROC", "8"))
    with Pool(nproc) as p:
        p.map(process, [(f, outdir) for f in files])


if __name__ == "__main__":
    main()
