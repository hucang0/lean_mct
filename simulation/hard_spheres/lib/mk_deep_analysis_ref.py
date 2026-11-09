#!/usr/bin/env python3
"""
Analysis of MK (lambda = infinity) Monte Carlo trajectories produced by mk_mc.py.

Observables (all as functions of the lag tau, in MC sweeps; averaged over all
saved time origins of a run, then over independent planted samples):

  MSD(tau)             mean squared displacement (unwrapped positions)
  alpha2(tau)          non-Gaussian parameter 3<r^4>/(5<r^2>^2) - 1
  Fs(k,tau)            self intermediate scattering function, k = 2 pi n / L with
                       |n|^2 fixed, averaged over all lattice directions with that
                       |n|^2 (targets k ~ 3, 5, 7, 10)
  Cpi(tau)             MK's C(q=pi, t) = (1/3N) sum_i sum_a cos(pi dx_ia)  (their Eq. 74)
  phi2_*(tau)          two-particle chain correlator <e^{i k1.dx_i} e^{i k2.dx_j}>
                       with |k1| = |k2| = k7 (lattice set near k=7):
                         'pp' : k2 = +k1   (centre-of-mass motion of the pair)
                         'pm' : k2 = -k1   (relative motion of the pair)
                       for (a) pairs in contact at the origin (|r_ij| < 1.05 in the
                       shifted metric), (b) random pairs; and the longitudinal
                       variants 'Lpp', 'Lpm' with k1 = k7 * rhat_ij (contact pairs).
  phi3_*(tau)          three-particle chain i-j-l (i-j and j-l both in contact):
                       <e^{i k.(dx_i + dx_j + dx_l)}>, lattice set near k=7, and
                       the same for random triples.

Derived: factorization ratio R(tau) = phi2 / Fs(k7)^2 (and phi3 / Fs^3),
relaxation times tau_alpha (Fs(k7) = 1/e), plateau heights, power-law fit of
tau_alpha(phi).

Usage: analysis.py [--data data] [--out results] [--figures figures] [--nproc 4]
Per-file results are cached in results/curves_<tag>.npz.
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

PER_DECADE = 12            # lag bins per decade (matches the saving schedule)
K_TARGETS = [3.0, 5.0, 7.0, 10.0]
K_FACT = 7.0               # k used for the factorization test
N_RAND_PAIRS = 4000
N_MAX_VEC = 24             # max lattice vectors per |k| (direction average)
CONTACT_CUT = 1.05


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def lattice_kvecs(k_target, L, nmax_vec=N_MAX_VEC, seed=0):
    """All integer vectors n (mod overall sign) with |n|^2 = m, m chosen so that
    2 pi sqrt(m)/L is closest to k_target. Returns (kvecs (nv,3), |k|)."""
    nmax = int(np.ceil(k_target * L / (2 * np.pi))) + 2
    r = np.arange(-nmax, nmax + 1)
    n = np.array(np.meshgrid(r, r, r, indexing="ij")).reshape(3, -1).T
    m = (n ** 2).sum(1)
    ms = np.unique(m[m > 0])
    kk = 2 * np.pi * np.sqrt(ms) / L
    mstar = ms[np.argmin(np.abs(kk - k_target))]
    sel = n[m == mstar]
    # remove overall sign duplicates: keep vectors whose first nonzero comp > 0
    keep = []
    for v in sel:
        nz = v[v != 0]
        if nz[0] > 0:
            keep.append(v)
    sel = np.array(keep, dtype=np.float64)
    if len(sel) > nmax_vec:
        rng = np.random.default_rng(seed)
        sel = sel[rng.choice(len(sel), nmax_vec, replace=False)]
    return sel * (2 * np.pi / L), 2 * np.pi * np.sqrt(mstar) / L


def find_contacts(x, A, L, cut=CONTACT_CUT):
    """Pairs (i<j) with shifted minimum-image distance < cut; returns idx (P,2),
    unit vectors rhat (P,3) of the shifted separation x_i - x_j - A_ij."""
    N = x.shape[0]
    d = x[:, None, :] - x[None, :, :] - A
    d -= L * np.rint(d / L)
    r2 = (d ** 2).sum(-1)
    iu = np.triu_indices(N, 1)
    mask = r2[iu] < cut * cut
    pi, pj = iu[0][mask], iu[1][mask]
    rv = d[pi, pj]
    rhat = rv / np.linalg.norm(rv, axis=1)[:, None]
    nmin = r2[iu].min()
    return np.stack([pi, pj], 1), rhat, nmin, rv


def chains_from_contacts(pairs, nmax=4000, rng=None):
    """Triples (i, j, l), i != l, with (i,j) and (j,l) both contacts."""
    adj = {}
    for i, j in pairs:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    tri = []
    for j, nb in adj.items():
        if len(nb) >= 2:
            for a in range(len(nb)):
                for b in range(a + 1, len(nb)):
                    tri.append((nb[a], j, nb[b]))
    tri = np.array(tri, dtype=np.int64).reshape(-1, 3)
    if len(tri) > nmax and rng is not None:
        tri = tri[rng.choice(len(tri), nmax, replace=False)]
    return tri


class Accum:
    """Binned accumulator over (origin, frame) pairs."""

    def __init__(self, nbins):
        self.nbins = nbins
        self.sums = {}
        self.cnt = np.zeros(nbins)
        self.lag = np.zeros(nbins)

    def add(self, b, lag, **vals):
        self.cnt[b] += 1
        self.lag[b] += np.log(lag)
        for k, v in vals.items():
            if k not in self.sums:
                self.sums[k] = np.zeros(self.nbins)
            self.sums[k][b] += v

    def result(self):
        ok = self.cnt > 0
        out = {"lag": np.exp(self.lag[ok] / self.cnt[ok]), "count": self.cnt[ok]}
        for k, v in self.sums.items():
            out[k] = v[ok] / self.cnt[ok]
        return out


# ----------------------------------------------------------------------------
# per-file processing
# ----------------------------------------------------------------------------

def process_file(path, outdir):
    tag = os.path.basename(path).replace("run_", "").replace(".npz", "")
    cache = os.path.join(outdir, f"curves_{tag}.npz")
    d = np.load(path)
    if os.path.exists(cache):
        c = np.load(cache, allow_pickle=True)
        if int(c["sweeps"]) == int(d["sweeps"]):
            return cache
    traj = d["traj"].astype(np.float64)
    t_save = d["t_save"].astype(np.int64)
    A = d["A"].astype(np.float64)
    L, N, phi = float(d["L"]), int(d["N"]), float(d["phi"])
    origins = d["origins"].astype(np.int64)
    T = int(d["sweeps"])                      # sweeps actually completed (partial files: < planned)
    origins = origins[origins < t_save[-1]]   # origins not yet reached in a partial file are skipped
    partial = bool(d["partial"]) if "partial" in d.files else False
    rng = np.random.default_rng(12345)

    kv = {}
    kmag = {}
    for kt in K_TARGETS:
        kv[kt], kmag[kt] = lattice_kvecs(kt, L)
    K7, k7 = kv[K_FACT], kmag[K_FACT]

    nbins = int(np.ceil(PER_DECADE * np.log10(T))) + 2
    acc = Accum(nbins)
    n_contacts, n_chains, rmins = [], [], []
    for t0 in origins:
        i0 = int(np.searchsorted(t_save, t0))
        assert t_save[i0] == t0
        x0 = traj[i0]
        pairs, rhat, rmin, rvec = find_contacts(x0, A, L)
        rmins.append(rmin)
        n_contacts.append(len(pairs))
        ci, cj = pairs[:, 0], pairs[:, 1]
        kpar = k7 * rhat                                    # (P,3)
        tmp = np.cross(rhat, np.where(np.abs(rhat[:, [0]]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
        kperp = k7 * tmp / np.linalg.norm(tmp, axis=1)[:, None]   # (P,3), perpendicular to rhat
        ri = rng.integers(0, N, N_RAND_PAIRS)
        rj = (ri + rng.integers(1, N, N_RAND_PAIRS)) % N   # j != i
        tri = chains_from_contacts(pairs, 4000, rng)
        n_chains.append(len(tri))
        rtri = np.stack([rng.permutation(N)[:3] for _ in range(1000)], 0)
        for f in range(i0 + 1, len(t_save)):
            lag = int(t_save[f] - t0)
            b = int(np.rint(PER_DECADE * np.log10(lag)))
            dx = traj[f] - x0
            cm = dx.mean(0)                     # centre-of-mass drift: an exact zero mode of the
            dx = dx - cm                        # all-pairs model, diffusing with MSD = acc*delta^2*t/N
            r2 = (dx ** 2).sum(1)
            vals = {"msd": r2.mean(), "r4": (r2 ** 2).mean(), "msd_cm": float(cm @ cm)}
            for kt in K_TARGETS:
                vals[f"Fs{int(kt)}"] = np.cos(dx @ kv[kt].T).mean()
            vals["Cpi"] = np.cos(np.pi * dx).mean()
            # two-particle chain correlator, lattice k7 set
            pi_, pj_ = dx[ci] @ K7.T, dx[cj] @ K7.T
            vals["c_pp"] = np.cos(pi_ + pj_).mean()
            vals["c_pm"] = np.cos(pi_ - pj_).mean()
            pi_, pj_ = dx[ri] @ K7.T, dx[rj] @ K7.T
            vals["r_pp"] = np.cos(pi_ + pj_).mean()
            vals["r_pm"] = np.cos(pi_ - pj_).mean()
            # longitudinal (k parallel to the contact vector), |k| = k7
            pi_, pj_ = (dx[ci] * kpar).sum(1), (dx[cj] * kpar).sum(1)
            vals["c_Lpp"] = np.cos(pi_ + pj_).mean()
            vals["c_Lpm"] = np.cos(pi_ - pj_).mean()
            vals["c_Lpm_sin"] = np.sin(pi_ - pj_).mean()      # Im part (zero for any factorized form)
            pi_, pj_ = (dx[ci] * kperp).sum(1), (dx[cj] * kperp).sum(1)
            vals["c_Tpp"] = np.cos(pi_ + pj_).mean()
            vals["c_Tpm"] = np.cos(pi_ - pj_).mean()
            # contact survival: fraction of initially-contacting pairs still within 1.05 (and 1.2)
            rt = np.linalg.norm(rvec + dx[ci] - dx[cj], axis=1)
            vals["surv105"] = (rt < 1.05).mean()
            vals["surv12"] = (rt < 1.2).mean()
            # three-particle chains
            if len(tri):
                p3 = dx[tri[:, 0]] @ K7.T + dx[tri[:, 1]] @ K7.T + dx[tri[:, 2]] @ K7.T
                vals["c3"] = np.cos(p3).mean()
            p3 = dx[rtri[:, 0]] @ K7.T + dx[rtri[:, 1]] @ K7.T + dx[rtri[:, 2]] @ K7.T
            vals["r3"] = np.cos(p3).mean()
            acc.add(b, lag, **vals)
    res = acc.result()
    np.savez(cache, tag=tag, phi=phi, N=N, L=L, sweeps=T, acc=float(d["acc"]),
             wall=float(d["wall"]), delta=float(d["delta"]), seed=int(d["seed"]),
             norig=len(origins), n_contacts=np.mean(n_contacts), n_chains=np.mean(n_chains),
             rmin=np.min(rmins), partial=partial, kmag=np.array([kmag[k] for k in K_TARGETS]),
             nvec=np.array([len(kv[k]) for k in K_TARGETS]), **res)
    print(f"  processed {tag}: phi={phi} N={N} T={T} contacts/origin={np.mean(n_contacts):.0f} "
          f"chains={np.mean(n_chains):.0f} rmin={np.min(rmins):.4f}", flush=True)
    return cache


# ----------------------------------------------------------------------------
# aggregation across samples
# ----------------------------------------------------------------------------

def load_curves(outdir):
    runs = []
    for f in sorted(glob.glob(os.path.join(outdir, "curves_*.npz"))):
        c = dict(np.load(f, allow_pickle=True))
        runs.append({k: (v.item() if v.shape == () else v) for k, v in c.items()})
    return runs


def group_runs(runs):
    groups = {}
    for r in runs:
        groups.setdefault((round(float(r["phi"]), 3), int(r["N"])), []).append(r)
    return dict(sorted(groups.items()))


def average_group(rs, key):
    """Average curve 'key' over samples on the union lag grid (bins by log lag).
    Returns lag, mean, sem (sample-to-sample), n_samples per bin."""
    bins = {}
    for r in rs:
        if key not in r:
            continue
        for lag, v in zip(r["lag"], r[key]):
            b = int(np.rint(PER_DECADE * np.log10(lag)))
            bins.setdefault(b, []).append((lag, v))
    out_lag, out_m, out_s, out_n = [], [], [], []
    for b in sorted(bins):
        arr = np.array(bins[b])
        out_lag.append(np.exp(np.log(arr[:, 0]).mean()))
        out_m.append(arr[:, 1].mean())
        n = len(arr)
        out_s.append(arr[:, 1].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan)
        out_n.append(n)
    return np.array(out_lag), np.array(out_m), np.array(out_s), np.array(out_n)


def ratio_group(rs, num_key, den_key, power):
    """R = num / den^power computed per sample, then mean +- sem across samples."""
    bins = {}
    for r in rs:
        if num_key not in r:
            continue
        for lag, nu, de in zip(r["lag"], r[num_key], r[den_key]):
            if de <= 0.15:
                continue
            b = int(np.rint(PER_DECADE * np.log10(lag)))
            bins.setdefault(b, []).append((lag, nu / de ** power))
    lag, m, s = [], [], []
    for b in sorted(bins):
        arr = np.array(bins[b])
        lag.append(np.exp(np.log(arr[:, 0]).mean()))
        m.append(arr[:, 1].mean())
        s.append(arr[:, 1].std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.nan)
    return np.array(lag), np.array(m), np.array(s)


def crossing_time(lag, F, level=np.exp(-1)):
    """First crossing of F below 'level' (log-lag interpolation). None if no crossing."""
    for i in range(1, len(F)):
        if F[i - 1] > level >= F[i]:
            x0, x1 = np.log(lag[i - 1]), np.log(lag[i])
            w = (F[i - 1] - level) / (F[i - 1] - F[i])
            return float(np.exp(x0 + w * (x1 - x0)))
    return None


def plateau_estimate(lag, F, tmin=20.0):
    """Plateau height: F at the point of smallest |d ln F / d log10 t| for
    t >= tmin (after the short-time cage-formation decay), using a 3-point
    smoothed log-derivative. Returns (f_plateau, t_plateau, min_slope) or None."""
    m = (lag >= tmin) & (F > 0.05)
    if m.sum() < 5:
        return None
    lt, lF = np.log10(lag[m]), np.log(F[m])
    # smooth F in log t a little (running mean of 3) before differentiating
    lFs = np.convolve(lF, np.ones(3) / 3, mode="same")
    lFs[0], lFs[-1] = lF[0], lF[-1]
    slope = np.gradient(lFs, lt)
    # ignore the last two points (edge effects of the smoothing)
    inner = slice(1, max(2, len(slope) - 2))
    i = np.argmin(np.abs(slope[inner])) + 1
    return float(np.exp(lF[i])), float(10 ** lt[i]), float(slope[i])


def fit_power_law(phis, taus, gamma_fixed=None):
    """Fit tau = tau0 (phi_d - phi)^(-gamma) by least squares in log tau."""
    from scipy.optimize import least_squares
    phis, taus = np.asarray(phis, float), np.asarray(taus, float)
    lt = np.log(taus)

    def resid(p):
        if gamma_fixed is None:
            lt0, phid, g = p
        else:
            lt0, phid = p
            g = gamma_fixed
        if phid <= phis.max():
            return 1e3 * np.ones_like(lt)
        return lt0 - g * np.log(phid - phis) - lt
    p0 = [0.0, phis.max() + 0.15] + ([4.5] if gamma_fixed is None else [])
    sol = least_squares(resid, p0, method="lm" if len(phis) > len(p0) else "trf")
    res = sol.fun
    dof = max(len(phis) - len(p0), 1)
    try:
        J = sol.jac
        cov = np.linalg.inv(J.T @ J) * (res @ res) / dof
        err = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        err = np.full(len(p0), np.nan)
    out = {"log_tau0": float(sol.x[0]), "phi_d": float(sol.x[1]), "phi_d_err": float(err[1]),
           "rms_log_resid": float(np.sqrt((res @ res) / len(res)))}
    if gamma_fixed is None:
        out["gamma"] = float(sol.x[2])
        out["gamma_err"] = float(err[2])
    else:
        out["gamma"] = gamma_fixed
        out["gamma_err"] = 0.0
    return out


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 130,
    })
    return plt


def phi_colors(phis):
    import matplotlib.cm as cm
    phis = sorted(phis)
    import matplotlib
    cmap = matplotlib.colormaps["viridis"]
    return {p: cmap(0.08 + 0.84 * i / max(len(phis) - 1, 1)) for i, p in enumerate(phis)}


def savefig(fig, figdir, name):
    fig.savefig(os.path.join(figdir, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(figdir, name + ".pdf"), bbox_inches="tight")


def make_figures(groups, tables, figdir):
    plt = setup_mpl()
    g1000 = {k: v for k, v in groups.items() if k[1] == 1000}
    phis = [k[0] for k in g1000]
    cols = phi_colors(phis)

    # --- MSD -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for (phi, N), rs in g1000.items():
        lag, m, s, n = average_group(rs, "msd")
        ax.plot(lag, m, color=cols[phi], label=f"φ = {phi:.2f}")
        ax.annotate(f"{phi:.2f}", (lag[-1], m[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=cols[phi], va="center")
    g2000 = [k for k in groups if k[1] == 2000]
    for i2, k2 in enumerate(g2000):
        lag, m, s, n = average_group(groups[k2], "msd")
        ax.plot(lag, m, color="k", ls="--", lw=1.0, label="N = 2000 (φ = %s)" % ", ".join(f"{k[0]:.2f}" for k in g2000) if i2 == 0 else None)
    if (2.0, 1000) in g1000:
        lag, m, s, n = average_group(g1000[(2.0, 1000)], "msd_cm")
        ax.plot(lag, m, color="0.5", lw=0.8, ls="-.", label="CM drift |⟨Δx⟩|² (φ=2.0, subtracted)")
    xx = np.logspace(0, 2, 10)
    ax.plot(xx, 0.0045 * xx, color="0.6", lw=0.8, ls=":")
    ax.text(30, 0.3, "slope 1", color="0.4", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("t  (MC sweeps)"); ax.set_ylabel("MSD(t)  (units of the diameter²)")
    ax.set_title("MK model, λ = ∞, N = 1000: mean squared displacement", fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    savefig(fig, figdir, "msd"); plt.close(fig)

    # --- Fs(k~7,t) -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for (phi, N), rs in g1000.items():
        lag, m, s, n = average_group(rs, "Fs7")
        ax.plot(lag, m, color=cols[phi], label=f"φ = {phi:.2f}")
        ax.fill_between(lag, m - np.nan_to_num(s), m + np.nan_to_num(s), color=cols[phi], alpha=0.2, lw=0)
    for i2, k2 in enumerate(g2000):
        lag, m, s, n = average_group(groups[k2], "Fs7")
        ax.plot(lag, m, color="k", ls="--", lw=1.0, label="N = 2000 (φ = %s)" % ", ".join(f"{k[0]:.2f}" for k in g2000) if i2 == 0 else None)
    ax.axhline(np.exp(-1), color="0.6", lw=0.8, ls=":")
    ax.text(1.2, np.exp(-1) + 0.015, "1/e", color="0.4", fontsize=8)
    k7 = tables["kvalues"]["1.5"]["k7"] if "1.5" in tables["kvalues"] else 7.0
    ax.set_xscale("log"); ax.set_xlabel("t  (MC sweeps)")
    ax.set_ylabel("F_s(k, t)")
    ax.set_title(f"Self intermediate scattering function, k ≈ 7 (lattice k closest to 7; k = {k7:.2f} at φ = 1.5)", fontsize=9)
    ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8, ncol=2, loc="lower left")
    savefig(fig, figdir, "fs_k7"); plt.close(fig)

    # --- Fs at all k for selected densities (small multiples) -------------------
    sel = [p for p in phis if p in (1.5, 1.65, 1.8, 2.0)] or phis[-4:]
    fig, axs = plt.subplots(1, len(sel), figsize=(3.1 * len(sel), 3.2), sharey=True)
    for ax, phi in zip(np.atleast_1d(axs), sel):
        rs = g1000[(phi, 1000)]
        for j, kt in enumerate(K_TARGETS):
            lag, m, s, n = average_group(rs, f"Fs{int(kt)}")
            ax.plot(lag, m, color=PAL[j], label=f"k ≈ {int(kt)}")
        lag, m, s, n = average_group(rs, "Cpi")
        ax.plot(lag, m, color="0.3", ls="--", lw=1.0, label="C(q=π,t) [MK Eq. 74]")
        ax.set_xscale("log"); ax.set_title(f"φ = {phi:.2f}", fontsize=10); ax.set_xlabel("t (sweeps)")
        ax.set_ylim(-0.02, 1.02)
    np.atleast_1d(axs)[0].set_ylabel("F_s(k,t)")
    np.atleast_1d(axs)[0].legend(fontsize=7, loc="lower left")
    savefig(fig, figdir, "fs_allk"); plt.close(fig)

    # --- plateau vs phi ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for j, kt in enumerate(K_TARGETS):
        xs, ys, es = [], [], []
        for phi_s, row in tables["plateaus"].items():
            if "_N" in phi_s:
                continue
            r = row.get(f"k{int(kt)}")
            if r and r["has_plateau"]:
                xs.append(float(phi_s)); ys.append(r["f_plateau"]); es.append(r["f_err"] or 0)
        if xs:
            ax.errorbar(xs, ys, yerr=es, color=PAL[j], marker="o", ms=5, capsize=2, label=f"k ≈ {int(kt)}")
        # Gaussian estimate from the MSD at the same time
        xs, ys = [], []
        for phi_s, row in tables["plateaus"].items():
            if "_N" in phi_s:
                continue
            r = row.get(f"k{int(kt)}")
            if r and r["has_plateau"] and r.get("f_gauss") is not None:
                xs.append(float(phi_s)); ys.append(r["f_gauss"])
        if xs:
            ax.plot(xs, ys, color=PAL[j], ls="none", marker="x", ms=6, alpha=0.7)
    ax.plot([], [], color="0.3", ls="none", marker="x", label="exp(−k² MSD/6) at plateau")
    # quasi-plateau: F_s(k, t = 10^4 sweeps) for every density (open symbols), whether or not a true plateau exists
    for j, kt in enumerate(K_TARGETS):
        xs, ys = [], []
        for phi_s, row in tables["plateaus"].items():
            if "_N" in phi_s:
                continue
            r = row.get(f"k{int(kt)}", {})
            if "Fs_at_t10000" in r:
                xs.append(float(phi_s)); ys.append(r["Fs_at_t10000"])
        if xs:
            ax.plot(xs, ys, color=PAL[j], ls=":", lw=1.0, marker="o", mfc="none", ms=5)
    ax.plot([], [], color="0.3", ls=":", marker="o", mfc="none", label="F_s(k, t = 10⁴) (quasi-plateau, all φ)")
    for phi_s, row in tables["plateaus"].items():
        if "_N2000" in phi_s:
            for j, kt in enumerate(K_TARGETS):
                r = row.get(f"k{int(kt)}", {})
                if r.get("has_plateau"):
                    ax.plot([float(phi_s.split("_")[0])], [r["f_plateau"]], color=PAL[j], marker="s", mfc="none", ms=8, ls="none")
    ax.plot([], [], color="0.3", marker="s", mfc="none", ms=8, ls="none", label="N = 2000")
    ax.set_xlim(0.7, 2.1)
    ax.axvline(1.82, color="0.5", lw=0.8, ls="--"); ax.text(1.825, 0.05, "MK φ_d = 1.82", fontsize=8, color="0.4")
    ax.set_xlabel("φ"); ax.set_ylabel("plateau height  f_s(k)")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
    ax.set_title("Non-ergodicity (plateau) estimate vs density", fontsize=10)
    savefig(fig, figdir, "plateau_vs_phi"); plt.close(fig)

    # --- tau(phi) and fit ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    tt = {p: v for p, v in tables["tau"].items() if "_N" not in p}
    for j, (obs, lab, mk) in enumerate((("tau_k3", "F_s(k≈3) = 1/e", "o"), ("tau_pi", "C(q=π,t) = 1/e [MK]", "s"),
                                        ("tau_k7", "F_s(k≈7) = 1/e", "^"), ("tau_k7_0.1", "F_s(k≈7) = 0.1", "v"),
                                        ("tau_msd_0.5", "MSD = 0.5", "D"))):
        xs = [float(p) for p in tt if tt[p][obs] is not None]
        ys = [tt[p][obs] for p in tt if tt[p][obs] is not None]
        es = [tt[p][obs + "_err"] or 0 for p in tt if tt[p][obs] is not None]
        ax.errorbar(xs, ys, yerr=es, color=PAL[j], marker=mk, ms=5, ls="none", capsize=2, label=lab)
    xs = [float(p) for p in tt if tt[p]["tau_k3"] is not None]
    for key, ls, col in (("fit_free", "-", "0.2"), ("fit_gamma4.5", "--", "0.5")):
        f = tables["tau_fit"].get("tau_k3", {}).get(key)
        if f and "phi_d" in f:
            xx = np.linspace(min(xs) - 0.02, f["phi_d"] - 0.01, 200)
            ax.plot(xx, np.exp(f["log_tau0"]) * (f["phi_d"] - xx) ** (-f["gamma"]), ls=ls, color=col, lw=1.0,
                    label=f"k≈3 {key}: φ_d={f['phi_d']:.2f}, γ={f['gamma']:.1f}")
    ax.set_yscale("log"); ax.set_xlabel("φ"); ax.set_ylabel("τ  (MC sweeps)")
    ax.legend(fontsize=7, loc="upper left"); ax.set_title("Relaxation time vs density", fontsize=10)
    savefig(fig, figdir, "tau_vs_phi"); plt.close(fig)

    # --- factorization ratio R(t) -----------------------------------------------
    sel = [p for p in (1.5, 2.0) if (p, 1000) in g1000] or phis[-2:]
    sel = sel if len(sel) else phis
    fig, axs = plt.subplots(1, len(sel), figsize=(3.6 * len(sel), 3.6), sharey=True)
    for ax, phi in zip(np.atleast_1d(axs), sel):
        rs = g1000[(phi, 1000)]
        for j, (key, lab) in enumerate((("c_pp", "contact, k₂ = +k₁"), ("c_pm", "contact, k₂ = −k₁"),
                                         ("r_pp", "random, k₂ = +k₁"), ("r_pm", "random, k₂ = −k₁"),
                                         ("c_Lpp", "contact, k ∥ r_ij, +"), ("c_Lpm", "contact, k ∥ r_ij, −"),
                                         ("c_Tpp", "contact, k ⊥ r_ij, +"), ("c_Tpm", "contact, k ⊥ r_ij, −"))):
            lag, m, s = ratio_group(rs, key, "Fs7", 2)
            ls = "-" if key.startswith("c_") and "L" not in key and "T" not in key else ("--" if key.startswith("r_") else (":" if "L" in key else "-."))
            ax.plot(lag, m, color=PAL[j], ls=ls, label=lab)
            ax.fill_between(lag, m - np.nan_to_num(s), m + np.nan_to_num(s), color=PAL[j], alpha=0.15, lw=0)
        ax.axhline(1, color="0.5", lw=0.8)
        ax.set_xscale("log"); ax.set_xlabel("t (sweeps)"); ax.set_title(f"φ = {phi:.2f}", fontsize=10)
        ax.set_ylim(0.5, 3.0)
    np.atleast_1d(axs)[0].set_ylabel("R(t) = φ₂(k₁,k₂,t) / [F_s(k₁,t) F_s(k₂,t)]   (shown where F_s > 0.15)", fontsize=8)
    np.atleast_1d(axs)[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Factorization test of the two-particle chain correlator, |k₁| = |k₂| = k ≈ 7", fontsize=9)
    savefig(fig, figdir, "factorization"); plt.close(fig)

    # --- raw phi2 vs Fs^2 at the two densities --------------------------------------
    fig, axs = plt.subplots(1, len(sel), figsize=(3.6 * len(sel), 3.6), sharey=True)
    for ax, phi in zip(np.atleast_1d(axs), sel):
        rs = g1000[(phi, 1000)]
        lag, m, s, n = average_group(rs, "Fs7")
        ax.plot(lag, m ** 2, color="k", lw=1.0, label="F_s(k,t)²")
        ax.plot(lag, m ** 3, color="k", lw=1.0, ls="--", label="F_s(k,t)³")
        for j, key, lab in ((0, "c_pp", "φ₂ contact (+)"), (1, "c_pm", "φ₂ contact (−)"),
                            (2, "r_pp", "φ₂ random (+)"), (6, "c3", "φ₃ chain i–j–l"), (7, "r3", "φ₃ random")):
            lag, m, s, n = average_group(rs, key)
            ax.plot(lag, m, color=PAL[j], label=lab)
        ax.set_xscale("log"); ax.set_xlabel("t (sweeps)"); ax.set_title(f"φ = {phi:.2f}", fontsize=10)
    np.atleast_1d(axs)[0].set_ylabel("correlator"); np.atleast_1d(axs)[0].legend(fontsize=7, loc="lower left")
    savefig(fig, figdir, "phi2_phi3_raw"); plt.close(fig)

    # --- contact survival and Im part --------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for (phi, N), rs in g1000.items():
        lag, m, s, n = average_group(rs, "surv105")
        axs[0].plot(lag, m, color=cols[phi], label=f"φ = {phi:.2f}")
        lag, m, s, n = average_group(rs, "c_Lpm_sin")
        axs[1].plot(lag, m, color=cols[phi])
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("t (sweeps)")
    axs[0].set_ylabel("P(pair still within 1.05 | contact at t=0)"); axs[0].legend(fontsize=7, ncol=2)
    xx = np.logspace(1, 3, 5); axs[0].plot(xx, 0.5 * xx ** -0.5, color="0.5", ls=":", lw=0.8); axs[0].text(100, 0.07, "t^-1/2", fontsize=8, color="0.4")
    axs[1].set_xscale("log"); axs[1].set_xlabel("t (sweeps)"); axs[1].set_ylabel("Im φ₂ (contact pairs, k ∥ r̂_ij, k₂ = −k₁)", fontsize=8)
    axs[1].axhline(0, color="0.5", lw=0.8)
    axs[0].set_ylabel("P(still within 1.05 | contact at t = 0)", fontsize=8)
    fig.tight_layout()
    savefig(fig, figdir, "contact_survival"); plt.close(fig)


# ----------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------

def make_tables(groups, outdir):
    tables = {"runs": [], "kvalues": {}, "tau": {}, "plateaus": {}, "factorization": {}, "tau_fit": {}}
    for (phi, N), rs in groups.items():
        for r in rs:
            tables["runs"].append({"tag": r["tag"], "phi": phi, "N": N, "sweeps": int(r["sweeps"]),
                                   "n_origins": int(r["norig"]), "acceptance": round(float(r["acc"]), 4),
                                   "wall_min": round(float(r["wall"]) / 60, 1), "delta": float(r["delta"]),
                                   "contacts_per_origin": round(float(r["n_contacts"]), 1),
                                   "chains_per_origin": round(float(r["n_chains"]), 1),
                                   "min_pair_distance_at_origins": round(float(r["rmin"]), 5),
                                   "partial": bool(r.get("partial", False))})
        key = f"{phi}" if N == 1000 else f"{phi}_N{N}"
        tables["kvalues"][key] = {f"k{int(kt)}": float(rs[0]["kmag"][i]) for i, kt in enumerate(K_TARGETS)}
        tables["kvalues"][key]["nvec"] = [int(v) for v in rs[0]["nvec"]]
        tables["kvalues"][key]["L"] = float(rs[0]["L"])

        # relaxation times (group average, and per-sample for the error)
        row = {"n_samples": len(rs), "sweeps_max": int(max(r["sweeps"] for r in rs))}
        for name, ck, lev in (("tau_k7", "Fs7", np.exp(-1)), ("tau_pi", "Cpi", np.exp(-1)), ("tau_k3", "Fs3", np.exp(-1)),
                              ("tau_k5", "Fs5", np.exp(-1)), ("tau_k10", "Fs10", np.exp(-1)),
                              ("tau_k7_0.1", "Fs7", 0.1), ("tau_msd_0.5", "msd", 0.5)):
            lag, m, s, n = average_group(rs, ck)
            if ck == "msd":
                m = -m; lev = -lev      # crossing_time looks for a downward crossing
            tau = crossing_time(lag, m, lev)
            per = [crossing_time(r["lag"], (-r[ck] if ck == "msd" else r[ck]), lev) for r in rs]
            per = [p for p in per if p is not None]
            row[name] = tau
            row[name + "_err"] = (float(np.std(np.log(per), ddof=1) / np.sqrt(len(per)) * tau)
                                  if (tau is not None and len(per) > 1) else None)
            row[name + "_per_sample"] = per
        # diffusion constant estimate from the last decade of the MSD, D = MSD/(6t)
        lag, m, s, n = average_group(rs, "msd")
        sel_ = lag > lag.max() / 10
        row["D_est_last_decade"] = float(np.mean(m[sel_] / (6 * lag[sel_])))
        row["msd_at_tmax"] = float(m[-1]); row["tmax"] = float(lag[-1])
        # MK-type check: is the MSD still sub-diffusive at the end?
        if sel_.sum() >= 3:
            sl = np.polyfit(np.log(lag[sel_]), np.log(m[sel_]), 1)[0]
            row["msd_log_slope_last_decade"] = float(sl)
        tables["tau"][key] = row

        # plateaus
        prow = {}
        lagm, msd, _, _ = average_group(rs, "msd")
        for i, kt in enumerate(K_TARGETS):
            lag, m, s, n = average_group(rs, f"Fs{int(kt)}")
            pe = plateau_estimate(lag, m)
            tau = crossing_time(lag, m)
            has = pe is not None and abs(pe[2]) < 0.03 and pe[0] > 0.1
            ent = {"has_plateau": bool(has)}
            if pe is not None:
                ent.update({"f_plateau": pe[0], "t_plateau": pe[1], "min_log_slope": pe[2]})
                # sample scatter at that time
                vals = []
                for r in rs:
                    j = np.argmin(np.abs(np.log(r["lag"]) - np.log(pe[1])))
                    vals.append(r[f"Fs{int(kt)}"][j])
                ent["f_err"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None
                j = np.argmin(np.abs(np.log(lagm) - np.log(pe[1])))
                ent["msd_at_plateau"] = float(msd[j])
                ent["f_gauss"] = float(np.exp(-rs[0]["kmag"][i] ** 2 * msd[j] / 6))
            ent["tau_1e"] = tau
            for tt in (1e2, 1e3, 1e4, 1e5, 1e6):
                if lag.max() >= tt * 0.9:
                    j = np.argmin(np.abs(np.log(lag) - np.log(tt)))
                    ent[f"Fs_at_t{int(tt):d}"] = float(m[j])
            prow[f"k{int(kt)}"] = ent
        tables["plateaus"][key] = prow

        # factorization table: R at selected times
        frow = {}
        for ck in ("c_pp", "c_pm", "r_pp", "r_pm", "c_Lpp", "c_Lpm", "c_Tpp", "c_Tpm"):
            lag, m, s = ratio_group(rs, ck, "Fs7", 2)
            ent = {}
            for tt in (1, 10, 100, 1e3, 1e4, 1e5, 1e6):
                if len(lag) and lag.max() >= tt * 0.9 and lag.min() <= tt * 1.1:
                    j = np.argmin(np.abs(np.log(lag) - np.log(tt)))
                    ent[f"R_t{int(tt)}"] = [float(m[j]), float(s[j]) if np.isfinite(s[j]) else None]
            # value of R at tau_alpha, where Fs = 1/e
            tau = tables["tau"][key]["tau_k7"]
            if tau is not None and len(lag):
                j = np.argmin(np.abs(np.log(lag) - np.log(tau)))
                ent["R_at_tau_alpha"] = [float(m[j]), float(s[j]) if np.isfinite(s[j]) else None]
            # max deviation from 1 over the window where Fs > 0.2
            lagF, mF, _, _ = average_group(rs, "Fs7")
            if len(lag):
                okm = np.interp(np.log(lag), np.log(lagF), mF) > 0.2
                if okm.any():
                    jm = np.argmax(np.abs(m[okm] - 1))
                    ent["max_abs_dev_Fs>0.2"] = [float(lag[okm][jm]), float(m[okm][jm])]
            frow[ck] = ent
        for ck in ("c3", "r3"):
            lag, m, s = ratio_group(rs, ck, "Fs7", 3)
            ent = {}
            for tt in (1, 10, 100, 1e3, 1e4, 1e5, 1e6):
                if len(lag) and lag.max() >= tt * 0.9 and lag.min() <= tt * 1.1:
                    j = np.argmin(np.abs(np.log(lag) - np.log(tt)))
                    ent[f"R3_t{int(tt)}"] = [float(m[j]), float(s[j]) if np.isfinite(s[j]) else None]
            frow[ck] = ent
        tables["factorization"][key] = frow

    # power-law fits tau = tau0 (phi_d - phi)^-gamma (N=1000 only, phi >= 1.2, only densities that relaxed)
    for obs in ("tau_k3", "tau_pi", "tau_k7", "tau_k7_0.1", "tau_msd_0.5"):
        pts = [(float(k), v[obs]) for k, v in tables["tau"].items()
               if "_N" not in k and v[obs] is not None and float(k) >= 1.2]
        pts0 = [(float(k), v[obs]) for k, v in tables["tau"].items()
                if "_N" not in k and v[obs] is not None]
        ent = {"points": {"phi": [p[0] for p in pts], obs: [p[1] for p in pts]}}
        if len(pts0) >= 4:
            try:
                ent["fit_free_incl_phi0.8"] = fit_power_law([p[0] for p in pts0], [p[1] for p in pts0])
            except Exception as e:  # noqa
                ent["fit_free_incl_phi0.8"] = {"error": str(e)}
        if len(pts) >= 3:
            ph, ta = zip(*pts)
            try:
                ent["fit_gamma4.5"] = fit_power_law(ph, ta, gamma_fixed=4.5)
            except Exception as e:  # noqa
                ent["fit_gamma4.5"] = {"error": str(e)}
            if len(pts) >= 4:
                try:
                    ent["fit_free"] = fit_power_law(ph, ta)
                except Exception as e:  # noqa
                    ent["fit_free"] = {"error": str(e)}
                try:
                    ent["fit_free_drop_lowest"] = fit_power_law(ph[1:], ta[1:])
                except Exception as e:  # noqa
                    ent["fit_free_drop_lowest"] = {"error": str(e)}
        tables["tau_fit"][obs] = ent

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    tables = clean(tables)
    with open(os.path.join(outdir, "run_table.json"), "w") as f:
        json.dump(tables["runs"], f, indent=1)
    with open(os.path.join(outdir, "tau_table.json"), "w") as f:
        json.dump({"tau": tables["tau"], "tau_fit": tables["tau_fit"], "kvalues": tables["kvalues"]}, f, indent=1)
    with open(os.path.join(outdir, "plateau_table.json"), "w") as f:
        json.dump(tables["plateaus"], f, indent=1)
    with open(os.path.join(outdir, "factorization_table.json"), "w") as f:
        json.dump(tables["factorization"], f, indent=1)
    # also dump the averaged curves for reuse
    curves = {}
    for (phi, N), rs in groups.items():
        key = f"{phi}" if N == 1000 else f"{phi}_N{N}"
        ent = {}
        for ck in ("msd", "msd_cm", "r4", "Fs3", "Fs5", "Fs7", "Fs10", "Cpi", "c_pp", "c_pm", "r_pp", "r_pm", "c_Lpp", "c_Lpm", "c_Lpm_sin", "c_Tpp", "c_Tpm", "surv105", "surv12", "c3", "r3"):
            lag, m, s, n = average_group(rs, ck)
            ent[ck] = {"t": lag.tolist(), "mean": m.tolist(), "sem": [None if not np.isfinite(v) else float(v) for v in s]}
        curves[key] = ent
    with open(os.path.join(outdir, "curves_averaged.json"), "w") as f:
        json.dump(curves, f)
    return tables


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data", default=os.path.join(here, "data"))
    ap.add_argument("--out", default=os.path.join(here, "results"))
    ap.add_argument("--figures", default=os.path.join(here, "figures"))
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--skip-process", action="store_true", help="rebuild tables/figures from cached curves only")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.figures, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.data, "run_*.npz")))
    print(f"{len(files)} trajectory files")
    if args.skip_process:
        pass
    elif args.nproc > 1:
        with Pool(args.nproc) as pool:
            pool.starmap(process_file, [(f, args.out) for f in files])
    else:
        for f in files:
            process_file(f, args.out)
    runs = load_curves(args.out)
    # keep only runs whose data file still exists
    tags = {os.path.basename(f).replace("run_", "").replace(".npz", "") for f in files}
    runs = [r for r in runs if r["tag"] in tags]
    groups = group_runs(runs)
    tables = make_tables(groups, args.out)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if "per_sample" not in kk and kk in ("tau_k7", "tau_pi", "n_samples")}
                      for k, v in tables["tau"].items()}, indent=1))
    print(json.dumps(tables["tau_fit"], indent=1))
    if not args.no_plots:
        make_figures(groups, tables, args.figures)
        print("figures written to", args.figures)


if __name__ == "__main__":
    main()
