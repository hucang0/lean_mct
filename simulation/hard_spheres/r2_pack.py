#!/usr/bin/env python3
"""(iii) Separate n_c from local packing.
Pass 'struct' (per run, per origin):
  * contact pairs r_ij(0) < 1.05 sigma_ij; n_c = common neighbours (neighbour: r < 1.35 sigma_ij) via the verbatim
    hs_analysis.common_neighbours on pairs_upto(x0, sig, L, 2.1).
  * radical (Laguerre / power) Voronoi cell volumes V_i with periodic images, computed on a fine grid: every grid point
    (M^3, M = 150, random sub-cell offset per configuration) is assigned to argmin_i (|x - x_i|^2 - R_i^2), R_i = sigma_i/2,
    searched among its k = 16 Euclidean nearest centres (periodic cKDTree); completeness is guaranteed when
    best power <= d_k^2 - R_max^2, otherwise the point is re-searched with k = 64 (counted).  Volumes = counts * h^3.
    The plain (unweighted) Voronoi volume comes from the same query (nearest centre).
  * pair packing: phi_pair = (v_i + v_j)/(V_i + V_j), v = pi sigma^3/6 (radical and plain Voronoi);
    probe-sphere packing phi_probe: sphere volume fraction inside a probe sphere of radius 1.5 (mean diameter units)
    centred at the pair midpoint (exact sphere-sphere overlap volumes).
Pass 'dyn': thresholds (terciles / quintiles) pooled over all seeds and origins of the same (phi, N, delta) group;
  contact-pair observables (parallel and transverse, as hs_analysis bond_obs) accumulated per (measure, n_c, packing bin).
Usage: r2_pack.py struct|dyn [hs]
"""
import os, sys, time, glob
sys.dont_write_bytecode = True
import numpy as np
from multiprocessing import Pool
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r2_common as C

MGRID = 150
KQ = 16
RPROBE = 1.5
NC_MAX = 10
# packing measures per contact pair (terciles / quintiles pooled per group).  The raw local volume fractions are strongly
# size-biased at 23% polydispersity (corr(v_i/V_i, sigma_i) ~ 0.94 even for the radical tessellation), so the primary
# measures are SIZE-CORRECTED: Vhat(sigma) = cubic-polynomial fit of the cell volume vs sigma pooled over the group,
# dens_pair = (Vhat_i + Vhat_j)/(V_i + V_j)  (> 1: denser than typical for these sizes);  probe: residual of phi_probe
# after a cubic fit in sigma_ij.
MEASURES = [("radc3", "dens_rad", 3), ("radc5", "dens_rad", 5), ("vorc3", "dens_vor", 3), ("probec3", "res_probe", 3),
            ("rad3", "phi_rad", 3), ("sig3", "sij", 3), ("sig5", "sij", 5),
            ("sig3xradc3", None, 9)]          # joint: 3 * (sigma_ij tercile) + (size-corrected radical tercile)
DF = ["cnt", "L_pm_re", "L_pm_im", "L_Fi_re", "L_Fi_im", "L_Fj_re", "L_Fj_im", "L_ui", "L_uj", "L_uii", "L_ujj", "L_uij",
      "T_pm_re", "T_pm_im", "T_Fi_re", "T_Fi_im", "T_Fj_re", "T_Fj_im", "T_ui", "T_uj", "T_uii", "T_ujj", "T_uij"]


def cell_volumes(x, sig, L, rng, M=MGRID, k=KQ):
    xw = C.wrap(x, L)
    tree = cKDTree(xw, boxsize=L)
    h = L / M
    off = rng.uniform(0, h, 3)
    ax = (np.arange(M) * h)
    R2 = (0.5 * sig) ** 2
    R2max = R2.max()
    cnt_rad = np.zeros(len(x), np.int64); cnt_vor = np.zeros(len(x), np.int64)
    nfix = 0
    for z0 in range(0, M, 6):
        zs = ax[z0:z0 + 6]
        gx, gy, gz = np.meshgrid(ax, ax, zs, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1) + off
        pts %= L
        d, idx = tree.query(pts, k=k)
        pw = d ** 2 - R2[idx]
        am = np.argmin(pw, 1)
        rows = np.arange(len(pts))
        best = pw[rows, am]
        own = idx[rows, am]
        bad = best > d[:, -1] ** 2 - R2max
        if bad.any():
            nfix += int(bad.sum())
            d2, i2 = tree.query(pts[bad], k=64)
            pw2 = d2 ** 2 - R2[i2]
            am2 = np.argmin(pw2, 1)
            own[bad] = i2[np.arange(bad.sum()), am2]
        cnt_rad += np.bincount(own, minlength=len(x))
        cnt_vor += np.bincount(idx[:, 0], minlength=len(x))
    return cnt_rad * h ** 3, cnt_vor * h ** 3, nfix


def overlap_vol(a, b, d):
    v = np.zeros_like(d)
    full = d <= np.abs(a - b)
    v[full] = 4 * np.pi / 3 * np.minimum(a, b)[full] ** 3
    part = (~full) & (d < a + b)
    aa, bb, dd = a[part], b[part], d[part]
    v[part] = np.pi * (aa + bb - dd) ** 2 * (dd ** 2 + 2 * dd * (aa + bb) - 3 * (aa - bb) ** 2) / (12 * dd)
    return v


def probe_phi(x, sig, L, centres, Rp=RPROBE):
    xw = C.wrap(x, L)
    tree = cKDTree(xw, boxsize=L)
    cw = C.wrap(centres, L)
    lists = tree.query_ball_point(cw, Rp + 0.5 * sig.max())
    out = np.zeros(len(centres))
    for n, lst in enumerate(lists):
        lst = np.array(lst, np.int64)
        d = xw[lst] - cw[n]; d -= L * np.rint(d / L)
        dd = np.linalg.norm(d, axis=1)
        out[n] = overlap_vol(0.5 * sig[lst], np.full(len(lst), Rp), dd).sum() / (4 * np.pi / 3 * Rp ** 3)
    return out


def struct(path):
    tag = C.tag_of(path)
    cache = os.path.join(C.CACHE, f"struct_{tag}.npz")
    if os.path.exists(cache):
        return cache
    t0_ = time.time()
    run = C.load_run(path)
    traj, L, N, sig = run["traj"], run["L"], run["N"], run["sig"]
    rng = np.random.default_rng(777)
    out = {k: [] for k in ("orig", "ci", "cj", "nc", "phi_rad", "phi_vor", "phi_probe", "sij", "Vrad", "Vvor")}
    nfix_tot = 0; vsum = []
    for oi, t0 in enumerate(run["origins"]):
        i0 = int(np.searchsorted(run["t_save"], t0))
        x0 = traj[i0]
        P, dv, r, s = C.pairs_upto(x0, sig, L, 2.1)
        cm = s < C.CONTACT
        ci, cj = P[cm, 0], P[cm, 1]
        nc = C.common_neighbours(N, P, s, ci, cj)
        Vr, Vv, nfix = cell_volumes(x0, sig, L, rng, M=int(round(MGRID * (N / 1000) ** (1 / 3))))
        nfix_tot += nfix; vsum.append(Vr.sum() / L ** 3)
        v = np.pi * sig ** 3 / 6
        mid = x0[cj] + 0.5 * dv[cm]           # x_j + (x_i - x_j)/2 (min image)
        out["orig"].append(np.full(len(ci), oi)); out["ci"].append(ci); out["cj"].append(cj); out["nc"].append(nc)
        out["phi_rad"].append((v[ci] + v[cj]) / (Vr[ci] + Vr[cj])); out["phi_vor"].append((v[ci] + v[cj]) / (Vv[ci] + Vv[cj]))
        out["phi_probe"].append(probe_phi(x0, sig, L, mid)); out["sij"].append(0.5 * (sig[ci] + sig[cj]))
        out["Vrad"].append(Vr); out["Vvor"].append(Vv)
    Vr2, Vv2 = np.array(out.pop("Vrad")), np.array(out.pop("Vvor"))
    if os.path.exists(cache):
        return cache
    tmp = cache.replace(".npz", f".tmp{os.getpid()}.npz")
    np.savez(tmp, tag=run["tag"], key=run["key"], phi=run["phi"], sig=sig, nfix=nfix_tot, vsum=np.array(vsum),
             Vrad=Vr2, Vvor=Vv2, **{k: np.concatenate(v) for k, v in out.items()})
    os.replace(tmp, cache)
    print(f"  struct {tag}: {time.time() - t0_:.0f}s nfix={nfix_tot} Vsum/L3={np.mean(vsum):.6f}", flush=True)
    return cache


def pair_measures(d, coef):
    """Per-contact-pair packing measures for struct cache d, given group fit coefficients."""
    sig = d["sig"]
    out = {"phi_rad": d["phi_rad"], "phi_vor": d["phi_vor"], "phi_probe": d["phi_probe"], "sij": d["sij"]}
    o, ci, cj = d["orig"], d["ci"], d["cj"]
    for lab, V in (("rad", d["Vrad"]), ("vor", d["Vvor"])):
        Vh = np.polyval(coef[f"V{lab}"], sig)
        out[f"dens_{lab}"] = (Vh[ci] + Vh[cj]) / (V[o, ci] + V[o, cj])
    out["res_probe"] = d["phi_probe"] - np.polyval(coef["probe"], d["sij"])
    return out


def thresholds():
    files = sorted(f for f in glob.glob(os.path.join(C.CACHE, "struct_*.npz")) if ".tmp" not in f)
    groups = {}
    for f in files:
        d = np.load(f)
        groups.setdefault(str(d["key"]), []).append(d)
    th = {}
    for key, ds in groups.items():
        sig_all = np.concatenate([np.tile(d["sig"], len(d["Vrad"])) for d in ds])
        coef = {"Vrad": np.polyfit(sig_all, np.concatenate([d["Vrad"].ravel() for d in ds]), 3),
                "Vvor": np.polyfit(sig_all, np.concatenate([d["Vvor"].ravel() for d in ds]), 3),
                "probe": np.polyfit(np.concatenate([d["sij"] for d in ds]), np.concatenate([d["phi_probe"] for d in ds]), 3)}
        pm = [pair_measures(d, coef) for d in ds]
        th[key] = {"coef": {k: v.tolist() for k, v in coef.items()}}
        for mname, var, nb in MEASURES:
            if var is None:
                continue
            allv = np.concatenate([p[var] for p in pm])
            th[key][mname] = np.quantile(allv, np.linspace(0, 1, nb + 1)[1:-1]).tolist()
    C.jdump(th, os.path.join(C.RES, "pack_thresholds.json"))
    return th


def dyn(args):
    path, th = args
    tag = C.tag_of(path)
    cache = os.path.join(C.CACHE, f"packdyn_{tag}.npz")
    if os.path.exists(cache):
        return cache
    t0_ = time.time()
    run = C.load_run(path)
    st = np.load(os.path.join(C.CACHE, f"struct_{tag}.npz"))
    traj, L, sig = run["traj"], run["L"], run["sig"]
    K7, k7 = C.A.lattice_kvecs(7.0, L)
    tk = th[run["key"]]
    nbin_tot = sum(11 * nb for _, _, nb in MEASURES)
    acc = C.LagAcc(C.n_lagbins(run))
    cur = -1
    oi_of = {int(t0): n for n, t0 in enumerate(run["origins"])}
    for i0, f, lag, b in C.frames_of(run):
        if i0 != cur:
            cur = i0
            oi = oi_of[int(run["t_save"][i0])]
            m = st["orig"] == oi
            ci, cj = st["ci"][m], st["cj"][m]
            x0 = traj[i0]
            dv = x0[ci] - x0[cj]; dv -= L * np.rint(dv / L)
            rhat = dv / np.linalg.norm(dv, axis=1)[:, None]
            kpar, kperp = k7 * rhat, k7 * C.perp_of(rhat)
            nc = np.minimum(st["nc"][m], NC_MAX)
            pmv = pair_measures(st, {k: np.array(v) for k, v in tk["coef"].items()})
            idx = []
            off = 0
            pbs = {}
            for mname, var, nb in MEASURES:
                if var is None:
                    pb = 3 * pbs["sig3"] + pbs["radc3"]
                else:
                    pb = np.searchsorted(np.array(tk[mname]), pmv[var][m], side="right")
                pbs[mname] = pb
                idx.append(off + nc * nb + pb)
                off += 11 * nb
            idx = np.concatenate(idx)
            nmeas = len(MEASURES)
        dx = traj[f] - x0
        dx = dx - dx.mean(0)
        cols = [np.ones(len(ci))]
        for kk in (kpar, kperp):
            ai, aj = (dx[ci] * kk).sum(1), (dx[cj] * kk).sum(1)
            ui, uj = ai / k7, aj / k7
            cols += [np.cos(ai - aj), np.sin(ai - aj), np.cos(ai), np.sin(ai), np.cos(aj), np.sin(aj), ui, uj, ui * ui, uj * uj, ui * uj]
        Q = np.tile(np.stack(cols, 1), (nmeas, 1))
        S = np.zeros((nbin_tot, len(DF)))
        for c in range(len(DF)):
            S[:, c] = np.bincount(idx, weights=Q[:, c], minlength=nbin_tot)
        acc.add(b, lag, S=S, Fs7=np.cos(dx @ K7.T).mean())
    res = acc.result()
    np.savez(cache, tag=run["tag"], key=run["key"], phi=run["phi"], N=run["N"], delta=run["delta"], fields=np.array(DF),
             measures=np.array([m[0] for m in MEASURES]), nbins=np.array([m[2] for m in MEASURES]), **res)
    print(f"  packdyn {tag}: {time.time() - t0_:.0f}s", flush=True)
    return cache


def main():
    mode = sys.argv[1]
    files = sorted(C.hs_files("all"), key=lambda f: -os.path.getsize(f))
    C.require_files(files, "hard-sphere trajectories")
    nproc = int(os.environ.get("NPROC", "16"))
    if mode == "struct":
        with Pool(nproc) as p:
            list(p.imap_unordered(struct, files))
        thresholds()
    elif mode == "dyn":
        th = thresholds()
        with Pool(nproc) as p:
            list(p.imap_unordered(dyn, [(f, th) for f in files]))


if __name__ == "__main__":
    main()
