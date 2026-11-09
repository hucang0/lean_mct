#!/usr/bin/env python3
"""(i) Co-mobility vs pair distance. Per run (HS MC / HS EDMD / MK planted MC), for every (origin, frame) as in
hs_analysis.py (CM drift removed, lattice k7 set, log-lag bins), pairs are binned by their INITIAL separation:
  HS : 's'-binning in r/sigma_ij  (edges 0,1.05,1.15,...,4.05) and 'r'-binning in absolute r (0.1-wide bins to 0.49 L)
  MK : 's' = shifted-frame distance |x_i - x_j - A_ij| (interaction distance), 'r' = real-space distance |x_i - x_j|
Per pair (unit vectors rhat = (x_i-x_j)/r [shifted for MK-'s'], e1 = perp_of(rhat) [as hs_analysis], e2 = rhat x e1):
  for each direction e in (L = rhat, T1 = e1, T2 = e2), a = k7 e.dx:
     cc = cos a_i cos a_j, ss = sin a_i sin a_j, cos/sin a_i, cos/sin a_j, Im phi2 = sin(a_i - a_j),
     u = e.dx moments (u_i, u_j, u_i^2, u_j^2, u_i u_j)            -> Re phi2 = cc+ss, R = Re phi2/Fs^2, R_marg, C, Pearson
  cross-direction: cos(k7 e1.dx_i) cos(k7 e2.dx_j) and (e2,e1)        -> covariance between DIFFERENT transverse directions
  orientation-averaged over the lattice k7 set: <cos cos>_k, <sin sin>_k
  mobility: q_i = <cos(k.dx_i)>_k (orientation-averaged self overlap), m2_i = |dx_i|^2 (products, squares)
  'detour' stats: D_par = (dx_i-dx_j).rhat, D_perp^2 = |(dx_i-dx_j) - D_par rhat|^2
Random pairs (4000 per origin, i != j): same quantities with rhat = own min-image separation (single bin).
Per frame globals: Fs7 = <q>, <q^2>, <q>^2 (chi4), msd, <m2^2>.
Writes cache/comob_<tag>.npz (cached; rerun is free).  Usage: r2_comob.py [hs|mk|all]
"""
import os, sys, time
sys.dont_write_bytecode = True
import numpy as np
from multiprocessing import Pool
from numba import njit
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r2_common as C

S_EDGES = np.concatenate([[0.0], np.round(np.arange(1.05, 4.0501, 0.1), 4)])
FIELDS = (["cnt"] + [f"{d}_{q}" for d in ("L", "T1", "T2") for q in
          ("cc", "ss", "ci", "si", "cj", "sj", "impm", "ui", "uj", "uii", "ujj", "uij")]
          + ["X12", "X21", "Icc", "Iss", "qi", "qj", "qij", "qii", "qjj", "mi", "mj", "mij", "mii", "mjj",
             "Dpar", "Dperp2", "DparDperp2", "Dpar2"])
FI = {f: i for i, f in enumerate(FIELDS)}
NQ = len(FIELDS)


@njit(cache=False, fastmath=False)
def pair_kernel(ci, cj, bs, br, rhat, e1, e2, dx, cz, sz, q, m2, k, out_s, out_r):
    P = ci.shape[0]
    nk = cz.shape[1]
    v = np.empty(55)
    for p in range(P):
        i = ci[p]; j = cj[p]
        v[0] = 1.0
        cT1i = 0.0; cT1j = 0.0; cT2i = 0.0; cT2j = 0.0
        uL_i = 0.0; uL_j = 0.0; uT1_i = 0.0; uT1_j = 0.0; uT2_i = 0.0; uT2_j = 0.0
        for d in range(3):
            if d == 0:
                ex = rhat[p, 0]; ey = rhat[p, 1]; ez = rhat[p, 2]
            elif d == 1:
                ex = e1[p, 0]; ey = e1[p, 1]; ez = e1[p, 2]
            else:
                ex = e2[p, 0]; ey = e2[p, 1]; ez = e2[p, 2]
            ui = dx[i, 0] * ex + dx[i, 1] * ey + dx[i, 2] * ez
            uj = dx[j, 0] * ex + dx[j, 1] * ey + dx[j, 2] * ez
            ai = k * ui; aj = k * uj
            ca = np.cos(ai); sa = np.sin(ai); cb = np.cos(aj); sb = np.sin(aj)
            o = 1 + 12 * d
            v[o] = ca * cb; v[o + 1] = sa * sb; v[o + 2] = ca; v[o + 3] = sa; v[o + 4] = cb; v[o + 5] = sb
            v[o + 6] = sa * cb - ca * sb
            v[o + 7] = ui; v[o + 8] = uj; v[o + 9] = ui * ui; v[o + 10] = uj * uj; v[o + 11] = ui * uj
            if d == 0:
                uL_i = ui; uL_j = uj
            elif d == 1:
                cT1i = ca; cT1j = cb; uT1_i = ui; uT1_j = uj
            else:
                cT2i = ca; cT2j = cb; uT2_i = ui; uT2_j = uj
        v[37] = cT1i * cT2j
        v[38] = cT2i * cT1j
        scc = 0.0; sss = 0.0
        for kk in range(nk):
            scc += cz[i, kk] * cz[j, kk]
            sss += sz[i, kk] * sz[j, kk]
        v[39] = scc / nk; v[40] = sss / nk
        v[41] = q[i]; v[42] = q[j]; v[43] = q[i] * q[j]; v[44] = q[i] * q[i]; v[45] = q[j] * q[j]
        v[46] = m2[i]; v[47] = m2[j]; v[48] = m2[i] * m2[j]; v[49] = m2[i] * m2[i]; v[50] = m2[j] * m2[j]
        Dp = uL_i - uL_j
        Dt = (uT1_i - uT1_j) ** 2 + (uT2_i - uT2_j) ** 2
        v[51] = Dp; v[52] = Dt; v[53] = Dp * Dt; v[54] = Dp * Dp
        b = bs[p]
        if b >= 0:
            for f in range(55):
                out_s[b, f] += v[f]
        b = br[p]
        if b >= 0:
            for f in range(55):
                out_r[b, f] += v[f]


def frame_basis(rhat):
    e1 = C.perp_of(rhat)
    e2 = np.cross(rhat, e1)
    return np.ascontiguousarray(e1), np.ascontiguousarray(e2)


def binidx(x, edges):
    b = np.searchsorted(edges, x, side="right") - 1
    b[(b < 0) | (b >= len(edges) - 1)] = -1
    return b.astype(np.int64)


def process(path):
    tag = C.tag_of(path)
    cache = os.path.join(C.CACHE, f"comob_{tag}.npz")
    if os.path.exists(cache):
        return cache
    t_start = time.time()
    run = C.load_run(path)
    traj, L, N, sig = run["traj"], run["L"], run["N"], run["sig"]
    K7, k7 = C.A.lattice_kvecs(7.0, L)
    rmax = min(0.49 * L, 4.97)
    R_EDGES = np.round(np.arange(0.0, rmax + 1e-9, 0.1), 4)
    nbs, nbr = len(S_EDGES) - 1, len(R_EDGES) - 1
    acc = C.LagAcc(C.n_lagbins(run))
    rng = np.random.default_rng(12345)
    cur_i0 = -1
    for i0, f, lag, b in C.frames_of(run):
        if i0 != cur_i0:
            cur_i0 = i0
            x0 = traj[i0]
            if run["mk"]:
                si, sj, sd, sr = C.mk_shifted_pairs(x0, run["A"], L, rmax)
                ri_, rj_, rd, rr = C.mk_real_pairs(x0, L, rmax)
                # two pair sets: shifted ('s' binning) and real-space ('r' binning)
                sets = [(si, sj, sd / sr[:, None], binidx(sr, S_EDGES), np.full(len(si), -1, np.int64)),
                        (ri_, rj_, rd / rr[:, None], np.full(len(ri_), -1, np.int64), binidx(rr, R_EDGES))]
            else:
                # all pairs with r < rmax (absolute); s = r / sigma_ij
                xw = C.wrap(x0, L)
                from scipy.spatial import cKDTree
                tr = cKDTree(xw, boxsize=L)
                Pp = np.sort(tr.query_pairs(rmax, output_type="ndarray"), axis=1)
                dv = x0[Pp[:, 0]] - x0[Pp[:, 1]]; dv -= L * np.rint(dv / L)
                r = np.linalg.norm(dv, axis=1)
                s = r / (0.5 * (sig[Pp[:, 0]] + sig[Pp[:, 1]]))
                sets = [(Pp[:, 0].copy(), Pp[:, 1].copy(), dv / r[:, None], binidx(s, S_EDGES), binidx(r, R_EDGES))]
            prep = []
            for (a, bb, rh, bs_, br_) in sets:
                rh = np.ascontiguousarray(rh)
                e1, e2 = frame_basis(rh)
                prep.append((a.astype(np.int64), bb.astype(np.int64), bs_, br_, rh, e1, e2))
            # random pairs, own separation direction (MK: real-space separation)
            ra = rng.integers(0, N, 4000); rb = (ra + rng.integers(1, N, 4000)) % N
            drr = x0[ra] - x0[rb]; drr -= L * np.rint(drr / L)
            rrh = np.ascontiguousarray(drr / np.linalg.norm(drr, axis=1)[:, None])
            re1, re2 = frame_basis(rrh)
            rand = (ra.astype(np.int64), rb.astype(np.int64), np.zeros(4000, np.int64), np.full(4000, -1, np.int64), rrh, re1, re2)
        dx = traj[f] - x0
        dx = dx - dx.mean(0)
        Ph = dx @ K7.T
        cz, sz = np.ascontiguousarray(np.cos(Ph)), np.ascontiguousarray(np.sin(Ph))
        q = cz.mean(1)
        m2 = (dx ** 2).sum(1)
        out_s = np.zeros((nbs, NQ)); out_r = np.zeros((nbr, NQ))
        for (a, bb, bs_, br_, rh, e1, e2) in prep:
            pair_kernel(a, bb, bs_, br_, rh, e1, e2, dx, cz, sz, q, m2, k7, out_s, out_r)
        out_rand = np.zeros((1, NQ)); dummy = np.zeros((1, NQ))
        a, bb, bs_, br_, rh, e1, e2 = rand
        pair_kernel(a, bb, bs_, br_, rh, e1, e2, dx, cz, sz, q, m2, k7, out_rand, dummy)
        Fs = q.mean()
        # per-frame-normalised copies of the Re phi2 sums (removes origin-to-origin global mobility fluctuation)
        sel = [FI["L_cc"], FI["L_ss"], FI["T1_cc"], FI["T1_ss"], FI["T2_cc"], FI["T2_ss"], FI["Icc"], FI["Iss"], FI["qij"], FI["qi"], FI["qj"]]
        g = {"Fs7": Fs, "Fs7sq": Fs * Fs, "q2": (q * q).mean(), "msd": m2.mean(), "m4": (m2 * m2).mean(),
             "Fs7_dir": cz.mean(0)}
        acc.add(b, lag, S_s=out_s, S_r=out_r, S_rand=out_rand, N_s=out_s[:, sel] / (Fs * Fs), N_r=out_r[:, sel] / (Fs * Fs),
                N_rand=out_rand[:, sel] / (Fs * Fs), **g)
    res = acc.result()
    np.savez(cache, tag=run["tag"], key=run["key"], phi=run["phi"], N=N, L=L, delta=run["delta"], sweeps=run["T"],
             mk=run["mk"], k7=k7, s_edges=S_EDGES, r_edges=R_EDGES, fields=np.array(FIELDS),
             norm_fields=np.array([FIELDS[i] for i in sel]), **res)
    print(f"  comob {tag}: {time.time() - t_start:.0f}s", flush=True)
    return cache


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    files = []
    if which in ("hs", "all"):
        files += C.hs_files("all")
    if which in ("mk", "all"):
        files += C.mk_files()
    if which.endswith(".npz"):
        files = sys.argv[1:]
    C.require_files(files, "simulation trajectories")
    # largest first (phi 0.60 / N2000 / MK) for load balance
    files = sorted(files, key=lambda f: -os.path.getsize(f))
    nproc = int(os.environ.get("NPROC", "24"))
    with Pool(nproc) as p:
        for c in p.imap_unordered(process, files):
            pass


if __name__ == "__main__":
    main()
