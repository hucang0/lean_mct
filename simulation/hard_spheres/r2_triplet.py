#!/usr/bin/env python3
"""(ii) Loop channel: closed vs open contact triplets.
For each origin: contact pairs r_ij(0) < 1.05 sigma_ij (MK: shifted metric). A triplet (i; j, k) is a centre i with two
contact partners j < k. Unit vectors e1 = rhat_ij = (x_i - x_j)/r_ij, e2 = rhat_ik (MK: shifted separations);
cos(theta) = e1.e2 is the bond angle at i.  Class by the j-k separation s_jk = r_jk/sigma_jk at t=0
(MK: shifted distance |x_j - x_k - A_jk|):  0 closed (s_jk < 1.05), 1 neighbour-closed (1.05 <= s_jk < 1.35), 2 open (>= 1.35).
Per frame (CM drift removed), per triplet:
   x1 = dx_i.e1, y1 = dx_j.e1, x2 = dx_i.e2, y2 = dx_k.e2   ->  b_ij = x1 - y1,  b_ik = x2 - y2   (bond-projected relative displ.)
   n = e1 x e2/|e1 x e2| (normal to the triplet plane, skipped if |e1 x e2| < 0.05): xt = dx_i.n, y1t = dx_j.n, y2t = dx_k.n
      -> t_ij = xt - y1t, t_ik = xt - y2t (transverse analogue)
   bond broken: |r_ij(0) + dx_i - dx_j| > 1.35 sigma_ij (endpoint at lag t); 'ok' = still < 1.05 sigma_ij.
All first and second moments are summed per (class, cos-theta bin [10 bins on -1..1]) and binned in log lag.
Writes cache/trip_<tag>.npz.  Usage: r2_triplet.py [hs|mk|all]
"""
import os, sys, time
sys.dont_write_bytecode = True
import numpy as np
from multiprocessing import Pool
from numba import njit
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r2_common as C

NANG = 10
TF = ["cnt", "cos", "x1", "x2", "y1", "y2", "x1x1", "x2x2", "y1y1", "y2y2", "x1x2", "x1y1", "x2y2", "x1y2", "x2y1", "y1y2",
      "cntt", "xt", "y1t", "y2t", "xtxt", "y1ty1t", "y2ty2t", "xty1t", "xty2t", "y1ty2t",
      "br1", "br2", "br12", "ok1", "ok2", "ok12"]
NF = len(TF)


@njit(cache=False)
def trip_kernel(I, J, K, cls, ab, e1, e2, nn, okn, d1, d2, s1, s2, dx, out):
    T = I.shape[0]
    for t in range(T):
        i = I[t]; j = J[t]; k = K[t]
        c = cls[t]; a = ab[t]
        x1 = dx[i, 0] * e1[t, 0] + dx[i, 1] * e1[t, 1] + dx[i, 2] * e1[t, 2]
        y1 = dx[j, 0] * e1[t, 0] + dx[j, 1] * e1[t, 1] + dx[j, 2] * e1[t, 2]
        x2 = dx[i, 0] * e2[t, 0] + dx[i, 1] * e2[t, 1] + dx[i, 2] * e2[t, 2]
        y2 = dx[k, 0] * e2[t, 0] + dx[k, 1] * e2[t, 1] + dx[k, 2] * e2[t, 2]
        o = out[c, a]
        o[0] += 1.0
        o[1] += e1[t, 0] * e2[t, 0] + e1[t, 1] * e2[t, 1] + e1[t, 2] * e2[t, 2]
        o[2] += x1; o[3] += x2; o[4] += y1; o[5] += y2
        o[6] += x1 * x1; o[7] += x2 * x2; o[8] += y1 * y1; o[9] += y2 * y2
        o[10] += x1 * x2; o[11] += x1 * y1; o[12] += x2 * y2; o[13] += x1 * y2; o[14] += x2 * y1; o[15] += y1 * y2
        if okn[t]:
            xt = dx[i, 0] * nn[t, 0] + dx[i, 1] * nn[t, 1] + dx[i, 2] * nn[t, 2]
            y1t = dx[j, 0] * nn[t, 0] + dx[j, 1] * nn[t, 1] + dx[j, 2] * nn[t, 2]
            y2t = dx[k, 0] * nn[t, 0] + dx[k, 1] * nn[t, 1] + dx[k, 2] * nn[t, 2]
            o[16] += 1.0; o[17] += xt; o[18] += y1t; o[19] += y2t
            o[20] += xt * xt; o[21] += y1t * y1t; o[22] += y2t * y2t
            o[23] += xt * y1t; o[24] += xt * y2t; o[25] += y1t * y2t
        # bond lengths at lag t
        r1 = 0.0; r2 = 0.0
        for q in range(3):
            w1 = d1[t, q] + dx[i, q] - dx[j, q]
            w2 = d2[t, q] + dx[i, q] - dx[k, q]
            r1 += w1 * w1; r2 += w2 * w2
        r1 = np.sqrt(r1) / s1[t]; r2 = np.sqrt(r2) / s2[t]
        b1 = 1.0 if r1 > 1.35 else 0.0
        b2 = 1.0 if r2 > 1.35 else 0.0
        k1 = 1.0 if r1 < 1.05 else 0.0
        k2 = 1.0 if r2 < 1.05 else 0.0
        o[26] += b1; o[27] += b2; o[28] += b1 * b2; o[29] += k1; o[30] += k2; o[31] += k1 * k2


def build_triplets(x0, sig, L, Ash=None):
    N = len(x0)
    if Ash is None:
        tr = cKDTree(C.wrap(x0, L), boxsize=L)
        P = np.sort(tr.query_pairs(1.05 * sig.max(), output_type="ndarray"), axis=1)
        d = x0[P[:, 0]] - x0[P[:, 1]]; d -= L * np.rint(d / L)
        sij = 0.5 * (sig[P[:, 0]] + sig[P[:, 1]])
        m = np.linalg.norm(d, axis=1) / sij < C.CONTACT
        P, d = P[m], d[m]
    else:
        ci, cj, d, r = C.mk_shifted_pairs(x0, Ash, L, C.CONTACT)
        P = np.stack([ci, cj], 1)
    # neighbour vectors from the centre's point of view: vec(c -> partner) ; rhat_{c,partner} = (x_c - x_p)/r = -vec
    nb = [[] for _ in range(N)]
    for (a, b), dv in zip(P, d):            # dv = x_a - x_b (shifted for MK: x_a - x_b - A_ab)
        nb[a].append((b, dv)); nb[b].append((a, -dv))
    I, J, K, D1, D2 = [], [], [], [], []
    for i in range(N):
        lst = nb[i]
        for u in range(len(lst)):
            for w in range(u + 1, len(lst)):
                I.append(i); J.append(lst[u][0]); K.append(lst[w][0]); D1.append(lst[u][1]); D2.append(lst[w][1])
    I, J, K = np.array(I, np.int64), np.array(J, np.int64), np.array(K, np.int64)
    D1, D2 = np.array(D1).reshape(-1, 3), np.array(D2).reshape(-1, 3)
    # D1 = r_ij vector (x_i - x_j [- A_ij]), D2 = r_ik vector
    e1 = D1 / np.linalg.norm(D1, axis=1)[:, None]; e2 = D2 / np.linalg.norm(D2, axis=1)[:, None]
    if Ash is None:
        djk = x0[J] - x0[K]; djk -= L * np.rint(djk / L)
    else:
        djk = x0[J] - x0[K] - Ash[J, K]; djk -= L * np.rint(djk / L)
    sjk = np.linalg.norm(djk, axis=1) / (0.5 * (sig[J] + sig[K]))
    cls = np.where(sjk < 1.05, 0, np.where(sjk < 1.35, 1, 2)).astype(np.int64)
    cos = (e1 * e2).sum(1)
    ab = np.clip(((cos + 1) / 2 * NANG).astype(np.int64), 0, NANG - 1)
    nv = np.cross(e1, e2); nl = np.linalg.norm(nv, axis=1)
    okn = nl > 0.05
    nn = np.where(okn[:, None], nv / np.maximum(nl, 1e-12)[:, None], 0.0)
    s1 = 0.5 * (sig[I] + sig[J]); s2 = 0.5 * (sig[I] + sig[K])
    return dict(I=I, J=J, K=K, cls=cls, ab=ab, e1=np.ascontiguousarray(e1), e2=np.ascontiguousarray(e2),
                nn=np.ascontiguousarray(nn), okn=okn, d1=np.ascontiguousarray(D1), d2=np.ascontiguousarray(D2), s1=s1, s2=s2,
                sjk=sjk, cos=cos, ncontacts=len(P))


def process(path):
    tag = C.tag_of(path)
    cache = os.path.join(C.CACHE, f"trip_{tag}.npz")
    if os.path.exists(cache):
        return cache
    t_start = time.time()
    run = C.load_run(path)
    traj, L, N, sig = run["traj"], run["L"], run["N"], run["sig"]
    acc = C.LagAcc(C.n_lagbins(run))
    cur = -1
    stats = {"ntrip": [], "ncls": [], "ncontacts": [], "sjk_hist": [], "cos_hist": []}
    sjk_edges = np.linspace(0, 4, 81)
    for i0, f, lag, b in C.frames_of(run):
        if i0 != cur:
            cur = i0
            x0 = traj[i0]
            T = build_triplets(x0, sig, L, run.get("A"))
            stats["ntrip"].append(len(T["I"])); stats["ncontacts"].append(T["ncontacts"])
            stats["ncls"].append(np.bincount(T["cls"], minlength=3))
            stats["sjk_hist"].append(np.histogram(T["sjk"], sjk_edges)[0])
            stats["cos_hist"].append(np.array([np.histogram(T["cos"][T["cls"] == c], np.linspace(-1, 1, 41))[0] for c in range(3)]))
        dx = traj[f] - x0
        dx = dx - dx.mean(0)
        out = np.zeros((3, NANG, NF))
        trip_kernel(T["I"], T["J"], T["K"], T["cls"], T["ab"], T["e1"], T["e2"], T["nn"], T["okn"], T["d1"], T["d2"],
                    T["s1"], T["s2"], dx, out)
        acc.add(b, lag, S=out, msd=(dx ** 2).sum(1).mean())
    res = acc.result()
    np.savez(cache, tag=run["tag"], key=run["key"], phi=run["phi"], N=N, L=L, delta=run["delta"], mk=run["mk"],
             fields=np.array(TF), ntrip=np.mean(stats["ntrip"]), ncontacts=np.mean(stats["ncontacts"]),
             ncls=np.sum(stats["ncls"], 0), sjk_hist=np.sum(stats["sjk_hist"], 0), sjk_edges=sjk_edges,
             cos_hist=np.sum(stats["cos_hist"], 0), **res)
    print(f"  trip {tag}: {time.time() - t_start:.0f}s ntrip={np.mean(stats['ntrip']):.0f} cls={np.sum(stats['ncls'], 0)}", flush=True)
    return cache


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    files = []
    if which in ("hs", "all"):
        files += C.hs_files("all")
    if which in ("mk", "all"):
        files += C.mk_files()
    C.require_files(files, "simulation trajectories")
    files = sorted(files, key=lambda f: -os.path.getsize(f))
    with Pool(int(os.environ.get("NPROC", "8"))) as p:
        for c in p.imap_unordered(process, files):
            pass


if __name__ == "__main__":
    main()
