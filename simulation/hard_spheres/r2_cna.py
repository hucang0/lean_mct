#!/usr/bin/env python3
"""(iv) Cluster level: common-neighbour analysis (CNA) of HS configurations.
Bond: r_ij < 1.35 sigma_ij (the n_c neighbour criterion).  For a bond (i,j) the common neighbours S = nb(i) & nb(j);
a 1551 bond has |S| = 5 and every member of S has exactly 2 bonded partners inside S (the only 2-regular graph on 5
vertices is the 5-ring).  Icosahedral centre: 12 neighbours, all 12 bonds 1551.  1551 clusters: connected components of
the graph of 1551 bonds.
"""
import os, sys, time, glob
sys.dont_write_bytecode = True
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r2_common as C


def cna(x, sig, L, cut=1.35):
    N = len(x)
    xw = C.wrap(x, L)
    tr = cKDTree(xw, boxsize=L)
    P = np.sort(tr.query_pairs(cut * sig.max(), output_type="ndarray"), axis=1)
    d = x[P[:, 0]] - x[P[:, 1]]; d -= L * np.rint(d / L)
    s = np.linalg.norm(d, axis=1) / (0.5 * (sig[P[:, 0]] + sig[P[:, 1]]))
    P = P[s < cut]
    adj = [set() for _ in range(N)]
    for a, b in P:
        adj[a].add(b); adj[b].add(a)
    is1551 = np.zeros(len(P), bool)
    ncn = np.zeros(len(P), np.int64)
    for n, (a, b) in enumerate(P):
        S = adj[a] & adj[b]
        ncn[n] = len(S)
        if len(S) == 5:
            ok = True
            for u in S:
                if len(adj[u] & S) != 2:
                    ok = False; break
            is1551[n] = ok
    deg = np.array([len(a) for a in adj])
    n1551 = np.bincount(P[is1551].ravel(), minlength=N)
    ico = (deg == 12) & (n1551 == 12)
    return P, is1551, ncn, deg, n1551, ico, adj


def clusters_1551(P, is1551, N):
    Q = P[is1551]
    if len(Q) == 0:
        return np.arange(N), np.ones(N, int)
    A = coo_matrix((np.ones(len(Q)), (Q[:, 0], Q[:, 1])), shape=(N, N))
    nc, lab = connected_components(A, directed=False)
    size = np.bincount(lab)
    return lab, size[lab]


GROUPS = ["n0", "n1-3", "n4-7", "n8+", "ico", "clus13", "all"]


def process(path):
    """Per origin: CNA at t0; per frame: CNA again (persistence of 1551 bonds / bonds / ico centres, endpoint) and
    the mobility q_i = <cos(k.dx_i)>_k (lattice k7 set), |dx_i|^2 averaged per group of the t0 structure."""
    from multiprocessing import current_process
    tag = C.tag_of(path)
    cache = os.path.join(C.CACHE, f"cna_{tag}.npz")
    if os.path.exists(cache):
        return cache
    t_start = time.time()
    run = C.load_run(path)
    traj, L, N, sig = run["traj"], run["L"], run["N"], run["sig"]
    K7, k7 = C.A.lattice_kvecs(7.0, L)
    acc = C.LagAcc(C.n_lagbins(run))
    cur = -1
    st = {"frac1551": [], "nico": [], "largest": [], "frac_clus13": [], "n1551_hist": [], "ncn_hist": []}
    for i0, f, lag, b in C.frames_of(run):
        if i0 != cur:
            cur = i0
            x0 = traj[i0]
            P0, is0, ncn0, deg0, n0, ico0, _ = cna(x0, sig, L)
            lab, csize = clusters_1551(P0, is0, N)
            inb = n0 > 0
            gm = np.zeros((len(GROUPS), N), bool)
            gm[0] = n0 == 0; gm[1] = (n0 >= 1) & (n0 <= 3); gm[2] = (n0 >= 4) & (n0 <= 7); gm[3] = n0 >= 8
            gm[4] = ico0; gm[5] = inb & (csize >= 13); gm[6] = True
            key0 = P0[:, 0] * N + P0[:, 1]
            k1551 = key0[is0]
            st["frac1551"].append(is0.mean()); st["nico"].append(ico0.sum()); st["largest"].append(csize[inb].max() if inb.any() else 0)
            st["frac_clus13"].append(gm[5].mean()); st["n1551_hist"].append(np.bincount(np.minimum(n0, 12), minlength=13))
            st["ncn_hist"].append(np.bincount(np.minimum(ncn0, 10), minlength=11))
        x = traj[f]
        P1, is1, _, _, _, ico1, _ = cna(x, sig, L)
        key1 = P1[:, 0] * N + P1[:, 1]
        pb = np.isin(key0, key1).mean()
        p1551 = np.isin(k1551, key1[is1]).mean() if len(k1551) else np.nan
        p1551_bond = np.isin(k1551, key1).mean() if len(k1551) else np.nan        # 1551 bond still a bond (any type)
        pico = (ico1[ico0]).mean() if ico0.any() else np.nan
        dx = traj[f] - x0; dx = dx - dx.mean(0)
        q = np.cos(dx @ K7.T).mean(1); m2 = (dx ** 2).sum(1)
        cnt = gm.sum(1).astype(float)
        acc.add(b, lag, q_sum=gm @ q, m2_sum=gm @ m2, cnt=cnt, Fs7=q.mean(),
                pers=np.array([pb, p1551, p1551_bond, pico if np.isfinite(pico) else 0.0, 1.0 if np.isfinite(pico) else 0.0]))
    res = acc.result()
    np.savez(cache, tag=tag, key=run["key"], phi=run["phi"], N=N, groups=np.array(GROUPS),
             frac1551=np.mean(st["frac1551"]), nico=np.mean(st["nico"]), largest=np.mean(st["largest"]),
             frac_clus13=np.mean(st["frac_clus13"]), n1551_hist=np.sum(st["n1551_hist"], 0), ncn_hist=np.sum(st["ncn_hist"], 0), **res)
    print(f"  cna {tag}: {time.time() - t_start:.0f}s", flush=True)
    return cache


def main():
    from multiprocessing import Pool
    files = sorted(C.hs_files("main"), key=lambda f: -os.path.getsize(f))
    C.require_files(files, "hard-sphere trajectories")
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        list(p.imap_unordered(process, files))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "run":
    main()
elif __name__ == "__main__":
    for f in sys.argv[1:]:
        run = C.load_run(f)
        t = time.time()
        P, is1551, ncn, deg, n1551, ico, adj = cna(run["traj"][0], run["sig"], run["L"])
        lab, csize = clusters_1551(P, is1551, run["N"])
        inb = n1551 > 0
        print(run["tag"], f"{time.time()-t:.2f}s bonds/N={len(P)/run['N']:.2f} frac1551={is1551.mean():.3f} ico_centres={ico.sum()} "
              f"frac(n1551>=8)={(n1551>=8).mean():.3f} mean deg={deg.mean():.2f} largest 1551 cluster={csize.max()} "
              f"frac particles in 1551 clusters size>=13: {(csize[inb]>=13).sum()/run['N']:.3f}")
