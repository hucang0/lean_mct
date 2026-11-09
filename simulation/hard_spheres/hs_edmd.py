#!/usr/bin/env python3
"""Event-driven molecular dynamics (Newtonian, elastic collisions, equal masses m = 1, kT = 1) for polydisperse HS,
as a dynamics control for the MC results. Starts from the same swap-equilibrated configurations as hs_dyn.py,
with Maxwell velocities (total momentum removed). Time unit sqrt(m sigma_bar^2 / kT).
Algorithm: per-particle next event (collision with a listed neighbour, or Verlet-list exit after skin/2), collision
counters to invalidate stale predictions, a tournament tree for the global minimum, lazy (per-particle) time
stamps; the Verlet list (sigma_ij + skin) is rebuilt at the first list-exit event.
Output mirrors hs_dyn.py; t_save / sweeps / origins are integer ticks of `tick` MD units (default 0.01), so
hs_analysis.py applies unchanged (delta is stored as -1 to tag MD runs; the CM is exactly conserved).
"""
import argparse, os, sys, time
import numpy as np
from numba import njit
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hs_core import build_list, nb_seed, pressure_Z
from hs_dyn import build_schedule

INF = 1e300


@njit(cache=True)
def tree_update(tree, P, times, i):
    k = P + i
    tree[k] = i
    k //= 2
    while k >= 1:
        a, b = tree[2 * k], tree[2 * k + 1]
        if b < 0 or (a >= 0 and times[a] <= times[b]):
            tree[k] = a
        else:
            tree[k] = b
        k //= 2


@njit(cache=True)
def predict(a, t, x, v, tl, sig, L, nl, nn, xref, h, count, et, ep, ec):
    """Next event of particle a from time t (positions of others advanced lazily)."""
    invL = 1.0 / L
    ax = x[a, 0] + v[a, 0] * (t - tl[a]); ay = x[a, 1] + v[a, 1] * (t - tl[a]); az = x[a, 2] + v[a, 2] * (t - tl[a])
    best = INF; bp = -1
    for m in range(nn[a]):
        b = nl[a, m]
        dx = ax - (x[b, 0] + v[b, 0] * (t - tl[b]))
        dy = ay - (x[b, 1] + v[b, 1] * (t - tl[b]))
        dz = az - (x[b, 2] + v[b, 2] * (t - tl[b]))
        dx -= L * np.rint(dx * invL); dy -= L * np.rint(dy * invL); dz -= L * np.rint(dz * invL)
        wx = v[a, 0] - v[b, 0]; wy = v[a, 1] - v[b, 1]; wz = v[a, 2] - v[b, 2]
        bb = dx * wx + dy * wy + dz * wz
        if bb >= 0.0:
            continue
        w2 = wx * wx + wy * wy + wz * wz
        s = 0.5 * (sig[a] + sig[b])
        r2 = dx * dx + dy * dy + dz * dz
        disc = bb * bb - w2 * (r2 - s * s)
        if disc < 0.0:
            continue
        dt = (-bb - np.sqrt(disc)) / w2
        if dt < 0.0:
            dt = 0.0
        if dt < best:
            best = dt; bp = b
    # list exit: |d0 + v dt| = h
    d0x = ax - xref[a, 0]; d0y = ay - xref[a, 1]; d0z = az - xref[a, 2]
    va2 = v[a, 0] ** 2 + v[a, 1] ** 2 + v[a, 2] ** 2
    if va2 > 0:
        bq = d0x * v[a, 0] + d0y * v[a, 1] + d0z * v[a, 2]
        cq = d0x * d0x + d0y * d0y + d0z * d0z - h * h
        dte = (-bq + np.sqrt(max(bq * bq - va2 * cq, 0.0))) / va2
        if dte < best:
            best = dte; bp = -2
    et[a] = t + best; ep[a] = bp
    ec[a] = count[bp] if bp >= 0 else -1


@njit(cache=True)
def advance_all(t, x, v, tl):
    for i in range(x.shape[0]):
        dt = t - tl[i]
        x[i, 0] += v[i, 0] * dt; x[i, 1] += v[i, 1] * dt; x[i, 2] += v[i, 2] * dt
        tl[i] = t


@njit(cache=True)
def full_rebuild(t, x, v, tl, sig, L, skin, smax, xref, nl, nn, count, et, ep, ec, tree, P):
    advance_all(t, x, v, tl)
    r = build_list(x, sig, L, skin, False, smax, xref, nl, nn)
    for i in range(x.shape[0]):
        predict(i, t, x, v, tl, sig, L, nl, nn, xref, 0.5 * skin, count, et, ep, ec)
    for i in range(x.shape[0]):
        tree_update(tree, P, et, i)
    return r


@njit(cache=True)
def run_until(tend, tnow, x, v, tl, sig, L, skin, smax, xref, nl, nn, count, et, ep, ec, tree, P, stats, vir):
    """Process events up to tend. stats: [collisions, invalid, rebuilds]. Returns current time (= tend)."""
    invL = 1.0 / L
    h = 0.5 * skin
    while True:
        i = tree[1]
        te = et[i]
        if te > tend:
            break
        p = ep[i]
        if p == -2:
            r = full_rebuild(te, x, v, tl, sig, L, skin, smax, xref, nl, nn, count, et, ep, ec, tree, P)
            stats[2] += 1
            if r < 0:
                return -1.0
            continue
        if p < 0 or count[p] != ec[i]:
            predict(i, te, x, v, tl, sig, L, nl, nn, xref, h, count, et, ep, ec)
            tree_update(tree, P, et, i)
            stats[1] += 1
            continue
        # valid collision i-p at time te
        for q in (i, p):
            dt = te - tl[q]
            x[q, 0] += v[q, 0] * dt; x[q, 1] += v[q, 1] * dt; x[q, 2] += v[q, 2] * dt
            tl[q] = te
        dx = x[i, 0] - x[p, 0]; dy = x[i, 1] - x[p, 1]; dz = x[i, 2] - x[p, 2]
        dx -= L * np.rint(dx * invL); dy -= L * np.rint(dy * invL); dz -= L * np.rint(dz * invL)
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        nx, ny, nz = dx / r, dy / r, dz / r
        vn = (v[i, 0] - v[p, 0]) * nx + (v[i, 1] - v[p, 1]) * ny + (v[i, 2] - v[p, 2]) * nz
        v[i, 0] -= vn * nx; v[i, 1] -= vn * ny; v[i, 2] -= vn * nz
        v[p, 0] += vn * nx; v[p, 1] += vn * ny; v[p, 2] += vn * nz
        count[i] += 1; count[p] += 1
        stats[0] += 1
        vir[0] += -vn * r          # r . dp_i  (vn < 0 on approach), for the collision virial
        predict(i, te, x, v, tl, sig, L, nl, nn, xref, h, count, et, ep, ec)
        tree_update(tree, P, et, i)
        predict(p, te, x, v, tl, sig, L, nl, nn, xref, h, count, et, ep, ec)
        tree_update(tree, P, et, p)
    return tend


def log_times_f(tmax, tmin, per_decade=12):
    n = int(np.ceil(np.log10(tmax / tmin) * per_decade)) + 1
    return np.logspace(np.log10(tmin), np.log10(tmax), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--tmax", type=float, required=True)
    ap.add_argument("--tick", type=float, default=0.01, help="time tick (MD units); saved times are integer ticks")
    ap.add_argument("--norig", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skin", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.init)
    x = np.ascontiguousarray(z["x"], dtype=np.float64); sig = np.ascontiguousarray(z["sig"], dtype=np.float64)
    L = float(z["L"]); N = len(sig)
    rng = np.random.default_rng(9000 + a.seed)
    v = rng.normal(size=(N, 3)); v -= v.mean(0); v *= np.sqrt(3 * N / np.sum(v ** 2))   # kT = 1 exactly (m = 1)
    tl = np.zeros(N); xref = np.empty_like(x); nl = np.empty((N, 64), np.int64); nn = np.empty(N, np.int64)
    count = np.zeros(N, np.int64); et = np.full(N, INF); ep = np.full(N, -1, np.int64); ec = np.zeros(N, np.int64)
    P = 1
    while P < N:
        P *= 2
    tree = np.full(2 * P, -1, np.int64)
    stats = np.zeros(3, np.int64)
    vir = np.zeros(1)
    smax = float(sig.max())
    full_rebuild(0.0, x, v, tl, sig, L, a.skin, smax, xref, nl, nn, count, et, ep, ec, tree, P)
    # integer time ticks of TICK MD units, so that the MC analysis (integer lags, log binning) applies unchanged
    TICK = a.tick
    Tt = int(round(a.tmax / TICK))
    t_save, origins = build_schedule(Tt, a.norig, 12)
    T = Tt
    traj = np.empty((len(t_save), N, 3), np.float32)
    traj[0] = x
    t0w = time.time(); tnow = 0.0
    Zs = []
    for k in range(1, len(t_save)):
        tnow = run_until(t_save[k] * TICK, tnow, x, v, tl, sig, L, a.skin, smax, xref, nl, nn, count, et, ep, ec, tree, P, stats, vir)
        if tnow < 0:
            raise RuntimeError("list overflow")
        xs = x + v * (t_save[k] * TICK - tl)[:, None]
        traj[k] = xs
        if t_save[k] in origins or k == len(t_save) - 1:
            Zc, nov, _ = pressure_Z(xs, sig, L)
            assert nov == 0, "overlap"
            Zs.append([t_save[k] * TICK, Zc])
    ke = 0.5 * np.sum(v ** 2) / N
    wall = time.time() - t0w
    Zvir = 1.0 + vir[0] / (3.0 * N * T * TICK)        # collision virial, m = kT = 1
    np.savez_compressed(a.out, x0=traj[0], sig=sig, L=L, phi=np.pi / 6 * np.sum(sig ** 3) / L ** 3, N=N, delta=-1.0,
                        seed=a.seed, t_save=t_save, traj=traj, origins=origins, acc=np.nan, wall=wall, sweeps=T,
                        sweeps_planned=T, norig=a.norig, tick=TICK, partial=False, Z=np.array(Zs), xfinal=xs, ke=ke,
                        collisions=stats[0], invalid=stats[1], rebuilds=stats[2], Z_virial=Zvir)
    print(f"[done] T={T*TICK} collisions={stats[0]} ({stats[0]*2/N/(T*TICK):.1f} per particle per unit time) invalid={stats[1]} "
          f"rebuilds={stats[2]} KE/N={ke:.6f} (1.5) Z_virial={Zvir:.3f} Z(xi->0)={[round(q[1],2) for q in Zs]} wall={wall/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
