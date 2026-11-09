#!/usr/bin/env python3
"""Standard (non-swap) single-particle Metropolis MC dynamics of polydisperse HS from a swap-equilibrated start.

Same move and schedule conventions as the MK engine (../mk_planted_mc/mk_mc.py): one attempted move = random particle,
cubic proposal of half-width delta, accepted iff no overlap; 1 sweep = N attempts; log-spaced save times from
norig equally spaced time origins (build_schedule copied verbatim); unwrapped positions saved as float32.
Output .npz keys mirror mk_mc.py (x0, L, phi, N, delta, seed, t_save, traj, origins, acc, wall, sweeps, partial)
plus sig (diameters, fixed during the run).
"""
import argparse, os, sys, time
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hs_core import HS, nb_seed, pressure_Z


def log_times(tmax, per_decade=12):
    if tmax < 1:
        return np.array([], dtype=np.int64)
    n = int(np.ceil(np.log10(max(tmax, 1)) * per_decade)) + 1
    t = np.unique(np.round(np.logspace(0, np.log10(tmax), max(n, 2))).astype(np.int64))
    return t[(t >= 1) & (t <= tmax)]


def build_schedule(T, norig, per_decade=12):
    origins = np.array([(m * T) // norig for m in range(norig)], dtype=np.int64)
    times = set([0, T])
    for t0 in origins:
        times.add(int(t0))
        for tau in log_times(T - t0, per_decade):
            times.add(int(t0 + tau))
    return np.array(sorted(times), dtype=np.int64), origins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="equilibrated configuration from hs_equil.py")
    ap.add_argument("--sweeps", type=int, required=True)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--norig", type=int, default=8)
    ap.add_argument("--per_decade", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skin", type=float, default=0.6)
    ap.add_argument("--ckpt", type=float, default=600.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    z = np.load(a.init)
    rng = np.random.default_rng(5000 + a.seed)
    nb_seed(int(rng.integers(0, 2 ** 31 - 1)))
    s = HS(z["x"], z["sig"], float(z["L"]), skin=a.skin, maxnb=64, swap=False)
    N = s.N
    t_start = time.time()
    if a.bench:
        s.run(50, a.delta)
        t0 = time.time(); st0 = s.stats.copy(); s.run(500, a.delta); dt = time.time() - t0; st = s.stats - st0
        print(f"[bench] phi={s.phi():.4f} delta={a.delta} acc={st[0]/st[1]:.3f} {dt/500*1e3:.3f} ms/sweep rebuilds/sweep={st[4]/500:.3f}")
        return
    T = a.sweeps
    t_save, origins = build_schedule(T, a.norig, a.per_decade)
    traj = np.empty((len(t_save), N, 3), dtype=np.float32)
    x0 = s.x.copy()
    traj[0] = s.x
    isave, t_now = 1, 0
    Zs = []
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    def save(nf, t_done, partial):
        tmp = a.out + ".tmp.npz"
        st = s.stats
        np.savez_compressed(tmp, x0=x0, sig=s.sig, L=s.L, phi=s.phi(), N=N, delta=a.delta, seed=a.seed,
                            t_save=t_save[:nf], traj=traj[:nf], origins=origins, acc=st[0] / max(st[1], 1),
                            wall=time.time() - t_start, sweeps=t_done, sweeps_planned=T, norig=a.norig,
                            partial=partial, init=os.path.basename(a.init), Z=np.array(Zs), xfinal=s.x)
        os.replace(tmp, a.out)

    last_rep = last_ck = time.time()
    for ts in t_save[1:]:
        nsw = int(ts - t_now)
        if nsw > 0:
            s.run(nsw, a.delta)
            t_now = int(ts)
        traj[isave] = s.x
        isave += 1
        if ts in origins or ts == T:
            Z, nov, _ = pressure_Z(s.x, s.sig, s.L)
            assert nov == 0
            Zs.append([ts, Z])
        if time.time() - last_rep > 300:
            el = time.time() - t_start
            print(f"  t={t_now}/{T} acc={s.stats[0]/max(s.stats[1],1):.3f} elapsed={el/60:.1f} min "
                  f"ETA={el/max(t_now,1)*(T-t_now)/60:.1f} min", flush=True)
            last_rep = time.time()
        if time.time() - last_ck > a.ckpt:
            save(isave, t_now, True)
            last_ck = time.time()
    save(isave, T, False)
    print(f"[done] T={T} acc={s.stats[0]/s.stats[1]:.4f} rebuilds={s.stats[4]} wall={(time.time()-t_start)/60:.1f} min Z={[round(z[1],3) for z in Zs]}", flush=True)


if __name__ == "__main__":
    main()
