#!/usr/bin/env python3
"""
Metropolis Monte Carlo for the Mari-Kurchan (MK) random-shift hard-sphere model
at lambda = infinity, d = 3, with planted (equilibrium) initial conditions.

Model
-----
N hard spheres of diameter 1 in a periodic cubic box of side L (rho = N/L^3,
phi = pi*rho/6).  Pair (i,j) interacts through the shifted separation
    r_ij = MI( x_i - x_j - A_ij ),
where MI is the minimum-image convention and the quenched shifts A_ij are
uniform in the box.  Consistency of the pair distance requires A_ji = -A_ij
(mod L), which is what we store.  Overlap <=> |r_ij| < 1.

Planting (MK Sec. IV A)
-----------------------
Draw all x_i uniformly in the box; for each pair i<j draw A_ij uniformly and
redraw while |MI(x_i - x_j - A_ij)| < 1.  Equivalently (and this is how it is
implemented) draw the shifted separation s_ij uniformly in the box conditioned
on |MI(s_ij)| >= 1 and set A_ij = x_i - x_j - s_ij.  At lambda = infinity this
is an equilibrium sample of the quenched ensemble (annealed = quenched).

Monte Carlo step
----------------
One attempted move: pick a particle i uniformly at random, propose
x_i' = x_i + delta*u with u uniform in [-1,1]^3, accept iff no overlap with any
j != i (all-pairs check, O(N) per move, early exit).  Random particle selection
+ symmetric proposal + hard-core acceptance => exact detailed balance.
Positions are kept UNWRAPPED (the minimum image convention takes care of the
periodicity), so the mean-squared displacement is directly available.

Time unit: 1 MC sweep = N attempted moves.

Output (.npz)
-------------
  x0      : planted positions (N,3)
  A       : shifts (N,N,3), A[j,i] = -A[i,j]
  L, phi, N, delta, seed
  t_save  : integer save times (sweeps), sorted
  traj    : unwrapped positions at t_save, shape (len(t_save), N, 3), float32
  origins : time origins used to build the schedule
  acc     : overall acceptance ratio
  wall    : wall-clock seconds
"""
import argparse
import os
import sys
import time

import numpy as np
from numba import njit

# ----------------------------------------------------------------------------
# numba kernels
# ----------------------------------------------------------------------------


@njit(cache=True)
def _seed(s):
    np.random.seed(s)


@njit(cache=True)
def plant(N, L, x, A):
    """Fill A (N,N,3) with planted shifts for given positions x (N,3)."""
    half = 0.5 * L
    for i in range(N):
        for j in range(i + 1, N):
            # draw shifted separation s uniformly in [-L/2, L/2)^3 with |s| >= 1
            while True:
                sx = np.random.uniform(-half, half)
                sy = np.random.uniform(-half, half)
                sz = np.random.uniform(-half, half)
                if sx * sx + sy * sy + sz * sz >= 1.0:
                    break
            ax = x[i, 0] - x[j, 0] - sx
            ay = x[i, 1] - x[j, 1] - sy
            az = x[i, 2] - x[j, 2] - sz
            # wrap into [-L/2, L/2)
            ax -= L * np.rint(ax / L)
            ay -= L * np.rint(ay / L)
            az -= L * np.rint(az / L)
            A[i, j, 0] = ax
            A[i, j, 1] = ay
            A[i, j, 2] = az
            A[j, i, 0] = -ax
            A[j, i, 1] = -ay
            A[j, i, 2] = -az
        A[i, i, 0] = 0.0
        A[i, i, 1] = 0.0
        A[i, i, 2] = 0.0


@njit(cache=True)
def count_overlaps(N, L, x, A, cut):
    """Number of pairs with shifted distance < cut (diagnostic)."""
    invL = 1.0 / L
    n = 0
    c2 = cut * cut
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i, 0] - x[j, 0] - A[i, j, 0]
            dy = x[i, 1] - x[j, 1] - A[i, j, 1]
            dz = x[i, 2] - x[j, 2] - A[i, j, 2]
            dx -= L * np.rint(dx * invL)
            dy -= L * np.rint(dy * invL)
            dz -= L * np.rint(dz * invL)
            if dx * dx + dy * dy + dz * dz < c2:
                n += 1
    return n


@njit(cache=True)
def run_moves(N, L, x, A, delta, nmoves):
    """Perform nmoves attempted single-particle moves. Returns #accepted."""
    invL = 1.0 / L
    acc = 0
    for m in range(nmoves):
        i = np.random.randint(0, N)
        nx = x[i, 0] + delta * (2.0 * np.random.random() - 1.0)
        ny = x[i, 1] + delta * (2.0 * np.random.random() - 1.0)
        nz = x[i, 2] + delta * (2.0 * np.random.random() - 1.0)
        ok = True
        for j in range(N):
            if j == i:
                continue
            dx = nx - x[j, 0] - A[i, j, 0]
            dy = ny - x[j, 1] - A[i, j, 1]
            dz = nz - x[j, 2] - A[i, j, 2]
            dx -= L * np.rint(dx * invL)
            dy -= L * np.rint(dy * invL)
            dz -= L * np.rint(dz * invL)
            if dx * dx + dy * dy + dz * dz < 1.0:
                ok = False
                break
        if ok:
            x[i, 0] = nx
            x[i, 1] = ny
            x[i, 2] = nz
            acc += 1
    return acc


# ----------------------------------------------------------------------------
# schedule
# ----------------------------------------------------------------------------


def log_times(tmax, per_decade=12):
    """Integer log-spaced times 1..tmax (sweeps), unique, sorted."""
    if tmax < 1:
        return np.array([], dtype=np.int64)
    n = int(np.ceil(np.log10(max(tmax, 1)) * per_decade)) + 1
    t = np.unique(np.round(np.logspace(0, np.log10(tmax), max(n, 2))).astype(np.int64))
    t = t[(t >= 1) & (t <= tmax)]
    return t


def build_schedule(T, norig, per_decade=12):
    """Union over origins m of {t0_m} U {t0_m + tau}, tau log-spaced up to T-t0_m.
    Origin 0 is at t=0 and spans the full run; the other origins are equally
    spaced in [0,T) so that they each have at least T/norig of lag available."""
    origins = np.array([(m * T) // norig for m in range(norig)], dtype=np.int64)
    times = set([0, T])
    for t0 in origins:
        times.add(int(t0))
        for tau in log_times(T - t0, per_decade):
            times.add(int(t0 + tau))
    return np.array(sorted(times), dtype=np.int64), origins


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--phi", type=float, required=True)
    ap.add_argument("--sweeps", type=int, required=True, help="total run length T (sweeps)")
    ap.add_argument("--delta", type=float, default=0.1, help="half-width of the cubic move")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--norig", type=int, default=8, help="number of time origins")
    ap.add_argument("--per_decade", type=int, default=12)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--bench", action="store_true", help="only time 20 sweeps and exit")
    ap.add_argument("--ckpt", type=float, default=600.0, help="checkpoint interval (s)")
    args = ap.parse_args()

    N, phi = args.N, args.phi
    rho = 6.0 * phi / np.pi
    L = (N / rho) ** (1.0 / 3.0)

    rng = np.random.default_rng(args.seed)
    _seed(int(rng.integers(0, 2**31 - 1)))

    # --- planting ---------------------------------------------------------
    t_start = time.time()
    x = rng.uniform(0.0, L, size=(N, 3)).astype(np.float64)
    A = np.empty((N, N, 3), dtype=np.float64)
    plant(N, L, x, A)
    nov = count_overlaps(N, L, x, A, 1.0)
    t_plant = time.time() - t_start
    print(f"[plant] N={N} phi={phi:.3f} rho={rho:.4f} L={L:.4f} overlaps={nov} ({t_plant:.1f}s)", flush=True)
    assert nov == 0

    if args.bench:
        run_moves(N, L, x, A, args.delta, N)  # compile
        t0 = time.time()
        acc = run_moves(N, L, x, A, args.delta, 20 * N)
        dt = time.time() - t0
        print(f"[bench] delta={args.delta} acc={acc/(20*N):.3f}  {dt/20*1e3:.2f} ms/sweep", flush=True)
        return

    # --- schedule ---------------------------------------------------------
    T = args.sweeps
    t_save, origins = build_schedule(T, args.norig, args.per_decade)
    traj = np.empty((len(t_save), N, 3), dtype=np.float32)
    x0 = x.copy()

    # --- run --------------------------------------------------------------
    t_run0 = time.time()
    acc_tot = 0
    t_now = 0
    isave = 0
    if t_save[0] == 0:
        traj[0] = x
        isave = 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def save(nframes, t_done, partial):
        # partial files contain the frames saved so far; 'sweeps' = sweeps completed
        tmp = args.out + ".tmp.npz"
        np.savez_compressed(
            tmp,
            x0=x0, A=A.astype(np.float32), L=L, phi=phi, N=N, delta=args.delta, seed=args.seed,
            t_save=t_save[:nframes], traj=traj[:nframes], origins=origins,
            acc=acc_tot / max(t_done * N, 1), wall=time.time() - t_start,
            sweeps=t_done, sweeps_planned=T, norig=args.norig, partial=partial,
        )
        os.replace(tmp, args.out)

    last_report = time.time()
    last_ckpt = time.time()
    for ts in t_save[isave:]:
        nsw = int(ts - t_now)
        if nsw > 0:
            acc_tot += run_moves(N, L, x, A, args.delta, nsw * N)
            t_now = int(ts)
        traj[isave] = x
        isave += 1
        if time.time() - last_report > 300:
            el = time.time() - t_run0
            print(f"  t={t_now}/{T} sweeps  acc={acc_tot/max(t_now*N,1):.3f}  "
                  f"elapsed={el/60:.1f} min  ETA={el/max(t_now,1)*(T-t_now)/60:.1f} min", flush=True)
            last_report = time.time()
        if time.time() - last_ckpt > args.ckpt:
            save(isave, t_now, True)
            last_ckpt = time.time()
            print(f"  [checkpoint] {isave} frames, t={t_now}", flush=True)
    wall = time.time() - t_start
    acc = acc_tot / (T * N)
    nov = count_overlaps(N, L, x, A, 1.0)
    print(f"[done] T={T} sweeps acc={acc:.4f} overlaps={nov} wall={wall/60:.1f} min", flush=True)
    assert nov == 0
    save(isave, T, False)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
