"""Numba kernels for polydisperse additive hard spheres (HS) in a periodic cubic box.

Model: continuous size polydispersity P(sigma) ~ sigma^-3 on [s_m, s_M], s_m/s_M = 0.4492 (Delta = 23%),
mean diameter sigma_bar = 1 (Berthier, Coslovich, Ninarello, Ozawa, PRL 116, 238002 (2016)).
Overlap <=> |r_ij| < sigma_ij = (sigma_i + sigma_j)/2.

Verlet neighbour list (built with a cell list) with per-pair cutoff  c_ij + skin, where c_ij = sigma_ij for the
fixed-size dynamics and c_ij = sigma_max (global) when diameters can be swapped.  The list is rebuilt
immediately when any particle has moved more than skin/2 since the last build, so the overlap test is exact.
Positions are kept UNWRAPPED (minimum image for distances), as in the MK engine (mk_mc.py).
"""
import numpy as np
from numba import njit

SIG_RATIO = 0.4492


def make_sizes(N, rng, ratio=SIG_RATIO):
    """Deterministic quantiles of P(s) ~ s^-3 on [ratio, 1], rescaled to mean 1, randomly permuted."""
    sM, sm = 1.0, ratio
    u = (np.arange(N) + 0.5) / N
    s = (sm ** -2 - u * (sm ** -2 - sM ** -2)) ** -0.5
    s = s / s.mean()
    return rng.permutation(s)


def box_for_phi(sig, phi):
    return (np.pi / 6.0 * np.sum(sig ** 3) / phi) ** (1.0 / 3.0)


@njit(cache=True)
def nb_seed(s):
    np.random.seed(s)


@njit(cache=True)
def build_list(x, sig, L, skin, swapmode, smax, xref, nl, nn):
    """Cell-list build of the Verlet list. Returns -1 on overflow else max neighbour count."""
    N = x.shape[0]
    maxnb = nl.shape[1]
    cmax = smax + skin
    nc = int(np.floor(L / cmax))
    if nc < 3:
        nc = 1
    cs = L / nc
    head = -np.ones(nc * nc * nc, np.int64)
    nxt = -np.ones(N, np.int64)
    cid = np.empty((N, 3), np.int64)
    for i in range(N):
        for a in range(3):
            w = x[i, a] - L * np.floor(x[i, a] / L)
            c = int(w / cs)
            if c >= nc:
                c = nc - 1
            cid[i, a] = c
        h = (cid[i, 0] * nc + cid[i, 1]) * nc + cid[i, 2]
        nxt[i] = head[h]
        head[h] = i
        xref[i, 0] = x[i, 0]
        xref[i, 1] = x[i, 1]
        xref[i, 2] = x[i, 2]
        nn[i] = 0
    invL = 1.0 / L
    mx = 0
    for i in range(N):
        for dxc in range(-1, 2):
            for dyc in range(-1, 2):
                for dzc in range(-1, 2):
                    if nc == 1 and (dxc != 0 or dyc != 0 or dzc != 0):
                        continue
                    cx = (cid[i, 0] + dxc) % nc
                    cy = (cid[i, 1] + dyc) % nc
                    cz = (cid[i, 2] + dzc) % nc
                    j = head[(cx * nc + cy) * nc + cz]
                    while j >= 0:
                        if j != i:
                            dx = x[i, 0] - x[j, 0]
                            dy = x[i, 1] - x[j, 1]
                            dz = x[i, 2] - x[j, 2]
                            dx -= L * np.rint(dx * invL)
                            dy -= L * np.rint(dy * invL)
                            dz -= L * np.rint(dz * invL)
                            if swapmode:
                                c = cmax
                            else:
                                c = 0.5 * (sig[i] + sig[j]) + skin
                            if dx * dx + dy * dy + dz * dz < c * c:
                                if nn[i] >= maxnb:
                                    return -1
                                nl[i, nn[i]] = j
                                nn[i] += 1
                        j = nxt[j]
        if nn[i] > mx:
            mx = nn[i]
    return mx


@njit(cache=True)
def overlap_one(i, px, py, pz, si, x, sig, L, nl, nn, skipj):
    invL = 1.0 / L
    for a in range(nn[i]):
        j = nl[i, a]
        if j == skipj:
            continue
        dx = px - x[j, 0]
        dy = py - x[j, 1]
        dz = pz - x[j, 2]
        dx -= L * np.rint(dx * invL)
        dy -= L * np.rint(dy * invL)
        dz -= L * np.rint(dz * invL)
        c = 0.5 * (si + sig[j])
        if dx * dx + dy * dy + dz * dz < c * c:
            return True
    return False


@njit(cache=True)
def run_moves(x, sig, L, delta, nmoves, pswap, dsmax, skin, smax, xref, nl, nn, stats):
    """nmoves attempted moves. With prob. pswap: swap diameters of two random particles (|ds| <= dsmax),
    else a cubic translation of half-width delta. stats[0..4] += (trans acc, trans att, swap acc, swap att, rebuilds).
    Returns 0 on success, -1 on neighbour-list overflow."""
    N = x.shape[0]
    swapmode = pswap > 0.0
    h2 = (0.5 * skin) ** 2
    for m in range(nmoves):
        if swapmode and np.random.random() < pswap:
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            stats[3] += 1
            if i == j or abs(sig[i] - sig[j]) > dsmax:
                continue
            si, sj = sig[j], sig[i]
            if overlap_one(i, x[i, 0], x[i, 1], x[i, 2], si, x, sig, L, nl, nn, j):
                continue
            # check j with its new diameter; sig[i] is still old here, pair (i,j) distance is unchanged
            if overlap_one(j, x[j, 0], x[j, 1], x[j, 2], sj, x, sig, L, nl, nn, i):
                continue
            sig[i] = si
            sig[j] = sj
            stats[2] += 1
        else:
            i = np.random.randint(0, N)
            stats[1] += 1
            px = x[i, 0] + delta * (2.0 * np.random.random() - 1.0)
            py = x[i, 1] + delta * (2.0 * np.random.random() - 1.0)
            pz = x[i, 2] + delta * (2.0 * np.random.random() - 1.0)
            ddx = px - xref[i, 0]
            ddy = py - xref[i, 1]
            ddz = pz - xref[i, 2]
            if ddx * ddx + ddy * ddy + ddz * ddz > h2:
                r = build_list(x, sig, L, skin, swapmode, smax, xref, nl, nn)
                stats[4] += 1
                if r < 0:
                    return -1
            if overlap_one(i, px, py, pz, sig[i], x, sig, L, nl, nn, -1):
                continue
            x[i, 0] = px
            x[i, 1] = py
            x[i, 2] = pz
            stats[0] += 1
    return 0


@njit(cache=True)
def min_gap_ratio(x, sig, L, nl, nn):
    """min over listed pairs of r_ij / sigma_ij (only valid if the list covers all near pairs)."""
    invL = 1.0 / L
    mn = 1e9
    N = x.shape[0]
    for i in range(N):
        for a in range(nn[i]):
            j = nl[i, a]
            if j <= i:
                continue
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            dz = x[i, 2] - x[j, 2]
            dx -= L * np.rint(dx * invL)
            dy -= L * np.rint(dy * invL)
            dz -= L * np.rint(dz * invL)
            q = np.sqrt(dx * dx + dy * dy + dz * dz) / (0.5 * (sig[i] + sig[j]))
            if q < mn:
                mn = q
    return mn


@njit(cache=True)
def gap_counts(x, sig, L, xis):
    """All-pairs counts n(xi) of pairs with 1 <= r/sigma_ij < 1 + xi, and the number of overlaps (r/sigma_ij < 1)."""
    N = x.shape[0]
    invL = 1.0 / L
    out = np.zeros(len(xis) + 1, np.int64)
    xm = xis.max()
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            dz = x[i, 2] - x[j, 2]
            dx -= L * np.rint(dx * invL)
            dy -= L * np.rint(dy * invL)
            dz -= L * np.rint(dz * invL)
            q = np.sqrt(dx * dx + dy * dy + dz * dz) / (0.5 * (sig[i] + sig[j]))
            if q < 1.0:
                out[len(xis)] += 1
            elif q < 1.0 + xm:
                for a in range(len(xis)):
                    if q < 1.0 + xis[a]:
                        out[a] += 1
    return out


def pressure_Z(x, sig, L, xis=np.array([0.0025, 0.005, 0.01, 0.02])):
    """Reduced pressure Z = beta P / rho = 1 + n(xi)/(3 N xi), linearly extrapolated xi -> 0 (virtual inflation)."""
    c = gap_counts(x, sig, L, xis)
    N = len(sig)
    y = c[:-1] / (3.0 * N * xis)
    p = np.polyfit(xis, y, 1)
    return 1.0 + p[1], int(c[-1]), (1.0 + y).tolist()


def Z_bmcsl(sig, phi):
    m1, m2, m3 = np.mean(sig), np.mean(sig ** 2), np.mean(sig ** 3)
    return 1 / (1 - phi) + 3 * m1 * m2 / m3 * phi / (1 - phi) ** 2 + m2 ** 3 / m3 ** 2 * phi ** 2 * (3 - phi) / (1 - phi) ** 3


class HS:
    """Convenience wrapper holding the state and the Verlet list."""

    def __init__(self, x, sig, L, skin=0.3, maxnb=96, swap=False):
        self.x = np.ascontiguousarray(x, dtype=np.float64)
        self.sig = np.ascontiguousarray(sig, dtype=np.float64)
        self.L = float(L)
        self.skin = skin
        self.swap = swap
        self.N = len(sig)
        self.smax = float(self.sig.max())
        self.xref = np.empty_like(self.x)
        self.nl = np.empty((self.N, maxnb), np.int64)
        self.nn = np.empty(self.N, np.int64)
        self.stats = np.zeros(5, np.int64)
        self.rebuild()

    def rebuild(self):
        r = build_list(self.x, self.sig, self.L, self.skin, self.swap, self.smax, self.xref, self.nl, self.nn)
        if r < 0:
            raise RuntimeError("neighbour list overflow")
        return r

    def run(self, nsweeps, delta, pswap=0.0, dsmax=10.0):
        r = run_moves(self.x, self.sig, self.L, delta, int(nsweeps * self.N), pswap if self.swap else 0.0, dsmax,
                      self.skin, self.smax, self.xref, self.nl, self.nn, self.stats)
        if r < 0:
            raise RuntimeError("neighbour list overflow")

    def scale(self, f):
        self.x *= f
        self.L *= f
        self.rebuild()

    def phi(self):
        return np.pi / 6.0 * np.sum(self.sig ** 3) / self.L ** 3
