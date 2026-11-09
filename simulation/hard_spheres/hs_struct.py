"""Structure diagnostics: Steinhardt bond-orientational order (crystallinity check) and partial g(r)."""
import numpy as np
from scipy.spatial import cKDTree
try:
    from scipy.special import sph_harm_y as _Y

    def Ylm(l, m, th, ph):
        return _Y(l, m, th, ph)
except ImportError:  # older scipy
    from scipy.special import sph_harm as _Y0

    def Ylm(l, m, th, ph):
        return _Y0(m, l, ph, th)


def neighbor_pairs(x, sig, L, rcut_scaled=1.35):
    """Pairs with r_ij < rcut_scaled * sigma_ij (first coordination shell of the scaled g(r))."""
    xw = x - L * np.floor(x / L)
    xw = np.clip(xw, 0, np.nextafter(L, 0))
    t = cKDTree(xw, boxsize=L)
    P = t.query_pairs(rcut_scaled * sig.max(), output_type="ndarray")
    d = xw[P[:, 0]] - xw[P[:, 1]]
    d -= L * np.rint(d / L)
    r = np.linalg.norm(d, axis=1)
    sij = 0.5 * (sig[P[:, 0]] + sig[P[:, 1]])
    keep = r < rcut_scaled * sij
    return P[keep], d[keep], r[keep]


def q6_stats(x, sig, L, l=6, dc=0.7, nsolid=8):
    """Global Q6 and fraction of crystal-like particles (ten Wolde-Frenkel: >= nsolid bonds with d6 > dc)."""
    N = len(sig)
    P, d, r = neighbor_pairs(x, sig, L)
    th = np.arccos(np.clip(d[:, 2] / r, -1, 1))
    ph = np.arctan2(d[:, 1], d[:, 0])
    ms = np.arange(-l, l + 1)
    Y = np.stack([Ylm(l, m, th, ph) for m in ms], 1)          # (P, 2l+1), for vector i->j reversed parity even for l=6
    q = np.zeros((N, 2 * l + 1), complex)
    nb = np.zeros(N)
    np.add.at(q, P[:, 0], Y)
    np.add.at(q, P[:, 1], Y)        # l even: Y(-r) = Y(r)
    np.add.at(nb, P[:, 0], 1)
    np.add.at(nb, P[:, 1], 1)
    Q6 = np.sqrt(4 * np.pi / (2 * l + 1) * np.sum(np.abs(q.sum(0)) ** 2)) / nb.sum()
    qn = q / np.maximum(np.linalg.norm(q, axis=1), 1e-12)[:, None]
    d6 = np.real(np.sum(qn[P[:, 0]] * np.conj(qn[P[:, 1]]), 1))
    sol = np.zeros(N)
    np.add.at(sol, P[d6 > dc, 0], 1)
    np.add.at(sol, P[d6 > dc, 1], 1)
    return float(Q6), float(np.mean(sol >= nsolid))


def gr_scaled(x, sig, L, rmax=2.5, nb=250):
    """g(r / sigma_ij) for all pairs (scaled radial distribution)."""
    N = len(sig)
    xw = x - L * np.floor(x / L)
    xw = np.clip(xw, 0, np.nextafter(L, 0))
    t = cKDTree(xw, boxsize=L)
    P = t.query_pairs(rmax * sig.max(), output_type="ndarray")
    d = xw[P[:, 0]] - xw[P[:, 1]]
    d -= L * np.rint(d / L)
    r = np.linalg.norm(d, axis=1)
    sij = 0.5 * (sig[P[:, 0]] + sig[P[:, 1]])
    s = r / sij
    h, e = np.histogram(s, bins=nb, range=(0, rmax))
    # ideal: expected count for pair with sigma_ij in shell [s, s+ds]: rho * 4 pi sigma_ij^3 s^2 ds summed / 2
    rho = N / L ** 3
    S3 = np.add.outer(sig, sig) / 2.0
    m3 = (np.sum(S3 ** 3) - np.sum(sig ** 3)) / 2.0 / N   # mean over pairs, per particle
    sh = 4 * np.pi / 3 * (e[1:] ** 3 - e[:-1] ** 3)
    ideal = rho / N * m3 * N * sh   # = rho * m3 * shell
    g = h / (ideal)
    return 0.5 * (e[1:] + e[:-1]), g
