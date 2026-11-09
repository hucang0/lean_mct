import numpy as np
ks = np.array([3.0, 5.0, 7.0, 10.0])
M = 590; i = np.arange(M) + 0.5; th = np.arccos(1 - 2 * i / M); ph = np.pi * (1 + 5 ** 0.5) * i
NV = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
def est(Xw):
    """f_meas (random frame pairs), f_gauss (same MSD), f_aniso (per-particle anisotropic Gaussian), C, MSD."""
    T = len(Xw); dx = Xw - Xw.mean(0); C = np.einsum('tna,tnb->nab', dx, dx) / (T - 1)
    rng = np.random.default_rng(0); pa = rng.integers(0, T, 6000); pb = rng.integers(0, T, 6000); keep = pa != pb
    dn = np.linalg.norm(Xw[pa[keep]] - Xw[pb[keep]], axis=2); msd = float(np.mean(dn ** 2))
    fm = np.array([np.mean(np.sinc(k * dn / np.pi)) for k in ks]); fg = np.exp(-ks ** 2 * msd / 6)
    q = np.einsum('ma,nab,mb->nm', NV, C, NV); fa = np.array([np.mean(np.exp(-k * k * q)) for k in ks])
    return fm, fg, fa, C, msd
