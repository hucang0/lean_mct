"""F8 (Appendix D): the exact Gaussian replica function I(m=0.96,K)/[d(m-1)] (d = 3) against points digitized from the
left panel of Fig. 18 of Mari and Kurchan (2011). Only digitized numbers are plotted; the published image is not used.
  crosses: their evaluated points; red: their 'largest gradient' line (slope 0.315); grey: their K -> inf asymptote;
  green: exact curve shifted to that asymptote; blue: exact curve halved;
  magenta: exact curve under y -> y/2, ln K -> (ln K)/2 (reproduces the points; slope unchanged);
  dotted magenta: tangent of the exact maximal slope 0.2933 at the steepest point of the magenta curve.
Data: data/brc/verify/fig18_digitized.json, data/brc/verify/fig18_curve_mine.dat."""
import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from _style import *

DD = Path(STEP6) / "verify"
D = json.load(open(DD / "fig18_digitized.json"))
lnK, y = np.loadtxt(DD / "fig18_curve_mine.dat", unpack=True)
pts = np.array(D["points_lnK_y"]); (xa, ya), (xb, yb) = D["largest_gradient_line"]; asym = D["asymptote_y"]
plateau = D["plateau_exact_K_to_inf"]

fig, ax = plt.subplots(figsize=(3.4, 2.55))
ax.axhline(asym, color=C["muted"], lw=0.8, zorder=1)
ax.plot(lnK, y + (asym - plateau), color="#2a9d3c", lw=1.2, label="exact, shifted to the asymptote")
ax.plot(lnK, y / 2, color="#2f5fd0", lw=1.2, label="exact, halved")
x3, y3 = lnK / 2, y / 2
ax.plot(x3, y3, color="#b0309f", lw=1.6, label=r"exact, $y\to y/2$, $\ln K\to\frac{1}{2}\ln K$")
g = np.gradient(y3, x3); i = int(np.argmax(g))
xt = np.array([x3[i] - 1.3, x3[i] + 1.3])
ax.plot(xt, y3[i] + D["exact_max_slope_m096"] * (xt - x3[i]), color="#b0309f", lw=0.9, ls=":", label=r"exact maximal slope $0.293$")
ax.plot([xa, xb], [ya, yb], color="#d03030", lw=1.2, label=r"Ref.: “largest gradient”, slope $0.315$")
ax.plot(pts[:, 0], pts[:, 1], "x", color="k", ms=4.5, mew=1.0, label="Ref.: evaluated points (digitized)")
ax.set_xlim(-3.62, 1.05); ax.set_ylim(-17.5, -16.68)
ax.set_xlabel(r"$\ln K$"); ax.set_ylabel(r"$I(m,K)/[d(m-1)]$,  $m=0.96$")
ax.legend(loc="lower right", fontsize=5.8, handlelength=1.8, frameon=True, framealpha=0.9, edgecolor="none")
fig.tight_layout()
save(fig, "F8_fig18_overlay")
