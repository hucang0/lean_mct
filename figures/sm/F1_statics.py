"""F1: MK statics and the structure of the frozen caging operator.
(a) shift-aware Mayer function f~(k) (analytic) with S(k) = 1;
(b) single-leg vertex weight q^4 f~(q)^2/(16 pi^2) -> cos^2 q (UV tail);
(c) Perron vectors v_P(k) of the frozen operator A(k,q) (M=100, q_max=40) for MK Bself, MK Bcoll and PY
    at phi_c(2), with the N=2 critical plateaus f_1*(k); data: numerics/results/critical_eigvec.json."""
import json, numpy as np, matplotlib.pyplot as plt
from _style import *

k = np.linspace(0.05, 40, 4000)
ft = -(4*np.pi/k**3)*(np.sin(k) - k*np.cos(k))          # f~(k), MK Eq. (25) / sec_theory

rec = json.load(open(f"{NUM}/results/critical_eigvec.json"))
R = {r["variant"]+("@%.2f" % r["phi"]): r for r in rec}
bself = R["Bself@2.80"]; bcoll = R["Bcoll@2.46"]; py = R["PY@0.52"]

fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.0))
ax = axs[0]
ax.plot(k, ft, color=COL_BSELF, lw=1.3, label=r"$\tilde f(k)$ (shift-aware)")
ax.axhline(0, color=C["muted"], lw=0.6)
ax.axhline(1, color=C["ink2"], lw=1.0, ls="--", label=r"$S(k)\equiv 1$ (ordinary density)")
ax.set_xlim(0, 20); ax.set_ylim(-4.5, 1.5)
ax.set_xlabel(r"$k$ (units of $1/\sigma$)"); ax.set_ylabel(r"$\tilde f(k),\ S(k)$")
ax.legend(loc="lower right", fontsize=6.5, frameon=True, facecolor="white", framealpha=0.88, edgecolor="none")
panel_label(ax, "(a)", x=-0.22)

ax = axs[1]
w = k**4*ft**2/(16*np.pi**2)
ax.plot(k, w, color=COL_BSELF, lw=1.0, label=r"$q^4\tilde f(q)^2/16\pi^2$")
ax.plot(k, np.cos(k)**2, color=C["muted"], lw=0.7, ls=":", label=r"$\cos^2 q$ (large-$q$ limit)")
ax.set_xlim(0, 40); ax.set_ylim(0, 1.42)
ax.set_xlabel(r"$q$"); ax.set_ylabel("single-leg vertex weight")
ax.legend(loc="upper right", fontsize=6.2, labelspacing=0.18, frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")
panel_label(ax, "(b)", x=-0.22)

ax = axs[2]
for r, col, lab in ((bself, COL_BSELF, "MK, $\\widetilde W_s$"), (bcoll, COL_BCOLL, "MK, $\\widetilde W_{\\rm coll}$"), (py, COL_PY, "PY, $\\varphi_c(2)$")):
    kk = np.array(r["k"]); v = np.array(r["vP"]); v = v/v.max()
    ax.plot(kk, v, color=col, lw=1.4, label=lab)
    f1 = np.array(r["f1"])
    ax.plot(kk, f1, color=col, lw=0.9, ls=":")
ax.set_xscale("log"); ax.set_xlim(0.2, 55); ax.set_ylim(0, 1.38)
ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
ax.set_xlabel(r"$k$"); ax.set_ylabel(r"$v_{\rm P}$ (solid), $f_1^*$ (dotted)")
ax.legend(loc="upper right", fontsize=6.0, handlelength=1.3, handletextpad=0.35, labelspacing=0.16, frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")
panel_label(ax, "(c)", x=-0.22)
fig.tight_layout(w_pad=1.2)
save(fig, "F1_statics")
