"""F6: factorization test of the two-particle chain correlator phi_2(k1,k2,t) = <exp(i k1.dx_i) exp(i k2.dx_j)>,
|k1|=|k2|~7, k2=-k1 (relative motion), from simulation/results/curves_averaged.json.
R(t) = Re phi_2 / F_s(k,t)^2 of the sample-averaged curves, shown where F_s > 0.15:
contact pairs direction-averaged, random pairs, contact pairs with k along the bond (||) and transverse (perp).
(c) contact return probability P_c(t) = P(|r_ij(t)|<1.05 | contact at 0), an endpoint probability; (d) Im phi_2 along the bond (zero for the isotropic factorized form)."""
import json, numpy as np, matplotlib.pyplot as plt
from _style import *

CV = json.load(open(f"{SIM}/results/curves_averaged.json"))
RC = json.load(open(f"{SIM}/reanalysis/conditional_summary.json"))
phis = ["0.8", "1.2", "1.5", "1.58", "1.65", "1.72", "1.8", "2.0"]
cmap = plt.get_cmap("Blues"); cols = [cmap(0.3 + 0.7*i/(len(phis)-1)) for i in range(len(phis))]
series = (("c_pm", COL_BSELF, "-", "contact, dir.-avg."), ("r_pm", C["ink2"], "--", "random pairs"),
          ("c_Lpm", COL_BCOLL, "-", r"contact, $\mathbf{k}\parallel\hat{\mathbf{r}}_{ij}$"),
          ("c_Tpm", COL_PY, "-", r"contact, $\mathbf{k}\perp\hat{\mathbf{r}}_{ij}$"))

fig, axs = plt.subplots(2, 2, figsize=(7.1, 3.6))
for ax, p, lab in ((axs[0,0], "1.5", "(a)"), (axs[0,1], "2.0", "(b)")):
    F = CV[p]["Fs7"]; tF = np.array(F["t"]); mF = np.array(F["mean"])
    for key, col, ls, name in series:
        g = CV[p][key]; t = np.array(g["t"]); m = np.array(g["mean"])
        Fi = np.interp(t, tF, mF); ok = Fi > 0.15
        ax.plot(t[ok], m[ok]/Fi[ok]**2, color=col, ls=ls, lw=1.4, label=name)
    ax.axhline(1, color=C["muted"], lw=0.7)
    ax.set_xscale("log"); ax.set_ylim(0.4, 3.55); ax.set_xlim(1, 1e4 if p=="1.5" else 2e6)
    ax.set_yticks([1, 2, 3])
    ax.set_xlabel(r"$t$ (MC sweeps)"); ax.set_ylabel(r"$R(t)=\mathrm{Re}\,\phi_2/F_s^2$")
    ax.set_title(r"$\varphi=%s$ (%s)" % (p, "liquid" if p=="1.5" else "plateau"), fontsize=8.5)
    panel_label(ax, lab, x=-0.19)
axs[0,1].legend(loc="upper left", ncol=2, fontsize=5.8, labelspacing=0.16, columnspacing=0.7, handlelength=1.4, frameon=True, facecolor="white", framealpha=0.88, edgecolor="none")

ax = axs[1,0]
for p, col in zip(phis, cols):
    g = CV[p]["surv105"]; t = np.array(g["t"]); m = np.array(g["mean"]); ok = m > 0
    ax.plot(t[ok], m[ok], color=col, lw=1.3)
    if p in ("2.0", "1.8", "1.65", "1.2", "0.8"):
        dy = 1.12 if p == "1.8" else (0.88 if p == "1.65" else 1.0)
        ax.text(t[ok][-1]*1.25, m[ok][-1]*dy, r"$%s$" % p, fontsize=6.1, color=col, va="center")
ax.text(0.03, 0.06, r"$\varphi=0.8$–$2.0$ (light to dark)", transform=ax.transAxes, fontsize=6.3, color=C["ink2"])
tt = np.array([10, 2000]); ax.plot(tt, 0.6*tt**-0.5, color=C["muted"], lw=0.7, ls=":"); ax.text(42, 0.032, r"$t^{-1/2}$", fontsize=6.8, color=C["ink2"])
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 3e7); ax.set_ylim(6e-4, 1.2)
ax.set_xlabel(r"$t$ (MC sweeps)"); ax.set_ylabel(r"$P_c(t)$ (contact return)")
panel_label(ax, "(c)", x=-0.19)

ax = axs[1,1]
for p, col in zip(phis, cols):
    g = CV[p]["c_Lpm_sin"]; t = np.array(g["t"]); m = np.array(g["mean"])
    ax.plot(t, m, color=col, lw=1.3)
    if p == "2.0": ax.text(t[-1]*1.25, m[-1], r"$2.0$", fontsize=6.1, color=col, va="center")
    elif p == "1.8": ax.text(t[-1]*1.20, m[-1]+0.005, r"$1.8$", fontsize=6.1, color=col, va="center")
    elif p == "1.65": ax.text(t[-1]*1.25, m[-1]+0.002, r"$1.65$", fontsize=6.1, color=col, va="center")
ip = np.array(RC["2.0"]["curves"]["L-:Im_prod"])
ax.plot(ip[:, 0], ip[:, 1], color=cols[-1], ls=(0, (3.2, 1.5)), lw=1.15)
ax.axhline(0, color=C["muted"], lw=0.7)
ax.set_xscale("log"); ax.set_xlim(1, 3e7); ax.set_ylim(-0.03, 0.45)
ax.set_xlabel(r"$t$ (MC sweeps)"); ax.set_ylabel(r"$\mathrm{Im}\,\phi_2$, contact, $\mathbf{k}\parallel\hat{\mathbf{r}}_{ij}$")
panel_label(ax, "(d)", x=-0.19)
fig.tight_layout(w_pad=1.5, h_pad=1.2)
save(fig, "F6_factorization")
