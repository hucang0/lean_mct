"""F4: grid dependence. (a) phi_c(2), phi_c(3) versus grid spacing h (F4_grid_convergence.json) with the
h -> 0 Aitken extrapolations (headline.json h_to_0_extrapolation); (b) lambda_P/phi and (c) Perron-vector peak
k_peak versus k_min on all grids (headline.json perron_grids = Table T4), with PY for reference."""
import json, numpy as np, matplotlib.pyplot as plt
from _style import *

G = json.load(open(f"{NUM}/figures/F4_grid_convergence.json"))
H = json.load(open(f"{NUM}/results/headline.json"))
ext = H["h_to_0_extrapolation"]; PG = H["perron_grids"]

fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.05))
ax = axs[0]
for v, col in (("Bself", COL_BSELF), ("Bcoll", COL_BCOLL)):
    for N, mk in ((2, "o"), (3, "s")):
        rows = G[f"{v}_N{N}"]
        q40 = sorted([r for r in rows if r["qmax"]==40.0], key=lambda r: r["h"]); q80 = [r for r in rows if r["qmax"]==80.0]
        ax.plot([r["h"] for r in q40], [r["phi_c"] for r in q40], color=col, marker=mk, ms=4.5, lw=1.2)
        ax.plot([r["h"] for r in q80], [r["phi_c"] for r in q80], color=col, marker=mk, mfc="white", ms=4.5, ls="none")
        e = ext[f"{v}_N{N}"]["phi_c_h0"]; ax.plot([0], [e], marker="_", color=col, ms=9, mew=1.5, ls="none")
ax.plot([], [], marker="o", color=C["ink2"], ls="-", lw=1.2, label=r"$N=2$, $q_{\max}=40$")
ax.plot([], [], marker="s", color=C["ink2"], ls="-", lw=1.2, label=r"$N=3$, $q_{\max}=40$")
ax.plot([], [], marker="o", mfc="white", color=C["ink2"], ls="none", label=r"$q_{\max}=80$")
ax.plot([], [], marker="_", color=C["ink2"], ms=9, mew=1.5, ls="none", label=r"$h\to0$ extrap.")
ax.text(0.55, 0.50, r"$\widetilde W_s$", transform=ax.transAxes, color=COL_BSELF, fontsize=7.5)
ax.text(0.55, 0.14, r"$\widetilde W_{\rm coll}$", transform=ax.transAxes, color=COL_BCOLL, fontsize=7.5)
ax.set_xlim(-0.05, 0.85); ax.set_ylim(2.28, 3.82)
ax.set_xlabel(r"grid spacing $h$"); ax.set_ylabel(r"$\varphi_c(N)$")
ax.legend(loc="upper left", ncol=2, fontsize=5.6, labelspacing=0.15, columnspacing=0.6, handlelength=1.3, handletextpad=0.35,
          borderaxespad=0.2, frameon=True, facecolor="white", framealpha=0.88, edgecolor="none")
panel_label(ax, "(a)", x=-0.25)

def rows(v, q): return sorted([dict(p, kmin=p.get("kmin", 20.0/p["M"])) for p in PG if p["variant"]==v and p["qmax"]==q], key=lambda p: p["kmin"])
ax = axs[1]
for v, col in (("Bself", COL_BSELF), ("Bcoll", COL_BCOLL)):
    r = rows(v, 40.0); ax.plot([p["kmin"] for p in r], [p["lamP_over_phi"] for p in r], color=col, marker="o", ms=4.5, lw=1.2)
    r = rows(v, 80.0); ax.plot([p["kmin"] for p in r], [p["lamP_over_phi"] for p in r], color=col, marker="o", mfc="white", ms=4.5, ls="none")
ax.text(0.05, 0.58, r"$\widetilde W_s$: $\lambda_1\to\approx 6.1$", transform=ax.transAxes, color=COL_BSELF, fontsize=7.2)
ax.text(0.05, 0.30, r"$\widetilde W_{\rm coll}$: $\lambda_1\to\approx 4.7$", transform=ax.transAxes, color=COL_BCOLL, fontsize=7.2)
ax.set_xscale("log"); ax.set_xlim(0.02, 0.5); ax.set_ylim(4.3, 6.3)
ax.set_xlabel(r"$k_{\min}=h/2$"); ax.set_ylabel(r"$\lambda_{\rm P}/\varphi=\lambda_1$")
panel_label(ax, "(b)", x=-0.25)

ax = axs[2]
for v, col in (("Bself", COL_BSELF), ("Bcoll", COL_BCOLL)):
    r = rows(v, 40.0); ax.plot([p["kmin"] for p in r], [p["k_peak_perron_vector"] for p in r], color=col, marker="o", ms=4.5, lw=1.2)
    r = rows(v, 80.0); ax.plot([p["kmin"] for p in r], [p["k_peak_perron_vector"] for p in r], color=col, marker="o", mfc="white", ms=4.5, ls="none")
r = rows("PY", 40.0); ax.plot([p["kmin"] for p in r], [p["k_peak_perron_vector"] for p in r], color=COL_PY, marker="x", ms=6, mew=1.2, lw=1.2)
ax.text(0.04, 0.66, r"PY: $k_{\rm peak}\simeq 7.0$ (cage scale)", transform=ax.transAxes, color=COL_PY, fontsize=7.0)
ax.text(0.04, 0.44, r"MK $\widetilde W_s$: $3k_{\min}$", transform=ax.transAxes, color=COL_BSELF, fontsize=7.2)
ax.text(0.36, 0.08, r"MK $\widetilde W_{\rm coll}$: $k_{\min}$", transform=ax.transAxes, color=COL_BCOLL, fontsize=7.2)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.02, 0.5); ax.set_ylim(0.02, 14)
ax.set_xlabel(r"$k_{\min}$"); ax.set_ylabel(r"Perron-vector peak $k_{\rm peak}$")
panel_label(ax, "(c)", x=-0.25)
fig.tight_layout(w_pad=1.3)
save(fig, "F4_grid")
