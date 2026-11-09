"""Additional diagnostic overview (not a numbered manuscript figure): restoring the bond hard core, matched ablation against planted MC of the MK model.
(a) MSD plateau vs phi.
(b) Critical density vs order: diagonal hierarchy order N and exclusion order ell.
(c) f_s(k ~ 7) on the plateau."""
import json, numpy as np
from _style import *
import matplotlib.pyplot as plt
G = json.load(open(f"{RES}/gauss_curves_dense.json"))
V = json.load(open(f"{RES}/vng_scan.json")) if __import__('os').path.exists(f"{RES}/vng_scan.json") else None
D = json.load(open(f"{RES}/diag_kres.json"))
S = json.load(open(f"{RES}/sim_summary.json"))['densities']
RS = json.load(open(f"{RES}/rescatter_series_gauss.json"))
COL_ON, COL_OFF, COL_SIM, COL_MIX = C["green"], C["blue"], C["ink"], C["magenta"]

fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.75), gridspec_kw=dict(wspace=0.38))
fig.subplots_adjust(top=0.76, bottom=0.23)
# (a)
ax = axs[0]
ax.plot(G['brc']['phi'], G['brc']['msd'], color=COL_ON, lw=1.4, label="restored (Gaussian)")
if V:
    vr = [r for r in V['full']['rows'] if r['status'] == 'glass']
    ax.plot([r['phi'] for r in vr], [r['msd'] for r in vr], 'o', mfc='none', mec=COL_ON, ms=4.2, mew=0.9, label="restored (static)")
ax.plot(G['diag']['phi'], G['diag']['msd'], color=COL_OFF, lw=1.4, ls=(0, (4, 1.6)), label=r"removed ($Z\equiv 1$)")
dr = [r for r in D if r['status'] == 'glass']
ax.plot([r['phi'] for r in dr], [r['msd_from_k2'] for r in dr], 's', mfc='none', mec=COL_OFF, ms=3.8, mew=0.9, label=r"diagonal $N=2$")
ph = np.array([float(p) for p in S]); m = np.array([S[p]['msd'][0] for p in S]); e = np.array([S[p]['msd'][1] for p in S])
drift = ph < 1.95
ax.errorbar(ph[~drift], m[~drift], yerr=e[~drift], fmt='o', color=COL_SIM, ms=3.6, capsize=1.5, elinewidth=0.8, label="simulation", zorder=5)
ax.errorbar(ph[drift], m[drift], yerr=e[drift], fmt='o', mfc='white', color=COL_SIM, ms=3.6, capsize=1.5, elinewidth=0.8, zorder=5)
ax.set_yscale('log'); ax.set_xlim(1.7, 4.6); ax.set_ylim(0.015, 1.2)
ax.set_xlabel(r"$\varphi$"); ax.set_ylabel("MSD plateau")
ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0]); ax.set_yticklabels(['0.02', '0.05', '0.1', '0.2', '0.5', '1'])
ax.minorticks_off()
panel_label(ax, "(a)", x=-0.28, y=1.05)
# (b)
ax = axs[1]
ax.axhspan(1.80, 1.93, color=C["grid"], lw=0)
ax.axhline(2.40, color=C["muted"], lw=0.8, ls=':')
ax.text(12.0, 2.45, r"$\varphi_p$", fontsize=6.5, color=C["ink2"], ha='right', va='bottom')
N = RS['diag_hierarchy']['N']; pcN = RS['diag_hierarchy']['phi_c']
ax.plot(N, pcN, 's-', color=COL_OFF, mfc='white', ms=3.8, lw=1.2)
ax.text(5.2, 3.88, r"$N$", fontsize=7.0, color=COL_OFF, ha='left', va='center')
import glob as _g
K = sorted((json.load(open(f)) for f in _g.glob(f"{RES}/kres_series_n*.json")), key=lambda d: d['n'])
kn = [d['n'] for d in K]; kp = [0.5 * (d['fold'][0] + d['fold'][1]) for d in K]; ke = [0.5 * (d['fold'][1] - d['fold'][0]) for d in K]
ax.errorbar(kn, kp, yerr=ke, fmt='o-', color=COL_ON, ms=3.5, lw=1.2, elinewidth=0.8, capsize=0, zorder=5)
g = [r for r in RS['gauss'] if r['n'] != 'inf' and r['n'] <= 12]
ax.plot([r['n'] for r in g], [r['phi_c'] for r in g], '^', color=COL_ON, mfc='white', ms=3.2, mew=0.75, lw=0, zorder=4)
ax.text(3.8, 2.15, r"$\ell$", fontsize=7.0, color=COL_ON, ha='left', va='bottom')
KINF = 1.776
ax.axhline(KINF, color=COL_ON, lw=0.8, ls=(0, (2, 1.5)))
ax.text(12.0, 1.62, r"$\varphi_G$", fontsize=6.5, color=COL_ON, ha='right', va='center')
ax.set_xlim(0.5, 12.3); ax.set_ylim(1.42, 4.70); ax.set_xticks([1, 2, 4, 6, 8, 10, 12])
ax.set_xlabel(r"order ($N$ or $\ell$)"); ax.set_ylabel(r"$\varphi_c$")
panel_label(ax, "(b)", x=-0.25, y=1.05)
# (c)
ax = axs[2]
ax.plot(G['brc']['phi'], G['brc']['f7'], color=COL_ON, lw=1.4)
fd = [r for r in D if r['status'] == 'glass']
ax.plot([r['phi'] for r in fd], [r['f']['k7'] for r in fd], 's', mfc='none', mec=COL_OFF, ms=3.8, mew=0.9)
f7 = np.array([S[p]['f_meas'][0][2] for p in S]); f7e = np.array([S[p]['f_meas'][1][2] for p in S]); fa7 = np.array([S[p]['f_aniso'][2] for p in S])
ax.plot(ph, fa7, 'D', color=COL_MIX, mec='white', mew=0.5, ms=3.4, label="cage mixture", zorder=6)
ax.errorbar(ph[~drift], f7[~drift], yerr=f7e[~drift], fmt='o', color=COL_SIM, ms=3.6, capsize=1.5, elinewidth=0.8, zorder=5)
ax.errorbar(ph[drift], f7[drift], yerr=f7e[drift], fmt='o', mfc='white', color=COL_SIM, ms=3.6, capsize=1.5, elinewidth=0.8, zorder=5)
ax.set_xlim(1.7, 4.6); ax.set_ylim(0, 1.0)
ax.set_xlabel(r"$\varphi$"); ax.set_ylabel(r"$f_s(k\simeq7)$")
panel_label(ax, "(c)", x=-0.25, y=1.05)
h0, l0 = axs[0].get_legend_handles_labels(); h2, l2 = axs[2].get_legend_handles_labels()
order = [l0.index("simulation")] + [i for i, l in enumerate(l0) if l != "simulation"]
fig.legend([h0[i] for i in order] + h2, [l0[i] for i in order] + l2, loc="upper center", bbox_to_anchor=(0.5, 1.02),
           ncol=6, fontsize=6.4, handlelength=1.8, columnspacing=1.1, frameon=False)
fig.savefig(f"{OUT}/fig3_restore.pdf"); fig.savefig(f"{OUT}/fig3_restore.png", dpi=220)
print("wrote fig3_restore")
