"""SM Fig. S5: testing the restored pair against planted MC of the MK model.
(a) MSD plateau vs phi across eight arrested densities.
(b) f_s(k ~ 7) on the plateau and per-particle cage covariance mixture."""
import json, numpy as np
from pathlib import Path
from _style import *
import matplotlib.pyplot as plt

RES = str(Path(__file__).resolve().parents[2] / "data" / "restore")
G = json.load(open(f"{RES}/gauss_curves_dense.json"))
V = json.load(open(f"{RES}/vng_scan.json")) if __import__('os').path.exists(f"{RES}/vng_scan.json") else None
D = json.load(open(f"{RES}/diag_kres.json"))
S = json.load(open(f"{RES}/sim_summary.json"))['densities']
COL_ON, COL_OFF, COL_SIM, COL_MIX = C["green"], C["blue"], C["ink"], C["magenta"]

fig, axs = plt.subplots(1, 2, figsize=(5.2, 1.85), gridspec_kw=dict(wspace=0.36))
fig.subplots_adjust(top=0.76, bottom=0.23, left=0.11, right=0.98)

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
panel_label(ax, "(a)", x=-0.24, y=1.05)

# (b)
ax = axs[1]
ax.plot(G['brc']['phi'], G['brc']['f7'], color=COL_ON, lw=1.4)
fd = [r for r in D if r['status'] == 'glass']
ax.plot([r['phi'] for r in fd], [r['f']['k7'] for r in fd], 's', mfc='none', mec=COL_OFF, ms=3.8, mew=0.9)
f7 = np.array([S[p]['f_meas'][0][2] for p in S]); f7e = np.array([S[p]['f_meas'][1][2] for p in S]); fa7 = np.array([S[p]['f_aniso'][2] for p in S])
ax.plot(ph, fa7, 'D', color=COL_MIX, mec='white', mew=0.5, ms=3.4, label="cage mixture", zorder=6)
ax.errorbar(ph[~drift], f7[~drift], yerr=f7e[~drift], fmt='o', color=COL_SIM, ms=3.6, capsize=1.5, elinewidth=0.8, zorder=5)
ax.errorbar(ph[drift], f7[drift], yerr=f7e[drift], fmt='o', mfc='white', color=COL_SIM, ms=3.6, capsize=1.5, elinewidth=0.8, zorder=5)
ax.set_xlim(1.7, 4.6); ax.set_ylim(0, 1.0)
ax.set_xlabel(r"$\varphi$"); ax.set_ylabel(r"$f_s(k\simeq7)$")
panel_label(ax, "(b)", x=-0.22, y=1.05)

h0, l0 = axs[0].get_legend_handles_labels(); h1, l1 = axs[1].get_legend_handles_labels()
order = [l0.index("simulation")] + [i for i, l in enumerate(l0) if l != "simulation"]
fig.legend([h0[i] for i in order] + h1, [l0[i] for i in order] + l1, loc="upper center", bbox_to_anchor=(0.5, 1.02),
           ncol=3, fontsize=6.2, handlelength=1.7, columnspacing=1.0, frameon=False)
save(fig, "F9_restore_msd")
