#!/usr/bin/env python3
"""Figures for R2 (figs/*.pdf + .png) from results/*.json.  Usage: r2_figs.py"""
import os, sys, json
sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r2_common as C

PHIS = ["0.500", "0.550", "0.570", "0.585", "0.600"]
KEYS = {p: f"phi{p}_N1000_d0.050" for p in PHIS}
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]     # ordinal blue (steps 250..650)
ORANGE, AQUA, VIOLET, GREY = "#eb6834", "#1baf7a", "#6250d6", "#8a8986"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.6, "legend.frameon": False,
                     "figure.dpi": 140, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(C.FIGS, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def J(name):
    return json.load(open(C.result_path(name)))


def fig_comob():
    cm, fits = J("comob.json"), J("xi_fits.json")
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))
    # (a) R_perp(s) - 1 vs s = r/sigma_ij at tau7 (s-bins complete for all pair sizes up to s = 3.08)
    for p, col in zip(PHIS, RAMP):
        e = cm[KEYS[p]]; a = e["at"]["tau7"]["S_s"]; x = np.array(e["s_centres"])
        m, s_ = np.array(a["R_T"][0], float) - 1, np.array(a["R_T"][1], float)
        ok = np.isfinite(m) & (x <= 3.06)
        ax[0].errorbar(x[ok], m[ok], s_[ok], color=col, lw=1.4, capsize=0, marker="o", ms=2.5, label=f"HS φ={p}")
    e = cm["mk1.80"]; a = e["at"]["tau7"]
    xs = np.array(e["s_centres"]); m = np.array(a["S_s"]["R_T"][0], float) - 1; ok = xs <= 3.06
    ax[0].plot(xs[ok], m[ok], color=ORANGE, ls="-", marker="s", ms=3, label="MK φ=1.8, shifted distance")
    xr = np.array(e["r_centres"]); m = np.array(a["S_r"]["R_T"][0], float) - 1
    ok = (np.array(a["S_r"]["cnt"][0], float) > 200) & (xr <= 3.06)
    ax[0].plot(xr[ok], m[ok], color=ORANGE, ls="--", lw=1.0, label="MK φ=1.8, real-space distance")
    ax[0].axhline(0, color=INK2, lw=0.6)
    ax[0].set_xlabel(r"initial separation $r/\sigma_{ij}$ (MK: $|x_i-x_j-a_{ij}|$ or $|x_i-x_j|$)")
    ax[0].set_ylabel(r"$R_\perp(r,\tau_7)-1$")
    ax[0].set_xlim(0.5, 3.1); ax[0].set_ylim(-0.35, 1.65); ax[0].legend(fontsize=6.5, loc="upper right", ncol=2)
    ax[0].set_title("(a) transverse factorization ratio vs distance", fontsize=9, loc="left")
    # (b) c_q(r) at tau7, log
    for p, col in zip(PHIS, RAMP):
        e = cm[KEYS[p]]; a = e["at"]["tau7"]["S_r"]; x = np.array(e["r_centres"])
        m, s = np.array(a["c_q"][0], float), np.array(a["c_q"][1], float)
        ok = np.isfinite(m) & (m > 0) & (x > 0.8)
        ax[1].errorbar(x[ok], m[ok], s[ok], color=col, lw=1.2, marker="o", ms=2.5, capsize=0, label=f"φ={p}")
        f = [q for q in fits[KEYS[p]]["tau7"] if q["binning"] == "S_r" and q["q"] == "c_q" and q["form"] == "exp" and q["window"] == [1.95, 4.95]]
        if f:
            A_, xi = f[0]["p"]; xx = np.linspace(1.95, 4.95, 50)
            ax[1].plot(xx, A_ * np.exp(-xx / xi), color=col, lw=0.8, ls=":")
    e = cm["mk1.80"]; a = e["at"]["tau7"]["S_s"]; xs = np.array(e["s_centres"]); m = np.array(a["c_q"][0], float)
    ok = m > 0
    ax[1].plot(xs[ok], m[ok], color=ORANGE, marker="s", ms=3, lw=1.0, ls="-", label="MK φ=1.8 (shifted)")
    ax[1].set_yscale("log"); ax[1].set_ylim(2e-3, 0.5); ax[1].set_xlim(0.6, 5.0)
    ax[1].set_xlabel("initial separation r (mean diameters)")
    ax[1].set_ylabel(r"mobility correlation $c_q(r,\tau_7)$")
    ax[1].legend(fontsize=7, loc="upper right")
    ax[1].set_title("(b) mobility correlation c_q (dotted: exp fits)", fontsize=9, loc="left")
    # (c) xi(phi)
    phis = np.array([float(p) for p in PHIS])
    for tn, mk, off, lab in (("tau7", "o", -0.0007, r"$\tau_7$"), ("tF045", "s", 0.0007, r"$F_s=0.45$")):
        for form, fill in (("exp", True), ("exp+c", False)):
            xi, se = [], []
            for p in PHIS:
                f = [q for q in fits[KEYS[p]][tn] if q["binning"] == "S_r" and q["q"] == "c_q" and q["form"] == form and q["window"] == [1.95, 4.95]]
                xi.append(f[0]["xi"] if f else np.nan); se.append(f[0].get("xi_jk_se", np.nan) if f else np.nan)
            xi, se = np.array(xi), np.array(se)
            ok = phis > 0.52              # phi = 0.50: amplitude too small for a fit
            ax[2].errorbar(phis[ok] + off + (0.0004 if form == "exp+c" else 0), xi[ok], se[ok], marker=mk, ls="-" if fill else "--",
                           color=RAMP[3] if tn == "tau7" else AQUA, mfc=(RAMP[3] if tn == "tau7" else AQUA) if fill else "white",
                           capsize=2, lw=1.0, label=f"{lab}, {form}")
    ctrl = [("phi0.585_N2000_d0.050", "N=2000", "D"), ("phi0.585_N1000_d0.025", "δ=0.025", "v"), ("phi0.585_N1000_d0.100", "δ=0.10", "^"),
            ("phi0.585_N1000_d-1.000", "EDMD", "P"), ("phi0.600_N1000_d-1.000", "EDMD", "P")]
    for k, lab, mk in ctrl:
        f = [q for q in fits[k]["tau7"] if q["binning"] == "S_r" and q["q"] == "c_q" and q["form"] == "exp" and q["window"] == [1.95, 4.95]]
        if f:
            x0 = float(k[3:8])
            ax[2].errorbar(x0 + 0.003, f[0]["xi"], f[0].get("xi_jk_se", np.nan), marker=mk, color=GREY, ls="none", capsize=2, ms=4,
                           label=f"{lab} (τ7, exp)" if not (lab == "EDMD" and x0 > 0.59) else None)
    ax[2].set_xlabel("φ"); ax[2].set_ylabel(r"$\xi$ (mean diameters)")
    ax[2].set_title(r"(c) ξ from c_q, fit window [1.95, 4.95]", fontsize=9, loc="left")
    ax[2].legend(fontsize=6.5, loc="upper left", ncol=1); ax[2].set_ylim(0, 2.2)
    fig.tight_layout(); save(fig, "fig1_comobility_length")


def fig_xi_t():
    cm = J("comob.json")
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for p, col in zip(PHIS[1:], RAMP[1:]):
        e = cm[KEYS[p]]; tau7 = e["times"]["tau7"]
        for i, qn in enumerate(("c_q", "CX_T")):
            rows = [r for r in e["xi_vs_t"][qn] if r["form"] == "exp" and "xi" in r]
            t = np.array([r["lag"] for r in rows]) / tau7
            xi = np.array([r["xi"] for r in rows]); se = np.array([r.get("xi_jk_se", np.nan) for r in rows])
            ok = (se < 0.5) & (xi < 5) & (xi > 0.2) & (t > 1e-3)     # early-time fits (amplitude ~ 0) are meaningless
            ax[i].errorbar(t[ok], xi[ok], se[ok], color=col, lw=1.0, marker="o", ms=2.5, capsize=0, label=f"φ={p}")
    for i, tt in enumerate((r"$c_q$ (orientation-averaged self-overlap)", r"$\mathrm{Cov}[\cos(k e_1\cdot\Delta x_i),\cos(k e_2\cdot\Delta x_j)]$")):
        ax[i].set_xscale("log"); ax[i].set_xlabel(r"$t/\tau_7$"); ax[i].set_ylabel(r"$\xi(t)$"); ax[i].set_ylim(0, 2.5)
        ax[i].set_title(f"({'ab'[i]}) " + tt, fontsize=8.5, loc="left"); ax[i].axvline(1, color=INK2, lw=0.6, ls=":")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); save(fig, "fig2_xi_vs_t")


def fig_variance():
    cm = J("comob.json")
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3))
    e = cm[KEYS["0.585"]]; a = e["at"]["tau7"]["S_r"]; x = np.array(e["r_centres"])
    for q, col, lab in (("Rc_T", INK, r"$R_\perp-R_{\rm marg,\perp}$ (connected)"), ("Ccc_T", RAMP[3], r"Cov$_{cc}$, same direction"),
                        ("CX_T", AQUA, r"Cov$_{cc}$, orthogonal directions"), ("Css_T", ORANGE, r"Cov$_{ss}$ (co-displacement)"),
                        ("Cqq", VIOLET, r"Cov$(q_i,q_j)$")):
        m, s = np.array(a[q][0], float), np.array(a[q][1], float); ok = np.isfinite(m)
        ax[0].errorbar(x[ok], m[ok], s[ok], color=col, lw=1.3, capsize=0, label=lab)
    ax[0].axhline(0, color=INK2, lw=0.6); ax[0].set_xlim(0.6, 5)
    ax[0].set_xlabel("initial separation r (mean diameters)"); ax[0].set_ylabel(r"term $/F_s^2$")
    ax[0].set_title(r"(a) HS φ=0.585, $\tau_7$: decomposition of $R_\perp(r)-1$", fontsize=9, loc="left")
    ax[0].legend(fontsize=7)
    # (b) contact decomposition HS vs MK at tau7
    labs, vals = [], []
    for k, lab in [(KEYS[p], f"HS {p}") for p in PHIS] + [("mk1.50", "MK 1.5"), ("mk1.80", "MK 1.8")]:
        c = cm[k]["at"]["tau7"]["S_s"]
        vals.append([c["Rm_T"][0][0] - 1, c["Ccc_T"][0][0], c["Css_T"][0][0]]); labs.append(lab)
    vals = np.array(vals); xx = np.arange(len(labs))
    w = 0.26
    for i, (col, lab) in enumerate(((GREY, r"$R_{\rm marg,\perp}-1$"), (RAMP[3], r"Cov$_{cc,\perp}/F_s^2$"), (ORANGE, r"Cov$_{ss,\perp}/F_s^2$"))):
        ax[1].bar(xx + (i - 1) * w, vals[:, i], w * 0.92, color=col, label=lab)
    tot = vals.sum(1)
    ax[1].plot(xx, tot, "k_", ms=12, mew=2, label=r"$R_\perp-1$ (sum)")
    ax[1].axhline(0, color=INK2, lw=0.6); ax[1].set_xticks(xx); ax[1].set_xticklabels(labs, rotation=30, fontsize=7.5)
    ax[1].set_title(r"(b) contact pairs at $\tau_7$: $R_\perp-1$ = marginal + cos·cos + sin·sin", fontsize=9, loc="left")
    ax[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); save(fig, "fig3_variance_decomposition")


def fig_triplets():
    tr, cm = J("triplets.json"), J("comob.json")
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.3))
    cb = np.linspace(-0.9, 0.9, 10)
    a = tr[KEYS["0.585"]]["at"]["tau7"]
    for cl, col, lab in ((0, RAMP[3], "closed"), (1, AQUA, "nb-closed"), (2, ORANGE, "open")):
        m = np.array(a["ang"]["Pb"][0][cl], float); n = np.array(a["ang"]["n"][0][cl], float)
        ok = n > 300
        ax[0].plot(cb[ok], m[ok], marker="o", ms=3, color=col, label=f"HS 0.585 {lab}")
    mk = tr["mk1.80"]["at"]["tau7"]
    ax[0].plot(cb, mk["ang_all"]["Pb"][0], marker="s", ms=3, color=VIOLET, label="MK 1.8 (all open)")
    ax[0].plot(cb, cb / 2, color=GREY, ls=":", lw=1, label=r"free particles: $\cos\theta/2$")
    ax[0].set_xlabel(r"bond angle at the centre, $\cos\theta_{jik}$"); ax[0].set_ylabel(r"Pearson$(b_{ij},b_{ik})$ at $\tau_7$")
    ax[0].legend(fontsize=6.5); ax[0].set_title("(a) raw bond–bond correlation", fontsize=9, loc="left")
    phis = np.array([float(p) for p in PHIS])
    for tn, ls, mkr in (("tstar", "--", "o"), ("tau7", "-", "s")):
        for cl, col, lab in ((0, RAMP[3], "closed"), (1, AQUA, "nb-closed"), (2, ORANGE, "open")):
            m = [tr[KEYS[p]]["at"][tn]["cls"]["dPb"][0][cl] for p in PHIS]; s = [tr[KEYS[p]]["at"][tn]["cls"]["dPb"][1][cl] for p in PHIS]
            ax[1].errorbar(phis, m, s, color=col, ls=ls, marker=mkr, ms=3, capsize=2, lw=1.0, label=f"{lab}, {'t*' if tn == 'tstar' else 'τ7'}")
    mkv = [tr[k]["at"]["tau7"]["all"]["dPb"][0] for k in ("mk1.50", "mk1.65", "mk1.72", "mk1.80")]
    ax[1].axhspan(min(mkv), max(mkv), color=VIOLET, alpha=0.2, label="MK 1.5–1.8, τ7")
    ax[1].axhline(0, color=INK2, lw=0.6); ax[1].set_xlabel("φ"); ax[1].set_ylabel(r"$\Delta P_b = P_b - P_b^{\rm tree}$")
    ax[1].legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2)); ax[1].set_title("(b) residual vs Gaussian-Markov tree", fontsize=9, loc="left")
    # (c) superposition test
    xs, ys, es = [], [], []
    for p, col in zip(PHIS, RAMP):
        for tn, mkr in (("tstar", "o"), ("tF045", "^"), ("tau7", "s")):
            c = cm[KEYS[p]]["at"][tn]["S_s"]; rp, rt = c["P_L"][0][0], c["P_T"][0][0]
            jd = -0.253 * rp + 0.747 * rt
            dj = tr[KEYS[p]]["at"][tn]["cls"]["dJ"]
            ax[2].errorbar(jd, dj[0][0], dj[1][0], color=col, marker=mkr, ms=4, capsize=2, ls="none")
            xs.append(jd); ys.append(dj[0][0])
    lim = [-0.16, 0.03]
    ax[2].plot(lim, lim, color=GREY, lw=0.8, ls=":")
    ax[2].set_xlabel(r"direct j–k pair term $-0.253\rho_\parallel+0.747\rho_\perp$")
    ax[2].set_ylabel(r"closed-triplet $\Delta J$ (outer–outer residual)")
    ax[2].set_title("(c) closed loops = superposed pairs", fontsize=9, loc="left")
    from matplotlib.lines import Line2D
    hand = [Line2D([], [], color=c, marker="o", ls="none", label=f"φ={p}") for p, c in zip(PHIS, RAMP)]
    hand += [Line2D([], [], color=GREY, marker=m, ls="none", label=l) for m, l in (("o", "t*"), ("^", "F_s=0.45"), ("s", "τ7"))]
    ax[2].legend(handles=hand, fontsize=6.5, ncol=2)
    # (d) co-breaking at tau7
    for cl, col, lab in ((0, RAMP[3], "closed"), (1, AQUA, "nb-closed"), (2, ORANGE, "open")):
        m = [tr[KEYS[p]]["at"]["tau7"]["cls"]["Kbr"][0][cl] for p in PHIS]; s = [tr[KEYS[p]]["at"]["tau7"]["cls"]["Kbr"][1][cl] for p in PHIS]
        ax[3].errorbar(phis, m, s, color=col, marker="o", ms=3, capsize=2, label=f"HS {lab}")
        ref = tr[KEYS["0.585"]]["mk_ref"]["mk1.80"]["tau7"]["Kbr"][cl]
        ax[3].axhline(ref, color=col, ls=":", lw=1)
    ax[3].plot([], [], color=GREY, ls=":", label="MK 1.8, angle-matched")
    ax[3].set_xlabel("φ"); ax[3].set_ylabel(r"$P(\rm both\ broken)/[P_{ij}P_{ik}]$ at $\tau_7$")
    ax[3].legend(fontsize=6.5); ax[3].set_title("(d) joint bond breaking (r>1.35σ)", fontsize=9, loc="left")
    fig.tight_layout(); save(fig, "fig4_triplets")


def fig_packing():
    pk = J("packing.json")
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    a = pk[KEYS["0.585"]]["at"]["tau7"]
    G = ["le3", "eq4", "ge5", "ge6"]; xl = ["≤3", "4", "≥5", "≥6"]; x = np.arange(4)
    def line(axx, m, pb, q, col, lab, ls="-"):
        mm = [a[m].get(f"{g}|{pb}", {}).get(q, [np.nan, np.nan]) for g in G]
        axx.errorbar(x, [v[0] for v in mm], [v[1] if v[1] is not None else np.nan for v in mm], color=col, marker="o", ms=3, capsize=2, ls=ls, label=lab)
    for pb, col in zip(range(3), [RAMP[1], RAMP[3], RAMP[4]]):
        line(ax[0], "radc3", pb, "R_L", col, f"packing tercile {pb + 1} (size-corrected radical)")
    line(ax[0], "radc3", "all", "R_L", INK, "all contacts", ls="--")
    ax[0].set_title(r"(a) $R_\parallel(\tau_7)$ at fixed local packing", fontsize=9, loc="left")
    q5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
    for pb in range(5):
        line(ax[1], "sig5", pb, "R_L", q5[pb], f"σ_ij quintile {pb + 1}")
    line(ax[1], "sig5", "all", "R_L", INK, "all contacts", ls="--")
    ax[1].set_title(r"(b) $R_\parallel(\tau_7)$ at fixed pair size $\sigma_{ij}$", fontsize=9, loc="left")
    for pb in range(5):
        line(ax[2], "sig5", pb, "ReG_L", q5[pb], f"σ_ij quintile {pb + 1}")
    line(ax[2], "sig5", "all", "ReG_L", INK, "all contacts", ls="--")
    ax[2].set_title(r"(c) Re$\Gamma_\parallel(\tau_7)$ at fixed pair size", fontsize=9, loc="left")
    for axx, yl in zip(ax, (r"$R_\parallel$", r"$R_\parallel$", r"Re$\Gamma_\parallel$")):
        axx.set_xticks(x); axx.set_xticklabels(xl); axx.set_xlabel(r"common neighbours $n_c$"); axx.set_ylabel(yl)
    ax[0].legend(fontsize=6.5, loc="upper left"); ax[0].set_ylim(1.5, 7.2)
    ax[1].legend(fontsize=6.5, loc="upper left", ncol=2); ax[1].set_ylim(0, 8.2)
    ax[2].legend(fontsize=6.5, loc="upper right", ncol=2); ax[2].set_ylim(1, 8.5)
    fig.suptitle("HS φ = 0.585, contact pairs (≥6 is a subset of ≥5)", fontsize=9, y=1.02)
    fig.tight_layout(); save(fig, "fig5_nc_vs_packing")


def fig_cna():
    cn = J("cna.json")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    phis = np.array([float(p) for p in PHIS])
    f = [cn[KEYS[p]]["frac1551"][0] for p in PHIS]; ico = [cn[KEYS[p]]["ico_per_1000"][0] / 1000 for p in PHIS]
    ax[0].plot(phis, f, marker="o", color=RAMP[3], label="1551 bonds / all bonds")
    ax[0].plot(phis, ico, marker="s", color=ORANGE, label="icosahedral centres / particles")
    ax[0].set_yscale("log"); ax[0].set_xlabel("φ"); ax[0].legend(fontsize=7); ax[0].set_title("(a) locally favoured structure", fontsize=9, loc="left")
    for p, col in zip(PHIS, RAMP):
        e = cn[KEYS[p]]; t = np.array(e["lags"]) / e["tau7"]
        ax[1].plot(t, e["pers_1551"][0], color=col, label=f"φ={p}")
        ax[1].plot(t, e["pers_bond"][0], color=col, ls=":", lw=1)
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"$t/\tau_7$"); ax[1].set_ylabel("endpoint persistence")
    ax[1].set_title("(b) 1551 bond still 1551 (solid); bond still bond (dotted)", fontsize=8.5, loc="left"); ax[1].legend(fontsize=7)
    ax[1].axvline(1, color=INK2, lw=0.6, ls=":")
    G = cn[KEYS["0.585"]]["groups"]
    for g, col, lab in (("n0", GREY, "no 1551 bond"), ("n1-3", RAMP[1], "1–3"), ("n4-7", RAMP[3], "4–7"), ("n8+", RAMP[4], "≥8")):
        gi = G.index(g)
        v = [cn[KEYS[p]]["at"]["tau7"]["q_over_all"][g] if cn[KEYS[p]]["frac_groups"][gi] > 0.005 else np.nan for p in PHIS]
        ax[2].plot(phis, v, marker="o", color=col, label=lab)
    ax[2].axhline(1, color=INK2, lw=0.6)
    ax[2].set_xlabel("φ"); ax[2].set_ylabel(r"$\langle q_i\rangle_{\rm group}/F_s$ at $\tau_7$  (>1: slower)")
    ax[2].legend(fontsize=7, title="1551 bonds per particle", title_fontsize=7); ax[2].set_title("(c) structure–mobility", fontsize=9, loc="left")
    fig.tight_layout(); save(fig, "fig6_cna")


if __name__ == "__main__":
    os.makedirs(C.FIGS, exist_ok=True)
    which = sys.argv[1:] or ["comob", "xi_t", "variance", "triplets", "packing", "cna"]
    for w in which:
        globals()[f"fig_{w}"]()
        print("made", w, flush=True)
