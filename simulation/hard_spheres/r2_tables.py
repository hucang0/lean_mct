#!/usr/bin/env python3
"""Markdown tables + key numbers from results/*.json -> results/tables_r2.md, results/key_numbers.json."""
import os, sys, json
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r2_common as C

PHIS = ["0.500", "0.550", "0.570", "0.585", "0.600"]
K = {p: f"phi{p}_N1000_d0.050" for p in PHIS}
CTRL = [("phi0.585_N1000_d0.050", "MC δ=0.05 (6)"), ("phi0.585_N1000_d0.025", "MC δ=0.025 (3)"), ("phi0.585_N1000_d0.100", "MC δ=0.10 (3)"),
        ("phi0.585_N2000_d0.050", "MC N=2000 (3)"), ("phi0.585_N1000_d-1.000", "EDMD (6)")]
J = lambda n: json.load(open(C.result_path(n)))
cm, fits, tr, pk, cn, times = J("comob.json"), J("xi_fits.json"), J("triplets.json"), J("packing.json"), J("cna.json"), J("times.json")
L = []; KN = {}


def f(v, s=None, d=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if s is None or (isinstance(s, float) and not np.isfinite(s)):
        return f"{v:.{d}f}"
    return f"{v:.{d}f} ± {s:.{d}f}"


def cval(k, tn, which, q, i=0):
    a = cm[k]["at"].get(tn, {}).get(which)
    if not a:
        return None, None
    m, s = a[q][0][i], a[q][1][i]
    return m, s


def xi(k, tn, q="c_q", form="exp", win=(1.95, 4.95), binning="S_r"):
    e = [x for x in fits.get(k, {}).get(tn, []) if x["binning"] == binning and x["q"] == q and x["form"] == form and tuple(x["window"]) == tuple(win)]
    return (e[0]["xi"], e[0].get("xi_jk_se"), e[0]["chi2_dof"]) if e else (None, None, None)


def rbin(k, r):
    x = np.array(cm[k]["r_centres"]); return int(np.argmin(np.abs(x - r)))


# ---------------------------------------------------------------- (i) contact decomposition
L.append("## (i) Co-mobility\n\n### T1. Contact pairs (r < 1.05 σ_ij): decomposition of R⊥ − 1 at τ₇ and F_s = 0.45\n")
L.append("R⊥ − 1 = (R_marg⊥ − 1) + Cov_cc⊥/F_s² + Cov_ss⊥/F_s² (identity). CX⊥ = cos·cos covariance between ORTHOGONAL transverse "
         "directions (no linear co-displacement can contribute; pure shared mobility). Cov_qq = covariance of orientation-averaged self overlaps. "
         "P⊥ = Pearson of transverse displacements. ⊥ = average over the two transverse directions. mean ± s.e.m. over samples.\n")
L.append("| system | t | R⊥ | R_marg⊥ − 1 | Cov_cc⊥/F_s² | Cov_ss⊥/F_s² | CX⊥/F_s² | Cov_qq/F_s² | Re C⊥ | P⊥ | R_iso | detour slope (rel.) |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
rows = [(K[p], f"HS {p}") for p in PHIS] + [(k, n) for k, n in CTRL[1:]] + [("phi0.600_N1000_d-1.000", "HS 0.600 EDMD (3)")] + \
       [(f"mk{p}", f"MK {p}") for p in ("0.80", "1.50", "1.80", "2.00")]
KN["contact"] = {}
for k, nm in rows:
    for tn in ("tau7", "tF045", "t1e4"):
        if tn not in cm[k]["at"]:
            continue
        g = lambda q: cval(k, tn, "S_s", q)
        Rm = g("Rm_T")
        L.append(f"| {nm} | {tn} | {f(*g('R_T'))} | {f(Rm[0] - 1, Rm[1])} | {f(*g('Ccc_T'))} | {f(*g('Css_T'))} | {f(*g('CX_T'))} | "
                 f"{f(*g('Cqq'))} | {f(*g('ReC_T'))} | {f(*g('P_T'), d=3)} | {f(*g('R_I'))} | {f(g('detour_rel')[0])} |")
        KN["contact"][f"{nm}|{tn}"] = {q: g(q) for q in ("R_T", "Rm_T", "Ccc_T", "Css_T", "CX_T", "Cqq", "ReC_T", "P_T", "R_I", "detour_rel")}
L.append("")
# ---------------------------------------------------------------- (i) distance dependence (absolute r)
L.append("### T2. Beyond contact (absolute-r bins, 0.1 wide, centre given): mobility correlation and connected R⊥\n")
L.append("c_q = Pearson(q_i, q_j); CX⊥/F_s² as in T1; R⊥ − R_marg⊥ = connected transverse factor. τ₇.\n")
L.append("| system | c_q(2.05) | c_q(2.55) | c_q(3.05) | c_q(3.55) | c_q(4.05) | CX⊥(2.05) | CX⊥(3.05) | (R⊥−R_marg⊥)(2.05) | (R⊥−R_marg⊥)(3.05) | R⊥(2.05) − 1 | R⊥(3.05) − 1 |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
KN["beyond"] = {}
for k, nm in [(K[p], f"HS {p}") for p in PHIS] + CTRL[1:] + [("phi0.600_N1000_d-1.000", "HS 0.600 EDMD (3)")]:
    a = cm[k]["at"]["tau7"]["S_r"]
    g = lambda q, r: (a[q][0][rbin(k, r)], a[q][1][rbin(k, r)])
    L.append(f"| {nm} | " + " | ".join(f(*g("c_q", r), d=3) for r in (2.05, 2.55, 3.05, 3.55, 4.05)) + " | " +
             " | ".join(f(*g("CX_T", r), d=3) for r in (2.05, 3.05)) + " | " + " | ".join(f(*g("Rc_T", r), d=3) for r in (2.05, 3.05)) + " | " +
             " | ".join(f(g("R_T", r)[0] - 1, g("R_T", r)[1], d=3) for r in (2.05, 3.05)) + " |")
    KN["beyond"][nm] = {f"c_q@{r}": g("c_q", r) for r in (2.05, 3.05, 4.05)} | {f"CX_T@{r}": g("CX_T", r) for r in (2.05, 3.05)}
L.append("")
# MK flatness
L.append("### T3. MK control (τ₇; φ = 2.0 at t = 10⁴): R − 1 beyond contact, shifted-frame vs real-space distance\n")
L.append("| MK φ | R⊥(contact) | ⟨R⊥ − 1⟩ shifted 1.45–3.05 | ⟨R_iso − 1⟩ shifted 1.45–3.05 | ⟨c_q⟩ shifted 1.45–3.05 | ⟨R⊥ − 1⟩ real 0.5–3.05 | max abs(R⊥ − 1) real | ⟨c_q⟩ real |")
L.append("|---|---|---|---|---|---|---|---|")
KN["mk_flat"] = {}
for p in ("0.80", "1.20", "1.50", "1.58", "1.65", "1.72", "1.80", "2.00"):
    k = f"mk{p}"; tn = "tau7" if "tau7" in cm[k]["at"] else "t1e4"
    a = cm[k]["at"][tn]; xs = np.array(cm[k]["s_centres"]); xr = np.array(cm[k]["r_centres"])
    ms = (xs > 1.45) & (xs < 3.06); mr = (xr > 0.5) & (xr < 3.06) & (np.array(a["S_r"]["cnt"][0], float) > 100)
    A = lambda w, q, m: np.array(a[w][q][0], float)[m]
    vals = [a["S_s"]["R_T"][0][0], np.nanmean(A("S_s", "R_T", ms)) - 1, np.nanmean(A("S_s", "R_I", ms)) - 1, np.nanmean(A("S_s", "c_q", ms)),
            np.nanmean(A("S_r", "R_T", mr)) - 1, np.nanmax(np.abs(A("S_r", "R_T", mr) - 1)), np.nanmean(A("S_r", "c_q", mr))]
    KN["mk_flat"][p] = vals
    L.append(f"| {p} ({tn}) | {vals[0]:.3f} | {vals[1]:+.4f} | {vals[2]:+.4f} | {vals[3]:+.4f} | {vals[4]:+.4f} | {vals[5]:.3f} | {vals[6]:+.4f} |")
L.append("")
# xi table
L.append("### T4. Decay length ξ (mean diameters) of the co-mobility beyond the first shell\n")
L.append("Fit of the sample-mean curve (weights = s.e.m.), absolute-r bins; ± = jackknife over samples; (χ²/dof). "
         "Primary: c_q(r) = A e^{−r/ξ}, window 1.95 ≤ r ≤ 4.95 (beyond the 2nd-shell peak). Variants: + constant; window 1.45–4.95 / 1.45–3.95; "
         "CX⊥ (orthogonal-direction cos·cos covariance); (R_iso − R_marg,iso) connected isotropic factor. φ = 0.50 omitted (c_q < 0.012 beyond r = 2; fit undetermined).\n")
L.append("| run | t | c_q exp [1.95,4.95] | c_q exp+c [1.95,4.95] | c_q exp [1.45,4.95] | c_q exp [1.45,3.95] | CX⊥ exp [1.95,4.95] | (R−R_m)_iso exp [1.95,4.95] | c_m2 exp [1.95,4.95] |")
L.append("|---|---|---|---|---|---|---|---|---|")
KN["xi"] = {}
for k, nm in [(K[p], f"HS {p}") for p in PHIS[1:]] + CTRL[1:] + [("phi0.600_N1000_d-1.000", "HS 0.600 EDMD (3)"), ("phi0.550_N1000_d-1.000", "HS 0.550 EDMD (3)")]:
    for tn in ("tau7", "tF045"):
        cells = []
        for q, form, win in (("c_q", "exp", (1.95, 4.95)), ("c_q", "exp+c", (1.95, 4.95)), ("c_q", "exp", (1.45, 4.95)), ("c_q", "exp", (1.45, 3.95)),
                             ("CX_T", "exp", (1.95, 4.95)), ("Rc_I", "exp", (1.95, 4.95)), ("c_m2", "exp", (1.95, 4.95))):
            v, s, c2 = xi(k, tn, q, form, win)
            cells.append(f"{f(v, s)} ({c2:.1f})" if v is not None else "–")
            KN["xi"][f"{nm}|{tn}|{q}|{form}|{win}"] = [v, s, c2]
        L.append(f"| {nm} | {tn} | " + " | ".join(cells) + " |")
L.append("")
# random pairs
L.append("### T5. Random-pair sanity check (4000 random pairs per origin, k along their own separation for ∥/⊥), τ₇\n")
L.append("R_pf: Re φ₂ normalised by the per-frame Fs² (removes the origin-to-origin global mobility fluctuation).\n")
L.append("| system | R_iso | R_iso,pf | R⊥ | R⊥,pf | R∥(own r̂) | Cov_ss∥/F_s² | Cov_ss⊥/F_s² | c_q |")
L.append("|---|---|---|---|---|---|---|---|---|")
KN["random"] = {}
for k, nm in [(K[p], f"HS {p}") for p in PHIS] + CTRL[1:] + [("mk1.50", "MK 1.5"), ("mk1.80", "MK 1.8")]:
    g = lambda q: cval(k, "tau7", "S_rand", q)
    L.append(f"| {nm} | {f(*g('R_I'), d=3)} | {f(*g('Rpf_I'), d=3)} | {f(*g('R_T'), d=3)} | {f(*g('Rpf_T'), d=3)} | {f(*g('R_L'), d=3)} | "
             f"{f(*g('Css_L'), d=3)} | {f(*g('Css_T'), d=3)} | {f(*g('c_q'), d=4)} |")
    KN["random"][nm] = {q: g(q) for q in ("R_I", "Rpf_I", "R_T", "Rpf_T", "R_L", "Css_L", "Css_T", "c_q")}
L.append("")
# ---------------------------------------------------------------- (ii) triplets
L.append("## (ii) Loop channel: contact triplets\n")
L.append("### T6. Triplet classes (pooled over origins): fractions and mean bond angle\n")
L.append("| system | triplets / config | closed (s_jk<1.05) | nb-closed (1.05–1.35) | open (≥1.35) | ⟨cos θ⟩ closed / nb / open |")
L.append("|---|---|---|---|---|---|")
for k, nm in [(K[p], f"HS {p}") for p in PHIS] + [(f"mk{p}", f"MK {p}") for p in ("0.80", "1.50", "1.80", "2.00")]:
    e = tr[k]; tn = "tau7" if "tau7" in e["at"] else "t1e4"
    cs = e["at"][tn]["cls"]["cos"][0]
    L.append(f"| {nm} | {e['ntrip_per_config']:.0f} | {e['class_frac'][0]:.3f} | {e['class_frac'][1]:.3f} | {e['class_frac'][2]:.3f} | "
             + " / ".join(f(c) for c in cs) + " |")
L.append("")
L.append("### T7. Bond–bond correlation: raw, tree null, residual; outer–outer residual vs the direct j–k pair term; co-breaking\n")
L.append("P_b = Pearson(b_ij, b_ik); P_b^tree = Gaussian-Markov tree prediction from the same triplets' measured self and pair covariances; "
         "ΔP_b = P_b − P_b^tree. ΔJ = (J − J_tree)/s_i² (outer–outer covariance residual). J_dir = −0.253 ρ∥ + 0.747 ρ⊥ = projection of an "
         "ordinary contact-pair covariance (ρ∥, ρ⊥ = contact-pair Pearson, this φ and t) onto the two bonds of an isosceles closed triplet "
         "(cos θ = 0.494). P_t = transverse analogue (normal to the triplet plane; free/tree value ≈ 0.5). K = P(both broken)/[P(ij)P(ik)], broken: r > 1.35σ.\n")
L.append("| system | t | P_b closed / nb / open | P_b^tree closed / nb / open | ΔP_b closed / nb / open | ΔJ closed | J_dir | ΔJ − J_dir | ΔP_t closed / open | K closed / nb / open |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
KN["trip"] = {}
for k, nm in [(K[p], f"HS {p}") for p in PHIS] + CTRL[1:] + [("phi0.600_N1000_d-1.000", "HS 0.600 EDMD (3)")]:
    for tn in ("tstar", "tF045", "tau7"):
        a = tr[k]["at"].get(tn)
        if not a:
            continue
        c = cm[k]["at"][tn]["S_s"]; rp, rt = c["P_L"][0][0], c["P_T"][0][0]; jd = -0.253 * rp + 0.747 * rt
        cl = a["cls"]
        t3 = lambda q, d=3: " / ".join(f(cl[q][0][i], cl[q][1][i], d) for i in range(3))
        L.append(f"| {nm} | {tn} | {t3('Pb')} | {t3('Pb_tree')} | {t3('dPb')} | {f(cl['dJ'][0][0], cl['dJ'][1][0], 3)} | {jd:.3f} | "
                 f"{cl['dJ'][0][0] - jd:+.3f} | {f(cl['dPt'][0][0], cl['dPt'][1][0], 3)} / {f(cl['dPt'][0][2], cl['dPt'][1][2], 3)} | {t3('Kbr', 2)} |")
        KN["trip"][f"{nm}|{tn}"] = {q: [cl[q][0], cl[q][1]] for q in ("Pb", "Pb_tree", "dPb", "dJ", "dPt", "Kbr")} | {"J_dir": jd}
for p in ("1.50", "1.65", "1.80", "2.00"):
    k = f"mk{p}"
    for tn in ("tF045", "tau7", "t1e4"):
        a = tr[k]["at"].get(tn)
        if not a:
            continue
        al = a["all"]
        L.append(f"| MK {p} (all ≈ open) | {tn} | – / – / {f(*al['Pb'], 3)} | – / – / {f(*al['Pb_tree'], 3)} | – / – / {f(*al['dPb'], 3)} | – | – | – | – / {f(*al['dPt'], 3)} | – / – / {f(*al['Kbr'], 2)} |")
        KN["trip"][f"MK {p}|{tn}"] = {q: al[q] for q in ("Pb", "Pb_tree", "dPb", "dJ", "dPt", "Kbr")}
L.append("")
L.append("MK angle-matched co-breaking reference (MK φ = 1.8, τ₇, reweighted to each HS class's angle histogram at φ = 0.585): "
         + " / ".join(f"{v:.2f}" for v in tr[K['0.585']]["mk_ref"]["mk1.80"]["tau7"]["Kbr"]) + " (closed / nb / open geometry). "
         "MK angle-matched ΔP_b: " + " / ".join(f"{v:+.3f}" for v in tr[K['0.585']]["mk_ref"]["mk1.80"]["tau7"]["dPb"]) + ".\n")
# ---------------------------------------------------------------- (iii) packing
L.append("## (iii) n_c versus local packing\n")
L.append("### T8. How n_c correlates with packing measures and with pair size (Spearman, all contact pairs, pooled)\n")
L.append("dens_rad = (V̂_i + V̂_j)/(V_i + V_j) with radical-Voronoi cell volumes V and V̂(σ) = cubic fit of ⟨V|σ⟩ (size-corrected); "
         "φ_rad = (v_i+v_j)/(V_i+V_j) (raw); probe = residual of the volume fraction in a sphere of radius 1.5 at the pair midpoint after a cubic fit in σ_ij.\n")
L.append("| φ | ⟨n_c⟩ | ρ_s(n_c, dens_rad) | ρ_s(n_c, dens_vor) | ρ_s(n_c, probe res.) | ρ_s(n_c, φ_rad raw) | ρ_s(n_c, σ_ij) | ρ_s(φ_rad, σ_ij) | ρ_s(dens_rad, σ_ij) |")
L.append("|---|---|---|---|---|---|---|---|---|")
for p in PHIS:
    s = pk[K[p]]["struct"]
    L.append(f"| {p} | {s['mean_nc']:.2f} | {s['radc3']['spearman_nc']:.3f} | {s['vorc3']['spearman_nc']:.3f} | {s['probec3']['spearman_nc']:.3f} | "
             f"{s['rad3']['spearman_nc']:.3f} | {s['spearman_nc_sij']:.3f} | {s['spearman_phirad_sij']:.3f} | {s['spearman_densrad_sij']:.3f} |")
L.append("")
G = ["le3", "eq4", "ge5", "ge6"]
KN["pack"] = {}
for p in ("0.550", "0.585", "0.600"):
    for tn in ("tstar", "tau7"):
        a = pk[K[p]]["at"][tn]
        L.append(f"### T9. φ = {p}, {tn}: contact-pair observables vs n_c (≤3 / 4 / ≥5 / ≥6) at fixed packing or size\n")
        L.append("| stratum | R∥ | Re Γ∥ | R_marg∥ | Pearson∥ | R⊥ |")
        L.append("|---|---|---|---|---|---|")
        for m, pbs, lab in (("radc3", ["all", 0, 1, 2], "size-corr. radical tercile"), ("probec3", [0, 1, 2], "probe-sphere tercile"),
                            ("sig5", [0, 1, 2, 3, 4], "σ_ij quintile"), ("rad3", [0, 1, 2], "raw radical φ tercile")):
            for pb in pbs:
                cells = []
                for q in ("R_L", "ReG_L", "Rm_L", "P_L", "R_T"):
                    vv = [a[m].get(f"{g}|{pb}", {}).get(q, [None, None]) for g in G]
                    cells.append(" / ".join(f(v[0], d=2) for v in vv))
                    KN["pack"][f"{p}|{tn}|{m}|{pb}|{q}"] = vv
                nm = "all contacts" if pb == "all" else f"{lab} {pb + 1}"
                L.append(f"| {nm} | " + " | ".join(cells) + " |")
        L.append("")
# s.e.m. example
a = pk[K["0.585"]]["at"]["tau7"]
L.append("Typical s.e.m. (φ = 0.585, τ₇, size-corrected radical tercile 2): R∥ " +
         " / ".join(f(*a["radc3"][f"{g}|1"]["R_L"]) for g in G) + "; Re Γ∥ " + " / ".join(f(*a["radc3"][f"{g}|1"]["ReG_L"]) for g in G) + ".\n")
L.append("### T10. Controls at φ = 0.585 (τ₇): R∥ vs n_c (≤3 / 4 / ≥5 / ≥6) in the middle size-corrected packing tercile and the middle σ_ij quintile\n")
L.append("| run | radc tercile 2: R∥ | radc tercile 2: Re Γ∥ | σ_ij quintile 3: R∥ | σ_ij quintile 3: Re Γ∥ |")
L.append("|---|---|---|---|---|")
for k, nm in CTRL:
    a = pk[k]["at"]["tau7"]
    cell = lambda m, pb, q: " / ".join(f(a[m].get(f"{g}|{pb}", {}).get(q, [None])[0]) for g in G)
    L.append(f"| {nm} | {cell('radc3', 1, 'R_L')} | {cell('radc3', 1, 'ReG_L')} | {cell('sig5', 2, 'R_L')} | {cell('sig5', 2, 'ReG_L')} |")
L.append("")
# ---------------------------------------------------------------- (iv) CNA
L.append("## (iv) Cluster level (common-neighbour analysis)\n")
L.append("| φ | 1551 bonds / bonds | ico centres per 1000 | largest 1551 cluster | frac. particles in 1551 clusters ≥13 | 1/e time of 1551 persistence | τ₇ | ratio | ⟨q⟩/F_s at τ₇: n₁₅₅₁=0 / 1–3 / 4–7 / ≥8 |")
L.append("|---|---|---|---|---|---|---|---|---|")
KN["cna"] = {}
for p in PHIS:
    e = cn[K[p]]; a = e["at"]["tau7"]; fr = e["frac_groups"]
    qo = " / ".join(f"{a['q_over_all'][g]:.2f}" if fr[i] > 0.005 else "–" for i, g in enumerate(["n0", "n1-3", "n4-7", "n8+"]))
    t1 = e["t_1e_1551"]
    L.append(f"| {p} | {f(*e['frac1551'], 3)} | {e['ico_per_1000'][0]:.2f} | {e['largest_1551_cluster'][0]:.0f} | {e['frac_in_1551_clusters_ge13'][0]:.2f} | "
             f"{t1:.3g} | {e['tau7']:.3g} | {t1 / e['tau7']:.2f} | {qo} |")
    KN["cna"][p] = {"frac1551": e["frac1551"], "ico_per_1000": e["ico_per_1000"], "largest": e["largest_1551_cluster"], "t1e_over_tau7": t1 / e["tau7"],
                    "q_over_all": a["q_over_all"]}
L.append("")
# times
L.append("## Time markers (recomputed; identical to hs_bond)\n")
L.append("| group | τ₇ | t* | t(F_s = 0.45) |")
L.append("|---|---|---|---|")
for k in sorted(times):
    t = times[k]
    L.append(f"| {k} | {t.get('tau7') or float('nan'):.4g} | {t.get('tstar') or float('nan'):.4g} | {t.get('tF045') or float('nan'):.4g} |")
os.makedirs(C.RES, exist_ok=True)
open(os.path.join(C.RES, "tables_r2.md"), "w").write("\n".join(L) + "\n")
C.jdump(KN, os.path.join(C.RES, "key_numbers.json"))
print("wrote tables_r2.md", len(L), "lines")
