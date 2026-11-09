#!/usr/bin/env python3
"""Markdown tables (SM Table S14 and controls): HS (this study) next to MK (SM Tables S11-S12 values from
data/simulation/reanalysis/conditional_summary.json, read-only; Pearson from mk_pearson.py).
Usage: make_tables.py [--hs-results DIR] [--mk-conditional FILE] > tables.md
  defaults: ../../data/simulation/hs_bond and ../../data/simulation/reanalysis/conditional_summary.json."""
import json, os, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "simulation")   # ../../data/simulation
_ap = argparse.ArgumentParser(description="markdown tables, HS vs MK")
_ap.add_argument("--hs-results", default=os.path.join(REPO_DATA, "hs_bond"))
_ap.add_argument("--mk-conditional", default=os.path.join(REPO_DATA, "reanalysis", "conditional_summary.json"))
_a = _ap.parse_args()
T = json.load(open(os.path.join(_a.hs_results, "summary_table.json")))
MKC = json.load(open(_a.mk_conditional))
MKP = json.load(open(os.path.join(_a.hs_results, "mk_reanalysis_pearson.json")))


def f(v, d=2):
    if not v or v[0] is None:
        return "–"
    return f"{v[0]:.{d}f} ± {v[1]:.{d}f}" if v[1] is not None else f"{v[0]:.{d}f}"


def mk(phi, tn, key, src="C"):
    if src == "C":
        e = MKC[phi]["values"].get(tn, {}).get(key)
    else:
        e = MKP[{"2.0": "2"}.get(phi, phi)]["values"].get(tn, {}).get(key)
    return [e["mean"], e["sem"]] if e else None


cols = [("L-:R_glob", "R∥"), ("T-:R_glob", "R⊥"), ("Ipm:R_glob", "R orient.-avg"), ("rand-:R", "R random"),
        ("L-:R_marg", "R_marg∥"), ("L-:ReC", "Re C∥"), ("T-:ReC", "Re C⊥"), ("L-:Im_phi2", "Im φ₂∥"),
        ("L-:Im_prod", "Im(F^c_i F^c_j*)∥"), ("L:pearson", "corr∥"), ("T:pearson", "corr⊥"), ("end105", "P_c")]
lines = ["| system | φ | t | F_s(7,t) | " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * (len(cols) + 4)]
order = ["phi0.500_N1000_d0.050", "phi0.550_N1000_d0.050", "phi0.570_N1000_d0.050", "phi0.585_N1000_d0.050", "phi0.600_N1000_d0.050"]
for k in order:
    if k not in T:
        continue
    r = T[k]
    for tn, lab in (("tau7", "τ₇"), ("tstar", "t*"), ("tF045", "F_s=0.45")):
        v = r["at"].get(tn)
        if not v or v.get("t") is None:
            continue
        lines.append(f"| HS | {r['phi']:.3f} | {lab} = {v['t']:.3g} | {f(v['Fs7'])} | " + " | ".join(f(v.get(c)) for c, _ in cols) + " |")
mkkeys = {"L-:R_glob": ("L-:R_glob", "C"), "T-:R_glob": ("T-:R_glob", "C"), "Ipm:R_glob": ("Ipm:R_glob", "C"), "rand-:R": None,
          "L-:R_marg": ("L-:R_marg", "C"), "L-:ReC": ("L-:ReC", "C"), "T-:ReC": ("T-:ReC", "C"), "L-:Im_phi2": ("L-:Im_phi2", "C"),
          "L-:Im_prod": ("L-:Im_prod", "C"), "L:pearson": ("L:pearson", "P"), "T:pearson": ("T:pearson", "P"), "end105": ("end105", "C")}
for phi, tn, lab in (("0.8", "tau7", "τ₇"), ("1.5", "tau7", "τ₇"), ("1.8", "tau7", "τ₇"), ("2.0", "t1e4", "10⁴ (plateau)")):
    fs = mk(phi, tn, "Fs7", "P")
    row = []
    for c, _ in cols:
        m = mkkeys[c]
        row.append("1.00 ± 0.01 (SM)" if m is None else f(mk(phi, tn, m[0], m[1])))
    lines.append(f"| MK | {phi} | {lab} | {f(fs)} | " + " | ".join(row) + " |")
print("\n".join(lines))
print()
# controls
cl = ["| run | t | F_s(7,t) | R∥ | R⊥ | R orient.-avg | Re C∥ | Im φ₂∥ | corr∥ | corr⊥ | τ₇ | acc |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for k in ["phi0.585_N1000_d0.050", "phi0.585_N1000_d0.025", "phi0.585_N1000_d0.100", "phi0.585_N2000_d0.050", "phi0.585_N1000_d-1.000",
          "phi0.550_N1000_d0.050", "phi0.550_N1000_d-1.000", "phi0.600_N1000_d0.050", "phi0.600_N1000_d-1.000"]:
    if k not in T:
        continue
    r = T[k]
    for tn in ("tau7", "tstar", "tF045"):
        v = r["at"].get(tn)
        if not v or v.get("t") is None:
            continue
        cl.append(f"| {k} | {tn} = {v['t']:.3g} | {f(v['Fs7'])} | {f(v['L-:R_glob'])} | {f(v['T-:R_glob'])} | {f(v['Ipm:R_glob'])} | {f(v['L-:ReC'])} | "
                  f"{f(v['L-:Im_phi2'])} | {f(v['L:pearson'])} | {f(v['T:pearson'])} | {r['tau7']:.4g} | {r['acc']:.3f} |")
print("\n".join(cl))
print()
sh = ["| φ | t | contact (<1.05) R∥ / corr∥ | rest of 1st shell (1.05–1.3) R∥ / corr∥ | gap (1.3–1.5) R∥ / corr∥ | 2nd shell (1.6–2.1) R∥ / R⊥ / corr∥ | random R∥(own r̂) / corr∥ |",
      "|---|---|---|---|---|---|---|"]
for k in order:
    if k not in T:
        continue
    r = T[k]
    for tn in ("tau7", "tstar"):
        v = r["at"][tn]
        sh.append(f"| {r['phi']:.3f} | {tn} | {f(v['L-:R_glob'])} / {f(v['L:pearson'])} | {f(v['nL-:R_glob'])} / {f(v['nL:pearson'])} | "
                  f"{f(v['gL-:R_glob'])} / {f(v['gL:pearson'])} | {f(v['sL-:R_glob'])} / {f(v['sT-:R_glob'])} / {f(v['sL:pearson'])} | "
                  f"{f(v['randL-:R'])} / {f(v['rL:pearson'])} |")
print("\n".join(sh))
print()
nc = ["| φ | t | frac n_c≤3 / =4 / ≥5 | R∥ (≤3 / =4 / ≥5 / ≥6) | Re C∥ (≤3 / =4 / ≥5 / ≥6) | corr∥ (≤3 / =4 / ≥5 / ≥6) |", "|---|---|---|---|---|---|"]
for k in order:
    if k not in T:
        continue
    r = T[k]; fr = r["n_common_groups"]["frac"]
    for tn in ("tau7", "tstar"):
        v = r["at"][tn]
        g = lambda q: " / ".join(f(v.get(f"nc_{x}:{q}")) for x in ("few", "nc4", "many", "ge6"))
        nc.append(f"| {r['phi']:.3f} | {tn} | {fr['few']:.2f} / {fr['nc4']:.2f} / {fr['many']:.2f} | {g('R_glob')} | {g('ReC')} | {g('pearson')} |")
print("\n".join(nc))
print()
kk = ["| φ | t | R∥(k≈5) | R∥(k≈7) | R∥(k≈10) | Re C∥(k≈5) | Re C∥(k≈7) | Re C∥(k≈10) |", "|---|---|---|---|---|---|---|---|"]
for k in order:
    if k not in T:
        continue
    r = T[k]
    for tn in ("tau7", "tstar"):
        v = r["at"][tn]
        kk.append(f"| {r['phi']:.3f} | {tn} | {f(v['L5-:R_glob'])} | {f(v['L-:R_glob'])} | {f(v['L10-:R_glob'])} | {f(v['L5-:ReC'])} | {f(v['L-:ReC'])} | {f(v['L10-:ReC'])} |")
print("\n".join(kk))
