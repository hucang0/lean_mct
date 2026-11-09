#!/usr/bin/env python3
"""Re-analysis of the stored planted-MC trajectories (no new simulation).

R1  Bond-conditioned single-particle marginals. For pairs in contact at the origin (|r_ij| < 1.05, shifted
    metric) and k along the bond (k = k7 rhat_ij, rhat = unit vector of x_i - x_j - A_ij), transverse to it, and
    direction-averaged over the lattice k7 set:
        phi2 = < e^{i k.(dx_i - dx_j)} >                (relative motion, '-')  and  < e^{i k.(dx_i + dx_j)} > ('+')
        Fi   = < e^{i k.dx_i} >_contact,   Fj = < e^{i k.dx_j} >_contact          (complex)
    Factorized forms: '-' -> Fi conj(Fj),  '+' -> Fi Fj.
    Ratios per sample:  R_glob = Re phi2 / Fs^2   (the published R, global isotropic Fs at k7)
                        R_marg = Re(Fi Fj*) / Fs^2   (what the bond-conditioned marginals alone give)
                        C      = phi2 / (Fi Fj*)     (connected part, complex)
R2  Contact persistence: fraction of initially contacting pairs within 1.05 (and 1.2) at EVERY saved frame up to
    the lag (frames are log-spaced, so this is an upper bound on unbroken survival), next to the endpoint
    probability P(|r_ij(t)| < 1.05 | contact at 0) of the published analysis.
Helpers are imported from ../analysis.py so contacts, k sets, binning and time origins are identical.
Writes only into reanalysis/results/.
"""
import glob, json, os, sys
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
sys.path.insert(0, SIM)
import analysis as A  # noqa: E402  (our own analysis module)

OUT = os.path.join(HERE, "results")


def process(path):
    tag = os.path.basename(path).replace("run_", "").replace(".npz", "")
    cache = os.path.join(OUT, f"cond_{tag}.npz")
    if os.path.exists(cache):
        return cache
    d = np.load(path)
    traj = d["traj"].astype(np.float64)
    t_save = d["t_save"].astype(np.int64)
    Ash = d["A"].astype(np.float64)
    L, N, phi = float(d["L"]), int(d["N"]), float(d["phi"])
    origins = d["origins"].astype(np.int64)
    origins = origins[origins < t_save[-1]]
    K7, k7 = A.lattice_kvecs(A.K_FACT, L)
    T = int(d["sweeps"])
    nb = int(np.ceil(A.PER_DECADE * np.log10(T))) + 2
    acc = A.Accum(nb)
    for t0 in origins:
        i0 = int(np.searchsorted(t_save, t0))
        x0 = traj[i0]
        pairs, rhat, rmin, rvec = A.find_contacts(x0, Ash, L)
        ci, cj = pairs[:, 0], pairs[:, 1]
        kpar = k7 * rhat
        tmp = np.cross(rhat, np.where(np.abs(rhat[:, [0]]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
        kperp = k7 * tmp / np.linalg.norm(tmp, axis=1)[:, None]
        alive105 = np.ones(len(ci), bool)
        alive12 = np.ones(len(ci), bool)
        for f in range(i0 + 1, len(t_save)):
            lag = int(t_save[f] - t0)
            b = int(np.rint(A.PER_DECADE * np.log10(lag)))
            dx = traj[f] - x0
            dx = dx - dx.mean(0)                                  # centre-of-mass subtraction, as published
            v = {"Fs7": np.cos(dx @ K7.T).mean()}
            for lab, kk in (("L", kpar), ("T", kperp)):
                ai, aj = (dx[ci] * kk).sum(1), (dx[cj] * kk).sum(1)
                pm, pp = np.exp(1j * (ai - aj)).mean(), np.exp(1j * (ai + aj)).mean()
                Fi, Fj = np.exp(1j * ai).mean(), np.exp(1j * aj).mean()
                v.update({f"{lab}_pm_re": pm.real, f"{lab}_pm_im": pm.imag, f"{lab}_pp_re": pp.real,
                          f"{lab}_pp_im": pp.imag, f"{lab}_Fi_re": Fi.real, f"{lab}_Fi_im": Fi.imag,
                          f"{lab}_Fj_re": Fj.real, f"{lab}_Fj_im": Fj.imag})
            # direction-averaged lattice set: per k vector, then averaged over k vectors
            Pi, Pj = dx[ci] @ K7.T, dx[cj] @ K7.T                   # (P, nv)
            Fi_k, Fj_k = np.exp(1j * Pi).mean(0), np.exp(1j * Pj).mean(0)
            v["I_pm_re"] = np.cos(Pi - Pj).mean()
            v["I_pp_re"] = np.cos(Pi + Pj).mean()
            v["I_prod_pm_re"] = (Fi_k * np.conj(Fj_k)).real.mean()
            v["I_prod_pp_re"] = (Fi_k * Fj_k).real.mean()
            v["I_Fc_re"] = 0.5 * (Fi_k.real.mean() + Fj_k.real.mean())   # conditional isotropic marginal
            rt = np.linalg.norm(rvec + dx[ci] - dx[cj], axis=1)
            alive105 &= rt < 1.05
            alive12 &= rt < 1.2
            v.update({"end105": (rt < 1.05).mean(), "end12": (rt < 1.2).mean(),
                      "pers105": alive105.mean(), "pers12": alive12.mean()})
            acc.add(b, lag, **v)
    res = acc.result()
    np.savez(cache, tag=tag, phi=phi, N=N, L=L, k7=k7, sweeps=T, **res)
    return cache


def per_sample_ratio(rs, fn, den_min=0.15):
    """fn(r, j) -> value for sample r at bin j (or None). Mean and s.e.m. over samples, binned by log lag."""
    bins = {}
    for r in rs:
        for j, lag in enumerate(r["lag"]):
            if r["Fs7"][j] <= den_min:
                continue
            val = fn(r, j)
            if val is None:
                continue
            b = int(np.rint(A.PER_DECADE * np.log10(lag)))
            bins.setdefault(b, []).append((lag, val))
    out = []
    for b in sorted(bins):
        arr = np.array(bins[b])
        n = len(arr)
        out.append((float(np.exp(np.log(arr[:, 0]).mean())), float(arr[:, 1].mean()),
                    float(arr[:, 1].std(ddof=1) / np.sqrt(n)) if n > 1 else None, n))
    return out


def at(curve, t):
    if not curve:
        return None
    lags = np.array([c[0] for c in curve])
    if lags.max() < 0.9 * t or lags.min() > 1.1 * t:
        return None
    j = int(np.argmin(np.abs(np.log(lags) - np.log(t))))
    return {"t": curve[j][0], "mean": curve[j][1], "sem": curve[j][2], "n": curve[j][3]}


def cplx(r, lab, part, j):
    return r[f"{lab}_{part}_re"][j] + 1j * r[f"{lab}_{part}_im"][j]


def summarize(files):
    runs = [dict(np.load(f)) for f in files]
    groups = {}
    for r in runs:
        groups.setdefault((round(float(r["phi"]), 3), int(r["N"])), []).append(r)
    taus = json.load(open(os.path.join(SIM, "results/tau_table.json")))["tau"]
    out = {}
    for (phi, N), rs in sorted(groups.items()):
        key = f"{phi}" if N == 1000 else f"{phi}_N{N}"
        tau7 = taus.get(key, {}).get("tau_k7")
        times = {"tau7": tau7, "t100": 1e2, "t1e4": 1e4}
        obs = {}
        for lab in ("L", "T"):
            Fs2 = lambda r, j: r["Fs7"][j] ** 2
            # '-' : relative motion, factorized form Fi conj(Fj)
            obs[f"{lab}-:R_glob"] = per_sample_ratio(rs, lambda r, j, l=lab: r[f"{l}_pm_re"][j] / Fs2(r, j))
            obs[f"{lab}-:R_marg"] = per_sample_ratio(
                rs, lambda r, j, l=lab: (cplx(r, l, "Fi", j) * np.conj(cplx(r, l, "Fj", j))).real / Fs2(r, j))
            obs[f"{lab}-:ReC"] = per_sample_ratio(
                rs, lambda r, j, l=lab: (cplx(r, l, "pm", j) / (cplx(r, l, "Fi", j) * np.conj(cplx(r, l, "Fj", j)))).real)
            obs[f"{lab}-:ImC"] = per_sample_ratio(
                rs, lambda r, j, l=lab: (cplx(r, l, "pm", j) / (cplx(r, l, "Fi", j) * np.conj(cplx(r, l, "Fj", j)))).imag)
            obs[f"{lab}-:Im_phi2"] = per_sample_ratio(rs, lambda r, j, l=lab: r[f"{l}_pm_im"][j])
            obs[f"{lab}-:Im_prod"] = per_sample_ratio(
                rs, lambda r, j, l=lab: (cplx(r, l, "Fi", j) * np.conj(cplx(r, l, "Fj", j))).imag)
            obs[f"{lab}:Fi"] = per_sample_ratio(rs, lambda r, j, l=lab: r[f"{l}_Fi_re"][j])
            obs[f"{lab}:Fi_im"] = per_sample_ratio(rs, lambda r, j, l=lab: r[f"{l}_Fi_im"][j])
            obs[f"{lab}:Fj_im"] = per_sample_ratio(rs, lambda r, j, l=lab: r[f"{l}_Fj_im"][j])
            # '+' : pair centre of mass, factorized form Fi Fj
            obs[f"{lab}+:R_glob"] = per_sample_ratio(rs, lambda r, j, l=lab: r[f"{l}_pp_re"][j] / Fs2(r, j))
            obs[f"{lab}+:R_marg"] = per_sample_ratio(
                rs, lambda r, j, l=lab: (cplx(r, l, "Fi", j) * cplx(r, l, "Fj", j)).real / Fs2(r, j))
            obs[f"{lab}+:ReC"] = per_sample_ratio(
                rs, lambda r, j, l=lab: (cplx(r, l, "pp", j) / (cplx(r, l, "Fi", j) * cplx(r, l, "Fj", j))).real)
        for s in ("pm", "pp"):
            obs[f"I{s}:R_glob"] = per_sample_ratio(rs, lambda r, j, s=s: r[f"I_{s}_re"][j] / r["Fs7"][j] ** 2)
            obs[f"I{s}:R_marg"] = per_sample_ratio(rs, lambda r, j, s=s: r[f"I_prod_{s}_re"][j] / r["Fs7"][j] ** 2)
            obs[f"I{s}:C"] = per_sample_ratio(rs, lambda r, j, s=s: r[f"I_{s}_re"][j] / r[f"I_prod_{s}_re"][j])
        obs["I:Fc_over_Fs"] = per_sample_ratio(rs, lambda r, j: r["I_Fc_re"][j] / r["Fs7"][j])
        for q in ("end105", "pers105", "end12", "pers12"):
            obs[q] = per_sample_ratio(rs, lambda r, j, q=q: r[q][j], den_min=-1.0)
        ent = {"n_samples": len(rs), "tau7": tau7, "values": {}}
        for tn, tv in times.items():
            if tv is None:
                continue
            ent["values"][tn] = {k: at(c, tv) for k, c in obs.items()}
        # persistence/endpoint also at the end of the run
        ent["values"]["tmax"] = {q: (lambda c: {"t": c[-1][0], "mean": c[-1][1], "sem": c[-1][2]})(obs[q])
                                 for q in ("end105", "pers105", "end12", "pers12") if obs[q]}
        ent["curves"] = {k: [[c[0], c[1], c[2]] for c in v] for k, v in obs.items()}
        out[key] = ent
    with open(os.path.join(OUT, "conditional_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SIM, "data", "run_*.npz")))
    with Pool(8) as p:
        caches = p.map(process, files)
    out = summarize(caches)
    for key, ent in out.items():
        for tn, vals in ent["values"].items():
            if tn == "tmax":
                continue
            def g(k):
                v = vals.get(k)
                return f"{v['mean']:.3f}" if v else "  -  "
            print(f"phi={key:8s} {tn:5s}  L-: Rglob {g('L-:R_glob')} Rmarg {g('L-:R_marg')} ReC {g('L-:ReC')} "
                  f"ImC {g('L-:ImC')} Imphi2 {g('L-:Im_phi2')} Improd {g('L-:Im_prod')} | "
                  f"T-: Rglob {g('T-:R_glob')} Rmarg {g('T-:R_marg')} ReC {g('T-:ReC')} | "
                  f"I-: Rglob {g('Ipm:R_glob')} C {g('Ipm:C')} | pers105 {g('pers105')} end105 {g('end105')}")


if __name__ == "__main__":
    main()
