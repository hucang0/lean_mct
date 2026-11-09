#!/usr/bin/env python3
"""Shared helpers for the R2 bond-resolved re-analysis.

Helpers are imported from vendored copies in ./lib: path-portable copies of ../mk_planted_mc/analysis.py and
conditional.py and of hs_analysis.py and hs_summary.py in this folder.
  A.lattice_kvecs, A.PER_DECADE, A.crossing_time, A.average_group  (lib/analysis.py)
  pairs_upto, perp_of, common_neighbours, bond_obs                   (lib/hs_analysis.py)
Data folders are read-only.
"""
import os, sys, glob, json
from pathlib import Path
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "lib")
if LIB not in sys.path:
    sys.path.insert(0, LIB)
import analysis as A                                   # noqa: E402  verbatim copy of orig_copies/analysis.py
from hs_analysis import pairs_upto, perp_of, common_neighbours, bond_obs, CONTACT, NB_CUT   # noqa: E402

ROOT = Path(HERE).parents[1]
MK = ROOT / "simulation" / "mk_planted_mc"
DATA = ROOT / "data" / "simulation" / "r2_bond"


def configured_path(name, default):
    return os.fspath(Path(os.environ.get(name, default)).expanduser().resolve())


HSD = configured_path("R2_HS_DATA", Path(HERE) / "data" / "dyn")
HSMD = configured_path("R2_HS_MD_DATA", Path(HERE) / "data" / "md")
MKD = configured_path("R2_MK_DATA", MK / "data")
MKTAU = configured_path("R2_MK_TAU", MK / "results" / "tau_table.json")
HSRES = configured_path("R2_HS_RESULTS", ROOT / "data" / "simulation" / "hs_bond")
CACHE = configured_path("R2_CACHE", Path(HERE) / "cache")
RES = configured_path("R2_RESULTS", Path(HERE) / "results")
FIGS = configured_path("R2_FIGURES", Path(HERE) / "figs")


def result_path(name):
    """Prefer regenerated results; otherwise read the published, immutable data."""
    local = Path(RES) / name
    if local.is_file():
        return local
    published = DATA / name
    if published.is_file():
        return published
    raise FileNotFoundError(f"Missing R2 result {name}: checked {local} and {published}")


def require_files(files, description):
    if not files:
        raise FileNotFoundError(
            f"No {description} found. See simulation/hard_spheres/README.md for "
            "trajectory generation and R2 path overrides."
        )
    Path(CACHE).mkdir(parents=True, exist_ok=True)
    return files
PD = A.PER_DECADE
BREAK = 1.35          # bond counted as broken when r(t) > 1.35 sigma_ij


def hs_files(kind="all"):
    main = sorted(glob.glob(f"{HSD}/run_phi*_N1000_d0.050_s*.npz"))
    ctrl = sorted(glob.glob(f"{HSD}/run_phi0.585_N1000_d0.025_s*.npz") + glob.glob(f"{HSD}/run_phi0.585_N1000_d0.100_s*.npz")
                  + glob.glob(f"{HSD}/run_phi0.585_N2000_d0.050_s*.npz"))
    md = sorted(glob.glob(f"{HSMD}/run_phi*_N1000_md_s*.npz"))
    main = [f for f in main if not f.endswith(".tmp.npz")]
    return {"main": main, "ctrl": ctrl, "md": md, "all": main + ctrl + md}[kind]


def mk_files(phis=("0.80", "1.20", "1.50", "1.58", "1.65", "1.72", "1.80", "2.00")):
    out = []
    for p in phis:
        out += sorted(glob.glob(f"{MKD}/run_phi{p}_N1000_s*.npz"))
    return out


def tag_of(path):
    return os.path.basename(path).replace("run_", "").replace(".npz", "")


def group_key(d):
    """HS group key identical to hs_summary: phi{phi:.3f}_N{N}_d{delta:.3f}; MK: mk{phi:.2f}."""
    if "A" in d.files:
        return f"mk{float(d['phi']):.2f}"
    return f"phi{float(d['phi']):.3f}_N{int(d['N'])}_d{float(d['delta']):.3f}"


def load_run(path):
    d = np.load(path)
    if bool(d.get("partial", False)):
        raise ValueError(f"Incomplete trajectory: {path}; wait for the production launcher to finish.")
    out = {"traj": d["traj"].astype(np.float64), "t_save": d["t_save"].astype(np.int64),
           "L": float(d["L"]), "N": int(d["N"]), "phi": float(d["phi"]), "T": int(d["sweeps"]),
           "origins": d["origins"].astype(np.int64), "key": group_key(d), "tag": tag_of(path)}
    out["origins"] = out["origins"][out["origins"] < out["t_save"][-1]]
    if "A" in d.files:
        out["A"] = d["A"].astype(np.float64); out["sig"] = np.ones(out["N"]); out["mk"] = True
    else:
        out["sig"] = d["sig"].astype(np.float64); out["mk"] = False
    out["delta"] = float(d["delta"])
    return out


def frames_of(run):
    """Yield (t0 index, frame index, lag, lag-bin) exactly as hs_analysis.process / analysis.process_file."""
    t_save = run["t_save"]
    for t0 in run["origins"]:
        i0 = int(np.searchsorted(t_save, t0))
        assert t_save[i0] == t0
        for f in range(i0 + 1, len(t_save)):
            lag = int(t_save[f] - t0)
            yield i0, f, lag, int(np.rint(PD * np.log10(lag)))


def n_lagbins(run):
    return int(np.ceil(PD * np.log10(run["T"]))) + 2


class LagAcc:
    """Sums of per-frame arrays binned in log lag (like analysis.Accum but array-valued, summed not averaged)."""

    def __init__(self, nb):
        self.nb = nb; self.cnt = np.zeros(nb); self.loglag = np.zeros(nb); self.S = {}

    def add(self, b, lag, **arrs):
        self.cnt[b] += 1; self.loglag[b] += np.log(lag)
        for k, v in arrs.items():
            v = np.asarray(v, dtype=np.float64)
            if k not in self.S:
                self.S[k] = np.zeros((self.nb,) + v.shape)
            self.S[k][b] += v

    def result(self):
        ok = self.cnt > 0
        out = {"lag": np.exp(self.loglag[ok] / self.cnt[ok]), "nframes": self.cnt[ok]}
        for k, v in self.S.items():
            out[k] = v[ok]
        return out


def wrap(x, L):
    xw = x - L * np.floor(x / L)
    return np.clip(xw, 0, np.nextafter(L, 0))


def mk_shifted_pairs(x, Ash, L, rmax):
    """All pairs i<j with shifted min-image distance < rmax: (ci, cj, dvec, r), dvec = x_i - x_j - A_ij (min image)."""
    N = len(x)
    iu = np.triu_indices(N, 1)
    d = x[iu[0]] - x[iu[1]] - Ash[iu[0], iu[1]]
    d -= L * np.rint(d / L)
    r = np.linalg.norm(d, axis=1)
    m = r < rmax
    return iu[0][m], iu[1][m], d[m], r[m]


def mk_real_pairs(x, L, rmax):
    N = len(x)
    iu = np.triu_indices(N, 1)
    d = x[iu[0]] - x[iu[1]]
    d -= L * np.rint(d / L)
    r = np.linalg.norm(d, axis=1)
    m = r < rmax
    return iu[0][m], iu[1][m], d[m], r[m]


# ----------------------------------------------------------------------------- aggregation over samples
def sample_mean_sem(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None, None, 0
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None), len(v)


def union_bins(runs):
    bs = set()
    for r in runs:
        for lag in r["lag"]:
            bs.add(int(np.rint(PD * np.log10(lag))))
    return sorted(bs)


def per_sample_curve(runs, fn, mask_fn=None):
    """fn(run, j) -> scalar or array for lag-bin j of run; returns dict b -> list over samples (aligned by bin)."""
    out = {}
    for r in runs:
        for j, lag in enumerate(r["lag"]):
            if mask_fn is not None and not mask_fn(r, j):
                continue
            b = int(np.rint(PD * np.log10(lag)))
            v = fn(r, j)
            if v is None:
                continue
            out.setdefault(b, []).append((lag, v))
    return out


def curve_stats(d):
    """dict b -> [(lag, v)] -> arrays lag, mean, sem, n (v may be arrays)."""
    lags, means, sems, ns = [], [], [], []
    for b in sorted(d):
        arr = d[b]
        lags.append(float(np.exp(np.mean([np.log(a[0]) for a in arr]))))
        V = np.array([a[1] for a in arr], float)
        means.append(np.nanmean(V, 0))
        n = len(V); ns.append(n)
        sems.append(np.nanstd(V, 0, ddof=1) / np.sqrt(n) if n > 1 else np.full_like(np.atleast_1d(V[0]), np.nan).squeeze())
    return np.array(lags), np.array(means), np.array(sems), np.array(ns)


def value_at(lags, arr, t, tol=0.12):
    """Nearest log-lag bin to t (as hs_summary.at): returns index or None."""
    if t is None or len(lags) == 0:
        return None
    if lags.max() < 0.9 * t or lags.min() > 1.1 * t:
        return None
    return int(np.argmin(np.abs(np.log(lags) - np.log(t))))


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)
