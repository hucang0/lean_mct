#!/usr/bin/env python3
"""Smoke test (e): multi-point hierarchy solvers (hierarchy/).
1. py_compile of every solver and `--help` of every command-line driver (no GPU needed).
2. Import check of the solver modules (needs PyTorch).
3. If a CUDA device is visible and has >= 16 GB free: PyTorch is capped at 8 GB of GPU memory for this process
   and two small production cases run through the shipped drivers (about 1 min, < 0.1 GB actually used):
     run_phic.py      MK, vertex W_s, correlator closure, N = 2, 3 on the benchmark grid (M = 100, q_max = 40)
                      -> SM Table S1 brackets (2.796484, 2.796680) and (3.257126, 3.257228);
     gmct_dense_gpu.py PY hard spheres, N = 2 -> phi_c(2) = 0.515912 (MCT, Franosch et al. 1997).
   Without a usable GPU: the dense N = 2 MK iteration runs on the CPU at phi = 2.79 (liquid) and 2.80 (glass).
Never touches other processes on a shared GPU."""
import glob, json, os, py_compile, runpy, shutil, subprocess, sys
from _check import ROOT, check, require, finish

HIER = ROOT / "hierarchy"
W = ROOT / "examples" / "_work" / "gpu"
shutil.rmtree(W, ignore_errors=True); W.mkdir(parents=True)
CAP_GB, NEED_FREE_GB = 8.0, 16.0

# 1. compile + CLI help
for f in sorted(HIER.glob("*.py")):
    try:
        py_compile.compile(str(f), cfile=str(W / (f.stem + ".pyc")), doraise=True); ok = True
    except py_compile.PyCompileError as e:
        print(e); ok = False
    require(f"py_compile {f.name}", ok)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
for f in ("run_phic.py", "run_linear.py", "campaign.py", "certified_liquid.py", "gmct_dense_gpu.py",
          "gmct_sym2_gpu.py", "linear_chain_bound.py"):
    p = subprocess.run([sys.executable, str(HIER / f), "--help"], env=env, capture_output=True, text=True)
    require(f"{f} --help", p.returncode == 0 and "usage" in p.stdout.lower())

# 2. imports
sys.path.insert(0, str(HIER))
import torch
import gmct_dense_gpu, gmct_sym2_gpu, linear_chain_bound, mk_vertex as V
require("import gmct_dense_gpu, gmct_sym2_gpu, linear_chain_bound, mk_vertex", True, f"torch {torch.__version__}")


def free_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return float(out[0]) / 1024
    except Exception:
        return 0.0


use_gpu = torch.cuda.is_available() and free_gb() >= NEED_FREE_GB
if use_gpu:
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, CAP_GB / total), 0)
    print(f"GPU: {torch.cuda.get_device_name(0)}, {free_gb():.0f} GB free; this process capped at {CAP_GB:.0f} GB", flush=True)
    # MK, W_s, correlator closure, N = 2 and 3 (exactly the SM Table S1 campaign settings)
    out = W / "phic_Bself_corr_M100.jsonl"
    sys.argv = ["run_phic.py", "--variant", "Bself", "--closure", "corr", "--Ns", "2,3", "--M", "100", "--qmax", "40",
                "--lo", "2.7", "--hi", "2.9", "--tol", "2e-4", "--out", str(out)]
    runpy.run_path(str(HIER / "run_phic.py"), run_name="__main__")
    res = {r["N"]: r for r in map(json.loads, open(out)) if r.get("type") == "result"}
    check("MK W_s phi_c(2) lo", res[2]["lo"], 2.796484, 2e-6); check("MK W_s phi_c(2) hi", res[2]["hi"], 2.796680, 2e-6)
    check("MK W_s phi_c(3) lo", res[3]["lo"], 3.257126, 2e-6); check("MK W_s phi_c(3) hi", res[3]["hi"], 3.257228, 2e-6)
    peak = max(r["gpu_peak_GB"] for r in map(json.loads, open(out)) if r.get("type") == "solve")
    require("GPU memory below cap", peak < CAP_GB, f"peak {peak:.3f} GB")
    # PY hard spheres, N = 2
    out = W / "py_N2.jsonl"
    sys.argv = ["gmct_dense_gpu.py", "--M", "100", "--N", "2", "--closure", "corr", "--lo", "0.50", "--hi", "0.53",
                "--tol", "1e-5", "--out", str(out)]
    runpy.run_path(str(HIER / "gmct_dense_gpu.py"), run_name="__main__")
    r = [json.loads(l) for l in open(out) if '"result"' in l][-1]
    check("PY phi_c(2)", 0.5 * (r["lo"] + r["hi"]), 0.515912, 2e-5)
else:
    print("no CUDA device with enough free memory: CPU fallback (dense N = 2, M = 100)", flush=True)
    for phi, want in ((2.79, "liquid"), (2.80, "glass")):
        r = V.solve_dense(100, 40.0, 2, "corr", phi, "Bself", torch.device("cpu"), torch.float64)
        require(f"MK W_s N=2 at phi={phi}", r["status"] == want, f"{r['status']} (want {want})")
finish()
