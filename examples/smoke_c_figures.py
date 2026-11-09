#!/usr/bin/env python3
"""Smoke test (c): regenerate two Letter figures, seven SM figures and one extra overview from data/ alone (CPU only,
under a minute). Checks that each PDF/PNG is freshly written and non-empty; outputs land next to the scripts in
figures/letter/ and figures/sm/ (git-ignored). Prints the image size of each PNG for a visual cross-check."""
import os, subprocess, sys, time
from _check import ROOT, require, finish

JOBS = [("letter", "fig0_cartoon_v2", ["fig0_cartoon_v2"]),       # Letter Fig. 1
        ("letter", "fig1_runoff", ["fig1_runoff"]),               # Letter Fig. 2
        ("letter", "fig3_restore", ["fig3_restore"]),             # extra diagnostic overview
        ("sm", "F1_statics", ["F1_statics"]),                     # Fig. S1
        ("sm", "F4_grid", ["F4_grid"]),                           # Fig. S2
        ("sm", "F5_simulation_Fs", ["F5_simulation_Fs"]),         # Fig. S3
        ("sm", "F6_factorization", ["F6_factorization"]),         # Fig. S4
        ("sm", "F9_restore_msd", ["F9_restore_msd"]),             # Fig. S5
        ("sm", "F7_hs_vs_mk_bond", ["F7_hs_vs_mk_bond"]),         # Fig. S6
        ("sm", "F8_fig18_overlay", ["F8_fig18_overlay"])]         # Fig. S7
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
from PIL import Image
for folder, script, outs in JOBS:
    d = ROOT / "figures" / folder
    t0 = time.time()
    p = subprocess.run([sys.executable, str(d / f"{script}.py")], cwd=d, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:])
    for o in outs:
        ok = p.returncode == 0
        for ext in ("pdf", "png"):
            f = d / f"{o}.{ext}"
            ok &= f.exists() and f.stat().st_size > 1000 and f.stat().st_mtime >= t0 - 1
        size = Image.open(d / f"{o}.png").size if ok else None
        require(f"figure {folder}/{o}", ok, f"pdf+png written ({time.time() - t0:.1f} s, png {size})")
finish()
