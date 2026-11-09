#!/usr/bin/env python3
"""Release regression checks: published R2 data and launcher completion/failures.

No simulation or physics solver runs. Launchers use a stub interpreter in a
temporary directory; R2 tables are regenerated from shipped aggregate results.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from _check import ROOT, require, finish

env = {k: v for k, v in os.environ.items() if not k.startswith("R2_")}
env.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")
with tempfile.TemporaryDirectory(prefix="mct_release_") as name:
    work = Path(name)
    env["R2_RESULTS"] = str(work / "results")
    p = subprocess.run(
        [sys.executable, str(ROOT / "simulation/hard_spheres/r2_tables.py")],
        cwd=work, env=env, capture_output=True, text=True, timeout=30,
    )
    require("R2 table generation from published data outside repository cwd", p.returncode == 0, p.stderr[-1000:])
    if p.returncode == 0:
        for file in ("tables_r2.md", "key_numbers.json"):
            actual = (work / "results" / file).read_text()
            expected = (ROOT / "data/simulation/r2_bond" / file).read_text()
            if file.endswith("json"):
                # Preserve existing NaN markers for undefined measurements.
                actual = json.dumps(json.loads(actual), sort_keys=True)
                expected = json.dumps(json.loads(expected), sort_keys=True)
            require(f"R2 {file} matches published result", actual == expected)

    # A fresh installation without trajectories must not silently emit empty aggregates.
    env.update(R2_HS_DATA=str(work / "missing_hs"), R2_HS_MD_DATA=str(work / "missing_md"),
               R2_MK_DATA=str(work / "missing_mk"), R2_CACHE=str(work / "cache"))
    for file in ("r2_triplet.py", "r2_sum_trip.py"):
        p = subprocess.run([sys.executable, str(ROOT / "simulation/hard_spheres" / file)],
                           cwd=work, env=env, capture_output=True, text=True, timeout=15)
        require(f"{file} rejects missing trajectories/caches", p.returncode != 0 and "FileNotFoundError" in p.stderr)

    # Check the wrapper's dependency handling without running any numerical tests.
    runner_dir = work / "examples"
    runner_dir.mkdir()
    shutil.copy2(ROOT / "examples/run_all.sh", runner_dir / "run_all.sh")
    wrapper_python = work / "wrapper python"
    wrapper_python.write_text(
        f"#!{sys.executable}\n"
        "import os,sys\n"
        "if sys.argv[1]=='-c': sys.exit(1)  # simulate missing PyTorch\n"
        "failed=os.environ.get('MOCK_TEST_FAIL') and 'c_figures' in sys.argv[1]\n"
        "print('RESULT: FAIL' if failed else 'RESULT: PASS')\n"
        "sys.exit(3 if failed else 0)\n"
    )
    wrapper_python.chmod(0o755)
    for failed in (False, True):
        runner_env = dict(env, PYTHON=str(wrapper_python), MOCK_TEST_FAIL="1" if failed else "")
        p = subprocess.run(["bash", str(runner_dir / "run_all.sh")], cwd=work,
                           env=runner_env, capture_output=True, text=True, timeout=15)
        require(f"suite skips unavailable PyTorch and {'propagates failure' if failed else 'succeeds'}",
                "e_hierarchy_gpu" in p.stdout and "SKIP" in p.stdout and
                ((p.returncode != 0) if failed else (p.returncode == 0)))

    sim = work / "simulation"
    hs, mk = sim / "hard_spheres", sim / "mk_planted_mc"
    hs.mkdir(parents=True)
    mk.mkdir()
    shutil.copy2(ROOT / "simulation/_jobs.sh", sim / "_jobs.sh")
    for folder, files in [(hs, ["launch_equil.sh", "launch_dyn.sh", "launch_md.sh"]),
                          (mk, ["launch_mc.sh"])]:
        for file in files:
            shutil.copy2(ROOT / "simulation" / folder.name / file, folder / file)
    # Only sentinel files are needed by the stub; these are not trajectories.
    eq = hs / "data/eq"
    eq.mkdir(parents=True)
    for phi in (0.500, 0.550, 0.570, 0.585, 0.600):
        for seed in range(1, 7):
            (eq / f"eq_phi{phi:.3f}_N1000_s{seed}.npz").write_text("stub")
    for seed in range(1, 4):
        (eq / f"eq_phi0.585_N2000_s{seed}.npz").write_text("stub")
    stub = work / "stub python"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import os,sys,time\n"
        "from pathlib import Path\n"
        "time.sleep(0.1)\n"
        "args=sys.argv[1:]\n"
        "seed=args[args.index('--seed')+1]\n"
        "out=args[args.index('--out')+1]\n"
        "marker=Path(os.environ['MOCK_MARKERS'])\n"
        "marker.mkdir(parents=True,exist_ok=True)\n"
        "(marker/Path(out).name).write_text('finished')\n"
        "sys.exit(7 if os.environ.get('MOCK_FAIL') and seed in ('1','11') else 0)\n"
    )
    stub.chmod(0o755)
    jobs = [(hs, "launch_equil.sh", [], 33), (hs, "launch_dyn.sh", [], 39),
            (hs, "launch_md.sh", [], 12), (mk, "launch_mc.sh", ["A"], 29),
            (mk, "launch_mc.sh", ["B"], 24)]
    for folder, file, args, count in jobs:
        for failed in (False, True):
            markers = work / f"markers_{file}_{''.join(args)}_{failed}"
            launch_env = dict(env, PYTHON=str(stub), MOCK_MARKERS=str(markers))
            if failed:
                launch_env["MOCK_FAIL"] = "1"
            else:
                launch_env.pop("MOCK_FAIL", None)
            p = subprocess.run(["bash", str(folder / file), *args], cwd=work,
                               env=launch_env, capture_output=True, text=True, timeout=30)
            done = len(list(markers.glob("*"))) if markers.exists() else 0
            require(f"{file} {' '.join(args)} waits for all {count} stub jobs ({'failure' if failed else 'success'})",
                    done == count and ((p.returncode != 0) if failed else (p.returncode == 0)),
                    f"completed={done}, exit={p.returncode}; {p.stderr[-200:] if done != count else ''}")
    # Missing equilibrated inputs must fail promptly instead of waiting forever.
    shutil.rmtree(eq)
    p = subprocess.run(["bash", str(hs / "launch_dyn.sh")], cwd=work,
                       env=dict(env, PYTHON=str(stub)), capture_output=True, text=True, timeout=10)
    require("missing equilibration inputs fail promptly", p.returncode != 0 and "Missing equilibrated start" in p.stderr)
finish()
