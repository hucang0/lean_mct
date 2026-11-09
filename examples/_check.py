"""Tiny helper shared by the smoke tests: compare a number with its expected value and keep a pass/fail tally."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # repository root
_fails = []


def check(name, value, expected, tol):
    ok = value is not None and abs(value - expected) <= tol
    print(f"CHECK {name}: {value:.7g} (expected {expected:.7g} +- {tol:g}) {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        _fails.append(name)
    return ok


def require(name, cond, detail=""):
    print(f"CHECK {name}: {detail} {'PASS' if cond else 'FAIL'}", flush=True)
    if not cond:
        _fails.append(name)
    return cond


def finish():
    if _fails:
        print(f"RESULT: FAIL ({len(_fails)} failed: {', '.join(_fails)})")
        sys.exit(1)
    print("RESULT: PASS")
