#!/usr/bin/env bash
# Rebuild the certificate and check that no `sorry` is present.
set -euo pipefail
cd "$(dirname "$0")"
if grep -rn "sorry" LeanMct/ ; then echo "FAIL: sorry found"; exit 1; fi
lake build LeanMct 2>&1 | grep -v '^warning: mathlib\|^trace:'
echo "OK: build finished; the eleven '#print axioms' lines above must list only propext, Classical.choice, Quot.sound"
