#!/usr/bin/env python3
"""Smoke test (a): Gaussian restored-pair closure on the Mari-Kurchan liquid (SM Sec. S5 B, Table S15).
phi_c = 1.776 with the bond hard core restored (weight 1/Z), 3.237 with Z = 1 (hard core removed), and the
plateau MSD at phi = 2.0 (0.119). CPU only, about 1 s."""
import math, sys
from _check import ROOT, check, finish
sys.path.insert(0, str(ROOT / "restored_pair"))
import brc_gauss as bg

pc_r, s2_r = bg.critical("brc", 8000)          # restored pair: sigma^2 (rho/3) int |J|^2 / Z = 1
pc_d, s2_d = bg.critical("diag", 8000)         # Z = 1: diagonal (single-collision) theory
s2 = bg.sigma2_glass(2.0, "brc", 8000, s2_r)
print(f"restored (1/Z):  phi_c = {pc_r:.6f}, sigma2_c = {s2_r:.6f}")
print(f"removed (Z = 1): phi_c = {pc_d:.6f}, sigma2_c = {s2_d:.6f}")
print(f"restored, phi = 2.0: MSD plateau 6 sigma^2 = {6 * s2:.4f}")
check("phi_c restored", pc_r, 1.776241, 5e-4)
check("phi_c Z=1", pc_d, 3.236939, 5e-4)
check("MSD(phi=2.0) restored", 6 * s2, 0.119, 2e-3)
finish()
