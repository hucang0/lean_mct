# `lean/`: Lean 4 certificate of the dilution theorems

Machine-checked proofs that accompany the Letter "More particles or the pair? An exact test of how to improve mode-coupling theory" (Supplemental Material Sec. S1 B–D). Six modules, 1,345 lines, no `sorry`.

- **Toolchain:** `leanprover/lean4:v4.30.0` (`lean-toolchain`).
- **Mathlib:** pinned to commit `1f32c4c0fbfa3dc9d2b7ac5716874fa87d3a28d5` (`lakefile.toml`, `lake-manifest.json`).

## Verify

With [elan](https://github.com/leanprover/elan) installed, from this folder:

    lake exe cache get   # downloads Mathlib and its prebuilt .olean files for the pinned commit (network, ~5 GB)
    ./verify.sh          # fails on any `sorry`, then builds LeanMct and prints the axioms of the exported theorems

Runtime: with the Mathlib cache in place, `./verify.sh` builds the six modules in about 20 s on one CPU core.
Without `lake exe cache get`, `lake build` compiles Mathlib from source (hours).

Each of the eleven `#print axioms` lines must read `depends on axioms: [propext, Classical.choice, Quot.sound]`.
Expected tail of the output:

    info: LeanMct/MKInstance.lean:176:0: 'GMCT.mk_infinite_hierarchy_bound' depends on axioms: [propext, Classical.choice, Quot.sound]
    ✔ [8498/8499] Built LeanMct
    Build completed successfully (8499 jobs).
    OK: build finished; the eleven '#print axioms' lines above must list only propext, Classical.choice, Quot.sound

## Statements (numbering of the Supplemental Material)

| SM | Lean (namespace `GMCT`) | File |
|---|---|---|
| Lemma 1 (the `C/n` bound) | `mem_le` | `GridTheorem.lean` |
| Theorem 1 (correlator closure: a glass needs `C ≥ N−1`) | `no_nontrivial_fixed_point_top` | `TopLevelBound.lean` |
| Theorem 2 (memory closure: a glass needs `2C ≥ N−2`) | `no_nontrivial_fixed_point_memclosure_top` | `TopLevelBound.lean` |
| Theorem 3 (transfer matrix, Collatz–Wielandt) | `no_nontrivial_fixed_point_chain` | `ChainBound.lean` |
| Theorem 3 (weighted chain bound, certified liquid) | `no_nontrivial_fixed_point_chain_g` | `CertifiedLiquid.lean` |
| finite-depth bound of the unclosed hierarchy | `infinite_hierarchy_bound` | `TopLevelBound.lean` |
| full-chain factorial bounds | `no_nontrivial_fixed_point`, `factorial_le_of_nontrivial`, `ratio_lower_bound_of_nontrivial`, `no_nontrivial_fixed_point_memclosure_canonical` | `GridTheorem.lean`, `MemoryClosure.lean` |
| Mari–Kurchan instance | `mk_no_glass_corr`, `mk_no_glass_mem`, `mk_infinite_hierarchy_bound` (with `VsMK_nonneg`, `VcMK_nonneg`, `mk_rate_ge`, `mk_wsum_eq`) | `MKInstance.lean` |

The witnesses `zero_isMFFixedPoint` and `zero_isMemFixedPoint` show that the hypothesis classes are not empty.
Index conventions: in `MFFixedPoint` (correlator closure), Lean parameter `L` closes at level `L + 2 = N`, so `L + 1 = N − 1`; in `MemFixedPoint` (memory closure), Lean parameter `L` retains correlators up to level `L + 2`, corresponding to `N − 1 = L + 2` (`N = L + 3`, `L + 1 = N − 2`) in the SM Sec. S1 C convention where `M_{N−1}` is factored. In `ChainBound.lean` and `CertifiedLiquid.lean`, the exported theorems formalize the Collatz–Wielandt certificate `T_N[g] v ≤ θ v` (`θ < 1 ⟹ f_n ≡ 0`) for any positive test vector `v > 0`; by Perron–Frobenius theory, taking the infimum over `v > 0` yields the spectral-radius criterion `r(T_N) < 1` quoted in the Letter.

## Scope

**What the statements assume.** They are abstract over grid data:
- positive rates `rate k = Ω₁²(k)`;
- the additive `Ωₙ²`;
- a nonnegative vertex `W k q p`;
- arbitrary insertion and removal maps (no permutation symmetry is assumed). The memory closure additionally needs a
  removal map that does not increase the frequency, which is proved for the canonical map (`Ωn_canonicalRem_le`).

`MKInstance.lean` builds the Mari–Kurchan rates `D₀(h(j+½))²` and both vertices from the hard-core Mayer function
`f̃(q) = −(4π/q³)(sin q − q cos q)`, then proves:
- the vertices are nonnegative;
- `wsum(ρ) = ρ·wsum(1)` (homogeneity);
- `ω_min = D₀(h/2)²`.

The module comments of `MKInstance.lean` refer to the MK vertices as those of "Paper II"; in the Letter these are the
vertices `W̃_s` and `W̃_coll` of SM Sec. S2 D. The Lean proof sources are unchanged from the accompanying ancillary archive.

**What is outside the formal statement:**
- the limit `m → ∞` of the finite-depth bound, which is elementary;
- floating-point values: `w_max`, the Collatz–Wielandt ratios and the monotone iterates (these are computed by the
  solvers in `../hierarchy/`);
- the derivation of the hierarchy from the dynamics.

The theorems hold on a fixed finite grid: the constant `C` grows as the grid is refined.
