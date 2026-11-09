import LeanMct.GridTheorem
import LeanMct.MemoryClosure
import LeanMct.TopLevelBound
/-!
# The Mari–Kurchan instance of the grid theorems

The companion paper (Paper II) runs the wavevector-diagonal hierarchy on the Mari–Kurchan (MK)
random-shift hard-sphere liquid at infinite shift range. On the equidistant grid
`k = h k̂`, `k̂ = i + 1/2` (`i = 0, …, M−1`), its static input is

* bare rates `Ω₁²(k) = D₀ k²` (`S ≡ 1`, no de Gennes narrowing);
* the derived single-leg vertex (Paper II, Eq. (Ws))
  `W̃ₛ(k;q,p) = ρ D₀ h⁵/(16π²) · q̂p̂/k̂³ · (k̂² + q̂² − p̂²)² · f̃(q)²`,
  and the robustness variant (Paper II, Eq. (Wc))
  `W̃_c(k;q,p) = ρ D₀ h⁵/(32π²) · q̂p̂/k̂³ · [(q̂²−p̂²+k̂²) f̃(q) + (p̂²−q̂²+k̂²) f̃(p)]²`,
  both restricted to the bipolar range `|k̂ − q̂| < p̂ < k̂ + q̂`, with the hard-core Mayer function
  `f̃(q) = −(4π/q³)(sin q − q cos q)`.

This file builds that input as a `GridData`, proves the two hypotheses of the grid theorems
(positive rates, nonnegative vertex), proves that the vertex is exactly linear in the density
(so `C(φ) = C₁ φ`), and instantiates the correlator-closure theorem, the memory-closure theorem
and the infinite-hierarchy bound with the explicit slowest rate `ω_min = D₀ (h/2)²`.

What is **not** done here: numerical values of `w_max` (the constants `C₁` quoted in the papers are
floating-point evaluations of `max_k Σ_{q,p} W̃(k;q,p)`, not checked in Lean), and the derivation of
the hierarchy itself from the MK dynamics (Paper II, Secs. III–IV).
-/
open Finset Real

namespace GMCT

noncomputable section

open scoped Classical

/-- Hard-core Mayer function in Fourier space (unit diameter), `f̃(q) = −(4π/q³)(sin q − q cos q)`. -/
def mayerFT (q : ℝ) : ℝ := -(4 * π / q ^ 3) * (Real.sin q - q * Real.cos q)

/-- Dimensionless grid wavenumber `k̂ = i + 1/2`. -/
def khat {M : ℕ} (i : Fin M) : ℝ := ((i : ℕ) : ℝ) + 1 / 2

lemma khat_pos {M : ℕ} (i : Fin M) : 0 < khat i := by unfold khat; positivity

lemma khat_ge_half {M : ℕ} (i : Fin M) : (1 : ℝ) / 2 ≤ khat i := by
  unfold khat; have : (0 : ℝ) ≤ ((i : ℕ) : ℝ) := Nat.cast_nonneg _; linarith

/-- Bipolar admissibility of `(k, q, p)`: `|k̂ − q̂| < p̂ < k̂ + q̂`. -/
def bipolar {M : ℕ} (k q p : Fin M) : Prop :=
  |khat k - khat q| < khat p ∧ khat p < khat k + khat q

/-- Derived MK vertex `W̃ₛ` at unit density (multiply by `ρ`). -/
def VsMK (M : ℕ) (h D₀ : ℝ) (k q p : Fin M) : ℝ :=
  if bipolar k q p then
    D₀ * h ^ 5 / (16 * π ^ 2) * (khat q * khat p / khat k ^ 3)
      * (khat k ^ 2 + khat q ^ 2 - khat p ^ 2) ^ 2 * mayerFT (h * khat q) ^ 2
  else 0

/-- Robustness-variant MK vertex `W̃_c` at unit density (multiply by `ρ`). -/
def VcMK (M : ℕ) (h D₀ : ℝ) (k q p : Fin M) : ℝ :=
  if bipolar k q p then
    D₀ * h ^ 5 / (32 * π ^ 2) * (khat q * khat p / khat k ^ 3)
      * ((khat q ^ 2 - khat p ^ 2 + khat k ^ 2) * mayerFT (h * khat q)
        + (khat p ^ 2 - khat q ^ 2 + khat k ^ 2) * mayerFT (h * khat p)) ^ 2
  else 0

lemma VsMK_nonneg (M : ℕ) (h D₀ : ℝ) (hh : 0 < h) (hD : 0 < D₀) (k q p : Fin M) :
    0 ≤ VsMK M h D₀ k q p := by
  unfold VsMK
  split_ifs
  · have hk := khat_pos k; have hq := khat_pos q; have hp := khat_pos p
    positivity
  · exact le_rfl

lemma VcMK_nonneg (M : ℕ) (h D₀ : ℝ) (hh : 0 < h) (hD : 0 < D₀) (k q p : Fin M) :
    0 ≤ VcMK M h D₀ k q p := by
  unfold VcMK
  split_ifs
  · have hk := khat_pos k; have hq := khat_pos q; have hp := khat_pos p
    positivity
  · exact le_rfl

/-- MK grid data at density `ρ`: rates `D₀ k²` and vertex `ρ · V` for a unit-density vertex
`V ≥ 0` (`V = VsMK` or `V = VcMK`). -/
def mkGrid (M : ℕ) (h ρ D₀ : ℝ) (hh : 0 < h) (hρ : 0 ≤ ρ) (hD : 0 < D₀)
    (V : Fin M → Fin M → Fin M → ℝ) (hV : ∀ k q p, 0 ≤ V k q p) : GridData M where
  rate k := D₀ * (h * khat k) ^ 2
  W k q p := ρ * V k q p
  rate_pos k := by have := khat_pos k; positivity
  W_nonneg k q p := mul_nonneg hρ (hV k q p)

/-- The two MK vertices of Paper II as grid data. -/
def mkGridS (M : ℕ) (h ρ D₀ : ℝ) (hh : 0 < h) (hρ : 0 ≤ ρ) (hD : 0 < D₀) : GridData M :=
  mkGrid M h ρ D₀ hh hρ hD (VsMK M h D₀) (VsMK_nonneg M h D₀ hh hD)

def mkGridC (M : ℕ) (h ρ D₀ : ℝ) (hh : 0 < h) (hρ : 0 ≤ ρ) (hD : 0 < D₀) : GridData M :=
  mkGrid M h ρ D₀ hh hρ hD (VcMK M h D₀) (VcMK_nonneg M h D₀ hh hD)

variable {M : ℕ} {h ρ D₀ : ℝ} (hh : 0 < h) (hρ : 0 ≤ ρ) (hD : 0 < D₀)
  {V : Fin M → Fin M → Fin M → ℝ} (hV : ∀ k q p, 0 ≤ V k q p)

/-- The slowest bare rate on the grid is `D₀ (h/2)²`, at `k̂ = 1/2`. -/
lemma mk_rate_ge (k : Fin M) : D₀ * (h / 2) ^ 2 ≤ (mkGrid M h ρ D₀ hh hρ hD V hV).rate k := by
  show D₀ * (h / 2) ^ 2 ≤ D₀ * (h * khat k) ^ 2
  have h1 : h / 2 ≤ h * khat k := by
    have := khat_ge_half k; nlinarith
  have h2 : (h / 2) ^ 2 ≤ (h * khat k) ^ 2 := pow_le_pow_left₀ (by positivity) h1 2
  exact mul_le_mul_of_nonneg_left h2 hD.le

/-- Exact homogeneity: the frozen memory weight is linear in the density. -/
lemma mk_wsum_eq (k : Fin M) :
    wsum (mkGrid M h ρ D₀ hh hρ hD V hV) k = ρ * ∑ q, ∑ p, V k q p := by
  unfold wsum
  simp only [mkGrid, Finset.mul_sum]

lemma mk_wsum_le {w₁ : ℝ} (hw₁ : ∀ k, ∑ q, ∑ p, V k q p ≤ w₁) (k : Fin M) :
    wsum (mkGrid M h ρ D₀ hh hρ hD V hV) k ≤ ρ * w₁ := by
  rw [mk_wsum_eq]; exact mul_le_mul_of_nonneg_left (hw₁ k) hρ

/-- **Correlator closure on MK.** With `w₁ ≥ max_k Σ_{q,p} V(k;q,p)` the unit-density weight and
`C = ρ w₁ / (D₀ (h/2)²)` (linear in density), a fixed point at closure level `N = L+2` is the liquid
whenever `C^{N−1} < (N−1)!`. -/
theorem mk_no_glass_corr (hM : 0 < M) (ins : Insertion M) (rem : Removal M) (L : ℕ)
    (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ)
    (F : MFFixedPoint (mkGrid M h ρ D₀ hh hρ hD V hV) ins rem L f)
    {w₁ : ℝ} (hw₁ : ∀ k, ∑ q, ∑ p, V k q p ≤ w₁)
    (hC : (ρ * w₁ / (D₀ * (h / 2) ^ 2)) ^ (L + 1) < (((L + 1).factorial : ℕ) : ℝ)) :
    ∀ n, 1 ≤ n → n ≤ L + 2 → ∀ K : Fin n → Fin M, f n K = 0 :=
  no_nontrivial_fixed_point _ hM ins rem L f F (D₀ * (h / 2) ^ 2) (ρ * w₁) (by positivity)
    (mk_rate_ge hh hρ hD hV) (mk_wsum_le hh hρ hD hV hw₁) hC

/-- **Memory closure on MK** (canonical removal), with the same explicit constants. -/
theorem mk_no_glass_mem (hM : 0 < M) (ins : Insertion M) (L : ℕ)
    (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ)
    (F : MemFixedPoint (mkGrid M h ρ D₀ hh hρ hD V hV) ins (canonicalRem M) L f)
    {w₁ : ℝ} (hw₁ : ∀ k, ∑ q, ∑ p, V k q p ≤ w₁)
    (hC : 2 * (ρ * w₁ / (D₀ * (h / 2) ^ 2)) ^ (L + 2)
      < ((L : ℝ) + 1) * (((L + 1).factorial : ℕ) : ℝ)) :
    ∀ n, 1 ≤ n → n ≤ L + 2 → ∀ K : Fin n → Fin M, f n K = 0 :=
  no_nontrivial_fixed_point_memclosure_canonical _ hM ins L f F (D₀ * (h / 2) ^ 2) (ρ * w₁)
    (by positivity) (mk_rate_ge hh hρ hD hV) (mk_wsum_le hh hρ hD hV hw₁) hC

/-- **Unclosed MK hierarchy**: the finite-depth bound behind Corollary 2, with the MK constants. -/
theorem mk_infinite_hierarchy_bound (hM : 0 < M) (ins : Insertion M)
    (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ)
    (hnn : ∀ n K, 0 ≤ f n K) (hle1 : ∀ n K, f n K ≤ 1)
    (hlevel : ∀ n, 1 ≤ n → ∀ K : Fin n → Fin M,
      f n K = mem (mkGrid M h ρ D₀ hh hρ hD V hV) ins K (f (n + 1))
        / (1 + mem (mkGrid M h ρ D₀ hh hρ hD V hV) ins K (f (n + 1))))
    {w₁ : ℝ} (hw₁ : ∀ k, ∑ q, ∑ p, V k q p ≤ w₁) (k m : ℕ) (K : Fin (k + 1) → Fin M) :
    f (k + 1) K ≤ bnd (ρ * w₁ / (D₀ * (h / 2) ^ 2)) 1 (k + m) k :=
  infinite_hierarchy_bound _ hM ins f hnn hle1 hlevel (D₀ * (h / 2) ^ 2) (ρ * w₁) (by positivity)
    (mk_rate_ge hh hρ hD hV) (mk_wsum_le hh hρ hD hV hw₁) k m K

/-- The derived vertex and the robustness variant both satisfy the hypotheses: the theorems above
apply to `mkGridS` and `mkGridC` verbatim. -/
example (hM : 0 < M) (ins : Insertion M) (rem : Removal M) (L : ℕ)
    (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ) (F : MFFixedPoint (mkGridS M h ρ D₀ hh hρ hD) ins rem L f)
    {w₁ : ℝ} (hw₁ : ∀ k, ∑ q, ∑ p, VsMK M h D₀ k q p ≤ w₁)
    (hC : (ρ * w₁ / (D₀ * (h / 2) ^ 2)) ^ (L + 1) < (((L + 1).factorial : ℕ) : ℝ)) :
    ∀ n, 1 ≤ n → n ≤ L + 2 → ∀ K : Fin n → Fin M, f n K = 0 :=
  mk_no_glass_corr hh hρ hD (VsMK_nonneg M h D₀ hh hD) hM ins rem L f F hw₁ hC

example (hM : 0 < M) (ins : Insertion M) (rem : Removal M) (L : ℕ)
    (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ) (F : MFFixedPoint (mkGridC M h ρ D₀ hh hρ hD) ins rem L f)
    {w₁ : ℝ} (hw₁ : ∀ k, ∑ q, ∑ p, VcMK M h D₀ k q p ≤ w₁)
    (hC : (ρ * w₁ / (D₀ * (h / 2) ^ 2)) ^ (L + 1) < (((L + 1).factorial : ℕ) : ℝ)) :
    ∀ n, 1 ≤ n → n ≤ L + 2 → ∀ K : Fin n → Fin M, f n K = 0 :=
  mk_no_glass_corr hh hρ hD (VcMK_nonneg M h D₀ hh hD) hM ins rem L f F hw₁ hC

end

end GMCT

#print axioms GMCT.mk_no_glass_corr
#print axioms GMCT.mk_no_glass_mem
#print axioms GMCT.mk_infinite_hierarchy_bound
