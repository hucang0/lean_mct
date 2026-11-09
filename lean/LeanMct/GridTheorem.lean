import Mathlib
/-!
# No ergodicity-breaking fixed point of mean-field GMCT at high closure level (finite grid)

## Physical statement

Generalized mode-coupling theory (GMCT) for an overdamped liquid is an infinite hierarchy of
equations for the normalised `2n`-point density correlators `φₙ(k₁,…,kₙ,t)`.  Its long-time limits
`fₙ(k₁,…,kₙ) = lim_{t→∞} φₙ` ("non-ergodicity parameters") satisfy, after discretisation on a
finite wavenumber grid with `M` points (Luo & Janssen, *J. Chem. Phys.* 153, 214507 (2020);
arXiv:2008.10550, Eqs. (10)–(14)),

  `fₙ(K) / (1 − fₙ(K)) = 𝓜ₙ(K)[f_{n+1}]`,
  `𝓜ₙ(K)[g] = Σ_{i=1}^{n} Ω₁²(kᵢ)/Ωₙ⁴(K) · Σ_{q,p} W̃(kᵢ;q,p) · g(q,p,{k_j}_{j≠i})`,
  `Ωₙ²(K) = Σ_{i=1}^{n} Ω₁²(kᵢ)`,

for `1 ≤ n ≤ N−1`, closed at level `N` by the mean-field (MF-N[(N−1)¹1¹]) factorisation

  `f_N(K) = (1/N) Σ_{i=1}^{N} f₁(kᵢ) · f_{N−1}({k_j}_{j≠i})`.

Here `W̃ ≥ 0` collects the squared static vertex, the structure factors and the Riemann-sum
prefactor `ρ D₀ h⁵/(32π²)`; it does not depend on the level `n`.  A liquid state is `f ≡ 0`;
an ideal glass is a fixed point with some `fₙ(K) > 0`.

## Theorem (informal)

Let `ω_min = min_k Ω₁²(k) > 0` and `w_max = max_k Σ_{q,p} W̃(k;q,p)`, and put `C = w_max/ω_min`.
If `C^{N−1} < (N−1)!` then the only fixed point with values in `[0,1]` is `f ≡ 0`.
Equivalently, a glass fixed point at closure level `N` forces `(N−1)! ≤ C^{N−1}`, hence
`C ≥ (N−1)/e`.  Since `C(φ)` is finite for every packing fraction `φ` at which the static input is
finite, the MF-N critical packing fraction `φ_c(N)` must leave every compact subset of the
domain of the static input as `N → ∞`: the infinite-order hierarchy has no sharp transition on a
fixed grid.

## Proof idea

Additivity of the bare frequencies gives `Ωₙ²(K) ≥ n ω_min`, so `𝓜ₙ(K)[g] ≤ (C/n) sup g`.
Chaining the levels, `sup f₁ ≤ C^{N−1}/(N−1)! · sup f_N`, and the closure gives
`sup f_N ≤ sup f₁`.  If `C^{N−1} < (N−1)!` this forces `sup f₁ = 0`, and then every level
vanishes.

## Generality of the formal statement

* Any grid size `M ≥ 1`, any positive rates `Ω₁²`, any nonnegative vertex `W̃`.
* The *insertion* convention (where `q,p` are placed in the `(n+1)`-tuple) and the *removal*
  convention used by the closure are arbitrary functions: the bound never uses them.
* No symmetry of `fₙ` under permutations is assumed.
* Only the inequality `fₙ ≤ 𝓜ₙ` and `0 ≤ fₙ ≤ 1` are used from the level equations.

What is **not** covered in this file: the continuum limit `k_min → 0` (there `ω_min → 0` and
`C → ∞`) and the time-dependent equations. The alternative *memory-level* closure of
arXiv:1909.00428 Eq. (8) is proved in the companion modules `MemoryClosure.lean` and
`TopLevelBound.lean`.
-/
open Finset

namespace GMCT

noncomputable section

/-- Static input on a finite wavenumber grid with `M` points.
`rate k` is the squared bare frequency `Ω₁²(k) = D₀ k²/S(k)` of a single density mode and
`W k q p` is the nonnegative discretised vertex weight `W̃(k;q,p)` (all static prefactors
absorbed). -/
structure GridData (M : ℕ) where
  rate : Fin M → ℝ
  W : Fin M → Fin M → Fin M → ℝ
  rate_pos : ∀ k, 0 < rate k
  W_nonneg : ∀ k q p, 0 ≤ W k q p

variable {M : ℕ} (D : GridData M)

/-- `Ωₙ²(K) = Σᵢ Ω₁²(kᵢ)`: additivity of the bare frequencies of an `n`-tuple. -/
def Ωn {n : ℕ} (K : Fin n → Fin M) : ℝ := ∑ i, D.rate (K i)

/-- An insertion convention: from an `n`-tuple `K`, an active leg `i` and two new wavenumbers
`q, p`, build the `(n+1)`-tuple fed to the next level.  Arbitrary. -/
abbrev Insertion (M : ℕ) :=
  ∀ n : ℕ, (Fin n → Fin M) → Fin n → Fin M → Fin M → (Fin (n + 1) → Fin M)

/-- A removal convention: drop the `i`-th entry of an `(n+1)`-tuple.  Arbitrary. -/
abbrev Removal (M : ℕ) :=
  ∀ n : ℕ, (Fin (n + 1) → Fin M) → Fin (n + 1) → (Fin n → Fin M)

/-- Level-`n` memory functional
`𝓜ₙ(K)[g] = Σᵢ Ω₁²(kᵢ)/Ωₙ⁴(K) · Σ_{q,p} W̃(kᵢ;q,p) g(ins K i q p)`
(arXiv:2008.10550, Eqs. (10), (11), (13)). -/
def mem (ins : Insertion M) {n : ℕ} (K : Fin n → Fin M)
    (g : (Fin (n + 1) → Fin M) → ℝ) : ℝ :=
  ∑ i, D.rate (K i) / (Ωn D K) ^ 2 * ∑ q, ∑ p, D.W (K i) q p * g (ins n K i q p)

/-- `w(k) = Σ_{q,p} W̃(k;q,p)`: the memory weight at complete freezing (`g ≡ 1`). -/
def wsum (k : Fin M) : ℝ := ∑ q, ∑ p, D.W k q p

/-- A mean-field closed fixed point of the long-time GMCT hierarchy at closure level `N = L+2`.
Levels `1,…,L+1` obey the plateau equations `fₙ = 𝓜ₙ/(1+𝓜ₙ)` (equivalently
`fₙ/(1−fₙ) = 𝓜ₙ`, arXiv:2008.10550 Eq. (12)); level `L+2` is given by the MF closure Eq. (14).
Correlators are normalised, hence valued in `[0,1]`. -/
structure MFFixedPoint (ins : Insertion M) (rem : Removal M) (L : ℕ)
    (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ) : Prop where
  nonneg : ∀ n K, 0 ≤ f n K
  le_one : ∀ n K, f n K ≤ 1
  level : ∀ n, 1 ≤ n → n ≤ L + 1 → ∀ K : Fin n → Fin M,
    f n K = mem D ins K (f (n + 1)) / (1 + mem D ins K (f (n + 1)))
  closure : ∀ K : Fin (L + 2) → Fin M,
    f (L + 2) K = (1 / ((L : ℝ) + 2)) * ∑ i, f 1 (fun _ => K i) * f (L + 1) (rem (L + 1) K i)

/-- Non-vacuity: the liquid solution `f ≡ 0` satisfies every hypothesis of `MFFixedPoint`. -/
theorem zero_isMFFixedPoint (ins : Insertion M) (rem : Removal M) (L : ℕ) :
    MFFixedPoint D ins rem L (fun _ _ => 0) where
  nonneg := fun _ _ => le_rfl
  le_one := fun _ _ => zero_le_one
  level := by intro n _ _ K; simp [mem]
  closure := by intro K; simp

/-! ### Elementary bounds -/

lemma wsum_nonneg (k : Fin M) : 0 ≤ wsum D k :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => D.W_nonneg _ _ _

lemma Ωn_pos {n : ℕ} (hn : 1 ≤ n) (K : Fin n → Fin M) : 0 < Ωn D K := by
  unfold Ωn
  have hne : (Finset.univ : Finset (Fin n)).Nonempty := ⟨⟨0, hn⟩, Finset.mem_univ _⟩
  exact Finset.sum_pos (fun i _ => D.rate_pos _) hne

/-- Additivity: `Ωₙ²(K) ≥ n ω_min`. -/
lemma Ωn_ge {n : ℕ} (K : Fin n → Fin M) (ωmin : ℝ) (hω : ∀ k, ωmin ≤ D.rate k) :
    (n : ℝ) * ωmin ≤ Ωn D K := by
  unfold Ωn
  have h := Finset.card_nsmul_le_sum (Finset.univ : Finset (Fin n))
    (fun i => D.rate (K i)) ωmin (fun i _ => hω _)
  simpa [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] using h

lemma mem_nonneg (ins : Insertion M) {n : ℕ} (K : Fin n → Fin M)
    (g : (Fin (n + 1) → Fin M) → ℝ) (hg : ∀ x, 0 ≤ g x) : 0 ≤ mem D ins K g := by
  unfold mem
  refine Finset.sum_nonneg fun i _ => mul_nonneg ?_ ?_
  · exact div_nonneg (D.rate_pos _).le (sq_nonneg _)
  · exact Finset.sum_nonneg fun q _ => Finset.sum_nonneg fun p _ =>
      mul_nonneg (D.W_nonneg _ _ _) (hg _)

/-- **Key bound.** If `0 ≤ g ≤ G` then `𝓜ₙ(K)[g] ≤ (w_max / (n ω_min)) · G`.
The factor `1/n` comes from the additivity `Ωₙ² ≥ n ω_min`. -/
theorem mem_le (ins : Insertion M) {n : ℕ} (hn : 1 ≤ n) (K : Fin n → Fin M)
    (g : (Fin (n + 1) → Fin M) → ℝ) (G : ℝ) (hG : 0 ≤ G) (hgG : ∀ x, g x ≤ G)
    (ωmin wmax : ℝ) (hωpos : 0 < ωmin) (hω : ∀ k, ωmin ≤ D.rate k)
    (hw : ∀ k, wsum D k ≤ wmax) :
    mem D ins K g ≤ wmax / ((n : ℝ) * ωmin) * G := by
  have hΩ : 0 < Ωn D K := Ωn_pos D hn K
  have hΩne : Ωn D K ≠ 0 := hΩ.ne'
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hnω : 0 < (n : ℝ) * ωmin := mul_pos hn' hωpos
  have hwmax : 0 ≤ wmax := le_trans (wsum_nonneg D (K ⟨0, hn⟩)) (hw _)
  -- inner double sum
  have h1 : ∀ i, ∑ q, ∑ p, D.W (K i) q p * g (ins n K i q p) ≤ wmax * G := by
    intro i
    calc ∑ q, ∑ p, D.W (K i) q p * g (ins n K i q p)
        ≤ ∑ q, ∑ p, D.W (K i) q p * G :=
          Finset.sum_le_sum fun q _ => Finset.sum_le_sum fun p _ =>
            mul_le_mul_of_nonneg_left (hgG _) (D.W_nonneg _ _ _)
      _ = wsum D (K i) * G := by
          unfold wsum
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl fun q _ => ?_
          rw [Finset.sum_mul]
      _ ≤ wmax * G := mul_le_mul_of_nonneg_right (hw _) hG
  -- sum over the active leg
  have h2 : mem D ins K g ≤ ∑ i, D.rate (K i) / (Ωn D K) ^ 2 * (wmax * G) := by
    unfold mem
    exact Finset.sum_le_sum fun i _ =>
      mul_le_mul_of_nonneg_left (h1 i) (div_nonneg (D.rate_pos _).le (sq_nonneg _))
  -- evaluate the geometric factor
  have h3 : ∑ i, D.rate (K i) / (Ωn D K) ^ 2 * (wmax * G) = wmax * G / Ωn D K := by
    rw [← Finset.sum_mul, ← Finset.sum_div]
    have hΩdef : ∑ i, D.rate (K i) = Ωn D K := rfl
    rw [hΩdef]
    field_simp
  -- additivity
  have h4 : wmax * G / Ωn D K ≤ wmax * G / ((n : ℝ) * ωmin) :=
    div_le_div_of_nonneg_left (mul_nonneg hwmax hG) hnω (Ωn_ge D K ωmin hω)
  calc mem D ins K g ≤ _ := h2
    _ = _ := h3
    _ ≤ _ := h4
    _ = wmax / ((n : ℝ) * ωmin) * G := by ring

/-! ### The chain of levels -/

/-- The bound carried by level `k+1`: `C^{L+1−k} · k! / (L+1)! · s`, where `s` bounds the top
level `L+2`.  At `k = L+1` it equals `s`; at `k = 0` it equals `C^{L+1}/(L+1)! · s`. -/
def bnd (C s : ℝ) (L k : ℕ) : ℝ :=
  C ^ (L + 1 - k) * ((k.factorial : ℕ) : ℝ) / (((L + 1).factorial : ℕ) : ℝ) * s

/-- One step down the hierarchy: a bound on level `k+2` gives a bound on level `k+1`. -/
theorem step (ins : Insertion M) (L : ℕ)
    (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ) (hnn : ∀ n K, 0 ≤ f n K)
    (hlevel : ∀ n, 1 ≤ n → n ≤ L + 1 → ∀ K : Fin n → Fin M,
      f n K = mem D ins K (f (n + 1)) / (1 + mem D ins K (f (n + 1))))
    (ωmin wmax : ℝ) (hωpos : 0 < ωmin) (hω : ∀ k, ωmin ≤ D.rate k)
    (hw : ∀ k, wsum D k ≤ wmax) (hwmax : 0 ≤ wmax) (s : ℝ) (hs : 0 ≤ s)
    (k : ℕ) (hk : k ≤ L)
    (hnext : ∀ K' : Fin (k + 2) → Fin M, f (k + 2) K' ≤ bnd (wmax / ωmin) s L (k + 1)) :
    ∀ K : Fin (k + 1) → Fin M, f (k + 1) K ≤ bnd (wmax / ωmin) s L k := by
  intro K
  have hC0 : 0 ≤ wmax / ωmin := div_nonneg hwmax hωpos.le
  have hlev := hlevel (k + 1) (by omega) (by omega) K
  have hmem0 : 0 ≤ mem D ins K (f (k + 1 + 1)) :=
    mem_nonneg D ins K _ (fun x => hnn _ x)
  have hG : 0 ≤ bnd (wmax / ωmin) s L (k + 1) := by
    unfold bnd
    exact mul_nonneg (div_nonneg (mul_nonneg (pow_nonneg hC0 _) (Nat.cast_nonneg _))
      (Nat.cast_nonneg _)) hs
  have hmem_le : mem D ins K (f (k + 1 + 1)) ≤
      wmax / (((k + 1 : ℕ) : ℝ) * ωmin) * bnd (wmax / ωmin) s L (k + 1) :=
    mem_le D ins (n := k + 1) (by omega) K (f (k + 1 + 1)) (bnd (wmax / ωmin) s L (k + 1)) hG
      (fun x => hnext x) ωmin wmax hωpos hω hw
  have halg : wmax / (((k + 1 : ℕ) : ℝ) * ωmin) * bnd (wmax / ωmin) s L (k + 1)
      = bnd (wmax / ωmin) s L k := by
    unfold bnd
    have e1 : L + 1 - (k + 1) = L - k := by omega
    have e2 : L + 1 - k = L - k + 1 := by omega
    rw [e1, e2, pow_succ, Nat.factorial_succ]
    push_cast
    have hω' : ωmin ≠ 0 := ne_of_gt hωpos
    have hk1 : ((k : ℝ) + 1) ≠ 0 := by positivity
    have hf : (((L + 1).factorial : ℕ) : ℝ) ≠ 0 := by positivity
    field_simp
  calc f (k + 1) K = _ := hlev
    _ ≤ mem D ins K (f (k + 1 + 1)) := div_le_self hmem0 (by linarith)
    _ ≤ _ := hmem_le
    _ = bnd (wmax / ωmin) s L k := halg

/-- Descending the whole hierarchy from a bound `s` on the closure level. -/
theorem chain (ins : Insertion M) (L : ℕ)
    (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ) (hnn : ∀ n K, 0 ≤ f n K)
    (hlevel : ∀ n, 1 ≤ n → n ≤ L + 1 → ∀ K : Fin n → Fin M,
      f n K = mem D ins K (f (n + 1)) / (1 + mem D ins K (f (n + 1))))
    (ωmin wmax : ℝ) (hωpos : 0 < ωmin) (hω : ∀ k, ωmin ≤ D.rate k)
    (hw : ∀ k, wsum D k ≤ wmax) (hwmax : 0 ≤ wmax) (s : ℝ) (hs : 0 ≤ s)
    (htop : ∀ K : Fin (L + 2) → Fin M, f (L + 2) K ≤ s) :
    ∀ k, k ≤ L + 1 → ∀ K : Fin (k + 1) → Fin M, f (k + 1) K ≤ bnd (wmax / ωmin) s L k := by
  have key : ∀ d, d ≤ L + 1 →
      ∀ K : Fin (L + 1 - d + 1) → Fin M,
        f (L + 1 - d + 1) K ≤ bnd (wmax / ωmin) s L (L + 1 - d) := by
    intro d
    induction d with
    | zero =>
      intro _ K
      simp only [Nat.sub_zero]
      have hb : bnd (wmax / ωmin) s L (L + 1) = s := by
        unfold bnd
        have hf : (((L + 1).factorial : ℕ) : ℝ) ≠ 0 := by positivity
        rw [Nat.sub_self, pow_zero, one_mul, div_self hf, one_mul]
      rw [hb]
      exact htop K
    | succ d ih =>
      intro hd K
      have hk : L + 1 - (d + 1) ≤ L := by omega
      have e : L + 1 - (d + 1) + 1 = L + 1 - d := by omega
      have hprev : ∀ K' : Fin (L + 1 - (d + 1) + 1 + 1) → Fin M,
          f (L + 1 - (d + 1) + 1 + 1) K' ≤ bnd (wmax / ωmin) s L (L + 1 - (d + 1) + 1) := by
        rw [e]
        exact ih (by omega)
      exact step D ins L f hnn hlevel ωmin wmax hωpos hω hw hwmax s hs (L + 1 - (d + 1)) hk hprev K
  intro k hk K
  have e : L + 1 - (L + 1 - k) = k := by omega
  have h := key (L + 1 - k) (by omega)
  rw [e] at h
  exact h K

/-! ### Main theorem -/

/-- **Grid theorem.** On a finite wavenumber grid, if `C^{N−1} < (N−1)!` with
`C = w_max/ω_min` and `N = L+2` the closure level, every mean-field closed fixed point of the
long-time GMCT hierarchy with values in `[0,1]` is trivial: `fₙ ≡ 0` for all levels. -/
theorem no_nontrivial_fixed_point (hM : 0 < M) (ins : Insertion M) (rem : Removal M) (L : ℕ)
    (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ) (F : MFFixedPoint D ins rem L f)
    (ωmin wmax : ℝ) (hωpos : 0 < ωmin) (hω : ∀ k, ωmin ≤ D.rate k)
    (hw : ∀ k, wsum D k ≤ wmax)
    (hC : (wmax / ωmin) ^ (L + 1) < (((L + 1).factorial : ℕ) : ℝ)) :
    ∀ n, 1 ≤ n → n ≤ L + 2 → ∀ K : Fin n → Fin M, f n K = 0 := by
  have hwmax : 0 ≤ wmax := le_trans (wsum_nonneg D ⟨0, hM⟩) (hw _)
  haveI : Nonempty (Fin 1 → Fin M) := ⟨fun _ => ⟨0, hM⟩⟩
  obtain ⟨K0, hK0⟩ := Finite.exists_max (fun K : Fin 1 → Fin M => f 1 K)
  set s := f 1 K0 with hs_def
  have hs : 0 ≤ s := F.nonneg _ _
  -- the closure level is bounded by the maximum of level 1
  have htop : ∀ K : Fin (L + 2) → Fin M, f (L + 2) K ≤ s := by
    intro K
    rw [F.closure K]
    have hterm : ∀ i ∈ (Finset.univ : Finset (Fin (L + 2))),
        f 1 (fun _ => K i) * f (L + 1) (rem (L + 1) K i) ≤ s := by
      intro i _
      calc f 1 (fun _ => K i) * f (L + 1) (rem (L + 1) K i)
          ≤ s * 1 := mul_le_mul (hK0 _) (F.le_one _ _) (F.nonneg _ _) hs
        _ = s := mul_one s
    have hsum := Finset.sum_le_card_nsmul _ _ _ hterm
    rw [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hsum
    have hL : (0 : ℝ) < (L : ℝ) + 2 := by positivity
    calc (1 / ((L : ℝ) + 2)) * ∑ i, f 1 (fun _ => K i) * f (L + 1) (rem (L + 1) K i)
        ≤ (1 / ((L : ℝ) + 2)) * (((L + 2 : ℕ) : ℝ) * s) :=
          mul_le_mul_of_nonneg_left hsum (by positivity)
      _ = s := by push_cast; field_simp
  have hchain := chain D ins L f F.nonneg F.level ωmin wmax hωpos hω hw hwmax s hs htop
  -- level 1 at the maximiser: s ≤ (C^{L+1}/(L+1)!) s
  have h0 := hchain 0 (by omega) K0
  simp only [bnd, Nat.sub_zero, Nat.factorial_zero, Nat.cast_one, mul_one] at h0
  have hfpos : (0 : ℝ) < (((L + 1).factorial : ℕ) : ℝ) := by positivity
  have hr : (wmax / ωmin) ^ (L + 1) / (((L + 1).factorial : ℕ) : ℝ) < 1 :=
    (div_lt_one hfpos).mpr hC
  have hs0 : s = 0 := by
    refine le_antisymm ?_ hs
    by_contra hpos
    have hpos' : 0 < s := not_le.mp hpos
    have hlt := mul_lt_mul_of_pos_right hr hpos'
    rw [one_mul] at hlt
    linarith
  -- every level is then squeezed to zero
  intro n hn1 hn2 K
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  have hb := hchain k (by omega) K
  rw [hs0] at hb
  simp only [bnd, mul_zero] at hb
  exact le_antisymm hb (F.nonneg _ _)

/-! ### Corollaries: what a glass fixed point forces -/

/-- Contrapositive: a nontrivial (glass) fixed point at closure level `N = L+2` forces
`(N−1)! ≤ C^{N−1}`. -/
theorem factorial_le_of_nontrivial (hM : 0 < M) (ins : Insertion M) (rem : Removal M) (L : ℕ)
    (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ) (F : MFFixedPoint D ins rem L f)
    (ωmin wmax : ℝ) (hωpos : 0 < ωmin) (hω : ∀ k, ωmin ≤ D.rate k)
    (hw : ∀ k, wsum D k ≤ wmax)
    (hne : ∃ n, 1 ≤ n ∧ n ≤ L + 2 ∧ ∃ K : Fin n → Fin M, f n K ≠ 0) :
    (((L + 1).factorial : ℕ) : ℝ) ≤ (wmax / ωmin) ^ (L + 1) := by
  by_contra h
  obtain ⟨n, hn1, hn2, K, hK⟩ := hne
  exact hK (no_nontrivial_fixed_point D hM ins rem L f F ωmin wmax hωpos hω hw (not_le.mp h)
    n hn1 hn2 K)

/-- Stirling-type consequence: a glass fixed point at closure level `N = L+2` forces
`C ≥ (N−1)/e`.  Hence `C(φ_c(N)) → ∞`, i.e. the critical point runs off to the edge of the
domain where the static input stays finite. -/
theorem ratio_lower_bound_of_nontrivial (hM : 0 < M) (ins : Insertion M) (rem : Removal M)
    (L : ℕ) (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ) (F : MFFixedPoint D ins rem L f)
    (ωmin wmax : ℝ) (hωpos : 0 < ωmin) (hω : ∀ k, ωmin ≤ D.rate k)
    (hw : ∀ k, wsum D k ≤ wmax)
    (hne : ∃ n, 1 ≤ n ∧ n ≤ L + 2 ∧ ∃ K : Fin n → Fin M, f n K ≠ 0) :
    ((L : ℝ) + 1) / Real.exp 1 ≤ wmax / ωmin := by
  have hwmax : 0 ≤ wmax := le_trans (wsum_nonneg D ⟨0, hM⟩) (hw _)
  have hC0 : 0 ≤ wmax / ωmin := div_nonneg hwmax hωpos.le
  have h1 := factorial_le_of_nontrivial D hM ins rem L f F ωmin wmax hωpos hω hw hne
  -- (L+1)^(L+1) / (L+1)! ≤ exp (L+1)
  have h2 := Real.pow_div_factorial_le_exp ((L : ℝ) + 1) (by positivity) (L + 1)
  have hfpos : (0 : ℝ) < (((L + 1).factorial : ℕ) : ℝ) := by positivity
  have hexp : Real.exp ((L : ℝ) + 1) = Real.exp 1 ^ (L + 1) := by
    rw [Real.exp_one_pow]; push_cast; ring_nf
  -- (L+1)^(L+1) ≤ exp(1)^(L+1) * (L+1)! ≤ (exp 1 * C)^(L+1)
  have h3 : ((L : ℝ) + 1) ^ (L + 1) ≤ (Real.exp 1 * (wmax / ωmin)) ^ (L + 1) := by
    have h2' : ((L : ℝ) + 1) ^ (L + 1) ≤ Real.exp ((L : ℝ) + 1) * (((L + 1).factorial : ℕ) : ℝ) := by
      rwa [div_le_iff₀ hfpos] at h2
    calc ((L : ℝ) + 1) ^ (L + 1)
        ≤ Real.exp ((L : ℝ) + 1) * (((L + 1).factorial : ℕ) : ℝ) := h2'
      _ ≤ Real.exp ((L : ℝ) + 1) * (wmax / ωmin) ^ (L + 1) :=
          mul_le_mul_of_nonneg_left h1 (Real.exp_pos _).le
      _ = (Real.exp 1 * (wmax / ωmin)) ^ (L + 1) := by rw [hexp, mul_pow]
  have h4 : (L : ℝ) + 1 ≤ Real.exp 1 * (wmax / ωmin) :=
    (pow_le_pow_iff_left₀ (by positivity) (by positivity) (by omega)).mp h3
  rw [div_le_iff₀ (Real.exp_pos 1)]
  linarith [h4]

end

end GMCT

#print axioms GMCT.no_nontrivial_fixed_point
#print axioms GMCT.ratio_lower_bound_of_nontrivial
