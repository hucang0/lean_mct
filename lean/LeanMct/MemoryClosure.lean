import LeanMct.GridTheorem

/-!
# The same conclusion for the memory-level mean-field closure (arXiv:1909.00428, Eq. (8))

The numerical series of Luo & Janssen, *J. Chem. Phys.* 153, 214507 (2020) (arXiv:1909.00428),
closes the hierarchy at the level of the **memory function**,

  `m_N(K) = (1/(N−1)) · (1/Ω_N²(K)) · Σ_{i=1}^{N} Ω_{N−1}²(K∖i) · m_{N−1}(K∖i) · f₁(kᵢ)`,

with `f_N/(1−f_N) = m_N/Ω_N²`, instead of factorising the correlator `f_N` itself.  Writing
`m_{N−1}(K∖i) = Ω_{N−1}²(K∖i) · 𝓜_{N−1}(K∖i)[f_N]`, the closure level reads

  `f_N(K)/(1−f_N(K)) = X(K)`,
  `X(K) = (1/(N−1)) Σ_{i=1}^{N} (Ω_{N−1}²(K∖i)/Ω_N²(K))² · 𝓜_{N−1}(K∖i)[f_N] · f₁(kᵢ)`.

Here all `N` levels `f₁,…,f_N` are unknowns.  The only extra structural fact used is that
removing a leg cannot increase the bare frequency, `Ω_{N−1}²(K∖i) ≤ Ω_N²(K)`, which holds for
the genuine removal map (proved below for the canonical one) and is assumed as a hypothesis for an
arbitrary removal convention.

**Theorem.**  With `N = L+2` and `C = w_max/ω_min`: if `2·C^{N} < (N−1)·(N−1)!` then the only
fixed point with values in `[0,1]` is `f ≡ 0`.  Equivalently a glass fixed point forces
`(N−1)·(N−1)! ≤ 2·C^{N}`, again factorial against power, so `C(φ_c(N)) → ∞` as `N → ∞`.
-/

open Finset

namespace GMCT

noncomputable section

variable {M : ℕ} (D : GridData M)

/-- The canonical removal: drop the `i`-th entry (`Fin.succAbove`). -/
def canonicalRem (M : ℕ) : Removal M := fun _ K i j => K (i.succAbove j)

/-- Removing a leg does not increase the total bare frequency. -/
lemma Ωn_canonicalRem_le {n : ℕ} (K : Fin (n + 1) → Fin M) (i : Fin (n + 1)) :
    Ωn D (canonicalRem M n K i) ≤ Ωn D K := by
  unfold Ωn canonicalRem
  rw [Fin.sum_univ_succAbove (fun j => D.rate (K j)) i]
  have h := D.rate_pos (K i)
  linarith

/-- The right-hand side `X(K)` of the memory-level closure at level `N = L+2`
(arXiv:1909.00428, Eq. (8), rewritten through `𝓜_{N−1}`). -/
def memClosureX (ins : Insertion M) (rem : Removal M) (L : ℕ)
    (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ) (K : Fin (L + 2) → Fin M) : ℝ :=
  (1 / ((L : ℝ) + 1)) * ∑ i, (Ωn D (rem (L + 1) K i) / Ωn D K) ^ 2 *
    mem D ins (rem (L + 1) K i) (f (L + 2)) * f 1 (fun _ => K i)

/-- A fixed point of the long-time hierarchy closed at level `N = L+2` by the memory-level
mean-field closure of arXiv:1909.00428, Eq. (8).  Levels `1,…,L+1` obey the plateau equations;
level `L+2` obeys `f_{L+2} = X/(1+X)`. -/
structure MemFixedPoint (ins : Insertion M) (rem : Removal M) (L : ℕ)
    (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ) : Prop where
  nonneg : ∀ n K, 0 ≤ f n K
  le_one : ∀ n K, f n K ≤ 1
  level : ∀ n, 1 ≤ n → n ≤ L + 1 → ∀ K : Fin n → Fin M,
    f n K = mem D ins K (f (n + 1)) / (1 + mem D ins K (f (n + 1)))
  closure : ∀ K : Fin (L + 2) → Fin M,
    f (L + 2) K = memClosureX D ins rem L f K / (1 + memClosureX D ins rem L f K)

/-- Non-vacuity: the liquid solution `f ≡ 0` satisfies every hypothesis of `MemFixedPoint`. -/
theorem zero_isMemFixedPoint (ins : Insertion M) (rem : Removal M) (L : ℕ) :
    MemFixedPoint D ins rem L (fun _ _ => 0) where
  nonneg := fun _ _ => le_rfl
  le_one := fun _ _ => zero_le_one
  level := by intro n _ _ K; simp [mem]
  closure := by intro K; simp [memClosureX, mem]

/-- **Grid theorem, memory-level closure.**  If `2·C^{L+2} < (L+1)·(L+1)!` then every fixed
point of the hierarchy closed at level `N = L+2` by the memory-level MF closure, with values in
`[0,1]` and a frequency-nonincreasing removal convention, is trivial. -/
theorem no_nontrivial_fixed_point_memclosure (hM : 0 < M) (ins : Insertion M) (rem : Removal M)
    (L : ℕ) (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ) (F : MemFixedPoint D ins rem L f)
    (hrem : ∀ (K : Fin (L + 2) → Fin M) i, Ωn D (rem (L + 1) K i) ≤ Ωn D K)
    (ωmin wmax : ℝ) (hωpos : 0 < ωmin) (hω : ∀ k, ωmin ≤ D.rate k)
    (hw : ∀ k, wsum D k ≤ wmax)
    (hC : 2 * (wmax / ωmin) ^ (L + 2) < ((L : ℝ) + 1) * (((L + 1).factorial : ℕ) : ℝ)) :
    ∀ n, 1 ≤ n → n ≤ L + 2 → ∀ K : Fin n → Fin M, f n K = 0 := by
  have hwmax : 0 ≤ wmax := le_trans (wsum_nonneg D ⟨0, hM⟩) (hw _)
  have hC0 : 0 ≤ wmax / ωmin := div_nonneg hwmax hωpos.le
  haveI : Nonempty (Fin 1 → Fin M) := ⟨fun _ => ⟨0, hM⟩⟩
  haveI : Nonempty (Fin (L + 2) → Fin M) := ⟨fun _ => ⟨0, hM⟩⟩
  obtain ⟨K0, hK0⟩ := Finite.exists_max (fun K : Fin 1 → Fin M => f 1 K)
  obtain ⟨K1, hK1⟩ := Finite.exists_max (fun K : Fin (L + 2) → Fin M => f (L + 2) K)
  set s := f 1 K0 with hs_def
  set t := f (L + 2) K1 with ht_def
  have hs : 0 ≤ s := F.nonneg _ _
  have ht : 0 ≤ t := F.nonneg _ _
  have ht1 : t ≤ 1 := F.le_one _ _
  have hfpos : (0 : ℝ) < (((L + 1).factorial : ℕ) : ℝ) := by positivity
  have hL1 : (0 : ℝ) < (L : ℝ) + 1 := by positivity
  -- descend from the top level bounded by `t`
  have hchain := chain D ins L f F.nonneg F.level ωmin wmax hωpos hω hw hwmax t ht (fun K => hK1 K)
  have h0 := hchain 0 (by omega) K0
  simp only [bnd, Nat.sub_zero, Nat.factorial_zero, Nat.cast_one, mul_one] at h0
  -- h0 : s ≤ C^(L+1) / (L+1)! * t
  -- the closure level at its maximiser
  have hX : t ≤ (((L : ℝ) + 2) / ((L : ℝ) + 1)) *
      (wmax / (((L + 1 : ℕ) : ℝ) * ωmin) * t) * s := by
    have hcl := F.closure K1
    have hXnn : 0 ≤ memClosureX D ins rem L f K1 := by
      unfold memClosureX
      refine mul_nonneg (by positivity) (Finset.sum_nonneg fun i _ => ?_)
      exact mul_nonneg (mul_nonneg (sq_nonneg _)
        (mem_nonneg D ins _ _ (fun x => F.nonneg _ x))) (F.nonneg _ _)
    have hterm : ∀ i ∈ (Finset.univ : Finset (Fin (L + 2))),
        (Ωn D (rem (L + 1) K1 i) / Ωn D K1) ^ 2 *
          mem D ins (rem (L + 1) K1 i) (f (L + 2)) * f 1 (fun _ => K1 i)
        ≤ 1 * (wmax / (((L + 1 : ℕ) : ℝ) * ωmin) * t) * s := by
      intro i _
      have hΩpos : 0 < Ωn D K1 := Ωn_pos D (by omega) K1
      have hratio : (Ωn D (rem (L + 1) K1 i) / Ωn D K1) ^ 2 ≤ 1 := by
        have h01 : 0 ≤ Ωn D (rem (L + 1) K1 i) / Ωn D K1 :=
          div_nonneg (Ωn_pos D (by omega) _).le hΩpos.le
        have h11 : Ωn D (rem (L + 1) K1 i) / Ωn D K1 ≤ 1 :=
          (div_le_one hΩpos).mpr (hrem K1 i)
        calc (Ωn D (rem (L + 1) K1 i) / Ωn D K1) ^ 2
            ≤ 1 ^ 2 := pow_le_pow_left₀ h01 h11 2
          _ = 1 := one_pow 2
      have hmemb : mem D ins (rem (L + 1) K1 i) (f (L + 2)) ≤
          wmax / (((L + 1 : ℕ) : ℝ) * ωmin) * t :=
        mem_le D ins (n := L + 1) (by omega) _ (f (L + 2)) t ht (fun x => hK1 x)
          ωmin wmax hωpos hω hw
      have hmem0 : 0 ≤ mem D ins (rem (L + 1) K1 i) (f (L + 2)) :=
        mem_nonneg D ins _ _ (fun x => F.nonneg _ x)
      have hb0 : 0 ≤ wmax / (((L + 1 : ℕ) : ℝ) * ωmin) * t := by positivity
      refine mul_le_mul ?_ (hK0 _) (F.nonneg _ _) (by positivity)
      exact mul_le_mul hratio hmemb hmem0 zero_le_one
    have hsum := Finset.sum_le_card_nsmul _ _ _ hterm
    rw [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hsum
    calc t = memClosureX D ins rem L f K1 / (1 + memClosureX D ins rem L f K1) := hcl
      _ ≤ memClosureX D ins rem L f K1 := div_le_self hXnn (by linarith)
      _ ≤ (1 / ((L : ℝ) + 1)) *
            (((L + 2 : ℕ) : ℝ) * (1 * (wmax / (((L + 1 : ℕ) : ℝ) * ωmin) * t) * s)) := by
          unfold memClosureX
          exact mul_le_mul_of_nonneg_left hsum (by positivity)
      _ = (((L : ℝ) + 2) / ((L : ℝ) + 1)) *
            (wmax / (((L + 1 : ℕ) : ℝ) * ωmin) * t) * s := by
          push_cast; ring
  -- combine: t ≤ A·s ≤ A·r·t with A·r < 1
  have hs0 : t = 0 := by
    refine le_antisymm ?_ ht
    by_contra hpos
    have htpos : 0 < t := not_le.mp hpos
    have hA : 0 ≤ (((L : ℝ) + 2) / ((L : ℝ) + 1)) *
        (wmax / (((L + 1 : ℕ) : ℝ) * ωmin) * t) := by positivity
    have h1 : t ≤ (((L : ℝ) + 2) / ((L : ℝ) + 1)) *
        (wmax / (((L + 1 : ℕ) : ℝ) * ωmin) * t) *
        ((wmax / ωmin) ^ (L + 1) / (((L + 1).factorial : ℕ) : ℝ) * t) :=
      hX.trans (mul_le_mul_of_nonneg_left h0 hA)
    -- divide by t > 0 and use t ≤ 1
    have h2 : 1 ≤ (((L : ℝ) + 2) / ((L : ℝ) + 1)) *
        (wmax / (((L + 1 : ℕ) : ℝ) * ωmin)) *
        ((wmax / ωmin) ^ (L + 1) / (((L + 1).factorial : ℕ) : ℝ)) := by
      have hB : 0 ≤ (((L : ℝ) + 2) / ((L : ℝ) + 1)) *
          (wmax / (((L + 1 : ℕ) : ℝ) * ωmin)) *
          ((wmax / ωmin) ^ (L + 1) / (((L + 1).factorial : ℕ) : ℝ)) := by positivity
      -- h1 says t ≤ B * t * t ≤ B * t  (as t ≤ 1), hence 1 ≤ B
      have h1' : t ≤ (((L : ℝ) + 2) / ((L : ℝ) + 1)) *
          (wmax / (((L + 1 : ℕ) : ℝ) * ωmin)) *
          ((wmax / ωmin) ^ (L + 1) / (((L + 1).factorial : ℕ) : ℝ)) * t * t := by
        calc t ≤ _ := h1
          _ = _ := by ring
      have h1'' : t ≤ (((L : ℝ) + 2) / ((L : ℝ) + 1)) *
          (wmax / (((L + 1 : ℕ) : ℝ) * ωmin)) *
          ((wmax / ωmin) ^ (L + 1) / (((L + 1).factorial : ℕ) : ℝ)) * t := by
        calc t ≤ _ := h1'
          _ ≤ _ := by
            apply mul_le_of_le_one_right (mul_nonneg hB ht) ht1
      by_contra hlt
      have hlt' := not_le.mp hlt
      have := mul_lt_mul_of_pos_right hlt' htpos
      rw [one_mul] at this
      linarith
    -- clear denominators: (L+1)^2 (L+1)! ≤ (L+2) C^(L+2)
    have hden : (0 : ℝ) < ((L : ℝ) + 1) ^ 2 * (((L + 1).factorial : ℕ) : ℝ) := by positivity
    have hrw : (((L : ℝ) + 2) / ((L : ℝ) + 1)) *
        (wmax / (((L + 1 : ℕ) : ℝ) * ωmin)) *
        ((wmax / ωmin) ^ (L + 1) / (((L + 1).factorial : ℕ) : ℝ))
        = (((L : ℝ) + 2) * (wmax / ωmin) ^ (L + 2)) /
          (((L : ℝ) + 1) ^ 2 * (((L + 1).factorial : ℕ) : ℝ)) := by
      have hω' : ωmin ≠ 0 := ne_of_gt hωpos
      have hf : (((L + 1).factorial : ℕ) : ℝ) ≠ 0 := hfpos.ne'
      have hL' : ((L : ℝ) + 1) ≠ 0 := hL1.ne'
      rw [pow_succ (wmax / ωmin) (L + 1)]
      generalize (wmax / ωmin) ^ (L + 1) = A
      push_cast
      field_simp
    rw [hrw, le_div_iff₀ hden, one_mul] at h2
    -- h2 : (L+1)^2 (L+1)! ≤ (L+2) C^(L+2);  hC : 2 C^(L+2) < (L+1)(L+1)!
    have hP : 0 ≤ (wmax / ωmin) ^ (L + 2) := pow_nonneg hC0 _
    have hL0 : (0 : ℝ) ≤ L := Nat.cast_nonneg _
    nlinarith [mul_nonneg hL0 hP, mul_lt_mul_of_pos_left hC hL1]
  -- every level is then squeezed to zero
  intro n hn1 hn2 K
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  have hb := hchain k (by omega) K
  rw [hs0] at hb
  simp only [bnd, mul_zero] at hb
  exact le_antisymm hb (F.nonneg _ _)

/-- Specialisation to the canonical removal map, where the frequency hypothesis is a theorem. -/
theorem no_nontrivial_fixed_point_memclosure_canonical (hM : 0 < M) (ins : Insertion M) (L : ℕ)
    (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ) (F : MemFixedPoint D ins (canonicalRem M) L f)
    (ωmin wmax : ℝ) (hωpos : 0 < ωmin) (hω : ∀ k, ωmin ≤ D.rate k)
    (hw : ∀ k, wsum D k ≤ wmax)
    (hC : 2 * (wmax / ωmin) ^ (L + 2) < ((L : ℝ) + 1) * (((L + 1).factorial : ℕ) : ℝ)) :
    ∀ n, 1 ≤ n → n ≤ L + 2 → ∀ K : Fin n → Fin M, f n K = 0 :=
  no_nontrivial_fixed_point_memclosure D hM ins (canonicalRem M) L f F
    (fun K i => Ωn_canonicalRem_le D K i) ωmin wmax hωpos hω hw hC

end

end GMCT

#print axioms GMCT.no_nontrivial_fixed_point_memclosure_canonical
