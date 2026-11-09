import LeanMct.MemoryClosure

/-!
# The two-level bound: `C < N − 1` suffices

A reviewer observed that the factorial chain is not needed for the sup-norm result. With
`sₙ = sup fₙ`, the top two levels alone give, for the correlator closure,

    s_{N−1} ≤ (C/(N−1)) s_N ≤ (C/(N−1)) s_{N−1},

so `C < N − 1` already forces `s_{N−1} = 0` and then every level vanishes. For the memory closure
the same two levels give `s_N ≤ (N/(N−1))(C/(N−1)) s_N s₁`, so `2C < N − 1` suffices.
These conditions are stronger than `C^{N−1} < (N−1)!` and `2C^N < (N−1)(N−1)!` (which need
`N ≳ eC`), and the proofs are shorter. The chain of levels remains the basis of the sharper
operator criterion in `ChainBound.lean`.
-/

open Finset

namespace GMCT

noncomputable section

variable {M : ℕ} (D : GridData M)

/-- **Two-level bound, correlator closure.** If `wmax/ωmin < L + 1` (i.e. `C < N − 1` with
`N = L + 2`), every mean-field closed fixed point with values in `[0,1]` is trivial. -/
theorem no_nontrivial_fixed_point_top (hM : 0 < M) (ins : Insertion M) (rem : Removal M) (L : ℕ)
    (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ) (F : MFFixedPoint D ins rem L f)
    (ωmin wmax : ℝ) (hωpos : 0 < ωmin) (hω : ∀ k, ωmin ≤ D.rate k)
    (hw : ∀ k, wsum D k ≤ wmax)
    (hC : wmax / ωmin < (L : ℝ) + 1) :
    ∀ n, 1 ≤ n → n ≤ L + 2 → ∀ K : Fin n → Fin M, f n K = 0 := by
  have hwmax : 0 ≤ wmax := le_trans (wsum_nonneg D ⟨0, hM⟩) (hw _)
  haveI : Nonempty (Fin (L + 1) → Fin M) := ⟨fun _ => ⟨0, hM⟩⟩
  obtain ⟨K1, hK1⟩ := Finite.exists_max (fun K : Fin (L + 1) → Fin M => f (L + 1) K)
  set t := f (L + 1) K1 with ht_def
  have ht : 0 ≤ t := F.nonneg _ _
  -- the closure level is bounded by t:  f_{L+2}(K) = (1/(L+2)) Σ f₁ f_{L+1} ≤ t
  have htop : ∀ K : Fin (L + 2) → Fin M, f (L + 2) K ≤ t := by
    intro K
    rw [F.closure K]
    have hterm : ∀ i ∈ (Finset.univ : Finset (Fin (L + 2))),
        f 1 (fun _ => K i) * f (L + 1) (rem (L + 1) K i) ≤ t := by
      intro i _
      calc f 1 (fun _ => K i) * f (L + 1) (rem (L + 1) K i)
          ≤ 1 * t := mul_le_mul (F.le_one _ _) (hK1 _) (F.nonneg _ _) zero_le_one
        _ = t := one_mul t
    have hsum := Finset.sum_le_card_nsmul _ _ _ hterm
    rw [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hsum
    calc (1 / ((L : ℝ) + 2)) * ∑ i, f 1 (fun _ => K i) * f (L + 1) (rem (L + 1) K i)
        ≤ (1 / ((L : ℝ) + 2)) * (((L + 2 : ℕ) : ℝ) * t) :=
          mul_le_mul_of_nonneg_left hsum (by positivity)
      _ = t := by push_cast; field_simp
  -- level L+1 at its maximiser:  t ≤ (C/(L+1)) t
  have hlev := F.level (L + 1) (by omega) (by omega) K1
  have hmem0 : 0 ≤ mem D ins K1 (f (L + 1 + 1)) := mem_nonneg D ins K1 _ (fun x => F.nonneg _ x)
  have hmem_le : mem D ins K1 (f (L + 1 + 1)) ≤ wmax / (((L + 1 : ℕ) : ℝ) * ωmin) * t :=
    mem_le D ins (n := L + 1) (by omega) K1 (f (L + 1 + 1)) t ht (fun x => htop x) ωmin wmax hωpos hω hw
  have hL1 : (0 : ℝ) < (L : ℝ) + 1 := by positivity
  have hr : wmax / (((L + 1 : ℕ) : ℝ) * ωmin) < 1 := by
    push_cast
    rw [div_lt_one (by positivity)]
    have := (div_lt_iff₀ hωpos).mp hC
    linarith
  have ht0 : t = 0 := by
    refine le_antisymm ?_ ht
    by_contra hpos
    have htpos : 0 < t := not_le.mp hpos
    have h1 : t ≤ wmax / (((L + 1 : ℕ) : ℝ) * ωmin) * t := by
      calc t = _ := hlev
        _ ≤ mem D ins K1 (f (L + 1 + 1)) := div_le_self hmem0 (by linarith)
        _ ≤ _ := hmem_le
    have := mul_lt_mul_of_pos_right hr htpos
    rw [one_mul] at this
    linarith
  -- everything vanishes: level L+1 and L+2 are zero, then descend with the chain at s = 0
  have hL1z : ∀ K : Fin (L + 1) → Fin M, f (L + 1) K = 0 := fun K =>
    le_antisymm (by rw [← ht0]; exact hK1 K) (F.nonneg _ _)
  have htop0 : ∀ K : Fin (L + 2) → Fin M, f (L + 2) K ≤ 0 := fun K => by rw [← ht0]; exact htop K
  have hchain := chain D ins L f F.nonneg F.level ωmin wmax hωpos hω hw hwmax 0 le_rfl htop0
  intro n hn1 hn2 K
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  have hb := hchain k (by omega) K
  simp only [bnd, mul_zero] at hb
  exact le_antisymm hb (F.nonneg _ _)

/-- **Two-level bound, memory closure.** If `2·wmax/ωmin < L + 1` (i.e. `2C < N_mem − 1` with
`N_mem = L + 2` the top retained correlator, or `2C < N − 2` in the SM Sec. S1 C convention
`N = L + 3` where `M_{N−1}` is factored), every fixed point of the memory-closed hierarchy with
values in `[0,1]` and a frequency-nonincreasing removal map is trivial. -/
theorem no_nontrivial_fixed_point_memclosure_top (hM : 0 < M) (ins : Insertion M) (rem : Removal M)
    (L : ℕ) (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ) (F : MemFixedPoint D ins rem L f)
    (hrem : ∀ (K : Fin (L + 2) → Fin M) i, Ωn D (rem (L + 1) K i) ≤ Ωn D K)
    (ωmin wmax : ℝ) (hωpos : 0 < ωmin) (hω : ∀ k, ωmin ≤ D.rate k)
    (hw : ∀ k, wsum D k ≤ wmax)
    (hC : 2 * (wmax / ωmin) < (L : ℝ) + 1) :
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
  have hs1 : s ≤ 1 := F.le_one _ _
  have ht : 0 ≤ t := F.nonneg _ _
  have hL1 : (0 : ℝ) < (L : ℝ) + 1 := by positivity
  -- closure level at its maximiser (same estimate as in MemoryClosure.lean)
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
        calc (Ωn D (rem (L + 1) K1 i) / Ωn D K1) ^ 2 ≤ 1 ^ 2 := pow_le_pow_left₀ h01 h11 2
          _ = 1 := one_pow 2
      have hmemb : mem D ins (rem (L + 1) K1 i) (f (L + 2)) ≤
          wmax / (((L + 1 : ℕ) : ℝ) * ωmin) * t :=
        mem_le D ins (n := L + 1) (by omega) _ (f (L + 2)) t ht (fun x => hK1 x)
          ωmin wmax hωpos hω hw
      have hmem0 : 0 ≤ mem D ins (rem (L + 1) K1 i) (f (L + 2)) :=
        mem_nonneg D ins _ _ (fun x => F.nonneg _ x)
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
  -- t ≤ ((L+2)/(L+1)) (C/(L+1)) t s ≤ (2C/(L+1)) t  <  t  unless t = 0
  have ht0 : t = 0 := by
    refine le_antisymm ?_ ht
    by_contra hpos
    have htpos : 0 < t := not_le.mp hpos
    have hA : 0 ≤ (((L : ℝ) + 2) / ((L : ℝ) + 1)) * (wmax / (((L + 1 : ℕ) : ℝ) * ωmin)) := by positivity
    have h1 : t ≤ (((L : ℝ) + 2) / ((L : ℝ) + 1)) * (wmax / (((L + 1 : ℕ) : ℝ) * ωmin)) * t := by
      calc t ≤ _ := hX
        _ = (((L : ℝ) + 2) / ((L : ℝ) + 1)) * (wmax / (((L + 1 : ℕ) : ℝ) * ωmin)) * t * s := by ring
        _ ≤ (((L : ℝ) + 2) / ((L : ℝ) + 1)) * (wmax / (((L + 1 : ℕ) : ℝ) * ωmin)) * t * 1 :=
          mul_le_mul_of_nonneg_left hs1 (mul_nonneg hA ht)
        _ = _ := mul_one _
    -- the coefficient is < 1: (L+2)/(L+1) * C/(L+1) ≤ 2C/(L+1) < 1
    have hcoef : (((L : ℝ) + 2) / ((L : ℝ) + 1)) * (wmax / (((L + 1 : ℕ) : ℝ) * ωmin)) < 1 := by
      push_cast
      have hω' : ωmin ≠ 0 := hωpos.ne'
      have hL' : (L : ℝ) + 1 ≠ 0 := hL1.ne'
      have hC' : wmax / ωmin < ((L : ℝ) + 1) / 2 := by linarith
      have hfrac : ((L : ℝ) + 2) / ((L : ℝ) + 1) ≤ 2 := by
        rw [div_le_iff₀ hL1]; linarith [Nat.cast_nonneg (α := ℝ) L]
      have hrewrite : wmax / (((L : ℝ) + 1) * ωmin) = (wmax / ωmin) / ((L : ℝ) + 1) := by
        field_simp
      rw [hrewrite]
      calc ((L : ℝ) + 2) / ((L : ℝ) + 1) * (wmax / ωmin / ((L : ℝ) + 1))
          ≤ 2 * (wmax / ωmin / ((L : ℝ) + 1)) :=
            mul_le_mul_of_nonneg_right hfrac (by positivity)
        _ < 2 * (((L : ℝ) + 1) / 2 / ((L : ℝ) + 1)) := by
            apply mul_lt_mul_of_pos_left _ (by norm_num)
            exact div_lt_div_of_pos_right hC' hL1
        _ = 1 := by field_simp
    have := mul_lt_mul_of_pos_right hcoef htpos
    rw [one_mul] at this
    linarith
  -- all levels vanish
  have htop0 : ∀ K : Fin (L + 2) → Fin M, f (L + 2) K ≤ 0 := fun K => by rw [← ht0]; exact hK1 K
  have hchain := chain D ins L f F.nonneg F.level ωmin wmax hωpos hω hw hwmax 0 le_rfl htop0
  intro n hn1 hn2 K
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  have hb := hchain k (by omega) K
  simp only [bnd, mul_zero] at hb
  exact le_antisymm hb (F.nonneg _ _)

/-- **Unclosed infinite-hierarchy factorial decay bound.** For any bounded solution `0 ≤ f_n ≤ 1`
of the unclosed stationary hierarchy (satisfying the level relation for all `n ≥ 1`), iterating
the `1/n` dilution `m + 1` steps down from level `k + m + 2` bounds level `n = k + 1` by
`bnd (wmax / ωmin) 1 (k + m) k = C^{m+1} k! / (k + m + 1)!`. -/
theorem infinite_hierarchy_bound (hM : 0 < M) (ins : Insertion M)
    (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ)
    (hnn : ∀ n K, 0 ≤ f n K) (hle1 : ∀ n K, f n K ≤ 1)
    (hlevel : ∀ n, 1 ≤ n → ∀ K : Fin n → Fin M,
      f n K = mem D ins K (f (n + 1)) / (1 + mem D ins K (f (n + 1))))
    (ωmin wmax : ℝ) (hωpos : 0 < ωmin) (hω : ∀ k, ωmin ≤ D.rate k)
    (hw : ∀ k, wsum D k ≤ wmax) (k m : ℕ) (K : Fin (k + 1) → Fin M) :
    f (k + 1) K ≤ bnd (wmax / ωmin) 1 (k + m) k := by
  have hwmax : 0 ≤ wmax := le_trans (wsum_nonneg D ⟨0, hM⟩) (hw _)
  exact chain D ins (k + m) f hnn (fun n hn1 _ K' => hlevel n hn1 K')
    ωmin wmax hωpos hω hw hwmax 1 zero_le_one (fun K' => hle1 _ K') k (by omega) K

end

end GMCT

#print axioms GMCT.no_nontrivial_fixed_point_top
#print axioms GMCT.no_nontrivial_fixed_point_memclosure_top
#print axioms GMCT.infinite_hierarchy_bound
