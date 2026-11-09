import LeanMct.ChainBound

/-!
# Chain bound with an arbitrary upper bound on the top unknown level

`ChainBound.lean` bounds the closure by `f_{N−1} ≤ 1`. Any pointwise upper bound `g ≥ f_{N−1}` can
be used instead, giving `f₁ ≤ T[g] f₁` with
`(P_g v)(K) = (1/N) Σᵢ v(kᵢ) g(K \ kᵢ)`. In practice `g = Φ^m(1)` is an iterate of the monotone
hierarchy map from the fully frozen state (every fixed point in `[0,1]` lies below it), and the
criterion `ρ(T[g]) < 1` certifies the liquid phase arbitrarily close to the critical point
(computer-assisted sharpening of the rigorous bound).
-/

open Finset

namespace GMCT

noncomputable section

variable {M : ℕ} (D : GridData M)

/-- Weighted closure linearisation `(P_g v)(K) = (1/N) Σᵢ v(kᵢ) g(rem K i)`. -/
def PclosG (rem : Removal M) (L : ℕ) (g : (Fin (L + 1) → Fin M) → ℝ) (v : Fin M → ℝ) :
    (Fin (L + 2) → Fin M) → ℝ :=
  fun K => (1 / ((L : ℝ) + 2)) * ∑ i, v (K i) * g (rem (L + 1) K i)

/-- The weighted chain operator `T[g] v = 𝓛₁ ⋯ 𝓛_{L+1} P_g v`. -/
def chainTG (ins : Insertion M) (rem : Removal M) (L : ℕ) (g : (Fin (L + 1) → Fin M) → ℝ)
    (v : Fin M → ℝ) : Fin M → ℝ :=
  fun k => descend D ins (L + 1) (PclosG rem L g v) (fun _ => k)

lemma PclosG_mono (rem : Removal M) (L : ℕ) (g : (Fin (L + 1) → Fin M) → ℝ) (hg : ∀ K, 0 ≤ g K)
    {v v' : Fin M → ℝ} (h : ∀ k, v k ≤ v' k) :
    ∀ K, PclosG (M := M) rem L g v K ≤ PclosG rem L g v' K := by
  intro K; unfold PclosG
  refine mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun i _ => ?_) (by positivity)
  exact mul_le_mul_of_nonneg_right (h _) (hg _)

lemma PclosG_smul (rem : Removal M) (L : ℕ) (g : (Fin (L + 1) → Fin M) → ℝ) (c : ℝ) (v : Fin M → ℝ) :
    PclosG (M := M) rem L g (fun k => c * v k) = fun K => c * PclosG rem L g v K := by
  funext K; unfold PclosG
  have h : ∑ i, c * v (K i) * g (rem (L + 1) K i) = c * ∑ i, v (K i) * g (rem (L + 1) K i) := by
    rw [Finset.mul_sum]; refine Finset.sum_congr rfl fun i _ => ?_; ring
  rw [h]; ring

/-- Chain inequality with the weight `g ≥ f_{L+1}`. -/
theorem chain_inequality_g (ins : Insertion M) (rem : Removal M) (L : ℕ)
    (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ) (F : MFFixedPoint D ins rem L f)
    (g : (Fin (L + 1) → Fin M) → ℝ) (hg : ∀ K, f (L + 1) K ≤ g K) :
    ∀ k, f 1 (fun _ => k) ≤ chainTG D ins rem L g (fun k => f 1 (fun _ => k)) k := by
  have hlev : ∀ n, 1 ≤ n → n ≤ L + 1 → ∀ K : Fin n → Fin M, f n K ≤ mem D ins K (f (n + 1)) := by
    intro n hn1 hn2 K
    rw [F.level n hn1 hn2 K]
    exact div_le_self (mem_nonneg D ins K _ (fun x => F.nonneg _ x))
      (by linarith [mem_nonneg D ins K _ (fun x => F.nonneg _ x)])
  have key : ∀ m, m ≤ L + 1 → ∀ x, f 1 x ≤ descend D ins m (f (m + 1)) x := by
    intro m
    induction m with
    | zero => intro _ x; exact le_rfl
    | succ m ih =>
      intro hm x
      have h1 := ih (by omega) x
      have h2 : ∀ K : Fin (m + 1) → Fin M, f (m + 1) K ≤ mem D ins K (f (m + 1 + 1)) :=
        fun K => hlev (m + 1) (by omega) (by omega) K
      exact h1.trans (descend_mono D ins m h2 x)
  have hclos : ∀ K : Fin (L + 2) → Fin M,
      f (L + 2) K ≤ PclosG rem L g (fun k => f 1 (fun _ => k)) K := by
    intro K
    rw [F.closure K]
    unfold PclosG
    refine mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun i _ => ?_) (by positivity)
    exact mul_le_mul_of_nonneg_left (hg _) (F.nonneg _ _)
  intro k
  exact (key (L + 1) le_rfl (fun _ => k)).trans (descend_mono D ins (L + 1) hclos (fun _ => k))

/-- **Certified-liquid criterion.** If `g ≥ f_{L+1}` pointwise with `g ≥ 0`, and some `v > 0`
satisfies `T[g] v ≤ θ v` with `θ < 1`, then the fixed point is trivial. -/
theorem no_nontrivial_fixed_point_chain_g (hM : 0 < M) (ins : Insertion M) (rem : Removal M)
    (L : ℕ) (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ) (F : MFFixedPoint D ins rem L f)
    (g : (Fin (L + 1) → Fin M) → ℝ) (hg0 : ∀ K, 0 ≤ g K) (hg : ∀ K, f (L + 1) K ≤ g K)
    (v : Fin M → ℝ) (hv : ∀ k, 0 < v k) (θ : ℝ) (hθ : θ < 1)
    (hT : ∀ k, chainTG D ins rem L g v k ≤ θ * v k) :
    ∀ n, 1 ≤ n → n ≤ L + 2 → ∀ K : Fin n → Fin M, f n K = 0 := by
  haveI : Nonempty (Fin M) := ⟨⟨0, hM⟩⟩
  obtain ⟨k0, hk0⟩ := Finite.exists_max (fun k : Fin M => f 1 (fun _ => k) / v k)
  set c := f 1 (fun _ => k0) / v k0 with hc
  have hc0 : 0 ≤ c := div_nonneg (F.nonneg _ _) (hv _).le
  have hfv : ∀ k, f 1 (fun _ => k) ≤ c * v k := by
    intro k; have := hk0 k; rwa [div_le_iff₀ (hv k)] at this
  have hchain := chain_inequality_g D ins rem L f F g hg
  have hmono : ∀ k, chainTG D ins rem L g (fun k => f 1 (fun _ => k)) k
      ≤ chainTG D ins rem L g (fun k => c * v k) k := by
    intro k; unfold chainTG
    exact descend_mono D ins (L + 1) (PclosG_mono rem L g hg0 hfv) _
  have hhom : ∀ k, chainTG D ins rem L g (fun k => c * v k) k = c * chainTG D ins rem L g v k := by
    intro k; unfold chainTG
    rw [PclosG_smul]
    exact descend_smul D ins (L + 1) c _ _
  have hk0' : f 1 (fun _ => k0) ≤ c * θ * v k0 := by
    calc f 1 (fun _ => k0) ≤ _ := hchain k0
      _ ≤ _ := hmono k0
      _ = c * chainTG D ins rem L g v k0 := hhom k0
      _ ≤ c * (θ * v k0) := mul_le_mul_of_nonneg_left (hT k0) hc0
      _ = c * θ * v k0 := by ring
  have hcv : c * v k0 = f 1 (fun _ => k0) := by rw [hc]; exact div_mul_cancel₀ _ (hv k0).ne'
  have hc_le : c * (1 - θ) * v k0 ≤ 0 := by nlinarith [hk0', hcv]
  have hc_zero : c = 0 := by
    have hpos : 0 < (1 - θ) * v k0 := mul_pos (by linarith) (hv k0)
    by_contra hne
    have hcpos : 0 < c := lt_of_le_of_ne hc0 (Ne.symm hne)
    have : 0 < c * (1 - θ) * v k0 := by rw [mul_assoc]; exact mul_pos hcpos hpos
    linarith
  have hf1 : ∀ k, f 1 (fun _ => k) = 0 := by
    intro k; have := hfv k; rw [hc_zero, zero_mul] at this
    exact le_antisymm this (F.nonneg _ _)
  have hf1' : ∀ K : Fin 1 → Fin M, f 1 K = 0 := by
    intro K; have := hf1 (K 0)
    have hK : K = fun _ => K 0 := by funext i; rw [Fin.fin_one_eq_zero i]
    rw [hK]; exact this
  have htop : ∀ K : Fin (L + 2) → Fin M, f (L + 2) K = 0 := by
    intro K; rw [F.closure K]; simp [hf1']
  have hdown : ∀ d, d ≤ L + 1 → ∀ K : Fin (L + 2 - d) → Fin M, f (L + 2 - d) K = 0 := by
    intro d
    induction d with
    | zero => intro _ K; simpa using htop K
    | succ d ih =>
      intro hd K
      have e : L + 2 - d = L + 2 - (d + 1) + 1 := by omega
      have hnext : ∀ K' : Fin (L + 2 - (d + 1) + 1) → Fin M, f (L + 2 - (d + 1) + 1) K' = 0 := by
        rw [← e]; exact ih (by omega)
      have hlev := F.level (L + 2 - (d + 1)) (by omega) (by omega) K
      have hzero : mem D ins K (f (L + 2 - (d + 1) + 1)) = 0 := by
        unfold mem; simp [hnext]
      rw [hlev, hzero]; simp
  intro n hn1 hn2 K
  have e : n = L + 2 - (L + 2 - n) := by omega
  have h := hdown (L + 2 - n) (by omega)
  rw [← e] at h
  exact h K

/-- The unweighted bound is the special case `g ≡ 1` (uses `f_{L+1} ≤ 1`). -/
theorem chainTG_one (ins : Insertion M) (rem : Removal M) (L : ℕ) (v : Fin M → ℝ) :
    chainTG D ins rem L (fun _ => 1) v = chainT D ins L v := by
  funext k; unfold chainTG chainT
  congr 1
  funext K; unfold PclosG Pclos
  simp

end

end GMCT

#print axioms GMCT.no_nontrivial_fixed_point_chain_g
