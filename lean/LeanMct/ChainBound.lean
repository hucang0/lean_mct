import LeanMct.GridTheorem

/-!
# The linear-chain (Collatz–Wielandt) bound

The level equations are linear in the next level: `fₙ ≤ 𝓛ₙ f_{n+1}` where `(𝓛ₙ g)(K) = 𝓜ₙ(K)[g]`
is a positive linear map. The correlator closure with `f_{N−1} ≤ 1` gives `f_N ≤ P f₁` with
`(P v)(K) = (1/N) Σᵢ v(kᵢ)`. Hence every fixed point satisfies `f₁ ≤ T f₁` with
`T = 𝓛₁ 𝓛₂ ⋯ 𝓛_{N−1} P`, a positive linear map on functions of one wavenumber.

**Theorem.** If some `v > 0` satisfies `T v ≤ θ v` with `θ < 1`, the fixed point is trivial.

This contains `GridTheorem.no_nontrivial_fixed_point` as the special case `v ≡ 1`,
`θ = C^{N−1}/(N−1)!`, and for `N = 2` the operator `T` is half the frozen linearisation of the
MCT map, whose spectral radius organises the empirical order dependence of the critical point.
-/

open Finset

namespace GMCT

noncomputable section

variable {M : ℕ} (D : GridData M)

/-- `(P v)(K) = (1/N) Σᵢ v(kᵢ)`, `v` a function of one wavenumber. -/
def Pclos (N : ℕ) (v : Fin M → ℝ) : (Fin N → Fin M) → ℝ :=
  fun K => (1 / (N : ℝ)) * ∑ i, v (K i)

/-- `descend m g = 𝓛₁ 𝓛₂ ⋯ 𝓛ₘ g` for `g` a function on `(m+1)`-tuples: the result is a function on
`1`-tuples. -/
def descend (ins : Insertion M) : (m : ℕ) → ((Fin (m + 1) → Fin M) → ℝ) → ((Fin 1 → Fin M) → ℝ)
  | 0, g => g
  | m + 1, g => descend ins m (fun K : Fin (m + 1) → Fin M => mem D ins K g)

/-- The chain operator `T v = 𝓛₁ ⋯ 𝓛_{L+1} P v` for closure level `N = L+2`, as a map on
functions of one wavenumber (`Fin M → ℝ`). -/
def chainT (ins : Insertion M) (L : ℕ) (v : Fin M → ℝ) : Fin M → ℝ :=
  fun k => descend D ins (L + 1) (Pclos (L + 2) v) (fun _ => k)

/-! ### Monotonicity and homogeneity of the level maps -/

lemma mem_mono (ins : Insertion M) {n : ℕ} (K : Fin n → Fin M)
    {g g' : (Fin (n + 1) → Fin M) → ℝ} (h : ∀ x, g x ≤ g' x) :
    mem D ins K g ≤ mem D ins K g' := by
  unfold mem
  refine Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left ?_
    (div_nonneg (D.rate_pos _).le (sq_nonneg _))
  exact Finset.sum_le_sum fun q _ => Finset.sum_le_sum fun p _ =>
    mul_le_mul_of_nonneg_left (h _) (D.W_nonneg _ _ _)

lemma mem_smul (ins : Insertion M) {n : ℕ} (K : Fin n → Fin M) (c : ℝ)
    (g : (Fin (n + 1) → Fin M) → ℝ) :
    mem D ins K (fun x => c * g x) = c * mem D ins K g := by
  unfold mem
  simp only [Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun q _ =>
    Finset.sum_congr rfl fun p _ => ?_
  ring

lemma descend_mono (ins : Insertion M) (m : ℕ) {g g' : (Fin (m + 1) → Fin M) → ℝ}
    (h : ∀ x, g x ≤ g' x) : ∀ x, descend D ins m g x ≤ descend D ins m g' x := by
  induction m with
  | zero => exact h
  | succ m ih =>
    intro x
    exact ih (fun K => mem_mono D ins K h) x

lemma descend_smul (ins : Insertion M) (m : ℕ) (c : ℝ) (g : (Fin (m + 1) → Fin M) → ℝ) :
    ∀ x, descend D ins m (fun y => c * g y) x = c * descend D ins m g x := by
  induction m with
  | zero => intro x; rfl
  | succ m ih =>
    intro x
    have : (fun K : Fin (m + 1) → Fin M => mem D ins K (fun y => c * g y))
        = fun K => c * mem D ins K g := by
      funext K; exact mem_smul D ins K c g
    show descend D ins m (fun K => mem D ins K (fun y => c * g y)) x = _
    rw [this]
    exact ih _ x

lemma Pclos_smul (N : ℕ) (c : ℝ) (v : Fin M → ℝ) :
    Pclos (M := M) N (fun k => c * v k) = fun K => c * Pclos N v K := by
  funext K; unfold Pclos
  have h : ∑ i, c * v (K i) = c * ∑ i, v (K i) := by rw [Finset.mul_sum]
  rw [h]; ring

lemma Pclos_mono (N : ℕ) {v v' : Fin M → ℝ} (h : ∀ k, v k ≤ v' k) :
    ∀ K, Pclos (M := M) N v K ≤ Pclos N v' K := by
  intro K; unfold Pclos
  exact mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun i _ => h _) (by positivity)

/-! ### The chain inequality `f₁ ≤ T f₁` -/

theorem chain_inequality (ins : Insertion M) (rem : Removal M) (L : ℕ)
    (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ) (F : MFFixedPoint D ins rem L f) :
    ∀ k, f 1 (fun _ => k) ≤ chainT D ins L (fun k => f 1 (fun _ => k)) k := by
  -- step 1: fₙ ≤ 𝓛ₙ f_{n+1} for 1 ≤ n ≤ L+1
  have hlev : ∀ n, 1 ≤ n → n ≤ L + 1 → ∀ K : Fin n → Fin M, f n K ≤ mem D ins K (f (n + 1)) := by
    intro n hn1 hn2 K
    rw [F.level n hn1 hn2 K]
    exact div_le_self (mem_nonneg D ins K _ (fun x => F.nonneg _ x)) (by linarith [mem_nonneg D ins K _ (fun x => F.nonneg _ x)])
  -- step 2: f₁ ≤ descend m (f (m+1)) for all m ≤ L+1
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
  -- step 3: f_{L+2} ≤ P f₁ (closure with f_{L+1} ≤ 1)
  have hclos : ∀ K : Fin (L + 2) → Fin M, f (L + 2) K ≤ Pclos (L + 2) (fun k => f 1 (fun _ => k)) K := by
    intro K
    rw [F.closure K]
    unfold Pclos
    push_cast
    refine mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun i _ => ?_) (by positivity)
    calc f 1 (fun _ => K i) * f (L + 1) (rem (L + 1) K i)
        ≤ f 1 (fun _ => K i) * 1 := mul_le_mul_of_nonneg_left (F.le_one _ _) (F.nonneg _ _)
      _ = f 1 (fun _ => K i) := mul_one _
  intro k
  have h := key (L + 1) le_rfl (fun _ => k)
  exact h.trans (descend_mono D ins (L + 1) hclos (fun _ => k))

/-! ### Collatz–Wielandt criterion -/

/-- **Chain bound.** If some positive `v` satisfies `T v ≤ θ v` with `θ < 1`, every mean-field
closed fixed point is trivial. -/
theorem no_nontrivial_fixed_point_chain (hM : 0 < M) (ins : Insertion M) (rem : Removal M) (L : ℕ)
    (f : ∀ n : ℕ, (Fin n → Fin M) → ℝ) (F : MFFixedPoint D ins rem L f)
    (v : Fin M → ℝ) (hv : ∀ k, 0 < v k) (θ : ℝ) (hθ : θ < 1)
    (hT : ∀ k, chainT D ins L v k ≤ θ * v k) :
    ∀ n, 1 ≤ n → n ≤ L + 2 → ∀ K : Fin n → Fin M, f n K = 0 := by
  -- c := max_k f₁(k)/v(k)
  haveI : Nonempty (Fin M) := ⟨⟨0, hM⟩⟩
  obtain ⟨k0, hk0⟩ := Finite.exists_max (fun k : Fin M => f 1 (fun _ => k) / v k)
  set c := f 1 (fun _ => k0) / v k0 with hc
  have hc0 : 0 ≤ c := div_nonneg (F.nonneg _ _) (hv _).le
  have hfv : ∀ k, f 1 (fun _ => k) ≤ c * v k := by
    intro k
    have := hk0 k
    rwa [div_le_iff₀ (hv k)] at this
  -- T is monotone and homogeneous, so f₁ ≤ T f₁ ≤ T (c v) = c T v ≤ c θ v
  have hchain := chain_inequality D ins rem L f F
  have hmono : ∀ k, chainT D ins L (fun k => f 1 (fun _ => k)) k ≤ chainT D ins L (fun k => c * v k) k := by
    intro k; unfold chainT
    exact descend_mono D ins (L + 1) (Pclos_mono (L + 2) hfv) _
  have hhom : ∀ k, chainT D ins L (fun k => c * v k) k = c * chainT D ins L v k := by
    intro k; unfold chainT
    rw [Pclos_smul]
    exact descend_smul D ins (L + 1) c _ _
  have hk0' : f 1 (fun _ => k0) ≤ c * θ * v k0 := by
    calc f 1 (fun _ => k0) ≤ _ := hchain k0
      _ ≤ _ := hmono k0
      _ = c * chainT D ins L v k0 := hhom k0
      _ ≤ c * (θ * v k0) := mul_le_mul_of_nonneg_left (hT k0) hc0
      _ = c * θ * v k0 := by ring
  -- so c v k0 ≤ c θ v k0, hence c (1-θ) ≤ 0, hence c = 0
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
  -- every level vanishes: descend from the closure level
  have hf1' : ∀ K : Fin 1 → Fin M, f 1 K = 0 := by
    intro K; have := hf1 (K 0)
    have hK : K = fun _ => K 0 := by funext i; rw [Fin.fin_one_eq_zero i]
    rw [hK]; exact this
  have htop : ∀ K : Fin (L + 2) → Fin M, f (L + 2) K = 0 := by
    intro K; rw [F.closure K]; simp [hf1']
  -- downward: f (L+2-d) = 0 for all d ≤ L+1
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

end

end GMCT

#print axioms GMCT.no_nontrivial_fixed_point_chain
