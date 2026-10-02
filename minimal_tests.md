# Minimal mathematical test suite

The smallest set of tests that pins down the mathematics of PCA / pPCA / FA. Each test checks one property, against an
**independent** calculation (SciPy, scikit-learn, a closed form, or a formula derived a different way) or against an
**invariant** of the model. None of them compares two copies of the same code path.

Notation (current code): data `X (N, d)`, loadings `W (d, k)`, noise `psi (d,)`, implied covariance
`C = W Wᵀ + diag(psi)`, posterior `Ez (N, k)`, `M = (I + Wᵀ Ψ⁻¹ W)⁻¹`, `sum_Ezz = N·M + EzᵀEz`.
After the rename to latents names: `X → Y`, `W → C`, `k → x_dim`, `d → y_dim`.

## Principles

1. **Compare what is identified.**
   - W is only defined up to rotation (W R gives the same model), so compare the **subspace** (`subspace_dist`) or the **rotation-invariant covariance** W Wᵀ + Ψ, never the elements of W.
   - PCA components are only defined up to sign, so compare `|cos|`.
2. **One reason per tolerance.**
   - **Exact identities** (the same number computed two ways): numpy's default `assert_allclose` (`rtol=1e-7`).
   - **EM fixed points**: near the optimum the ll is quadratic in the parameters. Stopping at relative ll error ≈ `tol` therefore leaves parameter error ≈ `√tol`. Fit with `tol=1e-12` and allow ll `1e-10` and parameters `1e-5`.
   - **Recovery from simulated data**: the error shrinks like `1/√N`. State the N and the bound together.
3. **Fit through `fit()`**, not a hand-rolled loop (R5), except for the primitives in section A.

## Fixtures (3, all seeded)

| Name | Generative model | Used for |
|---|---|---|
| `iso` | N=500, d=8, k=3, Ψ = 0.25·I | pPCA closed form and recovery |
| `aniso` | N=2000, d=8, k=3, ψⱼ ∈ [0.25, 2.25] | FA vs sklearn, scale equivariance |
| `params` | random W (6,3), ψ ∈ [0.3, 1.2], X ~ N(0,1) (40,6) | the EM primitives, no fitting |

---

## A. EM primitives, checked against independent formulas

**A1. Log-likelihood equals SciPy's dense Gaussian log-density.**
Our version uses Woodbury and the determinant lemma; SciPy factors the dense (d,d) covariance. Two different routes to the same number.
```
ll_ref = scipy.stats.multivariate_normal(0, W Wᵀ + diag(psi)).logpdf(X).sum()
assert_allclose(log_likelihood(X, W, psi), ll_ref)
```

**A2. The E-step equals Gaussian conditioning on the joint distribution.**
[z; x] ~ N(0, [[I, Wᵀ], [W, C]]), so E[z|x] = Wᵀ C⁻¹ x and Cov[z|x] = I − Wᵀ C⁻¹ W.
This uses the d×d C, not our k×k M, so a misplaced Ψ⁻¹ or a transpose shows up.
```
C        = W Wᵀ + diag(psi)
Ez_ref   = X @ solve(C, W)                 # rows: Wᵀ C⁻¹ xₙ
cov_ref  = I_k − Wᵀ solve(C, W)
Ez, sum_Ezz = e_step(X, W, psi)
assert_allclose(Ez, Ez_ref)
assert_allclose(sum_Ezz, N·cov_ref + Ez_refᵀ Ez_ref)   # posterior second moment
```

**A3. The posterior second moment is a valid second moment.**
Σₙ E[zₙzₙᵀ] is symmetric, and `sum_Ezz − EzᵀEz = N·Cov` is positive definite.
```
assert_allclose(sum_Ezz, sum_Ezzᵀ)
assert eigvalsh(sum_Ezz − Ezᵀ Ez).min() > 0
```

**A4. The W update solves the normal equations it claims to solve.**
```
W_new = m_step_W(X, Ez, sum_Ezz)
assert_allclose(W_new @ sum_Ezz, Xᵀ Ez)
```

**A5. The noise updates equal the expected residual, written out per sample.**
ψⱼ = (1/N) Σₙ E[(xₙⱼ − wⱼᵀzₙ)²], with E[zₙzₙᵀ] = M + EzₙEzₙᵀ. This is the textbook expectation, not our vectorized algebra. The isotropic update is the mean over features.
```
expected = mean over n of: xₙ² − 2 xₙ ⊙ (W Ezₙ) + diag(W (M + Ezₙ Ezₙᵀ) Wᵀ)
assert_allclose(AnisotropicNoise.m_step(...).psi, expected)
assert_allclose(IsotropicNoise.m_step(...).psi,   expected.mean())
```

## B. Closed forms and independent libraries

**B1. PCA matches scikit-learn.**
Same directions (up to sign); same variances after the ddof rescale; σ² is the mean of the discarded eigenvalues.
```
ours, sk = PCA(k).fit(X), sklearn.PCA(k).fit(X)
assert_allclose(|sum(ours.components_ ⊙ sk.components_ᵀ, axis=0)|, 1)        # |cos| = 1
assert_allclose(ours.explained_variance_ · N/(N−1), sk.explained_variance_)   # ddof=0 vs 1
eig = eigvalsh(cov(X, ddof=0))[::-1]
assert_allclose(ours.noise_variance_, eig[k:].mean())
```

**B2. pPCA's closed form matches sklearn's probabilistic-PCA covariance.**
sklearn's `PCA.get_covariance()` is the Tipping–Bishop covariance (ddof=1).
```
m = pPCA(k).fit(X)                                  # closed form
C = m.components_ m.components_ᵀ + psi·I
assert ‖C·N/(N−1) − sklearn.PCA(k).fit(X).get_covariance()‖ / ‖·‖ < 1e-10
```

**B3. FA matches sklearn's FactorAnalysis** (rotation-invariant covariance, ll, uniquenesses).
Use the exact SVD (`svd_method="lapack"`) and a tight `tol`, or sklearn stops early.
```
ours = FA(k).fit(X, tol=1e-12, max_iter=100_000)
sk   = FactorAnalysis(k, tol=1e-11, max_iter=10_000, svd_method="lapack").fit(X)
assert ‖C_ours − sk.get_covariance()‖ / ‖·‖ < 1e-5        # √tol
assert_allclose(ours.log_likelihood_, sk.score(X)·N, rtol=1e-10)
assert_allclose(ours.psi, sk.noise_variance_, rtol=1e-4)  # each ψⱼ is less well identified than C
```

## C. EM behavior, through `fit()`

**C1. EM never decreases the log-likelihood.**
```
m = FA(k).fit(X, tol=1e-12)
assert all(diff(m.ll_history_) >= −1e-12·|ll|)   # rounding only
assert not m.flags.decreasing_ll
```

**C2. EM from a random start reaches pPCA's closed form, from below.**
The closed form is the global maximum, and pPCA has no other local maxima (Tipping & Bishop).
```
closed = pPCA(k).fit(X)
em     = pPCA(k) with _init_W → (random W0, random σ²₀); fit(method="em", tol=1e-12)
assert em.flags.converged and em.n_iter_ > 2
assert em.log_likelihood_ <= closed.log_likelihood_ + 1e-12·|ll|
assert_allclose(em.log_likelihood_, closed.log_likelihood_, rtol=1e-10)
assert ‖C_em − C_closed‖_max / ‖C_closed‖_max < 1e-4     # √tol
```
(Exists: `test_em_from_random_start_reaches_the_closed_form`.)

**C3. The reported ll describes the returned parameters, after an ordinary early stop** (R5).
```
m = FA(k).fit(X)                                   # default tol, stops early
assert m.flags.converged and m.n_iter_ < max_iter
assert m.log_likelihood_ == m.ll_history_[-1]
assert_allclose(m.log_likelihood_, log_likelihood(X − m.mean_, m.components_, m.psi))
```

**C4. The stopping rule on synthetic sequences, with no model** (R4).
Needs the rule extracted into a pure function.
```
rule(ll_curr, ll_old, ll_base, tol) on:
  gain 1.0,   total 10            → continue
  gain 1e-9,  total 10, tol 1e-8  → converged
  ll_curr < ll_old                → decreased (never converged)
```

## D. Invariances and recovery

**D1. Loadings recover the true subspace, not the true matrix, and the error shrinks like 1/√N.**
A fixed threshold is fragile: measured on the fixtures, pPCA gives 0.094 (N=500) against the current bound of 0.1, and FA gives 0.26 (N=2000).
The consistency check is sturdier: 4× the data should roughly halve the error.
```
for model, gen in [(pPCA, iso_generator), (FA, aniso_generator)]:
    ratios = []
    for seed in range(5):                                  # one seed is too noisy (measured up to 1.01)
        e1 = subspace_dist(model(k).fit(gen(seed, N)).components_,  W_true)
        e4 = subspace_dist(model(k).fit(gen(seed, 4N)).components_, W_true)
        ratios.append(e4 / e1)
    assert mean(ratios) < 0.75       # 1/√4 = 0.5 expected; measured 0.51 for both models
```

**D2. FA is equivariant to per-feature rescaling; pPCA is not.**
This is the property that separates the two models. With X → X·diag(c), FA's covariance must become D C D.
```
c = [0.5, 2, 1, 3, 0.25, 1.5, 4, 0.75];  D = diag(c)
assert ‖C_FA(X·D) − D C_FA(X) D‖ / ‖·‖ < 1e-5          # √tol: each fit stops at its own iteration
assert ‖C_pPCA(X·D) − D C_pPCA(X) D‖ / ‖·‖ > 1e-2      # the contrast case
```

**D3. Rotating W does not change the model.**
This is the reason every other test compares subspaces or covariances.
```
R = qr(randn(k, k)).Q
assert_allclose(log_likelihood(X, W @ R, psi), log_likelihood(X, W, psi))
assert_allclose((W R)(W R)ᵀ, W Wᵀ)
```

---

## Coverage

| Area | Tests |
|---|---|
| Log-likelihood formula | A1, D3 |
| E-step: posterior mean, covariance, second moment | A2, A3 |
| M-steps | A4, A5 |
| PCA / pPCA closed form | B1, B2, C2 |
| FA fixed point | B3 |
| EM contract and stopping rule | C1, C3, C4 |
| Identifiability and model semantics | D1, D2, D3 |

That's 15 tests. Everything else in the current suite (shapes, `fit` returns `self`, repeated fits giving identical results, optional-argument equivalence) is API hygiene. Keep a few in one `test_api.py`, but they are not part of the mathematical core.
