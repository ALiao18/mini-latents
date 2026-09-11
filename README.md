# mini-latents

From-scratch implementations of three linear latent variable models:

| Model | Noise covariance `Ψ` | Fitted by |
|---|---|---|
| `PCA` | — | eigendecomposition of the sample covariance |
| `pPCA` | `σ²I` (isotropic) | EM |
| `FA` | `diag(ψ₁…ψ_d)` (per-feature) | EM |

pPCA and FA share one EM loop (`em_core.py`) and differ only in the noise model
plugged into it (`noise_model.py`).

```python
from mini_latents.pca import PCA
from mini_latents.ppca_fa import pPCA, FA

pca = PCA(n_components=3).fit(X)
Z = pca.transform(X)

ppca = pPCA(3).fit(X)
fa = FA(3).fit(X, max_iter=500, tol=1e-8)
```

## Running the tests

```bash
uv sync --group dev
uv run pytest
```

Expect **101 passed** in about 9 seconds. Useful variations:

```bash
uv run pytest tests/test_fa.py -v
uv run pytest -k monotone   # run tests matching a name
```

The suite reads the package straight from `src/` (`pythonpath = ["src"]` in
`pyproject.toml`), so it tests the working tree rather than an installed copy —
no reinstall needed after an edit.

### Layout

| File | Covers |
|---|---|
| `tests/conftest.py` | seeded synthetic datasets (isotropic, heteroscedastic, near-noiseless) |
| `tests/helpers.py` | `subspace_dist` and a hand-rolled EM log-likelihood trace |
| `tests/test_em_core.py` | the shared primitives: `_sym_inv_logdet`, `e_step`, `m_step_W`, `log_likelihood` |
| `tests/test_noise_model.py` | both M-steps, cross-checked against each other and against naive loops |
| `tests/test_pca.py` | orthonormality, explained variance, reconstruction optimality |
| `tests/test_ppca.py` | the Tipping–Bishop closed form, EM monotonicity, the σ→0 limit |
| `tests/test_fa.py` | agreement with `sklearn.FactorAnalysis`, per-feature scale equivariance |
| `tests/test_api.py` | contracts shared by all three models |

Every assertion is either an analytic invariant (a closed form, monotonicity, a
reconstruction-error identity) or a comparison against an independent oracle
(scikit-learn, SciPy, or a slow reference loop) — nothing is a snapshot of
current output.
## EM convergence

`pPCA.fit` and `FA.fit` stop when an iteration improves the **per-sample**
log-likelihood by less than `tol` (default `1e-6`). The per-sample
normalisation matters: as a raw total, the same `tol` is coarse on a small
dataset and, on a large one, can fall below the float64 resolution of the
log-likelihood itself — at which point EM is comparing rounding noise and the
iteration it stops on varies between machines.

Pass a negative `tol` to disable early stopping and always run `max_iter`
iterations. The tests that need EM to sit exactly at the optimum use that, so
they converge identically everywhere.

FA converges more slowly than pPCA and often wants `max_iter` above the
default of 100.

## Attributes after `fit`

| Attribute | PCA | pPCA / FA |
|---|---|---|
| `components_` | `(d, k)`, orthonormal columns | `(d, k)` loading matrix `W` |
| `explained_variance_` | top-k eigenvalues (`ddof=0`) | — |
| `noise_variance_` | mean discarded eigenvalue | — |
| `noise_cov_` | `noise_variance_ * I` | `noise_model.as_matrix()` |
| `cov_` | full sample covariance (`ddof=0`) | — |
| `n_iter_`, `log_likelihood_` | — | EM diagnostics |

`PCA` builds its covariance with `ddof=0` so that its eigenvalues line up with
pPCA's EM estimates — `PCA(k).noise_variance_` equals the `sigma^2` that
`pPCA(k)` converges to on the same data. `sklearn.decomposition.PCA` uses
`ddof=1`, so its `explained_variance_` is larger by a factor of `N/(N-1)`. Both
conventions are pinned by tests.
