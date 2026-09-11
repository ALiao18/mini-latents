# mini-latents

From-scratch implementations of

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
