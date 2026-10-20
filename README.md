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
Z = pca.infer_latents(X)

ppca = pPCA(3).fit(X)
fa = FA(3).fit(X, max_iter=500, tol=1e-8)
```

## Running the tests

```bash
uv sync --group dev
uv run pytest
```

```bash
uv run pytest tests/test_fa.py -v
uv run pytest -k monotone   # run tests matching a name
```

The suite reads the package straight from `src/` (`pythonpath = ["src"]` in
`pyproject.toml`), so it tests the working tree rather than an installed copy —
no reinstall needed after an edit.