"""Assertion helpers shared across the test modules.

Kept out of conftest.py so they can be imported normally -- conftest is a
pytest plugin module, not an import target.
"""
import numpy as np
from scipy.linalg import orth

from mini_latents.em_core import e_step, m_step_W, log_likelihood


def subspace_dist(A: np.ndarray, B: np.ndarray) -> float:
    '''
    Distance between the column spaces of A and B, as ||P_A - P_B||_F.

    pPCA/FA loading matrices are only identified up to a right-multiplied
    orthogonal matrix, so W itself is never comparable across fits -- but the
    subspace it spans is. Ranges from 0 to sqrt(2k) for k columns.
    '''
    PA = orth(A)
    PB = orth(B)
    return float(np.linalg.norm(PA @ PA.T - PB @ PB.T))


def em_ll_trace(noise_model, Xc: np.ndarray, W0: np.ndarray, n_iter: int) -> np.ndarray:
    '''
    Run n_iter EM iterations by hand, returning the log-likelihood after each.

    fit() only reports the final log-likelihood, so a monotonicity test has to
    drive the E/M primitives itself.
    '''
    noise_model.initialize(Xc)
    W = W0
    lls = []
    for _ in range(n_iter):
        Ez, Ezz = e_step(Xc, W, noise_model.noise_as_vec())
        W = m_step_W(Xc, Ez, Ezz)
        noise_model.m_step(Xc, W, Ez, Ezz)
        lls.append(log_likelihood(Xc, W, noise_model.noise_as_vec()))
    return np.array(lls)
