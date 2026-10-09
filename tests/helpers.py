"""Assertion helpers shared across the test modules.

Kept out of conftest.py so they can be imported normally -- conftest is a
pytest plugin module, not an import target.
"""

import numpy as np
from scipy.linalg import orth


def subspace_dist(A: np.ndarray, B: np.ndarray) -> float:
    """
    Distance between the column spaces of A and B, as ||P_A - P_B||_F.

    pPCA/FA loading matrices are only identified up to a right-multiplied
    orthogonal matrix, so W itself is never comparable across fits -- but the
    subspace it spans is. Ranges from 0 to sqrt(2k) for k columns.
    """
    PA = orth(A)
    PB = orth(B)
    return float(np.linalg.norm(PA @ PA.T - PB @ PB.T))
