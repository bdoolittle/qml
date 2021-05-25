import numpy as np
from typing import List


def operator_2_norm(R):
    """
    Calculate the operator two norm.
    """
    return np.sqrt(np.trace(R.conjugate().transpose() @ R))


def shadow_bound(M: int, error: float, max_k: int, observables: List[np.ndarray]):
    """
    Calculate the shadow bound for the pauli measurement scheme.
    """
    shadow_norm = lambda op: np.linalg.norm(op, ord=np.inf) ** 2
    return int(np.ceil(np.log(M) * 4 ** max_k * max(shadow_norm(o) for o in observables) / error ** 2))
