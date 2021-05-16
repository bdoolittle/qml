import numpy as np


def operator_2_norm(R):
    """
    Calculate the operator two norm
    """
    return np.sqrt(np.trace(R.conjugate().transpose() @ R))

