import numpy as np


def trace_fidelity(rho, sigma):
    """
    Calculate the trace fidelity F(rho, sigma) = Tr{sqr{sqrt{rho} @ sigma @ sqrt{rho}}}
    """
    S_rho = np.linalg.eigvalsh(rho)
    S_rho[np.isclose(S_rho, 0)] = 0.0
    S_sigma = np.linalg.eigvalsh(sigma)
    S_sigma[np.isclose(S_sigma, 0)] = 0.0

    if np.sum(S_rho < 0) > 0:
        raise ValueError(f"Negative eigenvalues detected in S_rho: {S_rho}")
    if np.sum(S_sigma < 0) > 0:
        raise ValueError(f"Negative eigenvalues detected in S_sigma: {S_sigma}")
    return np.sum(np.sqrt(np.sqrt(S_rho) * (S_sigma * np.sqrt(S_rho))))