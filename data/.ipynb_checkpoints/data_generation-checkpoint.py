# data_generation.py
import numpy as np

def generate_returns(T, N, mu, Sigma, seed=42, num_samples=1):
    """
    Generate num_samples sets of T days of returns for N assets from a multivariate Gaussian.

    Parameters
    ----------
    T : int
        Number of time periods (e.g. 30).
    N : int
        Number of assets.
    mu : array_like, shape (N,)
        Expected returns.
    Sigma : array_like, shape (N, N)
        Covariance matrix.
    seed : int
        Random seed for reproducibility.
    num_samples : int
        Number of independent samples (paths) to generate.

    Returns
    -------
    returns : np.ndarray, shape (num_samples, T, N) if num_samples > 1, else (T, N)
        Simulated return matrix or matrices.
    """
    rng = np.random.default_rng(seed)
    if num_samples == 1:
        return rng.multivariate_normal(mu, Sigma, size=T)
    else:
        return np.array([
            rng.multivariate_normal(mu, Sigma, size=T)
            for _ in range(num_samples)
        ])

# Example usage:
#T, N = 30, 3
#mu = np.array([0.001, 0.0015, 0.0012])
#Sigma = np.array([
#        [0.0001, 0.00002, 0.00001],
#        [0.00002, 0.00015, 0.00003],
#       [0.00001, 0.00003, 0.00012]
#    ])
#returns = generate_returns(T, N, mu, Sigma, num_samples=2)
#print("Generated returns:", returns)  # Should be (2, 30, 3)
