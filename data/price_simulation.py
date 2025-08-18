from data_generation import generate_returns
import numpy as np

# === price_simulation.py ===

def simulate_price_paths(T, N, mu, Sigma, P0, seed=42, method='arithmetic', num_samples=1):
    """
    Generate T days of returns using generate_returns, then simulate price paths.
    
    Parameters
    ----------
    T : int
        Number of time periods (e.g., 30).
    N : int
        Number of assets.
    mu : array_like, shape (N,)
        Expected daily returns.
    Sigma : array_like, shape (N, N)
        Covariance matrix.
    P0 : array_like, shape (N,)
        Initial prices for each asset.
    seed : int
        Random seed for reproducibility.
    method : str, {'arithmetic', 'log'}
        'arithmetic': P_{t+1} = P_t * (1 + r_t)
        'log':        P_{t+1} = P_t * exp(r_t)
    
    Returns
    -------
    prices : np.ndarray, shape (T+1, N)
        Simulated price paths for each asset.
    returns : np.ndarray, shape (T, N)
        Generated return series.
    """
    # Generate returns
    returns = generate_returns(T, N, mu, Sigma, seed, num_samples=num_samples)
    if num_samples == 1:
        returns = returns[np.newaxis, ...]  # transform shape (1, T, N)
    prices = np.zeros((num_samples, T + 1, N))
    for i in range(num_samples):
        prices[i, 0] = P0
        if method == 'arithmetic':
            for t in range(T):
                prices[i, t+1] = prices[i, t] * (1 + returns[i, t])
        elif method == 'log':
            for t in range(T):
                prices[i, t+1] = prices[i, t] * np.exp(returns[i, t])
        else:
            raise ValueError("method must be 'arithmetic' or 'log'")
    return prices, returns

# Example usage:
if __name__ == "__main__":
    T, N = 30, 3
    mu = np.array([0.001, 0.0015, 0.0012])
    Sigma = np.array([
        [0.0001, 0.00002, 0.00001],
        [0.00002, 0.00015, 0.00003],
       [0.00001, 0.00003, 0.00012]
    ])
    P0 = np.array([100, 120, 80])
    
    prices, returns = simulate_price_paths(T, N, mu, Sigma, P0)
    print("Simulated Prices:\n", prices.shape)
    print("Generated Returns:\n", returns.shape)
