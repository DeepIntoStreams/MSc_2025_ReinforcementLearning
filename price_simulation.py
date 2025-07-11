from data_generation import generate_returns
import numpy as np

# === price_simulation.py ===

def simulate_price_paths(T, N, mu, Sigma, P0, seed=42, method='arithmetic'):
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
    returns = generate_returns(T, N, mu, Sigma, seed)
    
    # Initialize price array
    prices = np.zeros((T + 1, N))
    prices[0] = P0
    
    # Simulate prices
    if method == 'arithmetic':
        for t in range(T):
            prices[t+1] = prices[t] * (1 + returns[t])
    elif method == 'log':
        for t in range(T):
            prices[t+1] = prices[t] * np.exp(returns[t])
    else:
        raise ValueError("method must be 'arithmetic' or 'log'")
    
    return prices, returns

# Example usage:
if __name__ == "__main__":
    T, N = 30, 5
    mu = np.array([0.001, 0.0012, 0.0008, 0.0015, 0.001])
    Sigma = np.array([
        [0.0001, 0.00002, 0.000015, 0.00001, 0.00002],
        [0.00002, 0.00012, 0.000025, 0.000015, 0.00001],
        [0.000015, 0.000025, 0.00009, 0.00002, 0.000015],
        [0.00001, 0.000015, 0.00002, 0.00011, 0.000017],
        [0.00002, 0.00001, 0.000015, 0.000017, 0.00013]
    ])
    P0 = np.array([100, 120, 80, 95, 110])
    
    prices, returns = simulate_price_paths(T, N, mu, Sigma, P0)
    print("Simulated Prices:\n", prices)
    print("Generated Returns:\n", returns)
