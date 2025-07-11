# dpp_dist.py
import numpy as np

def dp_mv_distribution(mu, Sigma, lambda_, W0=1.0, T=30, 
                        n_wealth=200, n_actions=100, n_return_samples=1000):
    """
    Distribution-based Dynamic Programming for multi-period mean-variance portfolio optimization.
    
    Solves:
      max_{a_0,...,a_{T-1}} E[W_T] - (lambda_/2)*Var(W_T)
    under i.i.d. Gaussian returns with known (mu, Sigma).
    
    Parameters
    ----------
    mu : array_like, shape (N,)
        True mean return vector.
    Sigma : array_like, shape (N, N)
        True return covariance matrix.
    lambda_ : float
        Risk aversion coefficient.
    W0 : float
        Initial wealth (normalized to 1.0 by default).
    T : int
        Number of time periods (horizon).
    n_wealth : int
        Number of discretized wealth grid points.
    n_actions : int
        Number of candidate portfolio vectors sampled on the simplex.
    n_return_samples : int
        Number of Monte Carlo samples for return expectation per period.
    
    Returns
    -------
    policy : ndarray, shape (T, n_wealth, N)
        Optimal portfolio weights at each (time, wealth) grid point.
    wealth_grid : ndarray, shape (n_wealth,)
        Discretized grid of possible wealth levels.
    """
    N = len(mu)
    # Discretize wealth around [0.5*W0, 2.0*W0]
    wealth_grid = np.linspace(0.5*W0, 2.0*W0, n_wealth)
    
    # Pre-sample return realizations for each period
    return_samples = np.random.multivariate_normal(mu, Sigma, 
                                                   size=(T, n_return_samples))
    
    # Initialize value function V and policy
    V = np.zeros((T + 1, n_wealth))
    policy = np.zeros((T, n_wealth, N))
    
    # Terminal utility: W - (lambda/2)*W^2
    V[T] = wealth_grid - 0.5 * lambda_ * (wealth_grid ** 2)
    
    # Sample candidate portfolio weights on simplex
    action_grid = np.random.dirichlet(np.ones(N), size=n_actions)
    
    # Backward induction with expectation over distribution
    for t in reversed(range(T)):
        R_t = return_samples[t]  # shape: (n_return_samples, N)
        for i, w in enumerate(wealth_grid):
            best_val = -np.inf
            best_a = None
            for a in action_grid:
                # Compute next-period wealth for each sample
                W_next = w * (1 + R_t.dot(a))
                # Clip to grid bounds
                W_next = np.clip(W_next, wealth_grid[0], wealth_grid[-1])
                # Interpolate next value and average
                V_next = np.interp(W_next, wealth_grid, V[t+1])
                exp_val = V_next.mean()
                if exp_val > best_val:
                    best_val = exp_val
                    best_a = a
            V[t, i] = best_val
            policy[t, i] = best_a
    
    return policy, wealth_grid

# Example usage
if __name__ == "__main__":
    # Known distribution parameters
    mu = np.array([0.001, 0.0015, 0.0012])
    Sigma = np.array([
        [0.0001, 0.00002, 0.00001],
        [0.00002, 0.00015, 0.00003],
        [0.00001, 0.00003, 0.00012]
    ])
    lambda_ = 5.0
    W0 = 1.0
    
    policy, wealth_grid = dp_mv_distribution(mu, Sigma, lambda_, W0)
    #print("Policy shape:", policy.shape)   # (T, n_wealth, N)
    #print("Wealth grid:", wealth_grid)
    for t in range(30):
        print(policy[t, 0])
