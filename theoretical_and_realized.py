# theoretical_and_realized.py
import numpy as np

def compute_theoretical_and_realized_returns(policy, wealth_grid, mu, test_returns, W0=1.0):
    """
    Compute theoretical expected return and realized return on a test path
    using the output of the distribution-based DP policy.

    Parameters
    ----------
    policy : np.ndarray, shape (T, n_wealth, N)
        DP policy table mapping (time, wealth_idx) -> weight vector.
    wealth_grid : np.ndarray, shape (n_wealth,)
        Discretized wealth levels used in DP.
    mu : np.ndarray, shape (N,)
        True mean return vector for the assets.
    test_returns : np.ndarray, shape (T, N)
        A simulated test path of N asset returns over T periods.
    W0 : float
        Initial wealth.

    Returns
    -------
    theoretical_return : float
        Theoretical expected return over T periods: (1 + w*·mu)^T - 1.
    realized_return : float
        Realized return on the provided test path using the DP policy.
    """
    T, n_wealth, N = policy.shape

    # Find the index in wealth_grid closest to W0
    idx0 = np.argmin(np.abs(wealth_grid - W0))
    # Extract the constant weight vector at t=0 (should be same each period)
    w_star = policy[0, idx0]

    # 1) Theoretical expected return
    per_period_exp_ret = w_star.dot(mu)
    expected_terminal_wealth = (1 + per_period_exp_ret) ** T
    theoretical_return = expected_terminal_wealth - 1

    # 2) Realized return on the test path
    W = W0
    for t in range(T):
        # For a dynamic policy, use:
        # idx = np.argmin(np.abs(wealth_grid - W))
        # w_t = policy[t, idx]
        # Here w_star is constant, but we show the general approach:
        idx = np.argmin(np.abs(wealth_grid - W))
        w_t = policy[t, idx]
        r_t = test_returns[t]
        W *= (1 + w_t.dot(r_t))

    realized_return = W - W0

    return theoretical_return, realized_return

# Example usage:
if __name__ == "__main__":
    # Suppose you have:
    # policy, wealth_grid = dp_mv_distribution(mu, Sigma, lambda_, W0)
    # and simulated test_returns of shape (30, N)
    mu = np.array([0.001, 0.0015, 0.0012])
    policy = np.random.rand(30, 200, 3)  # placeholder
    # normalize policy rows to sum to 1
    policy /= policy.sum(axis=2, keepdims=True)
    wealth_grid = np.linspace(0.5, 2.0, 200)
    test_returns = np.random.multivariate_normal(mu, np.eye(3)*0.0001, size=30)

    thr, rlr = compute_theoretical_and_realized_returns(
        policy, wealth_grid, mu, test_returns, W0=1.0
    )
    print(f"Theoretical expected return: {thr:.4f}")
    print(f"Realized return on test path: {rlr:.4f}")
