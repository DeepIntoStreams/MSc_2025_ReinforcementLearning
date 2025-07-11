from data_generation import generate_returns
from dpp_dist import dp_mv_distribution
from theoretical_and_realized import compute_theoretical_and_realized_returns
import numpy as np

# Define parameters
T, N = 30, 3
mu = np.array([0.001, 0.0015, 0.0012]) # Expected returns
# Covariance matrix
Sigma = np.array([
        [0.0001, 0.00002, 0.00001],
        [0.00002, 0.00015, 0.00003],
        [0.00001, 0.00003, 0.00012]
    ])
W0 = 1.0 # Initial wealth
lambda_ = 5.0 # Risk aversion parameter “mid‐range”
M = 50  # Number of independent test paths

# Generate M independent 30-day return paths for 3 assets from a multivariate Gaussian.
all_returns = generate_returns(T, N, mu, Sigma, num_samples=M)
print("Generated returns:", all_returns.shape)  # Should be (M, 30, 3)

# Compute the DP policy
policy, wealth_grid = dp_mv_distribution(mu, Sigma, lambda_, W0)

# Compute realized returns for each path
realized_returns = []
for i in range(M):
    returns = all_returns[i]
    theoretical_return, realized_return = compute_theoretical_and_realized_returns(
        policy, wealth_grid, mu, returns, W0=1.0
    )
    realized_returns.append(realized_return)

realized_returns = np.array(realized_returns)
mean_realized = np.mean(realized_returns)
std_error = np.std(realized_returns, ddof=1) / np.sqrt(M)

print(f"Average realized return over {M} paths: {mean_realized:.4f} ± {std_error:.4f}")
print(f"Theoretical expected return: {theoretical_return:.4f}, set this as the target-return-to-go for the DP policy.")
# Use the above "all_returns" for DP policy evaluation, generate another set for training.

# Compute theoretical and realized returns
# Note: W0 is the initial wealth, which is 1.0 in this case
#theoretical_return, realized_return = compute_theoretical_and_realized_returns(policy, wealth_grid, mu, returns, W0=1.0)
#print(f"Theoretical expected return: {theoretical_return:.4f}")
#print(f"Realized return on test path: {realized_return:.4f}")