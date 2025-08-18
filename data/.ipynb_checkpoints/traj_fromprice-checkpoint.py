import numpy as np
import pickle
from price_simulation import simulate_price_paths
from data_generation import generate_returns

# Simulate price paths for a fixed-length trajectory of 30 days
T, N = 30, 3
mu = np.array([0.001, 0.0015, 0.0012])
Sigma = np.array([
    [0.0001, 0.00002, 0.00001],
    [0.00002, 0.00015, 0.00003],
   [0.00001, 0.00003, 0.00012]
])
P0 = np.array([100, 120, 80])
num_sets = 50
num_trajectories = 1000

def random_actions(n_steps, n_stocks):
    a = np.zeros((n_steps, n_stocks))
    
    for i in range(n_steps):
        while True:
            # Control the weights in a reasonable range and sum up to 1
            weights = np.random.uniform(-1, 1, size=n_stocks-1)
            last_weight = 1 - np.sum(weights)
            # If not in the specified range, discard and try again
            if -1 <= last_weight <= 1:
                a[i] = np.concatenate([weights, [last_weight]])
                break
    
    return a

trajectories = []

# === Use a different seed for training set ===
train_seed = 0

# Simulate 50 sets of returns and prices
price_sets, return_sets = simulate_price_paths(T, N, mu, Sigma, P0, seed=train_seed, num_samples=num_sets)

for set_idx in range(num_sets):
    price_31 = price_sets[set_idx]
    returns_30 = return_sets[set_idx]
    for _ in range(num_trajectories):
        A = random_actions(T + 1, N)
        A[0] = 0
        R = (A[1:] * returns_30).sum(axis=1)
        # Set reward at time 0 as 0
        R = np.insert(R, 0, 0)
        D = np.zeros(T + 1, dtype=bool)
        # Done = True at end of time
        D[-1] = True
        obs = price_31
        trajectories.append({
            'observations': obs,
            'actions':      A,
            'rewards':      R,
            'dones':        D
        })

# Compute compounded returns for all trajectories
returns = np.array([np.prod(1 + traj['rewards'][1:]) - 1 for traj in trajectories])

# Sort indices by return
sorted_inds = np.argsort(returns)

# Define splits 
n = len(trajectories)
medium_start = int(0.3 * n)
medium_end = int(0.7 * n)
expert_start = int(0.9 * n)

# Medium: 30th–70th percentile
medium_inds = sorted_inds[medium_start:medium_end]
medium_trajectories = [trajectories[i] for i in medium_inds]

# Expert: top 10%
expert_inds = sorted_inds[expert_start:]
expert_trajectories = [trajectories[i] for i in expert_inds]

# Medium-Expert: top 40%
medexp_start = int(0.6 * n)
medium_expert_inds = sorted_inds[medexp_start:]
medium_expert_trajectories = [trajectories[i] for i in medium_expert_inds]

# Save each split
with open('data/simulated_stock_trajs_medium.pkl', 'wb') as f:
    pickle.dump(medium_trajectories, f)
with open('data/simulated_stock_trajs_expert.pkl', 'wb') as f:
    pickle.dump(expert_trajectories, f)
with open('data/simulated_stock_trajs_medium_expert.pkl', 'wb') as f:
    pickle.dump(medium_expert_trajectories, f)

print(f"Saved {len(medium_trajectories)} medium, {len(expert_trajectories)} expert, and {len(medium_expert_trajectories)} medium-expert trajectories.")


################################################################################################
# --- Generate evaluation set with a fixed seed ---
# Use seed = 42 to align with the mvp experiment
eval_seed = 42
eval_num_sets = 50
eval_num_trajectories = 1  # One trajectory per set of return for evaluation, 50 trajectories in total

eval_price_sets, eval_return_sets = simulate_price_paths(
    T, N, mu, Sigma, P0, seed=eval_seed, num_samples=eval_num_sets
)

eval_trajectories = []
for set_idx in range(eval_num_sets):
    price_31 = eval_price_sets[set_idx]
    returns_30 = eval_return_sets[set_idx]
    for _ in range(eval_num_trajectories):
        A = random_actions(T + 1, N)
        A[0] = 0
        R = (A[1:] * returns_30).sum(axis=1)
        R = np.insert(R, 0, 0)
        D = np.zeros(T + 1, dtype=bool)
        D[-1] = True
        obs = price_31
        eval_trajectories.append({
            'observations': obs,
            'actions':      A,
            'rewards':      R,
            'dones':        D
        })

with open('data/simulated_stock_eval_31x50.pkl', 'wb') as f:
    pickle.dump(eval_trajectories, f)

print(f"Saved {len(eval_trajectories)} evaluation trajectories of length 31.")

# For training set
all_train_returns = np.concatenate(return_sets, axis=0)  # shape: (num_sets * T, N)
print("Training set per-step returns:")
print(f"  Mean: {all_train_returns.mean():.6f}")
print(f"  Std: {all_train_returns.std():.6f}")
print(f"  Max: {all_train_returns.max():.6f}")
print(f"  Min: {all_train_returns.min():.6f}")

# For evaluation set
all_eval_returns = np.concatenate(eval_return_sets, axis=0)  # shape: (eval_num_sets * T, N)
print("Evaluation set per-step returns:")
print(f"  Mean: {all_eval_returns.mean():.6f}")
print(f"  Std: {all_eval_returns.std():.6f}")
print(f"  Max: {all_eval_returns.max():.6f}")
print(f"  Min: {all_eval_returns.min():.6f}")