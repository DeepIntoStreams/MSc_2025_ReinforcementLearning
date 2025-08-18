import numpy as np
import pandas as pd
import pickle
import random
import os
seed = 40
random.seed(seed)
# === Load and clean data ===
prices = pd.read_csv("data/combined_prices.csv", index_col=0, parse_dates=True)
prices = prices[["AAPL", "MSFT", "TSLA"]].dropna()
returns = prices.pct_change().dropna()

# === Parameters ===
n_assets = 3
traj_len = 30
eval_traj_count = 1
train_traj_per_group = 3000
train_group_count = 20
lookback_days = 60

# === Paths ===
os.makedirs("data", exist_ok=True)


# === 1. Evaluation Set: last 31-day price window ===
eval_prices = prices.iloc[-31:].values
# raw_prices = prices.iloc[-31:].values
# eval_prices = raw_prices / raw_prices[0] * 100 # normalize per trajectory
eval_returns = returns.iloc[-30:].values

eval_trajectories = []
for _ in range(eval_traj_count):
    actions = np.zeros((traj_len + 1, n_assets))
    rewards = np.zeros(traj_len + 1)
    # put action and reward as 0
    #for k in range(1, traj_len + 1):
        #a = np.random.rand(n_assets)
        #a /= a.sum()
        #actions[k] = a
        #rewards[k] = np.dot(a, eval_returns[k - 1])
    dones = np.zeros(traj_len + 1, dtype=bool)
    dones[-1] = True
    traj = {
        "observations": eval_prices,
        "actions": actions,
        "rewards": rewards,
        "dones": dones
    }
    eval_trajectories.append(traj)

with open("data/real_stock_eval_31x50.pkl", "wb") as f:
    pickle.dump(eval_trajectories, f)
print(f"Saved evaluation trajectory.")

# # === 2. Validation Set ===
# eval_prices = prices.iloc[-62:-31].values
# # raw_prices = prices.iloc[-31:].values
# # eval_prices = raw_prices / raw_prices[0] * 100 # normalize per trajectory
# eval_returns = returns.iloc[-61:31].values

# eval_trajectories = []
# for _ in range(eval_traj_count):
#     actions = np.zeros((traj_len + 1, n_assets))
#     rewards = np.zeros(traj_len + 1)
#     # put action and reward as 0
#     #for k in range(1, traj_len + 1):
#         #a = np.random.rand(n_assets)
#         #a /= a.sum()
#         #actions[k] = a
#         #rewards[k] = np.dot(a, eval_returns[k - 1])
#     dones = np.zeros(traj_len + 1, dtype=bool)
#     dones[-1] = True
#     traj = {
#         "observations": eval_prices,
#         "actions": actions,
#         "rewards": rewards,
#         "dones": dones
#     }
#     eval_trajectories.append(traj)

# with open("data/validation.pkl", "wb") as f:
#     pickle.dump(eval_trajectories, f)
# print(f"Saved validation trajectory.")


# Allowing short
def random_actions(n_stocks):
    a = np.zeros(n_stocks)
    while True:
        # Control the weights in a reasonable range and sum up to 1
        weights = np.random.uniform(-1, 1, size=n_stocks-1)
        last_weight = 1 - np.sum(weights)
        # If not in the specified range, discard and try again
        if -1 <= last_weight <= 1:
            return np.concatenate([weights, [last_weight]])

# === 2. Training Set: 20 random historical windows (close to eval period) ===
eval_start_date = prices.index[-31]
train_cutoff_date = eval_start_date - pd.Timedelta(days=lookback_days)

# Collect valid window start indices
valid_indices = [
    i for i in range(len(prices) - 31)
    if train_cutoff_date <= prices.index[i + 30] < eval_start_date
]

# Sample 20 distinct windows
sampled_indices = random.sample(valid_indices, train_group_count)

# Generate 20 groups × 3000 trajectories = 60,000
all_train_groups = []

for i in sampled_indices:
    price_window = prices.iloc[i:i + 31].values
    # raw_prices = prices.iloc[i:i + 31].values
    # price_window = raw_prices / raw_prices[0] * 100 # normalize per trajectory

    return_window = returns.iloc[i + 1:i + 31].values
    group = []
    for _ in range(train_traj_per_group):
        actions = np.zeros((traj_len + 1, n_assets))
        rewards = np.zeros(traj_len + 1)
        for k in range(1, traj_len + 1):
            #a = np.random.rand(n_assets)
            #a /= a.sum()
            a = random_actions(n_assets)
            actions[k] = a
            rewards[k] = np.dot(a, return_window[k - 1])
        dones = np.zeros(traj_len + 1, dtype=bool)
        dones[-1] = True
        traj = {
            "observations": price_window,
            "actions": actions,
            "rewards": rewards,
            "dones": dones
        }
        all_train_groups.append(traj)
    

with open("data/real_stock_train_20groups_3000each.pkl", "wb") as f:
    pickle.dump(all_train_groups, f)
print(f"Saved 20 training groups of 3000 trajectories each.")

# === 3. Split into quality groups (expert / medium / poor) ===
# === Flatten all trajectories from 50 groups × 1000 into a single list ===
all_trajectories = all_train_groups

# === Define return computation method (multiplicative) ===
def compute_total_return(traj):
    rewards = traj["rewards"][1:]  # exclude t=0 reward
    return np.prod(1 + rewards) - 1

# ✅ Use the correct flat list
returns_with_traj = [(compute_total_return(traj), traj) for traj in all_train_groups]

# === Sort trajectories by return descending ===
returns_with_traj.sort(key=lambda x: x[0], reverse=True)

# === Split into quality groups (40% poor, 35% medium, 25% expert) ===
n_total = len(returns_with_traj)
n_10 = int(0.30 * n_total)
n_40 = int(0.50 * n_total)

poor_trajectories = [x[1] for x in returns_with_traj[n_10 + n_40:]]
medium_trajectories = [x[1] for x in returns_with_traj[n_10:n_10 + n_40]]
expert_trajectories = [x[1] for x in returns_with_traj[:n_10]]


# === Save as separate .pkl files ===
with open("data/expert.pkl", "wb") as f:
    pickle.dump(expert_trajectories, f)

with open("data/medium.pkl", "wb") as f:
    pickle.dump(medium_trajectories, f)

with open("data/poor.pkl", "wb") as f:
    pickle.dump(poor_trajectories, f)

print("Saved trajectories into separate files:")
print(f"   - expert.pkl: {len(expert_trajectories)} trajectories")
print(f"   - medium.pkl: {len(medium_trajectories)} trajectories")
print(f"   - poor.pkl: {len(poor_trajectories)} trajectories")
