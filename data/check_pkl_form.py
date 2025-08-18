import pickle
import numpy as np
import matplotlib.pyplot as plt

# Path to the saved pickle file
pkl_path = 'data/medium.pkl'

# Load the pickle file
with open(pkl_path, 'rb') as f:
    trajectories = pickle.load(f)

# Inspect the first trajectory to check the structure
print("First trajectory:", trajectories[0])

# Optionally, inspect the keys of the first trajectory's dictionary
print("Keys of the first trajectory:", trajectories[0].keys())

# Check the shape of observations, actions, rewards, and dones in the first trajectory
print("Observations shape:", trajectories[0]['observations'].shape)
print("Actions shape:", trajectories[0]['actions'].shape)
print("Rewards shape:", trajectories[0]['rewards'].shape)
print("Dones shape:", trajectories[0]['dones'].shape)

# Print the number of trajectories
print("Number of trajectories in file:", len(trajectories))


states, traj_lens, returns = [], [], []
for path in trajectories:
    states.append(path['observations'])
    traj_lens.append(len(path['observations']))
    returns.append(np.prod(1 + path['rewards'][1:]) - 1)
traj_lens, returns = np.array(traj_lens), np.array(returns)


print(len(traj_lens))

print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')

print("Training set return stats:")
print(f"Mean: {np.mean(returns):.4f}, Std: {np.std(returns):.4f}, Max: {np.max(returns):.4f}, Min: {np.min(returns):.4f}")
for thresh in [0.05, 0.06, 0.07]:
    print(f"Trajectories >= {thresh}: {np.sum(returns >= thresh)}")

print(f"Trajectories with return between 0.05 and 0.051: {np.sum((returns >= 0.049) & (returns < 0.051))}")

# === Plot histogram of returns ===
plt.figure(figsize=(8, 4))
plt.hist(returns, bins=50, alpha=0.75, color='steelblue', edgecolor='black')
plt.title("Distribution of Total Returns (Medium Group)")
plt.xlabel("Total Return (Sum of Daily Returns)")
plt.ylabel("Number of Trajectories")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# Save the plot
plt.savefig("data/medium_return_histogram.png", dpi=300)
