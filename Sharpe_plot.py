import matplotlib.pyplot as plt

# Data
baselines = {
    "Static MVP": 0.1761,
    "Riskfolio": 0.3665
}

dt_targets = {
    0.09: 0.1724,
    0.10: 0.1757,
    0.13: 0.1783,
    0.15: 0.1735,
    0.17: 0.1618
}

# Plot
plt.figure(figsize=(8, 5))

# Plot DT Sharpe Ratios
plt.plot(list(dt_targets.keys()), list(dt_targets.values()), 
         marker="o", linestyle="-", label="Decision Transformer")

# Plot baselines as horizontal lines
for method, sharpe in baselines.items():
    plt.axhline(y=sharpe, linestyle="--", label=f"{method} (Sharpe={sharpe:.3f})")

# Labels and title
plt.xlabel("RTG Target")
plt.ylabel("Sharpe Ratio")
plt.title("Sharpe Ratio Comparison: DT vs Baselines")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.show()
