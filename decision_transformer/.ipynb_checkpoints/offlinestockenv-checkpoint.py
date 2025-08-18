import numpy as np

class OfflineStockEnv:
    """
    Minimal environment wrapper for offline stock trajectories.
    At each step, takes an action (portfolio weights), applies it to the price change,
    and returns the next observation, computed reward, and done flag.
    """
    def __init__(self, trajectory):
        self.prices = np.array(trajectory['observations'])  # Price matrix of shape (T+1, N)
        self.length = len(self.prices) - 1  # Number of timesteps: T = (T+1) - 1
        self.idx = 0  # Internal pointer to the current timestep

    def reset(self):
        """
        Reset the environment to the beginning of the trajectory.

        Returns:
        - The initial observation (prices at time 0)
        """
        self.idx = 0
        return self.prices[self.idx]

    def step(self, action):
        # action: portfolio weights, shape (N,)
        if self.idx >= self.length:
            # If already at the final timestep, return terminal state
            return self.prices[-1], 0.0, True, {}

        # Get current and next-day prices
        p_t = self.prices[self.idx]
        p_tp1 = self.prices[self.idx + 1]
        # Calculate per-asset returns from t to t+1
        returns = (p_tp1 - p_t) / p_t  
        # Calculate portfolio return as the dot product of weights and returns
        reward = np.dot(action, returns)
        self.idx += 1
        # done = true when reached the end of time window
        done = self.idx >= self.length
        next_obs = self.prices[self.idx]
        return next_obs, reward, done, {} # {} is an empty dictionary, representing the info field returned by step() in OpenAI Gym-style environments.