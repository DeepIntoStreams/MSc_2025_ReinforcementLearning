def project_sum_to_one(action):
    # action: (..., act_dim)
    # Projects action to the affine hyperplane sum(action) = 1
    return action - action.mean(dim=-1, keepdim=True) + 1.0 / action.shape[-1]

def project_portfolio_weights(action, max_iter=100):
    import torch
    # Ensures action is a PyTorch tensor. If it's a NumPy array or list, it's converted to a tensor.
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action)
    
    # Makes a clone of the input action tensor into x
    x = action.clone()
    # Initializes an auxiliary variable y, same shape as x, filled with zeros. This will store the cumulative correction.
    y = torch.zeros_like(x)
    
    for _ in range(max_iter):
        # Performs projection onto the box constraints [-1, 1]
        x_new = torch.clamp(x + y, min=-1.0, max=1.0)
        y = x + y - x_new
        # Projects x_new onto the sum-to-1 hyperplane.
        x = x_new - x_new.mean(dim=-1, keepdim=True) + 1.0 / x_new.shape[-1]
        if torch.norm(x - x_new) < 1e-6:
            break
    
    return x