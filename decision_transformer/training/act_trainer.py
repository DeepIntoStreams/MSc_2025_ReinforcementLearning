# ActTrainer remains mostly the same, but we ensure it doesn't expect environment interaction
import numpy as np
import torch

from decision_transformer.training.trainer import Trainer
class ActTrainer(Trainer):
    def train_step(self):
        # Use the offline data batch
        states, actions, rewards, dones, rtg, _, attention_mask = self.get_batch(self.batch_size)
        
        # Model predictions
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, attention_mask=attention_mask, target_return=rtg[:, 0],
        )

        # Loss calculation (same as before)
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)
        action_target = actions.reshape(-1, act_dim)

        loss = self.loss_fn(state_preds, action_preds, reward_preds, state_preds, action_target, reward_preds)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
