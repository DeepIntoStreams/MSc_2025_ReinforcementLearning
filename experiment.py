import numpy as np
import torch
import pickle
import random
import os

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.offlinestockenv import OfflineStockEnv  # import the custom environment

seed = 40
random.seed(seed)

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    # ---- Load stock trajectories ----
    dataset_path = variant.get('dataset_path', 'data/simulated_stock_trajs_medium.pkl')
    assert os.path.exists(dataset_path), f"Dataset not found: {dataset_path}"
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    

    # ---- Load evaluation trajectories ----
    eval_dataset_path = variant.get('eval_dataset_path', 'data/simulated_stock_eval_31x50.pkl')
    assert os.path.exists(eval_dataset_path), f"Eval dataset not found: {eval_dataset_path}"
    with open(eval_dataset_path, 'rb') as f:
        eval_trajectories = pickle.load(f)

    # ---- Prepare data ----
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(np.prod(1 + path['rewards']) - 1) # product of return instead of sum
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # ---- Normalization ----
    states_concat = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states_concat, axis=0), np.std(states_concat, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)
    print('=' * 50)
    print(f'Starting new experiment: Offline Stock RL')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.4f}, std: {np.std(returns):.4f}')
    print(f'Max return: {np.max(returns):.4f}, min: {np.min(returns):.4f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    # The top pct of trajectories to train on
    pct_traj = variant.get('pct_traj', 1.)
    model_type = variant['model_type']

    # only train on top pct_traj trajectories (for %BC experiment)
    # updated to be the maximum allowed total timesteps to use
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    # gives indices that sort trajectories from lowest to highest return
    sorted_inds = np.argsort(returns)
    num_trajectories = 1
    # Start by selecting the single highest-return trajectory
    timesteps = traj_lens[sorted_inds[-1]]
    # Start checking the next-best (second-highest) trajectory, moving in reverse.
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    # ---- Infer dimensions ----
    state_dim = states_concat.shape[1]
    act_dim = trajectories[0]['actions'].shape[1]

    # ---- Define batch sampling ----
    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,
        )
        s, a, r, d, rtg, timesteps_arr, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = 0 # always start at time 0, use the full trajectories
            max_avail_len = traj['rewards'].shape[0] - si
            cur_len = min(max_len, max_avail_len)
            # Pad to max_len
            s_i = traj['observations'][si:si + cur_len].reshape(1, -1, state_dim)
            a_i = traj['actions'][si:si + cur_len].reshape(1, -1, act_dim)
            r_i = traj['rewards'][si:si + cur_len].reshape(1, -1, 1)
            if 'dones' in traj:
                d_i = traj['dones'][si:si + cur_len].reshape(1, -1)
            else:
                d_i = traj['terminals'][si:si + cur_len].reshape(1, -1)
            t_i = np.arange(si, si + cur_len).reshape(1, -1)

            # returns-to-go calculation
            rewards_segment = traj['rewards'][si:si + cur_len]
            rtg_seq = np.zeros(cur_len)
            
            for t in range(cur_len):
                # from current step to end of time cumulative return using product
                future_rewards = rewards_segment[t:]
                cumulative_return = np.prod(1 + future_rewards) - 1
                rtg_seq[t] = cumulative_return
            
            rtg_i = rtg_seq.reshape(1, -1, 1)
            
            # Padding (in the case si = 0, K = 31, and all trajectories are of length 31, there will be no padding)
            pad_len = max_len - cur_len
            s_i = np.concatenate([np.zeros((1, pad_len, state_dim)), s_i], axis=1)
            s_i = (s_i - state_mean) / state_std
            a_i = np.concatenate([np.ones((1, pad_len, act_dim)) * -10., a_i], axis=1)
            r_i = np.concatenate([np.zeros((1, pad_len, 1)), r_i], axis=1)
            d_i = np.concatenate([np.ones((1, pad_len)) * 2, d_i], axis=1)
            if rtg_i.shape[1] < max_len + 1:
                rtg_i = np.concatenate([rtg_i, np.zeros((1, max_len + 1 - rtg_i.shape[1], 1))], axis=1)
            rtg_i = np.concatenate([np.zeros((1, pad_len, 1)), rtg_i], axis=1)[:, :max_len+1, :]
            
            # Original Decision Transformer: simple scale division
            rtg_i = rtg_i / variant.get('scale', 1.0)
            t_i = np.concatenate([np.zeros((1, pad_len)), t_i], axis=1)
            mask_i = np.concatenate([np.zeros((1, pad_len)), np.ones((1, cur_len))], axis=1)
            s.append(s_i)
            a.append(a_i)
            r.append(r_i)
            d.append(d_i)
            rtg.append(rtg_i)
            timesteps_arr.append(t_i)
            mask.append(mask_i)
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps_arr = torch.from_numpy(np.concatenate(timesteps_arr, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        return s, a, r, d, rtg, timesteps_arr, mask
    
    print(f"Using evaluation dataset: {len(eval_trajectories)} trajectories")

    # ---- Evaluation function ----
    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            all_actions = []
            for traj in eval_trajectories:
                env = OfflineStockEnv(traj)
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length, actions_array = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            len(traj['observations']),
                            variant.get('scale', 1.0),
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            target_return=target_rew/variant.get('scale', 1.0),
                            mode=mode,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            len(traj['observations']),
                            device=device,
                            target_return=target_rew/variant.get('scale', 1.0),
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                        )
                        actions_array = None
                returns.append(ret)
                lengths.append(length)
                all_actions.append(actions_array)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
                #f'target_{target_rew}_actions': all_actions,
            }
        return fn

    # ---- Model selection ----
    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max(traj_lens),
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    # ---- Optimizer and scheduler ----
    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    # ---- Trainer ----
    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in variant.get('env_targets', [0.040])],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in variant.get('env_targets', [0.040])],
        )

    # ---- Training loop ----
    for i, traj in enumerate(trajectories):
        if len(traj['observations']) != len(traj['rewards']):
            print(f"Trajectory {i} has mismatched lengths: obs={len(traj['observations'])}, rewards={len(traj['rewards'])}")
    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        # Optionally log to wandb or elsewhere

    # === Check embedding weights ===
    if model_type == 'dt':
    
        print("RTG embedding weight norm:", model.embed_return.weight.norm().item())
        if hasattr(model.embed_return, "bias") and model.embed_return.bias is not None:
            print("RTG embedding bias norm:", model.embed_return.bias.norm().item())
        print("State embedding weight norm:", model.embed_state.weight.norm().item())
        print("Action embedding weight norm:", model.embed_action.weight.norm().item())
        

        with torch.no_grad():
            test_rtg = torch.tensor([[0.03], [0.05], [0.07]]).to(device)
            rtg_emb = model.embed_return(test_rtg)
            print("RTG embedding output range:", rtg_emb.min().item(), "to", rtg_emb.max().item())
            print("RTG embedding output std:", rtg_emb.std().item())
    # ===========================

    # After the training loop
    final_actions = None
    for eval_fn in trainer.eval_fns:
        outputs = eval_fn(trainer.model)
        for k, v in outputs.items():
            if k.endswith('_actions'):
                final_actions = v  # This is a list of arrays, one per trajectory
    
    if model_type == 'dt':
        target_str = "_".join([f"{tar:.3f}" for tar in variant.get('env_targets', [0.040])])
        model_filename = f'dt_model_target_{target_str}_scale_{variant.get("scale", 1.0)}.pth'
        torch.save(model.state_dict(), model_filename)
        print(f"Saved trained model to {model_filename}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data/simulated_stock_trajs_medium.pkl')
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=31)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--env_targets', nargs='+', type=float, default=[0.040])
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=1)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--scale', type=float, default=0.1)  
    parser.add_argument('--eval_dataset_path', type=str, default='data/simulated_stock_eval_31x50.pkl')
    args = parser.parse_args()
    experiment('stock-experiment', variant=vars(args))