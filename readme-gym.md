
# OpenAI Gym

## Installation

Create the experiment environment with the required packages using the following command

```
conda env create -f conda_env.yml
```

## Datasets

Datasets are stored in the `data` directory.

For synthetic data, run the following command to obtain the data

```
python traj_fromprice.py
```

For empirical data, run the following command

```
python empirical_traj.py
```

## Example usage

Experiments can be reproduced with the following (synthetic case):

```
python experiment.py --dataset_path data/simulated_stock_trajs_medium.pkl --eval_dataset_path data/simulated_stock_eval_31x50.pkl --model_type dt --max_iters 20 --num_steps_per_iter 2000 --batch_size 32 --K 31 --embed_dim 128 --n_layer 3 --n_head 1 --scale 0.1 --learning_rate 1e-4 --device cuda --env_targets 0.02 0.04 0.05 0.066
```

For evaluation and experiment details see the notebooks mvp_dt.ipynb and empirical_exp.ipynb respectively.
