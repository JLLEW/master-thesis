import argparse
from stable_baselines3 import DQN, PPO, A2C
import utils
import train_models
import torch as th
import wandb
import copy
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import json
import pandas as pd
import random
from morl_baselines.common.evaluation import seed_everything

import warnings
warnings.filterwarnings("ignore")

def load_model(env, model_name, model_path, seed):
    model = None
    if model_name == "DQN":
        model = DQN.load(model_path, env=env, seed=seed)
    elif model_name == "PPO":
        model = PPO.load(model_path, env=env, seed=seed)
    elif model_name == "A2C":
        model = A2C.load(model_path, env=env, seed=seed)
    elif model_name == "PCN":
        model = th.load(model_path)
        pcn_model = train_models.create_model(env, model_name, seed, None)
        pcn_model.model = model
        model = pcn_model
    elif model_name == "Envelope":
        envelope_model = train_models.create_model(env, model_name, seed, None)
        envelope_model.load(model_path)
        model = envelope_model

    return model

def take_action(model_name, model, observation, episode_length, deterministic=True):
    if model_name in ["DQN", "PPO", "A2C"]:
        action = model.predict(observation, deterministic)[0]
    elif model_name == "PCN":
        desired_return = [240, -21]
        model.set_desired_return_and_horizon(desired_return, episode_length)
        action = model.eval(observation)
    else:
        action = model.eval(observation, np.array([0.99, 0.01]))

    return action

def init_wandb(model_name, scenario, seed):
     config = train_models.create_config(model_name, None)
     wandb.init(
        project=train_models.create_experiment_name(model_name, scenario, seed),
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
     
def compute_volatility(data, asset_names, allocations, i):
    vol = []
    for k in range(len(allocations)):
        assets_data = {}
        offset = 30

        # should be changed to log returns????
        # not prices
        for asset_name in asset_names:
            assets_data[asset_name] = data[asset_name].iloc[i - offset : i, 0].values

        #1 create covariance matrix
        cov_matrix = utils.get_covariance_matrix(assets_data)
        
        #2 compute variance
        w = allocations[k]
        p_var = cov_matrix.mul(w, axis=0).mul(w, axis=1).sum().sum()

        #3 take a square root of it
        p_sd = np.sqrt(p_var)

        # annualize
        annual_p_sd = p_sd * 16
        vol.append(annual_p_sd)

        i += 1
    return vol

def generate_evaluation_metrics(path, episode, data, initial_data_i, asset_names, episode_length):
    path = f"{path}/data_{episode}"
    allocations = []
    prices = []
    pnl = []
    # load data
    with open (f"{path}/allocations", 'rb') as fp:
        allocations = pickle.load(fp)
    with open (f"{path}/prices", 'rb') as fp:
        prices = pickle.load(fp)
    with open (f"{path}/pnl", 'rb') as fp:
        pnl = pickle.load(fp)

    portfolio_volatility = compute_volatility(data, asset_names, allocations, initial_data_i)

def save_metrics(pnl, avg_sharpe, avg_vol, cummaltive_rewads, path, episode):
    metrics = {
        'avg_sharpe': avg_sharpe,
        'avg_vol': avg_vol,
        'rewards': cummaltive_rewads,
        'pnl': pnl
    }

    with open(f"{path}/{episode}_metrics.txt", "w") as fp:
        fp.write(json.dumps(metrics))

class MetricsLogger:

    def __init__(self, env, path):
        self.actions = [0]
        self.allocations = [[0]*env.n_assets]
        self.portfolio_value = [env.init_cash]
        self.pnl = [0]
        self.portfolio_volatility = [0]
        self.env = env
        self.transaction_costs = [0]
        self.initial_data_i = env.data_i
        self.path = path
        self.profit_reward = [0]
        self.risk_reward = [0]

    def gather_metrics(self, action, rewards):
        self.actions.append(action)
        self.allocations.append(self.env.assets_allocation)
        self.transaction_costs.append(self.env.transaction_costs)
        self.portfolio_value.append(self.env.portfolio_value)
        self.pnl.append(self.env.daily_return)
        self.profit_reward.append(rewards[0])
        self.risk_reward.append(rewards[1])

    def save_metrics(self, x):
        df = {
            "portfolio value": self.portfolio_value,
            "pnl": self.pnl,
            "transaction costs": self.transaction_costs,
            "actions": self.actions,
            "profit reward": self.profit_reward,
            "risk reward": self.risk_reward
        }

        i = 0
        for data_name, data in self.env.data.items():
            df[f"{data_name} price"] = data.iloc[self.initial_data_i:self.initial_data_i + episode_length, 0].tolist()
            df[f"{data_name} allocation"] = [x[i] for x in self.allocations]
            i += 1
        
        df = pd.DataFrame(df)
        df.to_csv(f"{self.path}/evaluation_metrics_for_anova{x}.csv")

        
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", help="Model type e.g. DQN", type=str)
    parser.add_argument("--model-path", help="Path to the train model", type=str)
    parser.add_argument("--scenario", help="Trainig scenario to generate data", type=str)
    parser.add_argument("--ep-length", help="Episode length", type=int)
    parser.add_argument("--seed", help="Seed for experiments", type=int)

    args = parser.parse_args()

    train_models.register_environments()
    init_wandb(args.model_name, args.scenario, args.seed)
    _, data = utils.generate_crypto()
    #_, _, data = utils.generate_data_from_yfinance()
    assets = utils.get_assets_name_list(args.scenario)
    env = train_models.create_env(args.model_name, data, assets, args.ep_length, args.seed, False)
    initial_env_i = env.data_i
    model = load_model(env, args.model_name, args.model_path, args.seed)

    obs, _ = env.reset()
    episode_length = args.ep_length

    path = os.path.dirname(args.model_path)

    metrics_logger = MetricsLogger(env, path)
    seed_everything(args.seed)

    k = 0
    for i in range(30 * episode_length):
        action = take_action(args.model_name, model, obs, episode_length)
        obs, reward, terminated, _, _ = env.step(action)
        metrics_logger.gather_metrics(action, reward)
        if terminated:
            metrics_logger.save_metrics(k)
            env.reset()
            metrics_logger = MetricsLogger(env, path)
            k += 1
   
            

            
