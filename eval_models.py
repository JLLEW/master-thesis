import argparse
from stable_baselines3 import DQN
import utils
import train_models
import torch as th
import wandb
import copy
import matplotlib.pyplot as plt
import os
import pickle

def load_model(env, model_name, model_path, seed):
    model = None
    if model_name == "DQN":
        model = DQN.load(model_path, env=env, seed=seed)
    elif model_name == "PCN":
        model = th.load(model_path)
        pcn_model = train_models.create_model(env, model_name, seed, None)
        pcn_model.model = model
        model = pcn_model
    return model

def take_action(model_name, model, observation, deterministic=True):
    if model_name == "DQN":
        action = model.predict(observation, deterministic)
    elif model_name == "PCN":
        desired_return = [200, 0]
        model.set_desired_return_and_horizon(desired_return, 99)
        action = model.eval(observation)

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
     
def save_episode_data(allocations, prices, pnl, path, episode):
    path = f"{path}/data_{episode}"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f"{path}/allocations", 'wb') as fp:
        pickle.dump(allocations, fp)
    with open(f"{path}/prices", 'wb') as fp:
        pickle.dump(prices, fp)
    with open(f"{path}/pnl", 'wb') as fp:
        pickle.dump(pnl, fp)

def generate_graphs(env, allocations, prices, pnl, path, episode, episode_length=100):
    if not os.path.exists(f"{path}/graphs"):
        os.makedirs(f"{path}/graphs")
    for i in range(len(allocations[0])):
                fig, axs = plt.subplots(3)
                x = range(episode_length)
                axs[0].plot(x, [y[i] for y in prices])
                axs[1].plot(x, [y[i] for y in allocations])
                axs[2].plot(x, [y for y in pnl])
                stock_name = env.asset_names[i]
                plt.savefig(f'{path}/graphs/{stock_name}_{episode}.png')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", help="Model type e.g. DQN", type=str)
    parser.add_argument("--model-path", help="Path to the train model", type=str)
    parser.add_argument("--scenario", help="Trainig scenario to generate data", type=str)
    parser.add_argument("--seed", help="Seed for experiments", type=int)

    args = parser.parse_args()

    train_models.register_environments()
    init_wandb(args.model_name, args.scenario, args.seed)
    data, assets = utils.generate_test_data(args.scenario, args.seed)
    env = train_models.create_env(args.model_name, data, assets, args.seed)

    model = load_model(env, args.model_name, args.model_path, args.seed)

    obs, _ = env.reset(seed=args.seed)
    episode_length = 100
    n_episodes = 3
    episodes_seeds = [57, 28, 44]
    e = 0

    allocations = [copy.deepcopy(env.assets_allocation)]
    prices = [[env.get_asset_price(i) for i in range(env.n_assets)]]
    pnl = [env.pnl]
    actions = [0]
    path = os.path.dirname(args.model_path)

    for _ in range(n_episodes * episode_length):
        action = take_action(args.model_name, model, obs)
        obs, reward, terminated, _, _ = env.step(action)

        # save data
        allocations.append(copy.deepcopy(env.assets_allocation))
        prices += [[env.get_asset_price(i) for i in range(env.n_assets)]]
        pnl += [env.pnl]
        actions += [action]

        if terminated:
            env.reset(seed=episodes_seeds[e])
            save_episode_data(allocations, prices, pnl, path, e)
            generate_graphs(env, allocations, prices, pnl, path, e)
            allocations = [copy.deepcopy(env.assets_allocation)]
            prices = [[env.get_asset_price(i) for i in range(env.n_assets)]]
            pnl = [env.pnl]
            actions = [0]
            e += 1
