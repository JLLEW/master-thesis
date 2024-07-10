import argparse
import gymnasium as gym
import mo_gymnasium
from gymnasium.envs.registration import register
from wandb.integration.sb3 import WandbCallback
import wandb
from stable_baselines3 import DQN, PPO, A2C
import datetime
import json
import utils
import numpy as np
import time

def create_experiment_name(model_name, scenario, seed):
    today = datetime.datetime.now()
    today = today.strftime("%x").replace("/", "_")

    experiment_name = f"{model_name}_{scenario}_{seed}_{today}"
    return experiment_name

def create_config(model_name, experiment_name):
    hyperparameter_defaults = None
    if model_name == "DQN":
        hyperparameter_defaults = {
            'dqn_time_steps' : 1e6,
            'dqn_learning_starts': 500,
            'dqn_batch_size': 32,
            'dqn_learning_rate': 5e-4,
            'dqn_exploration_final_eps': 0.04,
            'dqn_exploration_fraction': 0.4
        }

    elif model_name == "A2C":
        hyperparameter_defaults = {
            'a2c_time_steps' : 200000,
            'a2c_n_steps': 10,
            'a2c_learning_rate': 3e-4,
        }

    elif model_name == "PPO":
        hyperparameter_defaults = {
            'ppo_time_steps' : 1e6,
            'ppo_n_steps': 100,
            'ppo_batch_size': 64,
            'ppo_learning_rate': 1e-3,
        }

    elif model_name == "PCN":
        hyperparameter_defaults = {
            'pcn_time_steps': 100000,
            'pcn_sf': [1, 0.01, 0.01],
            'pcn_learning_rate': 7e-3,
            'pcn_gamma': 1.0,
            'pcn_batch_size': 256,
            'pcn_hidden_dim': 64,
            'pcn_ref_point': [-15, -5],
            'pcn_num_er_episodes': 50,
            'pcn_num_step_episodes': 10,
            'pcn_num_model_updates': 2,
            'pcn_max_buffer_size': 365
        }
    elif model_name == "Envelope":
        hyperparameter_defaults = {
            'time_steps': 10000,
            'sf': [0.0001, 0.1, 0.01],
            'learning_rate': 7e-3,
            'gamma': 1.0,
            'batch_size': 256,
            'hidden_dim': 64,
            'ref_point': [-25, -10],
            'num_er_episodes': 20,
            'num_step_episodes': 30,
            'num_model_updates': 20,
            'max_buffer_size': 450,
            'seed': 31,
            'max_return': [1200,0]
        }

    hyperparameter_defaults['experiment_name'] = experiment_name
    
    return hyperparameter_defaults

def initialize_logger(experiment_name, config):

    wandb_run = wandb.init(
        project=experiment_name,
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    save_path = f"models/{experiment_name}_{wandb_run.id}"
    logger_callback = WandbCallback(gradient_save_freq=100, model_save_freq=1e4, model_save_path=save_path, verbose=2)

    return logger_callback, wandb_run.id, save_path

def register_environments():

    # single objective env
    register(
        id="simple-env-v0",
        entry_point="simple_env:SimpleTradingEnv",
    )

    # multi objective env
    register(
        id="simple-multienv-v0",
        entry_point="simple_env_multi:SimpleTradingEnv",
    )

def create_env(model_name, data, assets, ep_length=365, seed=None, eval=False):
    single_objective_models = ['DQN', 'PPO', 'A2C']
    multi_objective_models = ['PCN', 'Envelope']
    env = None
    
    if model_name in single_objective_models:
        env = gym.make(
            "simple-env-v0",
            asset_list = assets,
            asset_data = data,
            episode_length=ep_length,
            seed=seed,
            eval=eval
            )
    elif model_name in multi_objective_models:
        env = mo_gymnasium.make(
            "simple-multienv-v0",
            asset_list = assets,
            asset_data = data,
            episode_length=ep_length,
            eval=eval)
    else:
        raise NameError(f"{model} does not exist")
    
    return env

def create_model(env, model_name, seed, id):
    model = None

    if model_name == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            learning_starts=wandb.config['dqn_learning_starts'],
            verbose=1,
            batch_size=wandb.config['dqn_batch_size'],
            learning_rate=wandb.config['dqn_learning_rate'],
            tensorboard_log=f"runs/{id}",
            exploration_final_eps=wandb.config['dqn_exploration_final_eps'],
            exploration_fraction=wandb.config['dqn_exploration_fraction'],
            seed=seed
        )

    elif model_name == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=wandb.config['ppo_n_steps'],
            batch_size=wandb.config['ppo_batch_size'],
            learning_rate=wandb.config['ppo_learning_rate'],
            tensorboard_log=f"runs/{id}",
            verbose=1,
            seed=seed
        )
    elif model_name == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            n_steps=wandb.config['a2c_n_steps'],
            learning_rate=wandb.config['a2c_learning_rate'],
            tensorboard_log=f"runs/{id}",
            verbose=1,
            seed=seed
        )

    elif model_name == "PCN":
        from morl_baselines.multi_policy.pcn.pcn import PCN
        model = PCN(
            env,
            scaling_factor=np.array(wandb.config['pcn_sf']),
            seed=seed,
            wandb_entity='thesis-vu',
            device='mps'
        )
    elif model_name == "Envelope":
        from morl_baselines.multi_policy.envelope.envelope import Envelope
        model = Envelope(
        env,
        max_grad_norm=0.1,
        learning_rate=7e-4,
        gamma=1.0,
        batch_size=64,
        net_arch=[64, 64, 64],
        buffer_size=int(1e5),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=20000,
        initial_homotopy_lambda=0.0,
        final_homotopy_lambda=1.0,
        homotopy_decay_steps=10000,
        learning_starts=100,
        envelope=True,
        gradient_updates=1,
        target_net_update_freq=1000,  # 1000,  # 500 reduce by gradient updates
        tau=1,
        log=True,
        wandb_entity='thesis-vu'
    )

    return model

def save_config(path, config):
    with open(f"{path}/config.txt", 'w') as config_file: 
     config_file.write(json.dumps(config))
     print(f"config saved to {path}")

def train_model(model_name, model, logger_callback, path):
    if model_name == "DQN":
        model.learn(
            total_timesteps=wandb.config['dqn_time_steps'],
            callback=logger_callback
            )
    elif model_name == "PPO":
        model.learn(
            total_timesteps=wandb.config['ppo_time_steps'],
            callback=logger_callback
        )
    elif model_name == "A2C":
        model.learn(
            total_timesteps=wandb.config['a2c_time_steps'],
            callback=logger_callback
        )
    elif model_name == "PCN":
        model.train(
            wandb.config['pcn_time_steps'],
            env,
            ref_point=np.array(wandb.config['pcn_ref_point']),
            num_er_episodes=wandb.config['pcn_num_er_episodes'],
            num_step_episodes=wandb.config['pcn_num_step_episodes'],
            num_model_updates=wandb.config['pcn_num_model_updates'],
            max_buffer_size = wandb.config['pcn_max_buffer_size']
            )
        model.save(savedir = path)
        print(f"desired return: {model.desired_return}")
        print(f"desired horizon: {model.desired_horizon}")

if __name__ == '__main__':
    start = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", help="Model type e.g. DQN", type=str)
    parser.add_argument("--scenario", help="Trainig scenario to generate data", type=str)
    parser.add_argument("--ep-length", help="Length of an episode", type=int)
    parser.add_argument("--seed", help="Seed for experiments", type=int)


    args = parser.parse_args()

    register_environments()
    experiment_name = create_experiment_name(args.model_name, args.scenario, args.seed)
    config = create_config(args.model_name, experiment_name)
    
    logger_callback, id, save_path = initialize_logger(experiment_name, config)
    # data, _, _ = utils.generate_data_from_yfinance()
    data, _, _ = utils.generate_crypto()
    assets = utils.get_assets_name_list("crypto")
    env = create_env(args.model_name, data, assets, args.ep_length, args.seed)
    model = create_model(env, args.model_name, args.seed, id)
    train_model(args.model_name, model, logger_callback, save_path)
    save_config(save_path, config)

    print("--- %s seconds ---" % (time.time() - start))
