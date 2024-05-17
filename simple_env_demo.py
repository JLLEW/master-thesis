import utils
import gymnasium as gym
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from wandb.integration.sb3 import WandbCallback
import wandb
import copy

if __name__ == '__main__':
    # generate data
    
    episode_length = 50
    time_steps = 120000
    n_episodes = 4

    register(
        id="simple-env-v0",
        entry_point="simple_env:SimpleTradingEnv",
    )
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": time_steps,
        "env_name": "simple-env-v0",
        }

    run = wandb.init(
        project="simple_env",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    logger_callback = WandbCallback(gradient_save_freq=100, model_save_path=f"models/{run.id}", verbose=2)


    #data, assets = utils.generate_train_data('single_w')
    data, assets = utils.generate_train_data('1bear_1bull')
    env = gym.make("simple-env-v0", asset_list = assets, asset_data = data, episode_length=episode_length)
    env.reset()
    model = DQN("MlpPolicy", env, learning_starts=500,  verbose=1, learning_rate=0.0005, tensorboard_log=f"runs/{id}", exploration_final_eps=0.05, exploration_fraction=0.5)
    model = model.learn(total_timesteps=time_steps, callback=logger_callback)

    data, assets = utils.generate_test_data('1bear_1bull')
    eval_env = gym.make("simple-env-v0", asset_list = assets, asset_data = data, episode_length=episode_length)
    obs, _ = eval_env.reset()



    alloc = [copy.deepcopy(eval_env.assets_allocation)]
    prices = [[eval_env.get_asset_price(i) for i in range(eval_env.n_assets)]]
    pnl = [eval_env.pnl]
    e = 0
    s = 0

    for _ in range(n_episodes * episode_length):
        a = model.predict(obs)
        obs, reward, terminated, _, _ = eval_env.step(a[0])
        print(f"==== episode {e} - step {s} =====")
        print(f"action taken: {a[0]}")
        print(f"reward returned: {reward}")
        print(f"portfolio value: {eval_env.portfolio_value}")
        print(f"shares: {eval_env.shares}")
        print(f"allocation: {eval_env.assets_allocation}")
        print(f"cash: {eval_env.cash}")
        print("--------------------------\n\n")

        alloc.append(copy.deepcopy(eval_env.assets_allocation))
        prices += [[eval_env.get_asset_price(i) for i in range(eval_env.n_assets)]]
        pnl += [eval_env.pnl]
        if terminated:
            eval_env.reset()
            s = 0
            for i in range(len(alloc[0])):
                fig, axs = plt.subplots(3)
                x = range(episode_length)
                axs[0].plot(x, [y[i] for y in prices])
                axs[1].plot(x, [y[i] for y in alloc])
                axs[2].plot(x, [y for y in pnl])
                stock_name = eval_env.asset_names[i]
                plt.savefig(f'simple_env_dqn_multi_1/{stock_name}_{e}.png')
            alloc = [copy.deepcopy(eval_env.assets_allocation)]
            prices = [[eval_env.get_asset_price(i) for i in range(eval_env.n_assets)]]
            pnl = [eval_env.pnl]
            e += 1
        s += 1

    